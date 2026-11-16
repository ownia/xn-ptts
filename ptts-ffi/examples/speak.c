// Speak a line of text into a WAV file through the C interface, ../include/ptts.h.
//
//     speak [--quant auto|q8|q4k|q4|f32] [--rtf] MODEL LANG TEXT OUT.wav
//           [VOICE [TEMPERATURE [SEED]]]
//
// MODEL is a checkpoint folder, or on Apple platforms a Core ML bundle. ../README.md shows how to
// build the library and this program. A TEMPERATURE or SEED other than the model's gives another
// take of the same text.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ptts.h"

#define SAMPLE_RATE 24000

// Write `v` as `bytes` little-endian bytes, which WAV files use on every platform.
static void put(uint8_t *p, uint32_t v, int bytes) {
    for (int i = 0; i < bytes; i++) p[i] = (uint8_t)(v >> (8 * i));
}

// The 44-byte header of a 16-bit mono WAV file holding `samples` samples.
static void wav_header(uint8_t h[44], uint32_t samples) {
    memcpy(h, "RIFF", 4);
    put(h + 4, 36 + samples * 2, 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    put(h + 16, 16, 4);               // the size of the format chunk
    put(h + 20, 1, 2);                // PCM
    put(h + 22, 1, 2);                // one channel
    put(h + 24, SAMPLE_RATE, 4);
    put(h + 28, SAMPLE_RATE * 2, 4);  // bytes per second
    put(h + 32, 2, 2);                // bytes per sample
    put(h + 34, 16, 2);               // bits per sample
    memcpy(h + 36, "data", 4);
    put(h + 40, samples * 2, 4);
}

// Called with the audio as it is made, on the thread that called ptts_speak. A player would queue
// it here; this appends it to the file as 16-bit samples. Returning false stops early.
static bool on_audio(const float *pcm, size_t n, void *user) {
    FILE *out = user;
    for (size_t i = 0; i < n; i++) {
        float s = pcm[i] < -1.0f ? -1.0f : pcm[i] > 1.0f ? 1.0f : pcm[i];
        uint8_t b[2];
        put(b, (uint16_t)(int16_t)(s * 32767.0f), 2);
        if (fwrite(b, 1, 2, out) != 2) return false;
    }
    return true;
}

// Print a list from ptts_voices or ptts_languages: names separated by NULs, ending in a second NUL.
static void print_names(const char *names) {
    for (const char *n = names; *n; n += strlen(n) + 1) fprintf(stderr, "  %s\n", n);
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s [--quant auto|q8|q4k|q4|f32] [--rtf] MODEL LANG TEXT OUT.wav "
            "[VOICE [TEMPERATURE [SEED]]]\n", program);
}

int main(int argc, char **argv) {
    const char *positionals[7];
    size_t positional_count = 0;
    const char *quantization = "auto";
    bool show_rtf = false;
    bool parse_options = true;
    for (int argument_index = 1; argument_index < argc; argument_index++) {
        const char *argument = argv[argument_index];
        if (parse_options && strcmp(argument, "--") == 0) {
            parse_options = false;
        } else if (parse_options && strcmp(argument, "--rtf") == 0) {
            show_rtf = true;
        } else if (parse_options && strcmp(argument, "--quant") == 0) {
            if (++argument_index == argc) {
                fprintf(stderr, "--quant requires auto, q8, q4k (or q4), or f32\n");
                return 2;
            }
            quantization = argv[argument_index];
            if (strcmp(quantization, "auto") && strcmp(quantization, "q8") &&
                strcmp(quantization, "q8_0") && strcmp(quantization, "q4") &&
                strcmp(quantization, "q4k") && strcmp(quantization, "f32")) {
                fprintf(stderr, "unsupported quantization %s\n", quantization);
                return 2;
            }
        } else if (parse_options && strcmp(argument, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (parse_options && strncmp(argument, "--", 2) == 0) {
            fprintf(stderr, "unknown option %s\n", argument);
            return 2;
        } else {
            if (positional_count == sizeof positionals / sizeof positionals[0]) {
                usage(argv[0]);
                return 2;
            }
            positionals[positional_count++] = argument;
        }
    }
    if (positional_count < 4) {
        usage(argv[0]);
        return 2;
    }
    const char *model = positionals[0], *lang = positionals[1];
    const char *text = positionals[2], *path = positionals[3];

#ifdef __APPLE__
    uint32_t unit = PTTS_UNIT_ANE;  // the Neural Engine
#else
    uint32_t unit = PTTS_UNIT_CPU;  // the only unit elsewhere
#endif
    // The slow call. An app loads once and keeps the handle.
    PttsHandle *h = ptts_new_with_quant(model, unit, lang, quantization);
    if (!h) {
        fprintf(stderr, "cannot load %s: %s\n", model, ptts_last_error(NULL));
        return 1;
    }
    if (positional_count >= 5 && !ptts_set_voice(h, positionals[4])) {
        fprintf(stderr, "%s. The voices are:\n", ptts_last_error(h));
        print_names(ptts_voices(h));
        ptts_free(h);
        return 1;
    }
    char *end;
    if (positional_count >= 6) {
        float temperature = strtof(positionals[5], &end);
        if (end == positionals[5] || *end || !ptts_set_temperature(h, temperature)) {
            fprintf(stderr, "bad temperature %s\n", positionals[5]);
            ptts_free(h);
            return 1;
        }
    }
    if (positional_count == 7) {
        unsigned long long seed = strtoull(positionals[6], &end, 10);
        // strtoull reads "-1" as the largest seed rather than refusing it.
        if (end == positionals[6] || *end || positionals[6][0] == '-') {
            fprintf(stderr, "bad seed %s\n", positionals[6]);
            ptts_free(h);
            return 1;
        }
        ptts_set_seed(h, seed);
    }

    FILE *out = fopen(path, "wb");
    if (!out) {
        perror(path);
        ptts_free(h);
        return 1;
    }
    // The length is not known yet: write the header again once it is.
    uint8_t header[44];
    wav_header(header, 0);
    fwrite(header, 1, sizeof header, out);

    PttsResult result;
    bool ok = ptts_speak(h, text, on_audio, out, &result);
    if (!ok) {
        fprintf(stderr, "cannot speak: %s\n", ptts_last_error(h));
    } else {
        // This needs OUT.wav to be a file: a pipe cannot seek back to the header.
        wav_header(header, (uint32_t)result.samples);
        if (fseek(out, 0, SEEK_SET) != 0 || fwrite(header, 1, sizeof header, out) != sizeof header) {
            perror(path);
            ok = false;
        }
    }
    int write_error = ferror(out);
    if (fclose(out) != 0 || write_error) {
        perror(path);
        ok = false;
    }
    if (ok) {
        double audio_seconds = (double)result.samples / SAMPLE_RATE;
        printf("%s: %.1f s of audio\n", path, audio_seconds);
        if (show_rtf) {
            if (result.samples) {
                printf("RTF: %.3f", result.total_ms / (audio_seconds * 1000.0));
            } else {
                printf("RTF: n/a");
            }
            printf(" (%.1f ms synthesis, %.1f ms TTFA, %u frames, %zu samples)\n",
                   result.total_ms, result.ttfa_ms, result.frames, result.samples);
        }
    }
    ptts_free(h);
    return ok ? 0 : 1;
}
