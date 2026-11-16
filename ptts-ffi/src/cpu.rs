//! The engine everywhere but Apple: [`ptts::synth::Synth`] on the CPU, with `dir` a checkpoint
//! folder as the other frontends read it.
//!
//! The folder holds `tokenizer.json`, the weights (q8_0 or q4k GGUF, or f32 safetensors, named as in
//! [`WEIGHTS`]), its own `config.json` and its voices
//! in `voices/` or `embeddings/`, or as `default-voice.safetensors`.

use crate::{PTTS_UNIT_CPU, PttsResult};
use ptts::checkpoint::read_config;
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth, SynthBuilder};
use std::path::{Path, PathBuf};

/// Supported weight files. Automatic selection keeps the original q8-to-f32 order.
const WEIGHTS: [(&str, Quant); 3] = [
    ("model.q8.gguf", Quant::Q80),
    ("model.safetensors", Quant::F32),
    ("model.q4k.gguf", Quant::Q4k),
];

fn select_weights(dir: &Path, quantization: &str) -> Result<(PathBuf, Quant), String> {
    let candidates = match quantization {
        "auto" => &WEIGHTS[..2],
        "q8" | "q8_0" => &WEIGHTS[..1],
        "f32" => &WEIGHTS[1..2],
        "q4" | "q4k" => &WEIGHTS[2..],
        _ => {
            return Err(format!(
                "unsupported quantization {quantization:?}; expected auto, q8, q4k (or q4), or f32"
            ));
        }
    };
    candidates
        .iter()
        .map(|&(name, quant)| (dir.join(name), quant))
        .find(|(path, _)| path.is_file())
        .ok_or_else(|| {
            format!(
                "no weights in {}; expected one of {}",
                dir.display(),
                candidates.iter().map(|(name, _)| *name).collect::<Vec<_>>().join(", ")
            )
        })
}

pub struct Engine {
    synth: Synth,
    voices: Vec<String>,
    languages: Vec<String>,
    voice: Option<String>,
    /// Left `None`, the builder's defaults.
    temperature: Option<f32>,
    seed: Option<u64>,
    /// Applied here rather than by `synth`, which is built with `Normalize::OFF`, so that
    /// `set_language` can change it without a reload. `Synth` applies it first thing anyway.
    normalize: Normalize,
    /// Samples per frame, to count frames in what the stream hands back.
    frame_size: usize,
}

impl Engine {
    pub fn open(dir: &Path, unit: u32, lang: &str) -> Result<Self, String> {
        Self::open_with_quant(dir, unit, lang, "auto")
    }

    pub fn open_with_quant(
        dir: &Path,
        unit: u32,
        lang: &str,
        quantization: &str,
    ) -> Result<Self, String> {
        if unit != PTTS_UNIT_CPU {
            return Err(format!("compute unit {unit} is Apple only; use PTTS_UNIT_CPU"));
        }
        let normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        let config = dir.join("config.json");
        let cfg = read_config(&config).map_err(|e| e.to_string())?;
        let (weights, quant) = select_weights(dir, quantization)?;
        let frame_rate = cfg.mimi.frame_rate;
        let mut builder = SynthBuilder::new(cfg, weights, Normalize::OFF)
            .tokenizer_file(dir.join("tokenizer.json"))
            .device(DeviceKind::Cpu)
            .quant(quant);
        for (name, path) in ptts::loader::checkpoint_voices(dir) {
            builder = builder.add_voice(name, path);
        }
        let synth = builder.build().map_err(|e| e.to_string())?;
        let frame_size = (synth.sample_rate() as f64 / frame_rate).round() as usize;
        Ok(Self {
            voices: synth.voices(),
            languages: synth.languages().to_vec(),
            voice: synth.default_voice(),
            temperature: None,
            seed: None,
            normalize,
            synth,
            frame_size,
        })
    }

    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    /// Switch to one of [`Self::voices`].
    pub fn set_voice(&mut self, name: &str) -> Result<(), String> {
        self.voice = Some(name.to_string());
        Ok(())
    }

    /// Normalize text as `lang` from the next utterance on.
    pub fn set_language(&mut self, lang: &str) -> Result<(), String> {
        self.normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Sample at `temperature`, already checked, from the next utterance on.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = Some(temperature);
    }

    /// Seed the sampling from the next utterance on.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    pub fn speak(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<PttsResult, String> {
        let start = std::time::Instant::now();
        let opts = SpeechOptions {
            voice: self.voice.clone(),
            temperature: self.temperature,
            seed: self.seed,
            ..Default::default()
        };
        let text = self.normalize.apply(text);
        let stream = self.synth.stream_with(&text, &opts).map_err(|e| e.to_string())?;
        let (mut samples, mut ttfa) = (0, None);
        // Dropping the stream on an early return stops generation.
        for pcm in stream {
            let pcm = pcm.map_err(|e| e.to_string())?;
            if pcm.is_empty() {
                continue;
            }
            ttfa.get_or_insert(start.elapsed());
            samples += pcm.len();
            if !sink(&pcm) {
                break;
            }
        }
        Ok(PttsResult {
            frames: samples.div_ceil(self.frame_size) as u32,
            samples,
            ttfa_ms: ttfa.unwrap_or_default().as_secs_f64() * 1e3,
            total_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PTTS_UNIT_ANE;

    fn open_error(unit: u32, lang: &str) -> String {
        match Engine::open(Path::new("no-such-checkpoint"), unit, lang) {
            Ok(_) => panic!("opened a checkpoint that does not exist"),
            Err(e) => e,
        }
    }

    // Each is refused before anything is loaded, so none needs weights.
    #[test]
    fn open_refuses_before_loading() {
        assert!(open_error(PTTS_UNIT_ANE, "en").contains("Apple only"));
        assert!(!open_error(PTTS_UNIT_CPU, "klingon").contains("no weights"));
        assert!(open_error(PTTS_UNIT_CPU, "en").contains("config.json"));
    }
}
