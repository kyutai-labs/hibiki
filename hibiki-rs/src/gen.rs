use anyhow::Result;
use candle::{Device, IndexOp, Tensor};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub mimi_name: String,
    pub moshi_name: String,
    pub tokenizer_name: String,
    pub model: moshi::lm::Config,
}

pub struct Args {
    pub lm_config: moshi::lm::Config,
    pub lm_model_file: std::path::PathBuf,
    pub mimi_model_file: std::path::PathBuf,
    pub audio_input_file: std::path::PathBuf,
    pub text_tokenizer: std::path::PathBuf,
    pub audio_output_file: std::path::PathBuf,
    pub seed: u64,
    pub cfg_alpha: Option<f64>,
}

fn text(
    text_tokenizer: &sentencepiece::SentencePieceProcessor,
    prev_text_token: u32,
    text_token: u32,
    text_start_token: u32,
) -> Option<String> {
    if prev_text_token == text_start_token {
        text_tokenizer.decode_piece_ids(&[text_token]).ok()
    } else {
        let prev_ids = text_tokenizer.decode_piece_ids(&[prev_text_token]).ok();
        let ids = text_tokenizer.decode_piece_ids(&[prev_text_token, text_token]).ok();
        prev_ids.and_then(|prev_ids| {
            ids.map(|ids| {
                if ids.len() > prev_ids.len() {
                    ids[prev_ids.len()..].to_string()
                } else {
                    String::new()
                }
            })
        })
    }
}

pub fn run(args: &Args, dev: &Device) -> Result<()> {
    let dtype = dev.bf16_default_to_f32();
    let lm_config = &args.lm_config;
    tracing::info!(?dtype, ?dev);

    tracing::info!("loading the audio input");
    let (in_pcm, in_pcm_len) = {
        let (mut pcm, sample_rate) = crate::audio_io::pcm_decode(&args.audio_input_file)?;
        pcm.extend_from_slice(&vec![0.0; 12000]);
        let pcm = if sample_rate != 24_000 {
            crate::audio_io::resample(&pcm, sample_rate as usize, 24_000)?
        } else {
            pcm
        };
        let pcm_len = pcm.len();
        let pcm = Tensor::from_vec(pcm, (1, 1, pcm_len), dev)?;
        (pcm, pcm_len)
    };
    tracing::info!(in_pcm_len, "loaded the audio input");

    tracing::info!("loading the lm");
    let lm_model = moshi::lm::load_lm_model(lm_config.clone(), &args.lm_model_file, dtype, dev)?;
    tracing::info!("loading the audio tokenizer");
    let mut mimi = moshi::mimi::load(
        args.mimi_model_file.to_str().unwrap(),
        Some(lm_model.generated_audio_codebooks()),
        dev,
    )?;
    tracing::info!("loading the text tokenizer");
    let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&args.text_tokenizer)?;
    tracing::info!("done loading models");

    let audio_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
        args.seed,
        candle_transformers::generation::Sampling::TopK { k: 250, temperature: 0.8 },
    );
    let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
        args.seed,
        candle_transformers::generation::Sampling::TopK { k: 25, temperature: 0.8 },
    );
    let generated_audio_codebooks = lm_config.depformer.as_ref().map_or(8, |v| v.num_slices);

    let conditions = match lm_model.condition_provider() {
        None => None,
        Some(cp) => {
            let conditions = if args.cfg_alpha.is_some() {
                use moshi::conditioner::Condition::AddToInput;
                let AddToInput(c1) = cp.condition_lut("description", "very_good")?;
                let AddToInput(c2) = cp.condition_lut("description", "very_bad")?;
                AddToInput(Tensor::cat(&[c1, c2], 0)?)
            } else {
                cp.condition_lut("description", "very_good")?
            };
            tracing::info!(?conditions, "generated conditions");
            Some(conditions)
        }
    };
    let max_steps = (in_pcm_len / 1920).min(2500);
    let cfg_alpha = if args.cfg_alpha == Some(1.) { None } else { args.cfg_alpha };
    let mut state = {
        let config = moshi::lm_generate_multistream::Config {
            acoustic_delay: 2,
            audio_vocab_size: lm_config.audio_vocab_size,
            generated_audio_codebooks,
            input_audio_codebooks: lm_config.audio_codebooks - generated_audio_codebooks,
            text_start_token: lm_config.text_out_vocab_size as u32,
            text_eop_token: 0,
            text_pad_token: 3,
        };
        moshi::lm_generate_multistream::State::new(
            lm_model,
            max_steps + 20,
            audio_lp,
            text_lp,
            None,
            None,
            cfg_alpha,
            config,
        )
    };

    let text_start_token = state.config().text_start_token;
    let mut prev_text_token = text_start_token;
    let mut out_pcms = vec![];
    let mut text_tokens = vec![];
    let mut nsteps = 0;
    tracing::info!("starting the inference loop");
    let start_time = std::time::Instant::now();
    for start_index in 0..max_steps {
        nsteps += 1;
        let in_pcm = in_pcm.i((.., .., start_index * 1920..(start_index + 1) * 1920))?;
        let codes = mimi.encode_step(&in_pcm.into())?;
        if let Some(codes) = codes.as_option() {
            let (_b, _codebooks, steps) = codes.dims3()?;
            for step in 0..steps {
                let codes = codes.i((.., .., step..step + 1))?;
                let codes = codes.i((0, .., 0))?.to_vec1::<u32>()?;
                let text_token =
                    state.step_(Some(prev_text_token), &codes, None, None, conditions.as_ref())?;
                if text_token != 0 && text_token != 3 {
                    text_tokens.push(text_token);
                    if let Some(text) =
                        text(&text_tokenizer, prev_text_token, text_token, text_start_token)
                    {
                        use std::io::Write;
                        print!("{text}");
                        std::io::stdout().flush().unwrap();
                    }
                }
                prev_text_token = text_token;
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let audio_tokens =
                        Tensor::new(&audio_tokens[..generated_audio_codebooks], dev)?
                            .reshape((1, 1, ()))?
                            .t()?;
                    let out_pcm = mimi.decode_step(&audio_tokens.into())?;
                    if let Some(out_pcm) = out_pcm.as_option() {
                        out_pcms.push(out_pcm.clone());
                    }
                }
            }
        }
    }
    println!();
    let dt = start_time.elapsed().as_secs_f32();
    tracing::info!(
        "generated {nsteps} steps in {dt:.2}s, {:.0}ms/token",
        dt * 1000. / (nsteps as f32)
    );
    let str = text_tokenizer.decode_piece_ids(&text_tokens)?;
    tracing::info!(str, "generated text");
    let out_pcms = Tensor::cat(&out_pcms, 2)?;
    tracing::info!(shape = ?out_pcms.shape(), "generated audio");
    let out_pcms = out_pcms.i((0, 0))?.to_vec1::<f32>()?;
    let mut out_wav = std::fs::File::create(&args.audio_output_file)?;
    moshi::wav::write_pcm_as_wav(&mut out_wav, &out_pcms, 24_000)?;
    tracing::info!(audio = ?args.audio_output_file, "generated audio");
    Ok(())
}
