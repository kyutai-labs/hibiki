# hibiki - rust

![rust ci badge](https://github.com/kyutai-labs/moshi/workflows/Rust%20CI/badge.svg)
[![Latest version](https://img.shields.io/crates/v/hibiki.svg)](https://crates.io/crates/hibiki)
[![Documentation](https://docs.rs/hibiki/badge.svg)](https://docs.rs/hibiki)
![License](https://img.shields.io/crates/l/hibiki.svg)

See the [top-level README.md](../README.md) for more information.

This provides the Rust implementation for Hibiki, a real-time speech-to-speech
translation model.

## Requirements

You will need a recent version of the [Rust toolchain](https://rustup.rs/).
To compile GPU support, you will also need the [CUDA](https://developer.nvidia.com/cuda-toolkit) properly installed for your GPU, in particular with `nvcc`.

## Example

```bash
cd hibiki-rs
wget https://github.com/kyutai-labs/moshi/raw/refs/heads/main/data/sample_fr_hibiki_crepes.mp3
cargo run  --features metal -r -- gen sample_fr_hibiki_crepes.mp3 out_en.wav
```

## License

The present code is provided under the Apache license.
