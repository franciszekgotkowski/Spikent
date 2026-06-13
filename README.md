# Polish → English Tiny Transformer

A small encoder-decoder Transformer trained from scratch on the [HuggingFaceFW/finetranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) `pol_Latn` subset, using the [Helsinki-NLP/opus-mt-pl-en](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en) tokenizer.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> For ROCm on a local RX 680M, install the ROCm PyTorch wheel instead of the CUDA one. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Project layout

```
Spikent/
├── config.py            # Model + training hyperparameters
├── requirements.txt
├── README.md
└── src/
    ├── model.py         # TransformerTranslator (small encoder-decoder)
    ├── dataset.py       # Streaming dataset loader & collator
    ├── train.py         # Training loop (Hugging Face Accelerate)
    └── inference.py     # Local inference script
```

## Train

Run locally or on a rented cloud GPU:

```bash
python -m src.train \
  --max-samples 500000 \
  --epochs 3 \
  --batch-size 64 \
  --output-dir ./checkpoints
```

You can override any hyperparameter in `config.py` by editing it directly.

Default model size (~30–40M parameters):
- d_model = 512
- 4 encoder + 4 decoder layers
- 8 attention heads
- FFN dim = 1024
- max sequence length = 128

## Inference

After training:

```bash
python -m src.inference \
  --checkpoint ./checkpoints/best \
  --text "Dzień dobry. Jak się masz?"
```

## Notes on local RX 680M use

- This model is sized for inference on a 4GB-class iGPU.
- Training from scratch on an RX 680M is possible but very slow (weeks for a useful model). Use the cloud GPU for training and the RX 680M for inference.
- On Linux with ROCm, the inference script will automatically use `cuda:0`. On Windows/WSL or if ROCm is unavailable, it falls back to CPU.

## Data details

The dataset streams from Hugging Face. Polish sentences come from `og_full_text`, English translations from `translated_text`. Examples are filtered by `og_language_score >= 0.9` and very short examples are skipped.
