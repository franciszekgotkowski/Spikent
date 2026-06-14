# Training

Wajpierw będziesz musiał stworzyć wirtualne środowisko 
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

From the project root (`Spikent/`):

```bash
python3 -m src.train --small --max-samples 10000 --epochs 1 --batch-size 8 --output-dir ./checkpoints
```

Key flags:
- `--small` — uses `SmallModelConfig` (d_model=128, 2 layers); good for CPU/laptop testing.
- `--max-samples` — cap the streaming dataset (default 2M).
- `--epochs`, `--batch-size`, `--output-dir` — self-explanatory.
- `--no-amp` — disables mixed precision; required on CPU since fp16 needs CUDA.

The default config in `config.py` trains on `HuggingFaceFW/finetranslations` Polish (`pol_Latn`) using the `Helsinki-NLP/opus-mt-pl-en` tokenizer. Checkpoints are saved to `--output-dir` every `--save-every` steps (default 5000), and a final `best/` checkpoint is copied at the end.

## Inference

Point it at a checkpoint directory containing `checkpoint.pt` and the saved tokenizer:

```bash
python3 -m src.inference --checkpoint ./checkpoints/best --text "Dzień dobry. Jak się masz?"
```

Flags:
- `--checkpoint` (required) — checkpoint directory, e.g. `./checkpoints/step_5000` or `./checkpoints/best`.
- `--text` — Polish text to translate (default: `"Dzień dobry. Jak się masz?"`).
- `--small` — use `SmallModelConfig` if the model was trained with `--small`.
- `--device` — override device (`cuda`/`cpu`).

## Example workflow

```bash
# 1. Quick CPU smoke test
python3 -m src.train --small --no-amp --max-samples 1000 --epochs 1 --batch-size 4 --output-dir ./test-checkpoints

# 2. Translate with the final checkpoint
python3 -m src.inference --checkpoint ./test-checkpoints/best --small --text "Dzień dobry. Jak się masz?"
```

A couple of things to note:
- Training uses Hugging Face `accelerate`, so multi-GPU / mixed precision is handled automatically if CUDA is available.
- The script exits with `os._exit(0)` to avoid a known PyArrow streaming shutdown crash.
- If you expected an `interface.py` file, it isn't currently in the repo — only `src/train.py` and `src/inference.py` exist.
