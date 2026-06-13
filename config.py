"""Training and model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    src_vocab_size: int = 40000
    tgt_vocab_size: int = 40000
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 128


@dataclass
class TrainConfig:
    # Dataset
    dataset_name: str = "HuggingFaceFW/finetranslations"
    dataset_subset: str = "pol_Latn"
    src_column: str = "og_full_text"
    tgt_column: str = "translated_text"
    lang_score_column: str = "og_language_score"
    max_samples: int | None = 2_000_000
    max_seq_len: int = 128

    # Tokenizer
    tokenizer_name: str = "Helsinki-NLP/opus-mt-pl-en"

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 2500
    log_every_n_steps: int = 100

    # Misc
    seed: int = 42
    num_workers: int = 4
    use_amp: bool = True
