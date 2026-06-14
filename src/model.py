"""Small Transformer encoder-decoder for Polish -> English translation."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerTranslator(nn.Module):
    """Small encoder-decoder transformer using shared token embeddings.

    The same embedding table is used for source and target because both
    languages share the same tokenizer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len

        self.embedding = nn.Embedding(cfg.src_vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Embed + scale + encode positions
        src_emb = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))

        if tgt_mask is None:
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(
                dtype=tgt_emb.dtype, device=tgt.device
            )

        # Ensure key-padding masks share the same dtype as the attention mask
        # to avoid PyTorch deprecation warnings.
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(dtype=tgt_mask.dtype)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=tgt_mask.dtype)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)

    @torch.inference_mode()
    def translate(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int | None = None,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """Greedy or beam-search decoding. Defaults to greedy for simplicity."""
        if max_len is None:
            max_len = self.max_seq_len
        max_len = min(max_len, self.max_seq_len)

        self.eval()
        batch_size = src.size(0)
        device = src.device

        if beam_size == 1:
            tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                logits = self.forward(src, tgt)[:, -1, :]
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                tgt = torch.cat([tgt, next_token], dim=1)
                finished |= next_token.squeeze(1) == eos_id
                if finished.all():
                    break
            return tgt

        raise NotImplementedError("Beam search not implemented; use beam_size=1.")
