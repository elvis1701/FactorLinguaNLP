# translingua.py
# -*- coding: utf-8 -*-
"""
Translingua: NMT estilo Transformer com atenção hierárquica (local + global),
codificação posicional dinâmica e embeddings fatorizados.

Principais correções em relação ao esboço original:
- MultiheadAttention com batch_first=True (aceita (B, L, E)).
- Máscaras corretas: key_padding_mask (B, L) e attn_mask (L, L) booleana.
- Decoder com cross-attention para usar a memória do encoder.
- Atenção local implementada com máscara de janela deslizante.
- Remoção de residual/LayerNorm duplicados (apenas no bloco, não dentro da atenção).
- PositionalEncoding não recomputa embedding; forma (1, L, E).
- Dataset com BOS/EOS, truncamento e collate_fn com padding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from typing import List, Tuple

# ---------------------------
# Utilidades de máscara
# ---------------------------

def causal_mask(L: int, device) -> torch.Tensor:
    """
    Máscara causal booleana de forma (L, L).
    True = posição NÃO pode ser atendida (será mascarada).
    """
    return torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)

def local_window_mask(L: int, w: int, device) -> torch.Tensor:
    """
    Máscara booleana de atenção local (janela de ±w).
    True = posição fora da janela (deve ser mascarada).
    Forma: (L, L)
    """
    # Começa com tudo True (mascarado), e libera janela ao redor da diagonal
    m = torch.ones((L, L), dtype=torch.bool, device=device)
    for i in range(L):
        left = max(0, i - w)
        right = min(L, i + w + 1)
        m[i, left:right] = False  # False => permitido
    return m

def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    Combina várias máscaras booleanas (todas (L, L)) por OR lógico.
    Qualquer True mascara a posição.
    """
    out = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out | m)
    return out


# ---------------------------
# Componentes do modelo
# ---------------------------

class DynamicPositionalEncoding(nn.Module):
    """
    Codificação posicional sinusoidal com fator de escala treinável.
    Retorna tensor de forma (1, L, E) para ser somado ao embedding.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, E)
        B, L, E = x.size()
        device = x.device

        position = torch.arange(L, dtype=torch.float, device=device).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, E, 2, device=device).float() * (-math.log(10000.0) / E))

        pe = torch.zeros(L, E, device=device)  # (L, E)
        pe[:, 0::2] = torch.sin(position * div_term) * self.scale
        pe[:, 1::2] = torch.cos(position * div_term) * self.scale

        return pe.unsqueeze(0)  # (1, L, E)


class TokenEmbedding(nn.Module):
    """
    Embedding fatorizado: vocab -> factor_dim -> d_model
    Reduz parâmetros em vocabulários grandes.
    """
    def __init__(self, vocab_size: int, d_model: int, factor_dim: int = 64):
        super().__init__()
        self.main_embed = nn.Embedding(vocab_size, factor_dim)
        self.projection = nn.Linear(factor_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.main_embed(x))  # (B, L, d_model)


class HierarchicalSelfAttention(nn.Module):
    """
    Atenção hierárquica: local (janela) + global (full).
    - Local usa attn_mask de janela.
    - Global não usa máscara de janela (apenas causal/padding quando aplicável).
    Combinação simples por soma (ou gate aprendível opcional).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, local_window: int = 8):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.local_window = local_window
        # Opcional: gate aprendível para ponderar local vs global
        self.gate = nn.Parameter(torch.tensor(0.5))  # valor inicial 0.5

    def forward(
        self,
        x: torch.Tensor,                     # (B, L, E)
        key_padding_mask: torch.Tensor,      # (B, L) True=PAD
        base_attn_mask: torch.Tensor = None  # (L, L) booleana (e.g. causal)
    ) -> torch.Tensor:
        B, L, E = x.shape
        device = x.device

        # Máscara local: janela + (opcional) causal
        loc_mask = local_window_mask(L, self.local_window, device)
        if base_attn_mask is not None:
            loc_mask = combine_masks(loc_mask, base_attn_mask)

        # Atenção local
        loc_out, _ = self.local_attn(
            x, x, x,
            attn_mask=loc_mask,                   # (L, L) booleana
            key_padding_mask=key_padding_mask     # (B, L)
        )

        # Atenção global (sem janela; apenas causal se passado)
        glob_out, _ = self.global_attn(
            x, x, x,
            attn_mask=base_attn_mask,             # (L, L) booleana
            key_padding_mask=key_padding_mask
        )

        # Combinação hierárquica (gate aprendível)
        combined = self.gate * loc_out + (1.0 - self.gate) * glob_out
        return self.dropout(combined)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1, local_window: int = 8):
        super().__init__()
        self.self_attn = HierarchicalSelfAttention(d_model, n_heads, dropout, local_window)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_keypad: torch.Tensor):
        # Self-attention (encoder não usa máscara causal)
        sa = self.self_attn(x, key_padding_mask=src_keypad, base_attn_mask=None)
        x = self.norm1(x + self.drop(sa))
        ff = self.ff(x)
        x = self.norm2(x + self.drop(ff))
        return x


class DecoderBlock(nn.Module):
    """
    Decoder: (i) self-attn com causal + janela local; (ii) cross-attn com memória do encoder; (iii) FFN.
    """
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1, local_window: int = 8):
        super().__init__()
        self.self_attn = HierarchicalSelfAttention(d_model, n_heads, dropout, local_window)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        y: torch.Tensor,                   # (B, Lt, E)
        memory: torch.Tensor,              # (B, Ls, E)
        tgt_keypad: torch.Tensor,          # (B, Lt) True=PAD
        src_keypad: torch.Tensor,          # (B, Ls) True=PAD
        causal: torch.Tensor               # (Lt, Lt) booleana
    ):
        # 1) Self-attn com causal + janela local
        sa = self.self_attn(y, key_padding_mask=tgt_keypad, base_attn_mask=causal)
        y = self.norm1(y + self.drop(sa))

        # 2) Cross-attn (Q=dec, K/V=memória do encoder) — sem causal; só key_padding do src
        ca, _ = self.cross_attn(y, memory, memory, key_padding_mask=src_keypad)
        y = self.norm2(y + self.drop(ca))

        # 3) FFN
        ff = self.ff(y)
        y = self.norm3(y + self.drop(ff))

        return y


class TranslinguaModel(nn.Module):
    """
    Modelo completo Encoder-Decoder com:
    - TokenEmbedding fatorizado
    - Positional Encoding dinâmica
    - Encoder/Decoder com atenção hierárquica local+global
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        local_window: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder
        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.pos_encoder = DynamicPositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ff_dim, dropout, local_window) for _ in range(num_layers)
        ])

        # Decoder
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_decoder = DynamicPositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, ff_dim, dropout, local_window) for _ in range(num_layers)
        ])

        # Output layer
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor, src_keypad: torch.Tensor) -> torch.Tensor:
        # src: (B, Ls)
        x = self.src_embed(src)                       # (B, Ls, E)
        x = self.dropout(x + self.pos_encoder(x))     # (B, Ls, E)
        for layer in self.encoder_layers:
            x = layer(x, src_keypad)
        return x                                       # memória do encoder

    def decode(
        self,
        tgt_in: torch.Tensor,         # (B, Lt)
        memory: torch.Tensor,         # (B, Ls, E)
        tgt_keypad: torch.Tensor,     # (B, Lt)
        src_keypad: torch.Tensor      # (B, Ls)
    ) -> torch.Tensor:
        y = self.tgt_embed(tgt_in)                    # (B, Lt, E)
        y = self.dropout(y + self.pos_decoder(y))

        Lt = tgt_in.size(1)
        causal = causal_mask(Lt, device=tgt_in.device)  # (Lt, Lt)

        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_keypad, src_keypad, causal)

        logits = self.output_proj(y)                  # (B, Lt, Vt)
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_keypad: torch.Tensor,
        tgt_keypad: torch.Tensor
    ) -> torch.Tensor:
        memory = self.encode(src, src_keypad)
        logits = self.decode(tgt_in, memory, tgt_keypad, src_keypad)
        return logits


# ---------------------------
# Dataset e DataLoader
# ---------------------------

PAD = 0
BOS = 1
EOS = 2

class TranslationDataset(Dataset):
    """
    Pares de tradução tokenizados com SentencePiece.
    - Aplica truncamento a max_len-2 (para BOS/EOS).
    - Adiciona BOS no início e EOS no final.
    """
    def __init__(self, src_file: str, tgt_file: str, src_spm: spm.SentencePieceProcessor,
                 tgt_spm: spm.SentencePieceProcessor, max_len: int = 256):
        self.max_len = max_len
        self.src_spm = src_spm
        self.tgt_spm = tgt_spm

        with open(src_file, "r", encoding="utf-8") as f:
            src_lines = [ln.strip() for ln in f if ln.strip()]
        with open(tgt_file, "r", encoding="utf-8") as f:
            tgt_lines = [ln.strip() for ln in f if ln.strip()]

        assert len(src_lines) == len(tgt_lines), "Arquivos src e tgt com comprimentos diferentes."

        self.src_data = [self._encode_line(ln, src_spm) for ln in src_lines]
        self.tgt_data = [self._encode_line(ln, tgt_spm) for ln in tgt_lines]

    def _encode_line(self, text: str, sp: spm.SentencePieceProcessor) -> List[int]:
        ids = sp.encode(text, out_type=int)
        # Trunca para abrir espaço a BOS/EOS
        ids = ids[: max(0, self.max_len - 2)]
        return [BOS] + ids + [EOS]

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return torch.LongTensor(self.src_data[idx]), torch.LongTensor(self.tgt_data[idx])


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad de src e tgt para o maior comprimento do batch.
    """
    src_list, tgt_list = zip(*batch)
    src_pad = pad_sequence(src_list, batch_first=True, padding_value=PAD)
    tgt_pad = pad_sequence(tgt_list, batch_first=True, padding_value=PAD)
    return src_pad, tgt_pad


# ---------------------------
# Loss com Label Smoothing
# ---------------------------

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing para classificação token-a-token.
    - logits: (B, L, V)
    - targets: (B, L)
    - ignore_index: token PAD
    """
    def __init__(self, smoothing: float = 0.1, ignore_index: int = PAD):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, L, V = logits.shape
        logits = logits.view(-1, V)           # (B*L, V)
        targets = targets.reshape(-1)         # (B*L,)

        # Máscara de posições válidas
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return logits.new_tensor(0.0)

        log_probs = F.log_softmax(logits[valid], dim=-1)  # (N, V)

        # NLL da classe correta
        nll = -log_probs.gather(dim=-1, index=targets[valid].unsqueeze(1)).squeeze(1)  # (N,)

        # Loss suave: média sobre classes
        smooth = -log_probs.mean(dim=-1)  # (N,)

        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


# ---------------------------
# Treinamento
# ---------------------------

def train_model(
    src_model_path: str = "src.model",
    tgt_model_path: str = "tgt.model",
    src_file: str = "train.src",
    tgt_file: str = "train.tgt",
    d_model: int = 512,
    n_heads: int = 8,
    num_layers: int = 6,
    ff_dim: int = 2048,
    dropout: float = 0.1,
    local_window: int = 8,
    batch_size: int = 32,
    max_len: int = 256,
    epochs: int = 10,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = None
):
    # Reprodutibilidade básica
    torch.manual_seed(seed)

    # Dispositivo
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizers
    src_spm = spm.SentencePieceProcessor()
    tgt_spm = spm.SentencePieceProcessor()
    src_spm.load(src_model_path)
    tgt_spm.load(tgt_model_path)

    # Dataset/DataLoader
    dataset = TranslationDataset(src_file, tgt_file, src_spm, tgt_spm, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)

    # Modelo
    model = TranslinguaModel(
        src_vocab_size=src_spm.vocab_size(),
        tgt_vocab_size=tgt_spm.vocab_size(),
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        local_window=local_window,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=PAD)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in loader:
            src = src.to(device)  # (B, Ls)
            tgt = tgt.to(device)  # (B, Lt)

            # Teacher Forcing: input e target deslocados
            tgt_in  = tgt[:, :-1]  # exclui o último token
            tgt_out = tgt[:, 1:]   # exclui o primeiro token

            # Máscaras de padding (True = PAD)
            src_keypad = (src == PAD)       # (B, Ls)
            tgt_keypad = (tgt_in == PAD)    # (B, Lt-1)

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_keypad, tgt_keypad)  # (B, Lt-1, V)
            loss = criterion(logits, tgt_out)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Stats
            tokens = (tgt_out != PAD).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

        ppl_loss = total_loss / max(1, total_tokens)
        print(f"Epoch {epoch:02d} | Loss/token: {ppl_loss:.4f}")

    return model


# ---------------------------
# Inferência simples (greedy)
# ---------------------------

@torch.no_grad()
def greedy_decode(
    model: TranslinguaModel,
    src_spm: spm.SentencePieceProcessor,
    tgt_spm: spm.SentencePieceProcessor,
    text: str,
    max_len: int = 64,
    device: str = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    # Tokeniza src com BOS/EOS
    src_ids = [BOS] + src_spm.encode(text, out_type=int) + [EOS]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, Ls)
    src_keypad = (src == PAD)

    # Memória do encoder
    memory = model.encode(src, src_keypad)

    # Decodificação greedy
    y = torch.tensor([BOS], dtype=torch.long, device=device).unsqueeze(0)  # (1, 1)
    for _ in range(max_len):
        tgt_keypad = (y == PAD)
        logits = model.decode(y, memory, tgt_keypad, src_keypad)  # (1, Lt, V)
        next_token = logits[:, -1, :].argmax(dim=-1)  # (1,)
        y = torch.cat([y, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == EOS:
            break

    # Remove BOS e EOS
    out_ids = y.squeeze(0).tolist()
    out_ids = [t for t in out_ids if t not in (BOS, EOS)]
    return tgt_spm.decode(out_ids)


# ---------------------------
# Execução
# ---------------------------

if __name__ == "__main__":
    # Ajuste os caminhos conforme seus arquivos:
    # - src.model / tgt.model (SentencePiece)
    # - train.src / train.tgt
    train_model()

