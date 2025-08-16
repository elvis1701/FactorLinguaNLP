
# Translingua
**NMT estilo Transformer com atenção hierárquica (local + global), embeddings fatorizados e codificação posicional dinâmica**

> Este repositório contém uma implementação educacional e extensível de um sistema de tradução neural (seq2seq) no estilo Transformer, com **atenção hierárquica** (janela local + escopo global), **embeddings fatorizados** para reduzir parâmetros e **positional encoding** **dinâmico**. Inclui _dataset_ com BOS/EOS, _collate_ com padding, _label smoothing_, _greedy decoding_, e um loop de treinamento completo em PyTorch.

---

## ✨ Destaques do projeto
- **Atenção hierárquica**: combina **atenção local** (janela deslizante) com **atenção global**. Um _gate_ treinável pondera as duas saídas.
- **Decoder com cross-attention**: utiliza a memória do encoder corretamente.
- **Máscaras corretas e com _batch_first_**: `key_padding_mask` em `(B, L)` e `attn_mask` booleana em `(L, L)`.
- **Positional Encoding dinâmico**: seno/cosseno com **escala treinável**, forma `(1, L, E)` somada ao embedding.
- **Embeddings fatorizados**: `vocab -> factor_dim -> d_model`, útil para vocabulários grandes.
- **Dataset robusto**: adiciona BOS/EOS, trunca para **max_len-2**, e faz padding via `pad_sequence`.
- **Treinamento com Label Smoothing**: perda mais estável e menos _overconfidence_.
- **Decodificação greedy** para inferência rápida.

---

## 📁 Estrutura do arquivo principal
O arquivo principal é `translingua.py`, organizado em blocos:

1. **Utilidades de máscara**: `causal_mask`, `local_window_mask` e `combine_masks`.
2. **Componentes do modelo**:
   - `DynamicPositionalEncoding`
   - `TokenEmbedding` (fatorizado)
   - `HierarchicalSelfAttention` (local + global com _gate_)
   - `FeedForward`
   - `EncoderBlock` e `DecoderBlock`
   - `TranslinguaModel` (Encoder-Decoder completo)
3. **Dataset/DataLoader**:
   - `TranslationDataset` (com SentencePiece, BOS/EOS e truncamento)
   - `collate_fn` (padding)
4. **Perda**: `LabelSmoothingLoss`
5. **Treinamento**: `train_model(...)`
6. **Inferência**: `greedy_decode(...)`

---

## 🧠 Como o algoritmo funciona (visão geral)
### Pipeline de dados
1. **Tokenização** com SentencePiece (`.model` do idioma fonte e alvo).
2. **Preparação**: para cada sentença, codifica, trunca, adiciona `BOS` e `EOS`.
3. **_Batching_**: padding por `PAD=0` via `collate_fn`.

### Encoder
- **Entrada**: `src` `(B, Ls)` → `TokenEmbedding` → soma com `PositionalEncoding dinâmico` → `Dropout`.
- **Blocos**: cada `EncoderBlock` aplica
  - `HierarchicalSelfAttention` (sem máscara causal; usa apenas `key_padding_mask`).
  - `Residual + LayerNorm`
  - `FeedForward` + `Residual + LayerNorm`
- **Saída**: memória `(B, Ls, d_model)`.

### Decoder
- **Entrada**: `tgt_in` `(B, Lt)` → `TokenEmbedding` + `PositionalEncoding dinâmico`.
- **Self-Attention**: usa **máscara causal** (triangular superior) **+** máscara **local** de janela (±`w`).
- **Cross-Attention**: consulta a **memória do encoder** (com `key_padding_mask` da fonte).
- **FFN** + normalizações.
- **Projeção final**: `Linear(d_model → vocab_tgt)` produz `logits` por passo.

### Atenção hierárquica (local + global)
- **Local**: `attn_mask` de janela deslizante (±`w`) **restringe** o escopo por token.
- **Global**: atenção padrão sem janela (mas respeitando causal/padding).
- **Combinação**: `combined = gate * local + (1 - gate) * global`, com `gate` **treinável** (inicial 0.5).

### Máscaras
- `key_padding_mask`: `True` marca tokens **PAD** e os mascara.
- `attn_mask`: booleana `(L, L)`; `True` = posição proibida. Usada para **causal** e/ou **janela local**.
- Tudo opera com `batch_first=True`.

---

## 🔩 Diagrama do fluxo (Encoder → Decoder → Saída)

```mermaid
flowchart LR
    A[Texto Fonte] --> B[SentencePiece\nencode + BOS/EOS]
    B --> C[src: Tensor (B,Ls)]
    C --> D[TokenEmbedding (fatorizado)]
    D --> E[PosEnc Dinâmico (1,L,E) + Dropout]
    E --> F[Encoder x N camadas]
    subgraph ENCODER
      F1[Self-Attn Hierárquica\n Local (janela w)\n + Global (full)\n+ Gate treinável]
      F2[Residual + LayerNorm]
      F3[FeedForward + Residual + LayerNorm]
      F --repete N--> F
    end
    F --> G[Memória do Encoder (B,Ls,E)]

    H[Texto Alvo (shifted)] --> I[SentencePiece\nencode + BOS/EOS]
    I --> J[tgt_in: Tensor (B,Lt)]
    J --> K[TokenEmbedding (fatorizado)]
    K --> L[PosEnc Dinâmico + Dropout]

    subgraph DECODER
      L --> M[Self-Attn Hierárquica\n (Causal + Janela w)]
      M --> N[Residual + LayerNorm]
      N --> O[Cross-Attn\n Q=Decoder, K/V=Mem. Encoder]
      O --> P[Residual + LayerNorm]
      P --> Q[FeedForward + Residual + LayerNorm]
    end

    Q --> R[Linear(d_model → Vocab_Tgt)]
    R --> S[Logits → Greedy Decode]
    S --> T[Texto Traduzido]
```

**Modificações destacadas**: blocos com _Self-Attn Hierárquica_, _TokenEmbedding (fatorizado)_ e _PosEnc Dinâmico_.

---

## 🛠️ Instalação e requisitos

### Requisitos mínimos
- Python 3.9+
- GPU CUDA (opcional, mas recomendado)
- Pip

### Instalação
```bash
git clone <seu-repo-url>
cd <seu-repo>
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
```

### `requirements.txt` sugerido
> Ajuste versões conforme seu ambiente/CUDA.
```
torch>=2.2.0
sentencepiece>=0.1.99
numpy>=1.24.0
tqdm>=4.66.0
sacrebleu>=2.4.0     # avaliação (BLEU/chrF)
matplotlib>=3.8.0    # gráficos de treinamento
```

---

## 🧪 Preparando os dados e os modelos SentencePiece
Treine dois modelos SentencePiece (fonte/alvo) **ou** reutilize modelos existentes.

```bash
# Exemplo: treinar vocabulário de 16k para a língua fonte
spm_train --input=train.src --model_prefix=src --vocab_size=16000 --model_type=bpe --character_coverage=0.9995

# Para a língua alvo
spm_train --input=train.tgt --model_prefix=tgt --vocab_size=16000 --model_type=bpe --character_coverage=0.9995
```

Arquivos esperados no diretório do projeto:
```
src.model   tgt.model
train.src   train.tgt
```

---

## 🚀 Treinamento

Com o ambiente pronto e os arquivos `*.model` e `train.*` existentes, execute:

```bash
python translingua.py
```

Ou personalize os hiperparâmetros chamando a função `train_model(...)` em um _script_ separado ou REPL:

```python
from translingua import train_model

model = train_model(
    src_model_path="src.model",
    tgt_model_path="tgt.model",
    src_file="train.src",
    tgt_file="train.tgt",
    d_model=512,
    n_heads=8,
    num_layers=6,
    ff_dim=2048,
    dropout=0.1,
    local_window=8,
    batch_size=64,
    max_len=256,
    epochs=10,
    lr=1e-4,
    seed=42,
    device=None,      # "cuda" ou "cpu"
)
```

### Hiperparâmetros principais
| Parâmetro       | Padrão | Descrição |
|-----------------|--------|-----------|
| `d_model`       | 512    | Dimensão dos embeddings e das projeções do Transformer |
| `n_heads`       | 8      | Número de _heads_ de atenção |
| `num_layers`    | 6      | Nº de camadas no encoder e no decoder |
| `ff_dim`        | 2048   | Dimensão interna da FFN |
| `dropout`       | 0.1    | Dropout em atenção/FFN/embeddings |
| `local_window`  | 8      | Raio da janela local (±w) na atenção hierárquica |
| `batch_size`    | 32     | Tamanho do batch |
| `max_len`       | 256    | Tamanho máximo por sentença (com BOS/EOS) |
| `epochs`        | 10     | Épocas de treinamento |
| `lr`            | 1e-4   | Taxa de aprendizado (AdamW) |

> **Dicas**: aumente `batch_size` e `num_layers` com GPU; ajuste `local_window` se seus dados tiverem dependências mais longas/curtas.

---

## 🔎 Inferência (decodificação _greedy_)

```python
import torch, sentencepiece as spm
from translingua import TranslinguaModel, greedy_decode, PAD

# Carregue modelos SentencePiece
src_spm = spm.SentencePieceProcessor(); src_spm.load("src.model")
tgt_spm = spm.SentencePieceProcessor(); tgt_spm.load("tgt.model")

# Carregue pesos do modelo (se salvou após o treino)
# model = torch.load("translingua.pt", map_location="cpu")

# Ou treine e reutilize o objeto retornado
# model = train_model(...)

texto = "This is a small test."
traducao = greedy_decode(model, src_spm, tgt_spm, texto, max_len=64, device=None)
print(traducao)
```

> Para produção, considere **beam search** e penalizações de comprimento/cobertura.

---

## 📈 Gráficos e métricas (precisão/qualidade)
Para tradução, métricas de precisão token-a-token são menos informativas do que métricas de **qualidade de sequência**. Recomenda-se:

- **BLEU** e **chrF** via `sacrebleu`.
- (Opcional) **COMET/BLEURT** para avaliações baseadas em aprendizado.

### Registrando _loss_ por época
O `train_model` imprime `Loss/token`. Para salvar e **plotar**:

```python
# Exemplo simples: modifique o loop de treino para acumular losses
losses = []
for epoch in range(1, epochs + 1):
    ...
    ppl_loss = total_loss / max(1, total_tokens)
    losses.append(ppl_loss)
    print(f"Epoch {epoch:02d} | Loss/token: {ppl_loss:.4f}")

# Plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1, len(losses)+1), losses, marker="o")
plt.xlabel("Época"); plt.ylabel("Loss por token"); plt.title("Curva de treinamento")
plt.grid(True); plt.tight_layout()
plt.savefig("training_curve.png")
```

### Avaliando com BLEU/chrF
Supondo que você possua `dev.src` e `dev.ref` (referências) e um _script_ que gera `dev.hyp` com seu modelo:

```bash
# BLEU
sacrebleu dev.ref < dev.hyp

# chrF (mais sensível a similaridade de _character n-grams_)
sacrebleu -m chrf dev.ref < dev.hyp
```

> **Boas práticas**: use `sacrebleu` para garantir comparabilidade; não faça _tokenization_ manual distinta entre sistemas.

---

## 🧪 Casos de uso
- **Tradução de domínio geral**: paralelos genéricos (ex.: notícias, Wikipedia).
- **Adaptação de domínio**: jurídico, médico, e-commerce — treine/fine-tune com corpora específicos.
- **Legendas e subtítulos**: sentenças curtas com janelas locais menores podem acelerar.
- **Sistemas embarcados/edge**: **embeddings fatorizados** reduzem parâmetros de entrada.
- **Educação/Pesquisa**: estudar efeitos de **atenção local** vs **global** com o _gate_.

> **Limitações**: sem _beam search_, sem cobertura/penalidades, sem BPE dropout, sem _length normalization_ no _greedy_.

---

## 🧩 Detalhes de implementação importantes
- `nn.MultiheadAttention(..., batch_first=True)` aceita tensores `(B, L, E)` diretamente.
- **Atenção local**: `local_window_mask(L,w)` cria máscara booleana `(L,L)` com `True` fora da janela (±`w`). Pode ser combinada com causal por `combine_masks`.
- **Causal**: `causal_mask(L)` mascara posições futuras (`triu` com diagonal=1).
- **Cross-attn** no decoder**:** `key_padding_mask=src_keypad` (sem causal).
- **Label smoothing**: reduz overfitting e gradientes extremos, especialmente com vocabulários grandes.
- **Clipping de gradiente**: `clip_grad_norm_(..., max_norm=1.0)` para estabilidade.
- **Seed** reproduzível**:** `torch.manual_seed(seed)`.

---

## 💾 Salvamento/Carregamento de pesos
```python
# salvar
torch.save(model, "translingua.pt")

# carregar
model = torch.load("translingua.pt", map_location="cpu")
model.eval()
```

---

## 🧑‍💻 IDEs recomendadas
- **VS Code**: leve, excelente suporte a Python, Jupyter e debugging.
- **PyCharm** (Community/Professional): refatoração avançada, inspeções estáticas.
- **JupyterLab**: ideal para experimentos rápidos e visualizações.
- **VS Code Remote/Dev Containers**: ambiente reprodutível.

**Extensões úteis**: Python, Pylance, Jupyter, GitLens, EditorConfig, Markdown All in One.

---

## 🔧 _Troubleshooting_
- **`RuntimeError: The shape of the 2D attn_mask is not correct`**  
  Garanta que `attn_mask` seja `(L,L)` **booleana** e que `key_padding_mask` seja `(B,L)`.
- **`CUDA out of memory`**  
  Reduza `batch_size`, `d_model` ou `num_layers`; aumente `local_window` para reduzir custo? (local não muda custo global, mas você pode treinar com menos camadas).
- **Vazamento de tamanho** (sentenças muito longas)  
  Ajuste `max_len`; confirme truncamento `max_len-2` para BOS/EOS.
- **Resultados fracos**  
  Aumente `epochs`, use `beam search`, ajuste `lr`, treine SentencePiece com vocabulário maior e cobertura adequada.

---

## 📝 Roadmap (sugestões)
- [ ] Beam search com _length penalty_
- [ ] _Weight tying_ entre `tgt_embed.projection` e `output_proj`
- [ ] Máscara local com janelas **dilatadas** ou **dinâmicas**
- [ ] _Relative positional encoding_ (T5/DeBERTa)
- [ ] _Knowledge distillation_ para modelos menores

---

## 📚 Referências (sugestões de leitura)
- Vaswani et al., **"Attention Is All You Need"** (2017).  
- Kudo & Richardson, **"SentencePiece: A simple and language independent subword tokenizer and detokenizer"** (2018).
- Loshchilov & Hutter, **"Decoupled Weight Decay Regularization"** (AdamW, 2019).
- Hendrycks & Gimpel, **"Gaussian Error Linear Units (GELUs)"** (2016/2020).
- Szegedy et al., **Rethinking the Inception Architecture for Computer Vision** (label smoothing, 2016).

---

## 📄 Licença
Defina aqui a licença do seu projeto (ex.: MIT, Apache-2.0).
