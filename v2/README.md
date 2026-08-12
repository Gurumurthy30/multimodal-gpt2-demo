# GPT-2 + Vision (1-Stage On-the-Fly VLM Architecture Specification)

This document outlines the **1-Stage neural network architecture, layer dimensions, parameter counts, dataflow, and training specification** for **GPT2VL** featuring:
- **1-Stage On-the-Fly Feature Extraction**: Frozen ViT-B/16 vision encoder processing raw image batches dynamically (0 disk feature cache footprint).
- **Trainable Multimodal Adapter**: **Perceiver Resampler** (64 latent tokens, 2 layers, 8 heads) + **Gated Sparse Cross-Attention** (spliced before GPT-2 layers 3, 6, 9) + **GPT-2 Small** language backbone.
- **Exact Resumable Checkpointing**: Seamless pause & resume with exact batch-level DataLoader fast-forwarding, full optimizer/scheduler/scaler state restoration, and RNG state recovery.

---

## 1. Executive Summary & Parameter Overview

The **GPT2VL** model uses an on-the-fly multimodal pipeline to optimize training speed, VRAM efficiency, and disk space usage on Google Colab NVIDIA T4 GPUs:

1. **On-the-Fly ViT Vision Encoder**: ViT-B/16 processes RGB images into patch feature tensors `(B, 197, 768)` dynamically under FP16 mixed precision during training forward passes, eliminating disk storage bottlenecks.
2. **Trainable Multimodal Adapter & Decoder**:
   - **Perceiver Resampler**: 2 layers, 64 latent tokens, 8 attention heads (**Trainable**).
   - **Gated Sparse Cross-Attention**: Spliced before GPT-2 layers 3, 6, and 9 with a learnable gate $\alpha$:
     $$H_{\text{new}} = H + \alpha \cdot \text{CrossAttention}(\text{Norm}(H), \text{VisualContext})$$
     $\alpha$ is initialized to `0.0` so cross-attention starts with zero disruption to pre-trained GPT-2 text generation.
   - **GPT-2 Small Backbone**: FP16, frozen.

### Parameter Breakdown Summary

| Component | Layer / Module | Parameter Count | % of Total | Trainable Status |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Encoder** | Torchvision `vit_b_16` (ImageNet-1K) | 85,798,656 | 32.31% | **Frozen (On-the-Fly)** |
| **Vision Projection** | *Implicit Identity / Direct* | 0 | 0.00% | N/A |
| **Perceiver Resampler** | 64 Latents + 2 Blocks (Cross-Attn + FFN) | 9,506,304 | 3.58% | **TRAINABLE** |
| **Gated Cross-Attention** | 3 Blocks (spliced before L3, L6, L9; +3 $\alpha$ gates) | 7,091,715 | 2.67% | **TRAINABLE** |
| **GPT-2 Text Backbone** | Embeddings + 12 Decoder Blocks + LM Head | 163,119,592 | 61.44% | **Frozen** |
| **Total Model Parameters** | | **265,516,267** | **100.00%** | |
| **Total Trainable Parameters**| | **16,598,019** | **6.25%** | **Active** |

---

## 2. 1-Stage On-the-Fly Pipeline Architecture & Dataflow

```mermaid
graph TD
    HF["Hugging Face: Trickxter/COCO2017-captions"] -->|Dynamic DataLoader| Batch["Batch Images (B, 3, 224, 224) & Captions (B, L)"]
    Batch -->|Frozen FP16 ViT| ViT["Torchvision ViT-B/16 Encoder"]
    ViT -->|Patch Tokens (B, 197, 768)| Resampler["Perceiver Resampler (2 Blocks, 64 Latents)"]
    Resampler --> VisualContext["Compressed Visual Tokens (B, 64, 768)"]

    Batch -->|Clean Captions| Emb["GPT-2 WTE + WPE (B, L, 768)"]
    Emb --> L0_2["GPT-2 Blocks 0 - 2"]
    VisualContext -. Gated Cross-Attn (alpha) .-> Cross3["Gated Cross-Attn Block (L3)"]
    L0_2 --> Cross3
    Cross3 --> L3_5["GPT-2 Blocks 3 - 5"]
    VisualContext -. Gated Cross-Attn (alpha) .-> Cross6["Gated Cross-Attn Block (L6)"]
    L3_5 --> Cross6
    Cross6 --> L6_8["GPT-2 Blocks 6 - 8"]
    VisualContext -. Gated Cross-Attn (alpha) .-> Cross9["Gated Cross-Attn Block (L9)"]
    L6_8 --> Cross9
    Cross9 --> L9_11["GPT-2 Blocks 9 - 11"]
    L9_11 --> FinalNorm["Final LayerNorm"]
    FinalNorm --> LMHead["LM Head Linear (768 -> 50258)"]
    LMHead --> Logits["FP16 AMP CrossEntropyLoss"]
```

---

## 3. Dataset Preprocessing & Cleaning Rules

* **Source Dataset**: `Trickxter/COCO2017-captions`
  - Train: `Data/train-*.parquet` (37 files, 118,287 images)
  - Validation: `Data/validation-*.parquet` (2 files, 5,000 images)
* **Caption Normalization Rules**:
  1. Unicode NFKD normalization (`unicodedata.normalize('NFKD', cap)`).
  2. Non-printable & invalid character removal.
  3. Collapse repeated whitespace (`re.sub(r'\s+', ' ', cap)`).
  4. Strip leading & trailing spaces (`cap.strip()`).

---

## 4. Detailed Layer Specifications

### 4.1 Perceiver Resampler (`PerceiverResamplerSF`)
* **Latent Parameter Tensor**: `self.latents` of shape `(64, 768)` $\implies 49,152$ params.
* **Depth**: 2 blocks (`perceiver_depth = 2`, `perceiver_heads = 8`).
* **Block Composition**:
  - `norm_latent`: `LayerNormalization(768)` ($1,536$)
  - `norm_media`: `LayerNormalization(768)` ($1,536$)
  - `cross_attn`: `Cross_MultiHead_Attention(embed_dim=768, num_heads=8)` ($2,362,368$)
  - `ffn_norms`: `LayerNormalization(768)` ($1,536$)
  - `ffns`: `FF_GELU(embed_dim=768, hidden_dim=1536)` ($2,361,600$)
* **Total Resampler Parameters**: $49,152 + 2 \times 4,728,576 = 9,506,304$ (**Trainable**).

### 4.2 Gated Sparse Cross-Attention Blocks (`GatedSparseCrossAttnBlock`)
* **Splicing Positions**: Spliced before GPT-2 decoder layers **3, 6, and 9**.
* **Gating Formulation**:
  $$H_{\text{new}} = H + \alpha \cdot \text{Dropout}\big(\text{CrossAttn}(\text{LayerNorm}(H), \text{VisualContext})\big)$$
  - `alpha`: `nn.Parameter(torch.zeros(1))` $\implies$ Initialized to `0.0`.
* **Total Parameters for 3 Gated Blocks**: $3 \times (1,536 + 2,362,368 + 1) = 7,091,715$ (**Trainable**).

---

## 5. Hyperparameter Configuration (`CFG`)

```python
CFG = {
    # Dataset & Paths
    "dataset_name": "Trickxter/COCO2017-captions",
    "checkpoint_dir": "./gpt2vl_checkpoints",
    "gdrive_checkpoint_dir": "/content/drive/MyDrive/GPT2VL_Checkpoints",

    # GPT-2 Small Text Backbone
    "vocab_size": 50258,
    "embed_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 3072,
    "context_length": 128,
    "dropout": 0.1,
    "qkv_bias": True,

    # Vision & Multimodal Gated Cross-Attn
    "vision_dim": 768,
    "cross_attention_pos": [3, 6, 9],
    "num_visual_tokens": 64,    # 64 latent tokens
    "perceiver_depth": 2,       # 2 perceiver layers
    "perceiver_heads": 8,       # 8 perceiver heads

    # Training Setup & Scheduler
    "batch_size": 16,
    "grad_accum_steps": 2,      # Effective batch size = 32
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "warmup_pct": 0.05,         # 5% warmup steps
    "weight_decay": 0.01,
    "num_epochs": 10,
    "time_budget_minutes": 270, # 4.5 hours max Colab budget
    "amp_dtype": "fp16",        # T4 optimized FP16 mixed precision
}
```

---

## 6. Checkpointing, Google Drive Sync & Exact Resumption

To prevent loss from Google Colab session timeouts:
1. **Model Weights**: Saved as `safetensors` format (`model_trainable.safetensors`).
2. **Metadata & Training State**: Saved as `checkpoint_state.pt` (includes `optimizer`, `scheduler`, `scaler`, `step`, `epoch`, `loss_history`, `alpha_values`, `rng_torch`, `rng_cuda`, `rng_python`, `rng_numpy`).
3. **Google Drive Sync**: Automatically copied to `/content/drive/MyDrive/GPT2VL_Checkpoints/` after every epoch or checkpoint step.
4. **Exact Resumability**: On reload, the trainer restores weights and state dicts, calculates `start_batch_in_epoch = (start_step * grad_accum_steps) % steps_per_epoch`, and fast-forwards the DataLoader past completed batches in the epoch to resume execution seamlessly from the exact uncompleted batch.
