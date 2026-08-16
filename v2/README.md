# GPT2VL Version 2 — Production-Grade 1-Stage VLM Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)

Version 2 (**GPT2VL**) is a production-ready **Vision-Language Model (VLM)** built from scratch by combining a frozen **OpenAI CLIP ViT-B/16** vision encoder with a **GPT-2 Small** language backbone via a **Flamingo-style Perceiver Resampler** and **Gated Sparse Cross-Attention** fusion layers implemented using modular `Stackformer` primitives.

---

## 🌟 Key Technical Innovations & Highlights

* **Modular `Stackformer` Primitives:** Uses optimized `Stackformer` building blocks for fused QKV linear projections ($2304 \times 768$) and modular layer stacks.
* **Flamingo-Style Perceiver Resampler (`PerceiverResamplerSF`):** Compresses 197 variable spatial patch tokens (1 CLS + 196 image patch tokens from ViT-B/16) down to a fixed set of **64 latent query tokens** across 2 cross-attention resampler layers.
* **Gated Sparse Cross-Attention (`GatedSparseCrossAttnBlock`):** Inserted into every 4th GPT-2 block (layers 0, 4, 8). Uses a zero-initialized tanh-gating parameter $\alpha$ ($\tanh(\alpha_0) = 0.0$), allowing the language model to initially retain 100% of its language prior while smoothly integrating visual features during training:
  $$H_{\text{out}} = H_{\text{in}} + \tanh(\alpha) \cdot \text{Dropout}(\text{CrossAttn}(\text{LN}(H_{\text{in}}), \text{VisualContext}))$$
* **FP32-Upcast LayerNorm & Mixed Precision Stability:** Prevents FP16 underflow/overflow in deep residual cross-attention blocks by upcasting LayerNorm math to FP32, enforcing gradient clipping (`max_norm=1.0`), and dropping non-finite loss batches.
* **Object-Oriented `CheckpointManager`:** Manages state persistence with `safetensors` weight serialization, exact batch fast-forwarding for mid-epoch resumption, and optional background Google Drive synchronization.
* **Multi-Reference `CaptionEvaluator`:** Quantitative evaluation pipeline measuring corpus-level **BLEU-1..4** (with NLTK method3 smoothing) and **METEOR** scores on multi-reference ground-truth validation captions.

---

## 📐 Architecture Specifications

| Attribute | Specification |
| :--- | :--- |
| **Vision Backbone** | `openai/clip-vit-base-patch16` (Frozen, 197 patch tokens) |
| **Language Backbone** | GPT-2 Small via `Stackformer` (12 layers, 12 heads, 768 hidden dim) |
| **Multimodal Connector** | `PerceiverResamplerSF` (64 latents, depth 2, head dim 64, FFN dim 1536) |
| **Cross-Attention Sub-layers** | Gated Sparse Cross-Attention at layers [0, 4, 8] with tanh gating |
| **Total Model Parameters** | **276,561,408** (~276.56M) |
| **Trainable Parameters** | **28,413,699** (~28.41M / **10.27%** of total model) |
| **Dataset** | `AKCIT/coco2017-captioning` (118,287 training images, 5,000 val images) |
| **Context Length** | 128 tokens |
| **Precision** | AMP FP16 mixed precision with FP32 LayerNorm upcasting |

---

## 📊 Quantitative Benchmark Results

The model was trained on COCO 2017 training set (5x caption-expanded, 586,748 pairs) and evaluated on a held-out 5,000-image full validation split with multi-reference captions.

### Epoch Progress & Evaluation

| Epoch | Step / Total | Train Loss | Eval Loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 17,048 / 18,336 | 3.9583 | 2.2481 | 0.6642 | 0.4642 | 0.3028 | 0.2002 | 0.4212 |
| **Epoch 2** | 26,216 / 26,216 | 2.3041 | 2.0758 | 0.6950 | 0.5030 | 0.3450 | 0.2390 | 0.4630 |

### Final Full Validation Set Metrics (5,000 Images)

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Validation Loss** | **2.0513** | Teacher-forced cross-entropy loss over full 5,000 val set |
| **BLEU-1** | **0.6975** | Unigram precision (ground-truth caption overlap) |
| **BLEU-2** | **0.5052** | Bigram precision |
| **BLEU-3** | **0.3448** | Trigram precision |
| **BLEU-4** | **0.2362** | 4-gram precision (standard machine translation & captioning metric) |
| **METEOR** | **0.4597** | Harmonic mean of precision and recall with stemming & synonym matching |

---

## 📁 File Structure in `v2/`

```text
v2/
├── README.md                              <- Technical documentation & benchmark results (this file)
└── gpt2_vision_stackformer_v3_train.ipynb <- Unified Jupyter Notebook (Training, Resumption, Sampling & Eval)
```

---

## 🚀 Quickstart Guide

### 1. Requirements & Dependencies

To execute the training or evaluation pipeline locally or on Google Colab (T4 GPU runtime recommended), install the required Python packages:

```bash
pip install torch torchvision transformers datasets evaluate nltk matplotlib tqdm safetensors
git clone https://github.com/stackformer-labs/Stackformer.git
pip install -e Stackformer
```

### 2. Running the Unified Notebook

1. Open [`gpt2_vision_stackformer_v3_train.ipynb`](gpt2_vision_stackformer_v3_train.ipynb) in Google Colab or VS Code / Jupyter Lab.
2. Select a GPU runtime (CUDA device required for AMP FP16 training).
3. Execute all cells sequentially:
   - **Cells 1–3:** Installs dependencies and imports Stackformer modules.
   - **Cells 4–10:** Defines model architecture (`CLIPVisionEncoder`, `PerceiverResamplerSF`, `GatedSparseCrossAttnBlock`, `GPT2VL`).
   - **Cells 11–18:** Configures dataset loader, `CheckpointManager`, `CaptionEvaluator`, and `GPT2VLTrainer`.
   - **Cells 19–23:** Initializes weights, prepares dataloaders, and runs or resumes training.
   - **Cells 24–27:** Generates qualitative image captions and executes full 5,000-image BLEU/METEOR evaluation.

### 3. Checkpoint Management & Auto-Resuming

The training loop automatically checks for existing checkpoints in `./gpt2vl_v3_checkpoints/` (or Google Drive if mounted):

```python
# To resume training from an existing step:
ckpt_manager = CheckpointManager(checkpoint_dir="./gpt2vl_v3_checkpoints")
model, optimizer, scheduler, scaler, start_epoch, start_step = ckpt_manager.load_latest(
    model, optimizer, scheduler, scaler, device=device
)
```

---

## 💡 Python Inference Example

Below is a standalone snippet demonstrating how to load a trained V2 checkpoint and run greedy image captioning on a custom image:

```python
import torch
from PIL import Image
from transformers import GPT2TokenizerFast, CLIPImageProcessor
from safetensors.torch import load_file

# 1. Initialize tokenizer and image transform
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# 2. Load model architecture (ensure GPT2VL is imported from notebook)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2VL(cfg=CFG).to(device)

# 3. Load trained weights from safetensors checkpoint
trainable_state_dict = load_file("./gpt2vl_v3_checkpoints/model_trainable.safetensors")
model.load_state_dict(trainable_state_dict, strict=False)
model.eval()

# 4. Process image and generate caption
image = Image.open("sample.jpg").convert("RGB")
image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

with torch.no_grad():
    visual_context = model.encode_image(image_tensor)
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    gen_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    
    for _ in range(60):
        logits = model(gen_ids, visual_context=visual_context)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

caption = tokenizer.decode(gen_ids[0, 1:], skip_special_tokens=True)
print(f"Generated Caption: {caption}")
```
