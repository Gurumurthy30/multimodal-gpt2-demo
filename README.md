# Multimodal GPT-2 — Vision-Language Model Built From Scratch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces%20Demo-yellow.svg)](https://huggingface.co/spaces/gurumurthy3/Multimodal-Gpt2-Demo)

A lightweight **Vision-Language Model (VLM)** built from scratch by fusing a frozen **Vision Transformer (ViT-B/16)** encoder with a frozen **GPT-2 Small** language backbone via a trainable **Perceiver Resampler** and **Sparse Cross-Attention** fusion layers.

This repository tracks the complete engineering evolution of the model from its initial hackathon prototype (**V1**) to a production-grade 1-Stage VLM with gated cross-attention, numerical stability safeguards, exact resumable checkpointing, and quantitative BLEU/METEOR evaluation (**V2**).

---

## 📜 Project Story: Evolution from V1 to V2

```
                       ┌──────────────────────────────────────────────┐
                       │  V1 Baseline (SIH 2025 Hackathon Prototype)  │
                       │  - Flickr8k dataset (8,000 images)           │
                       │  - Ungated cross-attention layers            │
                       │  - Custom PyTorch building blocks            │
                       │  - Basic FP32 single-GPU training loop       │
                       └──────────────────────┬───────────────────────┘
                                              │
                                       Lessons Learned
                                  (FP16 instability, small
                                    data, no resume/eval)
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │  V2 GPT2VL (1-Stage Production VLM Engine)   │
                       │  - COCO 2017 dataset (118,287 images)        │
                       │  - Gated Sparse Cross-Attention (alpha = 0)  │
                       │  - Stackformer modular architecture          │
                       │  - FP32-upcast LayerNorm (NaN-stable FP16)   │
                       │  - OO CheckpointManager & Colab Auto-Sync    │
                       │  - Quantitative BLEU-1..4 & METEOR Eval      │
                       └──────────────────────────────────────────────┘
```

* **Version 1 (V1)** was built as an initial proof-of-concept for the **Smart India Hackathon 2025**. It demonstrated that inserting a Perceiver Resampler (64 tokens) and cross-attention blocks into a frozen GPT-2 backbone could generate coherent image captions on the Flickr8k dataset with only 11M trainable parameters (~5% of the total model).
* **Version 2 (V2)** completely redesigned the architecture into **GPT2VL** using the modular `Stackformer` framework, scaling training to **COCO 2017 (118,287 images)** on Colab T4 GPUs. V2 introduced **gated cross-attention** ($\alpha$ initialization), **1-stage on-the-fly feature extraction**, **FP32-internal LayerNorm math** to solve deep residual FP16 overflow, exact batch fast-forwarding checkpoint resumption, and an end-of-epoch evaluation suite tracking **BLEU-1..4** and **METEOR** scores.

---

## 📊 Comprehensive Version Comparison (V1 vs V2)

| Feature / Metric | [Version 1 (V1 Baseline)](v1/) | [Version 2 (V2 GPT2VL)](v2/) |
| :--- | :--- | :--- |
| **Language Backbone** | Custom PyTorch GPT-2 Small (124M, 12 layers, 12 heads) | Modular `Stackformer` GPT-2 Small with fused QKV linear projections ($2304 \times 768$) |
| **Vision Encoder** | Torchvision `vit_b_16` (ImageNet-1K), frozen, 196 patch tokens | OpenAI CLIP ViT-B/16 (`openai/clip-vit-base-patch16`), frozen, 197 patch tokens (includes CLS token) |
| **Multimodal Resampler** | Custom `PerceiverResampler` (64 latents, depth 2, FFN hidden dim 256) | `PerceiverResamplerSF` using Stackformer primitives (64 latents, depth 2, FFN hidden dim 1536) |
| **Cross-Attention Fusion** | Ungated standard residual: $x + \text{CrossAttn}(x)$ | **Gated Cross-Attention**: $H + \tanh(\alpha) \cdot \text{Dropout}(\text{CrossAttn}(H))$, $\alpha_0 = 0.0$ at layers [0, 4, 8] |
| **Total Model Parameters** | 222,092,520 (222.09M) | 276,561,408 (276.56M) |
| **Trainable Parameters** | 11,084,288 (11.08M / 5.0%) | 28,413,699 (28.41M / 10.27%) |
| **Training Dataset** | Flickr8k (8,000 images, 40,000 text captions) | `AKCIT/coco2017-captioning` (118,287 training images, 586,748 5x-expanded pairs) |
| **Caption Preprocessing** | Basic BOS/EOS tokenization, context length 256 | Strict Unicode NFKD cleaning, control char removal, whitespace collapse, `"a photo"` fallback, context length 128 |
| **Training Pipeline** | FP32 standard PyTorch loop, 2 epochs, batch size 16, fixed LR ($1\times 10^{-4}$) | AMP FP16 mixed precision, FP32 LayerNorm math, dynamic batching, step fast-forward auto-resumption |
| **Numerical Stability** | None | FP32-upcast LayerNorm math, grad clipping (`max_norm=1.0`), non-finite loss batch dropping, gating $\alpha$ bounds |
| **State Persistence** | Basic end-of-run `torch.save` | `CheckpointManager` OO class (`safetensors` + `pt`), Google Drive auto-sync, exact batch fast-forwarding |
| **Validation & Metrics** | Qualitative manual sampling (39 images) | Quantitative full 5,000-image test set tracking **BLEU-1..4** & **METEOR** |
| **Best Results** | Train Loss: **2.6757** | Train Loss: **2.3041** \| Eval Loss: **2.0513** \| **BLEU-1**: **0.6975** \| **BLEU-4**: **0.2362** \| **METEOR**: **0.4597** |

---

## 📁 Repository Structure

```
.
├── README.md               <- Master project documentation (this file)
├── v1/                     <- Version 1 (Flickr8k baseline, custom PyTorch implementation)
│   ├── README.md           <- V1 architecture specifications, setup, and results
│   ├── trained_model.ipynb <- Self-explanatory V1 training notebook
│   └── examples/           <- Output caption samples (example1.png - example6.png)
└── v2/                     <- Version 2 (COCO2017 1-Stage VLM, Stackformer, Gated Cross-Attn)
    ├── README.md           <- V2 architecture specifications, stability fixes, and eval tables
    └── gpt2_vision_stackformer_v3_train.ipynb <- Unified V2 training, resumption & full 5k eval notebook
```

---

## 🚀 Quickstart & How to Run

### Running Version 1 (V1 Baseline)
1. Navigate to the [`v1/`](v1/) directory.
2. Open [`trained_model.ipynb`](v1/trained_model.ipynb) in Google Colab or Jupyter.
3. Run all cells sequentially. The notebook will automatically download ViT-B/16 and GPT-2 weights, stream the Flickr8k dataset, and train for 2 epochs.
4. Try the interactive Hugging Face demo: [Multimodal GPT-2 Space](https://huggingface.co/spaces/gurumurthy3/Multimodal-Gpt2-Demo).

### Running Version 2 (V2 Gated VLM)
1. Navigate to the [`v2/`](v2/) directory.
2. Open [`gpt2_vision_stackformer_v3_train.ipynb`](v2/gpt2_vision_stackformer_v3_train.ipynb) in Google Colab (T4 GPU runtime recommended).
3. Run all cells. The notebook automatically installs dependencies, initializes CLIP ViT-B/16 + GPT-2 Small via Stackformer, trains/resumes training on COCO 2017, logs quantitative BLEU-1..4 / METEOR evaluation metrics, and saves checkpoints to `./gpt2vl_v3_checkpoints/`.

---

## 📄 License & Acknowledgments

* Built with [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html), and [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).
* V2 utilizes modular transformer primitives from [Stackformer](https://github.com/stackformer-labs/Stackformer.git).
* Image datasets provided by Flickr8k and [COCO 2017](https://cocodataset.org/).
