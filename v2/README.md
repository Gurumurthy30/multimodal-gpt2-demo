# Multimodal GPT-2 — Version 2 (GPT2VL / Stackformer V2)

This directory contains **Version 2 (V2)** of **GPT2VL**, a 1-Stage On-the-Fly Vision-Language Model built with modular Object-Oriented design, gated cross-attention fusion, and comprehensive numerical stability safeguards for NVIDIA T4 GPUs.

---

## 🎯 Key Architectural Innovations in V2

```
Raw Image (224x224) ──► Frozen ViT-B/16 (FP16 On-The-Fly) ──► Patch Tokens (197 x 768)
                                                                    │
                                                           Perceiver Resampler (SF)
                                                               (64 Latents, Depth 2)
                                                                    │
                                                            Visual Tokens (64 x 768)
                                                                    │
Text Tokens ──► GPT-2 Small (Stackformer, Frozen) ◄─── Gated Cross-Attn (alpha = 0.0 -> trainable)
                                                              (Inserted at L3, L6, L9)
                                                                    │
                                                            LM Head Logits (50258)
```

1. **Gated Sparse Cross-Attention**: Cross-attention blocks inserted before GPT-2 layers 3, 6, and 9 incorporate a learnable gating parameter $\alpha$ initialized to `0.0`:
   $$H_{\text{new}} = H + \alpha \cdot \text{Dropout}\big(\text{CrossAttn}(\text{LayerNorm}(H), \text{VisualContext})\big)$$
   Starting $\alpha = 0.0$ guarantees **zero disruption** to pretrained GPT-2 language generation at step 0. $\alpha$ gradually adjusts as multimodal alignment trains (clamped to $[-1.0, 1.0]$).
2. **1-Stage Dynamic Pipeline**: Raw images from `Trickxter/COCO2017-captions` are transformed and passed directly into frozen ViT-B/16 during training batch forward passes, eliminating disk feature cache footprint.
3. **Modular Stackformer Integration**: Built on top of `Stackformer` primitives, supporting fused QKV projections ($2304 \times 768$) and modular layer structures.
4. **FP32-Upcasted Normalization Fix**: Overwrites `LayerNorm` and `RMSNorm` to perform `mean`/`var`/`sqrt` reductions internally in `float32` before returning in `float16`. Prevents catastrophic `inf`/`NaN` loss explosion when deep GPT-2 residual activations exceed FP16's ~65,504 range.

---

## 📊 Parameter Overview

| Component | Layer / Module | Parameters | % Total | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Encoder** | Torchvision `vit_b_16` (ImageNet-1K) | 85,798,656 | 32.31% | **Frozen (On-the-Fly)** |
| **Perceiver Resampler** | 64 Latents + 2 Blocks (Cross-Attn + FFN) | 9,506,304 | 3.58% | **TRAINABLE** |
| **Gated Cross-Attn** | 3 Blocks (before L3, L6, L9; +3 $\alpha$ gates) | 7,091,715 | 2.67% | **TRAINABLE** |
| **GPT-2 Text Backbone** | Embeddings + 12 Decoder Blocks + LM Head | 163,119,592 | 61.44% | **Frozen** |
| **Total Model** | | **265,516,267** | **100.00%** | |
| **Trainable Total** | | **16,598,019** | **6.25%** | **Active** |

---

## 📈 Dataset & Quantitative Evaluation Metrics

* **Dataset**: `Trickxter/COCO2017-captions` (118,287 training images, 5,000 validation images).
* **Held-Out Evaluation Split**: 1,000 images reserved deterministically (seed 42) for end-of-epoch quantitative metric tracking.
* **Text Preprocessing**: Normalization via `unicodedata.normalize('NFKD')`, control char removal, whitespace collapsing, and `"a photo"` fallback.
* **Hyperparameters**: Batch size 32, Grad Accumulation 2 (Effective Batch 64), `OneCycleLR` (max LR $1\times 10^{-4}$, 5% warmup), 10 Epochs (18,330 total steps).

### End-of-Epoch Quantitative Evaluation Results

| Epoch | Train Loss | Eval Loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Gating $\alpha$ [L3, L6, L9] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 12.2966 | 2.7970 | 0.222 | 0.093 | 0.050 | 0.034 | 0.155 | `[0.0068, -0.0107, -0.0207]` |
| **2** | 7.5842 | 2.5204 | 0.266 | 0.130 | 0.079 | 0.056 | 0.209 | `[0.0176, -0.0285, -0.0505]` |
| **3** | 5.9370 | 2.3991 | 0.294 | 0.152 | 0.098 | 0.069 | 0.237 | `[0.0268, -0.0462, -0.0795]` |
| **4** | 5.0872 | 2.3261 | 0.309 | 0.163 | 0.106 | 0.076 | 0.253 | `[0.0367, -0.0627, -0.1066]` |
| **5** | 4.5624 | 2.2809 | 0.312 | 0.167 | 0.110 | 0.080 | 0.258 | `[0.0458, -0.0780, -0.1301]` |
| **6** | 4.2035 | 2.2584 | 0.316 | 0.169 | 0.109 | 0.079 | 0.261 | `[0.0531, -0.0905, -0.1493]` |
| **7** | 3.9418 | 2.2450 | 0.319 | 0.171 | 0.112 | 0.081 | 0.265 | `[0.0594, -0.0101, -0.1625]` |
| **8** | 3.7420 | 2.2331 | 0.322 | 0.174 | 0.113 | 0.080 | 0.268 | `[0.0628, -0.1055, -0.1710]` |
| **9** | 3.5851 | 2.2280 | **0.323** | **0.177** | **0.117** | **0.084** | **0.271** | `[0.0645, -0.1081, -0.1744]` |
| **10** | **3.4590** | **2.2281** | **0.323** | **0.177** | 0.116 | 0.083 | **0.271** | `[0.0649, -0.1085, -0.1749]` |

---

## 🛠️ Object-Oriented Software Architecture

- `GPT2VL`: Full 1-Stage VLM module managing ViT encoder, Perceiver Resampler, Stackformer GPT-2, and gated cross-attention.
- `OnTheFlyVisionTextDataset`: Handles dynamic image loading, unicode normalization, and batch collation.
- `CheckpointManager`: Handles saving trainable weights in `safetensors` format, complete metadata/optimizer/scheduler/scaler/RNG states in PyTorch checkpoints, and automatic Google Drive syncing. Fast-forwards DataLoaders past processed batches upon resumption.
- `GPT2VLTrainer`: Object-Oriented training loop with gradient unscaling, norm clipping, alpha parameter clamping, live progress monitoring, time budget exit (4.5 hours), and automatic end-of-epoch evaluation.
- `CaptionEvaluator`: NLTK-based evaluator computing BLEU-1..4 and METEOR scores on greedy-decoded hypotheses.

---

## 🚀 How to Run

1. Open [`gpt2_vision_stackformer_v2_train.ipynb`](gpt2_vision_stackformer_v2_train.ipynb) in Google Colab (GPU runtime recommended, e.g., T4/V100/A100).
2. Execute cells top-to-bottom:
   - Cell 01 & 03 automatically clone `Stackformer` and apply the FP32-upcasted `Normalization.py` patch on disk.
   - Cell 23 transfers pretrained GPT-2 weights into the Stackformer backbone and runs a **Backbone Sanity Check** verifying English completion ("The cat sat on the" $\rightarrow$ floor, bed, couch).
   - Cell 27 launches training for 10 epochs. Checkpoints save locally to `./gpt2vl_checkpoints/` and sync to Google Drive.
3. Cell 30 displays a 6-sample visual comparison grid showing Ground Truth (GT) vs Model Generated (Gen) captions.
