# Vision-GPT: Multimodal Image Captioning

A lightweight multimodal model combining GPT-2 and Vision Transformer for image captioning, built for **Smart India Hackathon 2025**.

## 🎯 Architecture

**Sparse Cross-Attention Fusion** (inspired by Llama 3.2)

- **GPT-2** (124M params) - Language model backbone ❄️ frozen
- **ViT-B/16** (87M params) - Visual encoder ❄️ frozen  
- **Cross-Attention** (11M params) - Vision-language fusion 🔥 trainable
  - Inserted at layers 3, 6, 9
  - Perceiver Resampler for efficient visual token compression

**Total**: 222M params | **Trainable**: 11M params (5%)

## 📊 Training

- **Dataset**: Flickr8k
- **Epochs**: 2
- **Final Loss**: 2.632
- **Strategy**: Freeze pretrained models, train only cross-attention layers

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch torchvision transformers pillow

# Load model
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download(
    repo_id="gurumurthy3/vision-gpt-flickr8k",
    filename="model_fp32/model_checkpoint.pth"
)
```

## 🎨 Demo

Try it on Hugging Face Spaces: [Multimodal GPT-2 Demo](https://huggingface.co/spaces/gurumurthy3/Multimodal-Gpt2-Demo)

## 📝 Citation

Inspired by the sparse cross-attention approach from Meta's Llama 3.2 Vision paper.