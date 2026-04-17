# Compositional Image Retrieval for E-Commerce (Fashion-IQ) 🔍👕

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Milvus](https://img.shields.io/badge/Milvus-Lite-0D6EFD.svg)](https://milvus.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blueviolet.svg)](https://github.com/facebookresearch/faiss)



## 📖 Overview
This project tackles the "Compositional Image Retrieval" (CIR) problem in e-commerce. Traditional search engines rely strictly on either text or images. This system bridges the gap by allowing users to query using a **Reference Image + Relative Text Modification** (e.g., providing a picture of a dress and typing *"is shorter and has floral patterns"*).

Designed with hardware efficiency in mind, the pipeline successfully trains and evaluates on the **Fashion-IQ** dataset using constrained hardware (e.g., 4GB VRAM GPUs) without sacrificing production-level scalability.

>ONLINE DEMO: [DEMO on Huggingface](https://huggingface.co/spaces/zekifurkan/fashion-image-retrieval)

<img width="1526" height="664" alt="Screenshot from 2026-03-27 15-38-56" src="https://github.com/user-attachments/assets/2e18ef0d-fd65-4ec2-a8b5-2c0017f7a4d4" />

## 🏗️ Architecture & Features
* **Frozen Backbone:** Uses OpenAI's `CLIP (ViT-B/32)` as a frozen feature extractor to save memory.
* **Residual Combiner:** A lightweight, trainable MLP with `LayerNorm` that learns the cross-modal interaction (Image Feature + Text Delta = Target Query).
* **Memory Optimization:** Implements Offline Feature Caching, Automatic Mixed Precision (AMP), and Gradient Accumulation to maximize effective batch sizes on limited VRAM.
* **Vector Database:** Integrates both `FAISS` and `Milvus Lite` for millisecond-latency similarity search across thousands of products.
* **In-batch Negative Sampling:** Uses Symmetric Cross-Entropy / Triplet Loss with in-batch negatives for robust contrastive learning.


## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/fashion-iq-cir.git](https://github.com/YOUR_USERNAME/fashion-iq-cir.git)
cd fashion-iq-cir

# Install requirements
pip install torch torchvision torchaudio
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
pip install faiss-cpu pymilvus tqdm pyyaml
```

## 📂 Dataset Preparation

Download the [Fashion-IQ Dataset](https://github.com/XiaoxiaoGuo/fashion-iq) and organize it into your root directory as follows:

```text
fashionIQ_dataset/
├── captions/           
│   ├── cap.dress.train.json
│   └── ...
├── image_splits/       
│   ├── split.dress.train.json
│   └── ...
└── images/             
    ├── B0012345.jpg
    └── ...
```

## 🚀 Usage

### 1. Training the Model
The training script automatically extracts and caches CLIP features on the first run, drastically speeding up subsequent epochs.

```bash
python train.py
```

### 2. Evaluation
Evaluation (Recall@10, Recall@50) is integrated into the training loop and runs dynamically based on the `config.yml` settings. You can also run it standalone:

```bash
python eval.py
```

### 3. Interactive Search (Inference)
Gradio based web interface

```bash
python demo_interface.py
```

## 📊 Benchmarking & Results

*Results are reported on the Fashion-IQ validation set.*

| Category | Zero-Shot CLIP (R@10) | Residual Combiner (R@10) |
| :--- | :---: | :---: |
| **Dress** | ~12.0% | ~17.15% |
| **Shirt** | ~10.5% | ~16.65% |
| **Toptee** | ~11.8% | ~19.5% |

> *Note: Zero-Shot CLIP baselines represent the standard mathematical addition of normalized image and text features without a trained combiner module, as cited in CIR literature.*

## 📚 References

* Wu, H., Gao, J., Guo, X., Al-Halah, Z., Rennie, S., Kondadadi, R., & Jain, L. (2021). The Fashion-IQ Dataset: Retrieving Images by Combining Text and Image Queries. *CVPR*.
