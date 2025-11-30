# MultiModal_AI_Search_Engine
A multimodal AI agent powered by Qwen2.5-VL-3B that sees, reasons, and 'grounds' answers by drawing bounding boxes. Optimized for Google Colab's free T4 GPU using 4-bit quantization. Features a full end-to-end pipeline for visual reasoning and object detection.
# 👁️ Multimodal Vision Agent (Qwen2.5-VL-3B)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B-violet)
![Hardware](https://img.shields.io/badge/GPU-T4%20Compatible-green)
![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)

A lightweight but powerful Multimodal AI Agent capable of seeing, reasoning, and "grounding" its answers physically in an image. Built on **Qwen2.5-VL-3B**, this agent can chat about images and detect objects (draw bounding boxes) using natural language prompts.

**Key Feature:** Optimized to run entirely on free cloud resources (Google Colab T4 GPU) using 4-bit quantization.

---

## 🏗️ Project Architecture & Stages

This project is built in four distinct stages to ensure performance and stability on limited hardware.

### Stage 1: The Environment & Dependencies
We utilize the bleeding-edge versions of the Hugging Face ecosystem to support the Qwen2.5 architecture.
* **Transformers:** Handles model loading and tokenization.
* **Qwen-VL-Utils:** Essential helper functions for processing vision inputs (videos/images) specifically for Qwen.
* **BitsAndBytes:** Provides the 4-bit quantization kernels required to shrink the model size.
* **Accelerate:** Manages model placement across CPU/GPU automatically.
### Do restart the session after installing the essential libraries 
### Stage 2: Optimized Model Loading (The "Brain")

The core model is `Qwen/Qwen2.5-VL-3B-Instruct`. To fit this into a standard 15GB GPU (Colab T4), we employ **NF4 (NormalFloat 4-bit) Quantization**.
* **Without Optimization:** The model requires ~7-8GB VRAM + overhead for high-res images, leading to crashes.
* **With 4-Bit:** The model weights are compressed to ~2.5GB, leaving ample room for image processing and inference.

### Stage 3: Dynamic Resolution & Safety Checks
Qwen2.5-VL uses "Native Resolution" (NaViT), meaning it does not squash images into small squares. While this improves accuracy, it can cause memory explosions with 4K images.
* **Safety Mechanism:** The code includes a pre-processor that checks image dimensions.
* **Logic:** If an image exceeds `1024px` on any side, it is automatically resized (preserving aspect ratio) to prevent `CUDA OutOfMemory` errors.

### Stage 4: Visual Grounding & Visualization

Unlike standard chatbots that just output text, this agent performs **Visual Grounding**.
1.  **Prompt:** The user asks to "Detect the dog."
2.  **Reasoning:** The model generates coordinates in a `[0-1000]` normalized space (e.g., `[250, 250, 500, 500]`).
3.  **Parsing:** We use Regex to extract these coordinates from the text response.
4.  **Drawing:** We map the coordinates back to the original image pixel size and draw red bounding boxes using `PIL`.

---

## 🛠️ Installation

### Option 1: Google Colab (Recommended)
This project is designed to run in a notebook environment.
1.  Open Google Colab.
2.  Set Runtime to **T4 GPU**.
3.  Copy the code from `main.py` (or the provided notebook cells).

### Option 2: Local Setup
Requires a GPU with at least 6GB VRAM and CUDA installed.

```bash
# Clone the repository
git clone [https://github.com/yourusername/vision-agent-qwen.git](https://github.com/yourusername/vision-agent-qwen.git)
cd vision-agent-qwen

# Install dependencies (must use updated versions)
pip install git+[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
pip install qwen-vl-utils accelerate bitsandbytes pillow requests
