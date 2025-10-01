# Lightweight‑Mark

## [Lightweight-Mark: Rethinking Deep Learning-Based Watermarking](https://openreview.net/pdf?id=ag3uveGZCb)



Yupeng Qiu, Han Fang*, and Ee-Chien Chang.

> This is the source code of paper Lightweight-Mark: Rethinking Deep Learning-Based Watermarking, which is received by ICML' 25.



This repository contains the implementation of a lightweight watermarking framework and two training methods: **DO (Decoding‑Oriented surrogate loss)** and **PH (Detachable Projection Head)**.

---

## 🔍 What this repo provides
- A lightweight encoder/decoder.
- **DO** or **PH** training methods.
- Examples of **pretrained checkpoints** (under `experiments/`).


---


## 🚀 Quick start

### 1) Prepare data
- Training: COCO (or your own dataset). Update dataset paths in **config.py** or your own config.
- Testing: classic USC‑SIPI or your custom images.

### 2) Train
```bash
python train.py
```


### 3) Evaluate
```bash
python test.py 
```
- Replace the checkpoint path with yours.
- The script reports decoding accuracy and visual quality metrics (e.g., PSNR).

---

## 📬 Contact
- Yupeng Qiu — <qiuyupeng1999@gmail.com>



