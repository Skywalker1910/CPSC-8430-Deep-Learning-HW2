# **Deep Learning HW2 - Video Caption Generation**
# Video Captioning with Seq2Seq Model

## Installation
Make sure you have Python 3.x and PyTorch installed. Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

This repository contains the implementation of a Seq2Seq (Sequence-to-Sequence) model for video caption generation as part of Deep Learning Homework 2. The code files, along with the necessary scripts for training, inference, and evaluation, are provided in the **`hw2_1`** branch.

## **1. Branch Information**
All the code files for this homework are available in the branch **`hw2_1`**.

## **2. Dataset Download**
You can download the dataset used in this homework from the following link:
**[(https://drive.google.com/file/d/1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ/view)](#)**

The dataset contains video feature files in `.npy` format, along with `training_label.json` and `testing_label.json` files, which provide the ground truth captions for training and testing, respectively.

## **3. Code Files**

### **Important Files:**
- **`train.py`**: Script to train the Seq2Seq model on the video caption dataset.
- **`dataset.py`**: Custom PyTorch `Dataset` class to load the video features and captions.
- **`seq2seq.py`**: Defines the Seq2Seq model with an encoder, decoder, and attention mechanism.
- **`run_inference.py`**: Script for generating captions for test videos using the trained model.
- **`bleu_eval.py`**: Script to evaluate the generated captions using the BLEU score.

### **Directory Structure:**
```
- data/
  - training_data/
    - feat/
      - *.npy  (video features for training)
  - testing_data/
    - feat/
      - *.npy  (video features for testing)
  - training_label.json (Ground truth captions for training)
  - testing_label.json  (Ground truth captions for testing)
  
- train.py
- dataset.py
- seq2seq.py
- run_inference.py
- bleu_eval.py
- hw2_seq2seq.sh
```

Full directory structure can be found in file_structure.txt file.
