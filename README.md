# 🔊 AcoustiClass: Environmental Sound Classification with CNN

## 🎯 Project Overview
**AcoustiClass** is a deep learning-based audio classification system designed to identify urban environmental sounds. It leverages the **UrbanSound8K** dataset and a custom-built **Convolutional Neural Network (CNN)** using **PyTorch** and **torchaudio**. The model processes audio clips into mel spectrograms and predicts their associated sound class, such as sirens, dog barks, and more.

---

## 🧠 Sound Categories
The system can classify the following 10 sound types:
- Air Conditioner  
- Car Horn  
- Children Playing  
- Dog Bark  
- Drilling  
- Engine Idling  
- Gun Shot  
- Jackhammer  
- Siren  
- Street Music  

---

## ⚙️ Key Features
- Custom `Dataset` class for UrbanSound8K  
- Real-time mel spectrogram generation  
- Multi-layer CNN model for feature extraction  
- GPU acceleration support  
- Model saving/loading with `.pth` format  
- Predicts and compares actual vs. predicted sound labels

---


## 🔧 Getting Started

### 🔹 Prerequisites
Ensure you have Python 3.8+ and the following libraries:
```bash
pip install torch torchaudio pandas torchsummary
```
## 🔹 Dataset
Download the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) and update the following paths in the script:

```python
ANNOTATIONS_FILE = "path_to/UrbanSound8K.csv"
AUDIO_DIR = "path_to/audio"
```

## 🏗️ Model Architecture
The CNN includes four convolutional blocks followed by a fully connected layer:

```
[Conv2D → ReLU → MaxPool] × 4  
→ Flatten  
→ Fully Connected (Linear)  
→ Softmax Output (10 Classes)
```
## 🧪 Training
To train the model, run:
```
python Overall_pgm_own.py
```
After training, the model is saved automatically as:

```
feedforwardnet.pth
```
## 🔍 Inference
You can test the model on new audio clips (formatted similarly to UrbanSound8K). A prediction might look like:

```
Predicted: 'siren', expected: 'siren'
```
## 📈 Evaluation
The current setup provides predicted vs. expected labels. Additional evaluation metrics (e.g., accuracy, confusion matrix) can be added in future iterations.

## 🚀 Future Improvements
Real-time audio input and live classification

Web interface with Streamlit or Flask

Augmentation techniques for better generalization

Metrics dashboard with detailed evaluation visuals
