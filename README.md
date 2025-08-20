# CNN_Classification_HandGesture
Hand gesture recognition using CNNs and transfer learning (Pre-trained Architecture). Implements MobileNetV2, ResNet50, VGG16, and a custom CNN to classify Thumbs Up, Peace, and Fist gestures. Achieved up to 95.89% accuracy with MobileNetV2, while the custom CNN excelled in task-specific recognition.

# ✋ Hand Gesture Recognition using CNNs

## 📌 Project Overview
This project focuses on **hand gesture recognition** as an intuitive, non-verbal communication channel for human-computer interaction. Using **Convolutional Neural Networks (CNNs)**, we classify three hand gestures:
- 👍 Thumbs Up  
- ✌ Peace Sign  
- ✊ Fist  

The study compares **pretrained models** (MobileNetV2, ResNet50, VGG16) via transfer learning and a **custom CNN built with Keras Sequential API**.  

---

## 📊 Dataset
- **Collected from 30+ individuals** (different age groups, backgrounds, and orientations).  
- 3 gesture classes: **Thumbs Up, Peace, Fist**.  
- Images resized to **128×128 pixels** for efficiency.  
- **Data Augmentation**: rotation, flipping, zooming, shifting.  
- Expanded dataset size: ~360 images  
  - Training: 297  
  - Validation: 73  

👉 Dataset link: [Google Drive Folder]([https://drive.google.com/drive/folders/1AAhKf8HhBpPOL3C9ae3bvDFjO6zumBem?usp=sharing](https://drive.google.com/drive/folders/1yGLGg8yjLV5OGjN5AYH_3WECdPSh-TSD?usp=sharing))

---

## ⚙️ Installation & Dependencies
### 1. Clone the repository
```bash
git clone https://github.com/your-username/CNN_Classification_HandGesture.git
cd CNN_Classification_HandGesture
````

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and place dataset

Download the dataset from the [Google Drive link]([https://drive.google.com/drive/folders/1AAhKf8HhBpPOL3C9ae3bvDFjO6zumBem?usp=sharing](https://drive.google.com/drive/folders/1yGLGg8yjLV5OGjN5AYH_3WECdPSh-TSD?usp=sharing)) and place it inside the `data/` folder.

### 5. Train a model

* Train custom CNN:

```bash
python train.py --model custom
```

* Fine-tune pretrained model (MobileNetV2, ResNet50, VGG16):

```bash
python train.py --model mobilenet
python train.py --model resnet
python train.py --model vgg16
```

### 6. Evaluate the model

```bash
python evaluate.py --model mobilenet
```

---

## 🧠 Models Used

### 🔹 Pretrained (Transfer Learning)

* **MobileNetV2** → Lightweight, efficient, achieved **95.89% validation accuracy**.
* **ResNet50** → Deep residual network, accuracy **91.78%**.
* **VGG16** → Classic architecture, accuracy **90.41%**.

### 🔹 Custom CNN (Sequential API)

* Input: 128×128×3 images
* Conv + ReLU + MaxPooling layers (3 blocks)
* Dense (64 units, ReLU) + Dropout (0.5)
* Softmax output layer (3 classes)
* Validation accuracy: **76.71%**, but classified **all test images correctly**.

---

## 📈 Test Results

| Model       | Validation Accuracy |
| ----------- | ------------------- |
| MobileNetV2 | 95.89%              |
| ResNet50    | 91.78%              |
| VGG16       | 90.41%              |
| Custom CNN  | 76.71%              |

---

## 🚀 Future Applications

* Real-time webcam gesture recognition
* Larger dataset collection
* Hybrid models combining pretrained + custom CNNs
* Assistive technology (sign language recognition)
* Touchless device control in smart homes
* Gesture-based interaction in VR/gaming

---

## 🛠 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy, Pandas, Matplotlib** (data processing & visualization)
* **OpenCV** (image preprocessing & augmentation)

---

## ✨ Acknowledgements

* Dataset collected by the author.
* Architectures: MobileNetV2, ResNet50, VGG16 (Keras Applications).
* Visualization tools: [NN-SVG](https://alexlenail.me/NN-SVG/LeNet.html).

---

👩‍💻 **Author**: Aditi Prasad Chandavarkar
🎓 MSc Statistics, Christ University (2025)
