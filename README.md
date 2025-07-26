# 🧠 Fake vs Real Face Detector

An AI-powered web application that detects whether a face image is **real** or **AI-generated** (e.g., deepfake, GAN-generated).  
Built using **TensorFlow**, **MobileNetV2**, and **Streamlit** for an easy-to-use interface.

---

## 📌 Features

- ✅ Upload face image and get prediction
- 🔍 Classifies image as **REAL** or **FAKE**
- 📈 Displays confidence score
- 🖼 Side-by-side result display
- 🧠 Trained using transfer learning (MobileNetV2)
- 🌐 Deployed online using Streamlit Cloud

---

---

## 🚀 Live Demo

🔗 [Click here to use the app](https://ai-fake-image-detector.streamlit.app)


---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** TensorFlow, Keras
- **Model:** MobileNetV2 (transfer learning)
- **Language:** Python
- **Libraries:** NumPy, Pillow, Matplotlib

---

## 🧠 Model Training

- Dataset: 2000+ real and fake face images
- Model: MobileNetV2 with top-30 layers fine-tuned
- Accuracy: ~61% validation accuracy (can be improved)
- Trained using Keras and TensorFlow 2.x

---

## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/your-username/ai-fake-image-detector.git
cd ai-fake-image-detector
pip install -r requirements.txt
streamlit run app.py

