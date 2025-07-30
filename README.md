# 🎵 Music Genre Classification

This project is a machine learning-based system that classifies music tracks into different genres using audio features. It uses datasets like GTZAN and applies feature extraction techniques (MFCC, Spectral Contrast, etc.) and classification algorithms to achieve accurate predictions.

---

## 📌 Features
- Extracts audio features using `librosa`
- Supports multiple music genres (e.g., Pop, Rock, Jazz, Classical, Hip-Hop, etc.)
- Trains ML models using `RandomForest`, `SVM`, and/or `Neural Networks`
- Provides accuracy metrics and visualizations

---

## 🛠️ Technologies Used
- **Python**
- **Librosa** – for audio feature extraction  
- **NumPy / Pandas** – for data handling  
- **Matplotlib / Seaborn** – for visualizations  
- **Scikit-learn / TensorFlow / Keras** – for model training  

---

## 📂 Dataset
This project uses the **GTZAN Music Genre Dataset**:  
- 1000 audio tracks (30 sec each)
- 10 genres  
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets) or other sources.

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/music-genre-classification.git
   cd music-genre-classification
