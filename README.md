# 🧠 NLP Emotion Classifier

This project is a **Natural Language Processing (NLP) application** that detects the **emotion behind a piece of text**. It predicts five basic emotions: **Joy, Anger, Sadness, Fear, and Surprise**. The app is built using **Streamlit, NLTK, and Scikit-learn**.

---

## 📌 Features

- Detects emotions from text input.
- Uses **TF-IDF Vectorization** and **Multinomial Naive Bayes** for classification.
- Shows **emoji-based interactive visualization** for each emotion:
  - 😊 Joy  
  - 😡 Anger  
  - 😢 Sadness  
  - 😱 Fear  
  - 😮 Surprise
- Animated emoji display using CSS for a fun, interactive UI.
- Sidebar with project information and option to **view the training dataset**.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Streamlit** – for the interactive UI
- **NLTK** – for text preprocessing (tokenization, stopword removal)
- **Scikit-learn** – for TF-IDF vectorization and Naive Bayes classification
- **Pandas** – for handling dataset

---

## 💾 Dataset

The dataset contains **~200 sentences** labeled with five emotions: `Joy`, `Anger`, `Sadness`, `Fear`, `Surprise`. The dataset is balanced to improve the accuracy of predictions.

---

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/nlp-emotion-classifier.git
cd nlp-emotion-classifier
