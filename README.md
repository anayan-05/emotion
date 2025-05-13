# emotion
emotion classifier using nlp
# Emotion-Informed Analogy-Based Text Recommender System

This project is an NLP-based web application that:
- Predicts the **emotion** of a given input sentence using **Logistic Regression**
- Recommends similar texts based on **emotion category** and **semantic similarity** using **GloVe word embeddings**
- Supports a user-friendly **Flask** frontend for emotion prediction and text recommendations

---

## 💡 Features

- Trained on a small custom emotion dataset (`emotion_data.csv`)
- Uses **GloVe 100d vectors** for semantic similarity
- Combines **cosine similarity** and basic analogy reasoning (`king - man + woman ≈ queen`)
- Fully functional **Flask web app**
- Extendable and lightweight

---

