import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from utils import sentence_to_avg, load_glove_embeddings

# Load dataset
data = pd.read_csv("emotion_data.csv")

# Load embeddings
embeddings = load_glove_embeddings("glove.6B.50d.txt")

# Feature extraction
X = np.vstack([sentence_to_avg(text, embeddings) for text in data['text']])
y = data['emotion']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
