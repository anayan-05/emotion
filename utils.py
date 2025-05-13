import numpy as np

def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def sentence_to_avg(sentence, embeddings):
    words = sentence.lower().split()
    valid_vectors = [embeddings[word] for word in words if word in embeddings]
    if not valid_vectors:
        return np.zeros(100,)
    return np.mean(valid_vectors, axis=0)

def predict_emotion(text, model, embeddings):
    avg_vec = sentence_to_avg(text, embeddings).reshape(1, -1)
    return model.predict(avg_vec)[0]

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def recommend_text(user_input, emotion, embeddings):
    recommendations = {
        'happy': ["Keep smiling!", "Happiness looks good on you!", "Spread the joy!"],
        'sad': ["It's okay to feel sad.", "Take time to heal.", "You are not alone."],
        'angry': ["Take a deep breath.", "Channel your energy wisely.", "Let’s find calm."],
        'fear': ["You are strong.", "Fear is natural.", "Bravery starts with fear."],
        'love': ["Love conquers all.", "Cherish every moment.", "Love is powerful."],
        'surprise': ["Life is full of surprises!", "Embrace the unexpected.", "Whoa, that's exciting!"]
    }
    return recommendations.get(emotion, ["Stay positive!"])
