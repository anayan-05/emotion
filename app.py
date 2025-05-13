from flask import Flask, request, render_template
from utils import predict_emotion, recommend_text, load_glove_embeddings
import pickle

app = Flask(__name__)

# Load model and embeddings once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
embeddings = load_glove_embeddings("glove.6B.50d.txt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        emotion = predict_emotion(user_input, model, embeddings)
        recommendations = recommend_text(user_input, emotion, embeddings)
        return render_template("index.html", emotion=emotion, recommendations=recommendations, user_input=user_input)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
