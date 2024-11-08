from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import re

app = Flask(__name__)

model = tf.keras.models.load_model(r"path_to_your_model", custom_objects={'KerasLayer': hub.KerasLayer})

sentiment_labels = {0: 'Negative 🤬', 1: 'Neutral 😐', 2: 'Positive 😊'}

def clean(text):
    """Clean input text by removing non-alphabet characters and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(sentence):
    """Predict the sentiment of a given sentence."""
    sentence = clean(sentence)
    sentence = np.array([sentence])
    preds = model.predict(sentence)
    index = np.argmax(preds, axis=1)[0]
    sentiment = sentiment_labels[index]
    return sentiment

@app.route('/', methods=['GET', 'POST'])
def home():
    """Render home page with a form and handle form submissions for sentiment prediction."""
    if request.method == 'POST':
        user_input = request.form.get('sentence')
        if user_input:
            sentiment = predict_sentiment(user_input)
            return render_template('index.html', prediction=sentiment, user_input=user_input)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
