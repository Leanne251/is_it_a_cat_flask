
import fastai
from flask import Flask, request, jsonify
from fastai.learner import load_learner
import os
from pathlib import Path
import numpy as np
import torch
from flask_cors import CORS  # Import CORS for Flask
import builtins

# Couldnt read this function from the model.pkl file.
# its saying if the first letter is uppercase then it is a cat.
# This is how the oxford list of pets is structured.
# So we are using this to determine if the image is a cat or not.
def is_cat(x): return x[0].isupper()  

builtins.is_cat = is_cat

# Use environment variable to switch between local/remote
USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "False") == "True"

model_dir = Path("model")
model_path = model_dir / "cat_model.pkl"
model_url = "https://huggingface.co/datasets/Leanne251/is_it_a_cat_model/resolve/main/a_cat_model.pkl"

os.makedirs(model_dir, exist_ok=True)

if USE_REMOTE_MODEL and not model_path.exists():
    print("Downloading model from Hugging Face...")
    r = requests.get(model_url)
    r.raise_for_status()
    with open(model_path, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully.")


learn = load_learner(model_path, cpu=True)


app = Flask(__name__)
CORS(app) # Allow cross-origin requests / This enables CORS for all routes


@app.route("/")
def hello():
    return "Flask app running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    img_file = request.files['image']
    print(f"Received image: {img_file.filename}")

    try:
        img = PILImage.create(img_file)
        print(f"Image created successfully: {img_file.filename}")
    except Exception as e:
        return jsonify({'error': f'Image creation failed: {str(e)}'}), 400

    try:
        pred, _, probs = learn.predict(img)
        print(f"Prediction: {pred}, probabilities: {probs}")
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500
       

    return jsonify({
        'prediction': str(pred),
        'confidence': float(probs.max())
    })
   

if __name__ == "__main__":
    app.run(debug=True, port=5000)



# Run with python3 fastai_classifier.py

