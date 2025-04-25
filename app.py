import sys
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

from config import CLASS_NAMES, MODEL_PATH
from pipelines.prediction import make_prediction
from pipelines.preprocess import preprocess_image
from utils.exception import CustomException



app = Flask(__name__)



# Loading the model and class names
if not MODEL_PATH.exists():
    print(f"Error: Model file not found at {MODEL_PATH}")
else:
    tf_model = tf.keras.models.load_model(MODEL_PATH)

class_names = CLASS_NAMES


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.content_type not in ['image/jpeg', 'image/png']:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    try:
        img_bytes = request.files["file"].read()
        preprocessed_image_arr = preprocess_image(img_bytes)
        predicted_class, confidence = make_prediction(preprocessed_image_arr, tf_model, class_names)
    except Exception as e:
        return jsonify({'error': CustomException(e, sys)})    
    
    return jsonify({
        'predicted_class': str(predicted_class).title(),
        'confidence': float(confidence)
    })
    
    
    
    
if __name__=="__main__":
    app.run(
        host="0.0.0.0", port=5000, debug=True, threaded=True
    )
