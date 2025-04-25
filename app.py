import logging
import sys
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

from config import CLASS_NAMES, MODEL_PATH, TFLITE_MODEL_PATH
from pipelines.prediction import predict_with_keras_model, predict_with_tflite_model
from pipelines.preprocess import preprocess_image
from utils.exception import CustomException
from utils.logger import logging


app = Flask(__name__)

# Initialize model at the beginning
try:
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    logging.info("Successfully loaded TFLite model")
    keras_model = None
except Exception as e:
    raise CustomException(sys, e)
    # logging.info("TFLite model not found. Loading the Keras model")
    # keras_model = tf.keras.models.load_model(MODEL_PATH)
    # interpreter = None

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
        
        if interpreter:
            predicted_class, confidence = predict_with_tflite_model(
                preprocessed_image_arr,
                interpreter,
                class_names
            )
        else:
            predicted_class, confidence = predict_with_keras_model(
                preprocessed_image_arr, 
                keras_model, 
                class_names
            )
            
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
