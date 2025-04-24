import sys
from flask import Flask, render_template, request, jsonify

from pipelines.prediction import make_prediction
from pipelines.preprocess import preprocess_image
from utils.exception import CustomException



app = Flask(__name__)


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
        img = request.files["file"].read()
        preprocessed_image = preprocess_image(img)
        predicted_class, confidence = make_prediction(preprocessed_image)
    except Exception as e:
        return jsonify({'error': CustomException(e, sys)})    
    
    return jsonify({
        'predicted_class': str(predicted_class).title(),
        'confidence': float(confidence)
    })
    
    
    
    
if __name__=="__main__":
    app.run(
        host="0.0.0.0", port=8000, debug=True, threaded=True
    )
