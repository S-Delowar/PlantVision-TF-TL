import sys
import tensorflow as tf
import numpy as np
from utils.exception import CustomException
from utils.logger import logging




def predict_with_keras_model(preprocessed_img_arr, model, class_names):
    """Make prediction on preprocessed image array

    Args:
        preprocessed_img_arr (array): preprocessed array of the image 

    Raises:
        CustomException: exception if prediction not possible for any reason

    Returns:
        (str, float): returns predicted class as str and confidence level of predicting that class
    """
    try:
        img_batch = tf.expand_dims(preprocessed_img_arr, axis=0)
        pred_prob = model.predict(img_batch)
        # logging.info(f"Class Names :{class_names}")
        pred_class = class_names[np.argmax(pred_prob)]
        confidence = float(np.max(pred_prob))
        logging.info(f"Predicted Class: {pred_class}, Confidence: {confidence}")
        return pred_class, confidence
    except Exception as e:
        raise CustomException(e, sys)
    
    
def predict_with_tflite_model(preprocessed_img_arr, interpreter, class_names):
    try:
        input_img_arr = tf.expand_dims(preprocessed_img_arr, axis=0)
        
        logging.info("Get TFLite input output details")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.info("Model input shape:", input_details[0]['shape'])

        logging.info("Run inference")
        interpreter.set_tensor(input_details[0]['index'], input_img_arr)
        interpreter.invoke()
        pred_prob = interpreter.get_tensor(output_details[0]['index'])
        pred_class = class_names[np.argmax(pred_prob)]
        confidence = float(np.max(pred_prob))
        logging.info(f"Predicted Class: {pred_class}, Confidence: {confidence}")

        return pred_class, confidence
    except Exception as e:
        raise CustomException(sys, e)
        
    