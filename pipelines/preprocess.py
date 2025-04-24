import sys
import tensorflow as tf
from config import IMAGE_SIZE
from utils.exception import CustomException


def preprocess_image(image_bytes, image_size=IMAGE_SIZE):
    """preprocess the input image

    Args:
        image_bytes (file bytes): input image
        image_size (int, optional): required size of the image to be processed. Defaults to 300.

    Returns:
        array: the processed image array
    """
    try:
        img_arr = tf.image.decode_image(image_bytes, channels=3)
        img_arr = tf.image.resize(img_arr, image_size)
        return img_arr
    except Exception as e:
        raise CustomException(e, sys)
