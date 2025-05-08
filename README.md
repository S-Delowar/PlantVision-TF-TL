# Plant Classification App using Transfer Learning with EfficientNetB3

This project demonstrates the use of Transfer Learning to classify flower species using the Oxford Flowers 102 dataset. By leveraging powerful pre-trained deep learning models (specifically EfficientNet), the project fine-tunes and evaluates classification performance using various training strategies. It also includes conversion to TFLite format to deploy with Flask API endpoint for predictions.

**Live on Render:**  [https://plantvision-tf-tl.onrender.com/](https://plantvision-tf-tl.onrender.com/)

## Tech Stack
- Language: Python
- Deep Learning Framework: TensorFlow / Keras
- Dataset Source: TensorFlow Datasets (TFDS)
- Model Architecture: EfficientNetB0 (Pre-trained)
- Utilities: NumPy, Matplotlib, TensorFlow Datasets (TFDS)
- Model Conversion: TensorFlow Lite (TFLite)
- Deployment: Flask API
  

# Dataset and Data Preprocessing

* **Dataset:** [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), consisting of 8189 images across 102 flower categories.
* **Split:** Training (1020), Validation (1020), Testing (6149)
* **Preprocessing Steps:**

  * Resize all images to 300x300
  * Convert from `uint8` to `float32`
  * EfficientNet-specific preprocessing (no manual normalization required)
  * Batch and prefetch using `tf.data.AUTOTUNE` for optimized input pipeline

---

## 📈 Model Training Strategy

### 🔢 Feature Extractor Model Architecture

* **Base Model:** EfficientNetB3 from `tf.keras.applications`
* **Strategy:**

  * Load pre-trained weights (`imagenet`)
  * Freeze the base model
  * Include Data Augmentation Layer
  * Add custom classification head (GlobalAveragePooling, Dense Softmax)

### ⚖️ Fine-tuning & Comparison

Three training strategies were applied:

| Strategy           | Description                                | Trainable Layers  | Learning Rate | Train Acc. | Val Acc.  | Test Acc. | Val Loss | Test Loss |
| ------------------ | ------------------------------------------ | ----------------- | ------------- | ---------- | --------- | --------- | -------- | --------- |
| Feature Extraction | Freeze all EfficientNet layers             | 0                 | 1e-3          | \~94%      | 79.5%     | \~74%     | \~0.65   | -         |
| Fine-Tuning (1st)  | Unfreeze last 30 layers (trainable = -30:) | Last 30           | **1e-6**      | 74.6%      | 68.7%     | 62.9%     | 1.56     | 1.75      |
| Fine-Tuning (2nd)  | Unfreeze last 10 + Dropout & reinit Dense  | Last 10 + Dropout | **1e-3**      | **98.5%**  | **84.9%** | **83.4%** | 0.60     | 0.61      |

#### 🔍 Key Observations:

* **Initial Fine-Tuning Attempt (1e-6):**

  * Training accuracy dropped significantly from \~94% to **74.6%**.
  * Validation accuracy decreased to **68.73%**.
  * Validation loss showed steady increase (**1.56**).
  * Test set accuracy also dropped to **62.86%**.
  * Likely due to overly conservative learning rate and too many layers being unfrozen too early.

* **Re-Fine-Tuning Attempt (1e-3):**

  * Training accuracy jumped to **98.5%**.
  * Validation accuracy improved significantly to **84.9%**.
  * Validation loss decreased to **0.60**, indicating better generalization.
  * Test accuracy also improved sharply to **83.4%**, validating the changes.
  * This shows the importance of proper learning rate and controlled unfreezing.


## 🔄 Converting to TFLite Version

The trained model is converted to TensorFlow Lite format to support edge devices and future mobile deployment.
* Using TFLiteConverter from Tensorflow for the conversion
* TFLite model is significantly smaller and optimized for inference speed.

---


## 🔍 Prediction Endpoint

A lightweight Flask API is implemented to serve predictions from the TFLite model.

* User Interface with HTML, BootStrap & JavaScript
* Accepts image uploads
* Preprocesses image as per model input
* Uses the TFLite Interpreter for prediction
* Returns class name with highest probability


## Dockerization & Deployment
To ensure consistency across environments and simplify deployment, the Flask API is containerized using Docker.
  ### Dockerfile
    - Uses a lightweight Python base image (python:3.10-slim)
    - Installs all dependencies from requirements.txt
    - Sets up the working directory and environment variables
    - Copies source code into the container
    - Uses Gunicorn to run the Flask app on port 5000
  ### Deployment
    - Deploy the dockerized app to Render (Uses free tier)
    - Render can automatically build and deploy your app by detecting the Dockerfile in your GitHub repository.

  ** Live Prediction on Render:**  [https://plantvision-tf-tl.onrender.com](https://plantvision-tf-tl.onrender.com/)
