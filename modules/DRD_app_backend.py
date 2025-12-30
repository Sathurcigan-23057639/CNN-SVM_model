import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# ------------------ Load Feature Extractor ------------------

IMAGE_SIZE = (224, 224)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# ------------------ Load Trained Models ------------------
KDR_model = joblib.load("KDR_Hybrid_ResNet50_SVM.pkl")
MS_model  = joblib.load("MS_Hybrid_ResNet50_SVM.pkl")

# ------------------ Labels ------------------
DR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

# ------------------ Image → Feature Extraction ------------------
def process_image(image: Image.Image):
    """ Preprocess image and extract deep features """
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image.convert("RGB"))
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    features = feature_extractor.predict(img_array, verbose=0)
    return features 


# ------------------ Model Prediction ------------------
def predict_with_model(features, model, dataset_name):
    """ Predict class + confidence score using SVM model """
    pred_class = model.predict(features)[0]

    prob = model.predict_proba(features)[0][pred_class] if hasattr(model, "predict_proba") else 0.0

    return {
        "disease": DR_CLASSES[int(pred_class)],
        "confidence": float(prob),
        "model": dataset_name 
    }


# ------------------ Main Function ------------------
def analyze_image_backend(image):
    """ Returns prediction from both KDR and MS models """

    features = process_image(image)

    result_KDR = predict_with_model(features, KDR_model, "Model trained using KDR dataset")
    result_MS  = predict_with_model(features, MS_model,  "Model trained using MS dataset")

    return result_KDR, result_MS
