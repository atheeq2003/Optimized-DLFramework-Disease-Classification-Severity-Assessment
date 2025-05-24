# ---------------------- Imports ----------------------
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import joblib
import streamlit as st
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------- Load Models ----------------------
segmentation_model = load_model("segmenter/multiresunet_lung_segmentation.h5", custom_objects={
    'dice_coefficient': lambda y_true, y_pred: 2 * np.sum(y_true * y_pred) / (np.sum(y_true) + np.sum(y_pred) + 1e-6),
    'jaccard_index': lambda y_true, y_pred: np.sum(y_true * y_pred) / (
            np.sum(y_true) + np.sum(y_pred) - np.sum(y_true * y_pred) + 1e-6)
})

softmax_classifier = joblib.load("models/softmax_classifier_New.pkl")
scaler = joblib.load("models/feature_scaler_New.pkl")
label_encoder = joblib.load("models/label_encoder_New.pkl")

cnn_model = load_model("models/severity models/cnn_feature_extractor.h5")
elm = np.load("models/severity models/elm_weights.npz")
severity_label_encoder = joblib.load("models/severity models/label_encoder.pkl")

W = elm['W']
b = elm['b']
beta = elm['beta']

base_model = EfficientNetV2L(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
feature_extractor = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)

hog_train_csv = "handcrafted_features/hog_features_train.csv"
glcm_train_csv = "handcrafted_features/glcm_features_train.csv"
hog_test_csv = "handcrafted_features/hog_features_test.csv"
glcm_test_csv = "handcrafted_features/glcm_features_test.csv"

img_size = (224, 224)

# ---------------------- Utility Functions ----------------------
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)

def segment_lungs(image):
    img_resized = cv2.resize(image, (128, 128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    predicted_mask = segmentation_model.predict(img_input)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(predicted_mask, (image.shape[1], image.shape[0]))
    segmented = image * np.expand_dims(mask_resized, axis=-1)
    return mask_resized, apply_clahe(segmented)

def load_handcrafted_features(image_name, hog_csv, glcm_csv):
    def load_features(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        feature_row = df[df["image_name"] == image_name]
        return feature_row.drop(columns=["image_name", "disease"]).values if not feature_row.empty else None

    hog_features = load_features(hog_csv)
    glcm_features = load_features(glcm_csv)

    return np.hstack([hog_features, glcm_features]) if hog_features is not None and glcm_features is not None else None

def predict_disease(image, image_name):
    hog_csv = hog_test_csv
    glcm_csv = glcm_test_csv

    img_resized = cv2.resize(image, img_size) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    deep_features = feature_extractor.predict(img_array)

    handcrafted_features = load_handcrafted_features(image_name, hog_csv, glcm_csv)
    if handcrafted_features is None:
        return "Error: Handcrafted features not found for this image."

    fused_features = np.hstack([deep_features, handcrafted_features])
    fused_scaled = scaler.transform(fused_features)

    predicted_label = softmax_classifier.predict(fused_scaled)
    return label_encoder.inverse_transform(predicted_label)[0]

def predict_severity(image):
    resized = cv2.resize(image, (224, 224))
    preprocessed = preprocess_input(resized)
    preprocessed = np.expand_dims(preprocessed, axis=0)

    features = cnn_model.predict(preprocessed)

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    H = sigmoid(np.dot(features, W) + b)
    output = np.dot(H, beta)
    pred = np.argmax(output, axis=1)

    return severity_label_encoder.inverse_transform(pred)[0]

# ---------------------- Streamlit UI ----------------------
# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Lung Disease & Severity Classification", layout="centered")

st.markdown(
    """
    <style>
        .main-title { text-align: center; font-size: 40px; font-weight: bold; }
        .stButton > button {
            background-color: #0072B5; color: white; font-size: 16px; border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🩺 Lung Disease & Severity Classification</p>', unsafe_allow_html=True)

# ------------------ Disease Prediction ------------------
st.header("🧬 Disease Prediction")
uploaded_disease_file = st.file_uploader("📤 Upload Chest X-ray for Disease Prediction", type=["png", "jpg", "jpeg"])

if uploaded_disease_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_disease_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_rgb, caption="Original Image", width=250)

    with st.spinner("🫁 Segmenting lungs..."):
        mask, segmented_image = segment_lungs(image_rgb)

    with col2:
        st.image(mask * 255, caption="Lung Mask", width=250)

    with col3:
        st.image(segmented_image, caption="Segmented Image", width=250)

    image_name = f"segmented_{uploaded_disease_file.name}"

    with st.spinner("🔬 Predicting Disease..."):
        predicted_disease = predict_disease(segmented_image, image_name)

    if "Error" in predicted_disease:
        st.error(predicted_disease)
    else:
        st.success(f"✅ Predicted Disease: **{predicted_disease}**")


# ------------------ Severity Prediction ------------------
st.header("📊 Severity Prediction for COVID")
uploaded_severity_file = st.file_uploader("📤 Upload Chest X-ray for Severity Prediction", type=["png", "jpg", "jpeg"])

if uploaded_severity_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_severity_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Severity Input Image", width=300)

    with st.spinner("🧪 Predicting Severity Level..."):
        predicted_severity = predict_severity(image_rgb)
    st.info(f"📈 Predicted Severity Level: **{predicted_severity}**")
