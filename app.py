import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import joblib
import streamlit as st
import requests
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ---------------------- GitHub Download Helper ----------------------
def download_file_from_github(tag, filename, subfolder=""):
    base_url = f"https://github.com/atheeq2003/Optimized-DLFramework-Disease-Classification-Severity-Assessment/releases/download"
    url = f"{base_url}/{tag}/{filename}"
    local_path = os.path.join("downloads", subfolder, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Lung Disease & Severity Classification", layout="centered", page_icon="🩺")

# ---------------------- Load Models ----------------------
@st.cache_resource
def load_segmentation_model():
    model_path = download_file_from_github("segmentation-model", "multiresunet_lung_segmentation.h5")
    return load_model(model_path, custom_objects={
        'dice_coefficient': lambda y_true, y_pred: 2 * np.sum(y_true * y_pred) / (np.sum(y_true) + np.sum(y_pred) + 1e-6),
        'jaccard_index': lambda y_true, y_pred: np.sum(y_true * y_pred) / (np.sum(y_true) + np.sum(y_pred) - np.sum(y_true * y_pred) + 1e-6)
    })

@st.cache_resource
def load_classification_models():
    clf = joblib.load(download_file_from_github("disease_classification_models", "softmax_classifier_New.pkl"))
    scaler = joblib.load(download_file_from_github("disease_classification_models", "feature_scaler_New.pkl"))
    encoder = joblib.load(download_file_from_github("disease_classification_models", "label_encoder_New.pkl"))
    return clf, scaler, encoder

@st.cache_resource
def load_severity_models():
    cnn_model = load_model(download_file_from_github("severity_assessment_models", "cnn_feature_extractor.h5"))
    elm_weights = np.load(download_file_from_github("severity_assessment_models", "elm_weights.npz"))
    label_encoder = joblib.load(download_file_from_github("severity_assessment_models", "label_encoder.pkl"))
    return cnn_model, elm_weights, label_encoder

@st.cache_resource
def load_feature_extractor():
    base_model = EfficientNetV2L(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    return tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)

# ---------------------- UI Styling ----------------------
st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 40px; font-weight: bold; }
        .stButton > button {
            background-color: #0072B5; color: white; font-size: 16px; border-radius: 5px;
        }
        .sample-image { max-width: 100%; border-radius: 8px; border: 2px solid #0072B5; }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Utility Functions ----------------------
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)

def segment_lungs(image, model):
    img_resized = cv2.resize(image, (128, 128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    predicted_mask = model.predict(img_input)[0]
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

def predict_disease(image, image_name, extractor, classifier, scaler, encoder, hog_csv, glcm_csv):
    img_resized = cv2.resize(image, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    deep_features = extractor.predict(img_array)
    handcrafted_features = load_handcrafted_features(image_name, hog_csv, glcm_csv)
    if handcrafted_features is None:
        return "Error: Handcrafted features not found for this image."
    fused_features = np.hstack([deep_features, handcrafted_features])
    fused_scaled = scaler.transform(fused_features)
    predicted_label = classifier.predict(fused_scaled)
    return encoder.inverse_transform(predicted_label)[0]

def predict_severity(image, cnn_model, elm, label_encoder):
    resized = cv2.resize(image, (224, 224))
    preprocessed = preprocess_input(resized)
    preprocessed = np.expand_dims(preprocessed, axis=0)
    features = cnn_model.predict(preprocessed)

    def sigmoid(x): return 1 / (1 + np.exp(-x))
    H = sigmoid(np.dot(features, elm["W"]) + elm["b"])
    output = np.dot(H, elm["beta"])
    pred = np.argmax(output, axis=1)
    return label_encoder.inverse_transform(pred)[0]

def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------- Main App ----------------------
def main():
    st.markdown('<p class="main-title">🩺 Lung Disease & Severity Classification</p>', unsafe_allow_html=True)

    # Load models
    segmentation_model = load_segmentation_model()
    classifier, scaler, label_encoder = load_classification_models()
    cnn_model, elm, severity_label_encoder = load_severity_models()
    extractor = load_feature_extractor()

    hog_test_csv = download_file_from_github("csv_files", "hog_features_test.csv")
    glcm_test_csv = download_file_from_github("csv_files", "glcm_features_test.csv")

    disease_samples = {
        "COVID-19": download_file_from_github("sample-images-diseases", "img_38.png"),
        "Tuberculosis": download_file_from_github("sample-images-diseases", "img_12.png"),
        "Viral Pneumonia": download_file_from_github("sample-images-diseases", "img_400.png"),
        "Normal": download_file_from_github("sample-images-diseases", "img_144.png")
    }

    severity_samples = {
        "COVID-19 (Mild)": download_file_from_github("sample-images-severity", "img_384.png"),
        "COVID-19 (Moderate)": download_file_from_github("sample-images-severity", "img_137.png"),
        "COVID-19 (Severe)": download_file_from_github("sample-images-severity", "835897948125275878.png")
    }

    tab1, tab2 = st.tabs(["🧬 Disease Prediction", "📊 COVID-19 Severity Prediction"])

    with tab1:
        st.header("Lung Disease Classification")
        uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"], key="disease_upload")

        if uploaded_file:
            st.session_state.disease_sample = "Select"

        disease_sample = st.selectbox("Or select a sample image", ["Select"] + list(disease_samples.keys()), index=0,
                                      disabled=uploaded_file is not None, key="disease_sample")

        if 'disease_prediction' not in st.session_state:
            st.session_state.disease_prediction = None

        if disease_sample == "Select" and not uploaded_file:
            st.session_state.disease_prediction = None

        if uploaded_file or disease_sample != "Select":
            if uploaded_file:
                image = load_image(uploaded_file)
                image_name = f"segmented_{uploaded_file.name}"
            else:
                sample_path = disease_samples[disease_sample]
                image = cv2.imread(sample_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_name = f"segmented_{os.path.basename(sample_path)}"

            with st.spinner("Segmenting lungs..."):
                mask, segmented_image = segment_lungs(image, segmentation_model)

            col1, col2, col3, col4 = st.columns(4)
            col1.image(image, caption="Input Image", use_column_width=True)
            col2.image(mask * 255, caption="Lung Mask", use_column_width=True)
            col3.image(segmented_image, caption="Segmented Image", use_column_width=True)

            with st.spinner("Predicting Disease..."):
                st.session_state.disease_prediction = predict_disease(
                    segmented_image, image_name, extractor, classifier,
                    scaler, label_encoder, hog_test_csv, glcm_test_csv
                )

        if st.session_state.disease_prediction:
            if "Error" in st.session_state.disease_prediction:
                st.error(st.session_state.disease_prediction)
            else:
                st.success(f"✅ Predicted Disease: **{st.session_state.disease_prediction}**")

    with tab2:
        st.header("COVID-19 Severity Assessment")
        uploaded_file = st.file_uploader("Upload COVID-19 Chest X-ray", type=["png", "jpg", "jpeg"],
                                         key="severity_upload")

        if uploaded_file:
            st.session_state.severity_sample = "Select"

        severity_sample = st.selectbox("Or select a severity sample image", ["Select"] + list(severity_samples.keys()),
                                       index=0, disabled=uploaded_file is not None, key="severity_sample")

        if 'severity_prediction' not in st.session_state:
            st.session_state.severity_prediction = None

        col1, col2 = st.columns(2)
        image = None

        if uploaded_file:
            image = load_image(uploaded_file)
            col1.image(image, caption="Uploaded Image", use_column_width=True)
            col2.empty()
        elif severity_sample != "Select":
            sample_path = severity_samples.get(severity_sample)
            image = cv2.imread(sample_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            col1.image(image, caption=f"Sample: {severity_sample}", use_column_width=True)
            col2.image(image, caption="Input Image", use_column_width=True)
        else:
            col1.empty()
            col2.empty()

        if image is not None:
            with st.spinner("Predicting Severity..."):
                try:
                    st.session_state.severity_prediction = predict_severity(image, cnn_model, elm, severity_label_encoder)
                except Exception as e:
                    st.session_state.severity_prediction = f"Error: {str(e)}"

        if st.session_state.severity_prediction:
            if "Error" in st.session_state.severity_prediction:
                st.error(st.session_state.severity_prediction)
            else:
                st.info(f"📈 Predicted Severity Level: **{st.session_state.severity_prediction}**")

if __name__ == "__main__":
    main()
