import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Cats vs Dogs SVM Classifier", layout="centered")
st.title("🐱 Cats vs Dogs 🐶 - SVM Classifier")

# 2. Load the trained SVM model
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error("Model error. Please retrain your model.")


uploaded_file = st.file_uploader("Upload a Cat or Dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
        
    img_resized = cv2.resize(img_gray, (64, 64))
    img_flattened = img_resized.flatten().reshape(1, -1) / 255.0
    
    
    if st.button("Classify Image"):
        
        probabilities = model.predict_proba(img_flattened)[0]
        max_probability = max(probabilities)
        prediction = model.predict(img_flattened)[0]
        
        
        if max_probability < 0.70:
            st.warning("Please enter a correct picture of a cat or dog")
        else:
            if prediction == 0:
                st.success("Prediction: It is a Cat 🐱")
            else:
                st.success("Prediction: It is a Dog 🐶")