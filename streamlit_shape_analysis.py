import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Advanced Shape Analysis & Effects",
    page_icon="🎨",
    layout="wide"
)

st.title("Advanced Shape Analysis & Effects")
st.markdown("Upload an image and apply real-time shape analysis and visual effects.")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Define image processing function
def apply_effects(image):
    """Apply selected effects to the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Display original image
    st.subheader("Original Image")
    st.image(image, channels="BGR")

    # Process and display image
    processed_image = apply_effects(image)
    st.subheader("Processed Image")
    st.image(processed_image, channels="BGR")
