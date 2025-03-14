import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Streamlit page configuration
st.set_page_config(
    page_title="Advanced Shape Analysis & Effects",
    page_icon="🎨",
    layout="wide"
)

# Sidebar for settings
st.sidebar.header("Upload an Image or Video")

# File uploader (Streamlit replacement for `tkinter.filedialog`)
uploaded_file = st.sidebar.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # Load image
    if file_extension in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.sidebar.success(f"Uploaded Image: {uploaded_file.name}")
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Processing function: Apply Edge Detection
        def apply_edge_detection(img, threshold):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold, threshold * 2)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Edge detection slider
        edge_threshold = st.sidebar.slider("Edge Detection Threshold", 0, 255, 100)

        # Processed image
        processed_image = apply_edge_detection(image, edge_threshold)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Download button for processed image
        st.sidebar.download_button(
            label="Download Processed Image",
            data=cv2.imencode(".jpg", processed_image)[1].tobytes(),
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )

    elif file_extension == "mp4":
        st.sidebar.success(f"Uploaded Video: {uploaded_file.name}")
        st.video(uploaded_file)

else:
    st.sidebar.warning("Upload an image or video to process.")

st.sidebar.info("Use the sliders to adjust effects in real-time.")
