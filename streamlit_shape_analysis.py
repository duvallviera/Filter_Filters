import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Advanced Shape Analysis & Effects",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("Advanced Shape Analysis & Effects")
st.markdown("""
Upload an image or video and apply real-time shape analysis and visual effects.
""")


# Sidebar settings
st.sidebar.header("Settings")
apply_edge_detection = st.sidebar.checkbox("Apply Edge Detection", value=True)
apply_cartoon_effect = st.sidebar.checkbox("Apply Cartoon Effect")
apply_sharpening = st.sidebar.checkbox("Apply Sharpening")

# File uploader (for image or video)
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

# Define image processing functions
def apply_effects(image):
    """
    Apply selected effects to the image.
    """
    if apply_edge_detection:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if apply_cartoon_effect:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
        image = cv2.bitwise_and(color, color, mask=edges)

    if apply_sharpening:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

    return image

# Process uploaded image
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension in ["jpg", "jpeg", "png"]:
        # Load image
        image = Image.open(uploaded_file)
        image = np.array(image)
        if len(image.shape) == 2:  # Convert grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR")

        # Process and display image
        processed_image = apply_effects(image)
        st.subheader("Processed Image")
        st.image(processed_image, channels="BGR")

    elif file_extension == "mp4":
        st.subheader("Processing Video...")

        # Save video to temp file
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process video frame-by-frame
        cap = cv2.VideoCapture(temp_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = "processed_video.mp4"
        out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply selected effects
            frame = apply_effects(frame)
            out.write(frame)

            # Update progress bar
            frame_num += 1
            progress_bar.progress(frame_num / frame_count)

        cap.release()
        out.release()
        st.success("Video processing complete!")

        # Display processed video
        st.subheader("Processed Video")
        st.video(output_video_path)

        # Provide download button
        with open(output_video_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(label="Download Processed Video", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")

    else:
        st.error("Unsupported file format. Please upload a JPG, PNG, or MP4 file.")
