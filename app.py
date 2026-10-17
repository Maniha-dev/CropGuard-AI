import cv2
import numpy as np
import streamlit as st

from src.detection.detector import load_model
from src.detection.inference import detect_frame


@st.cache_resource
def get_model():
    return load_model(require_trained=True)


st.title("CropGuard-AI: Pest Detection")
confidence_threshold = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.60,
    step=0.05,
)
image_source = st.radio("Image source", ("Upload an image", "Take a photo"), horizontal=True)

if image_source == "Upload an image":
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo")

if image_file is not None:
    encoded_image = np.frombuffer(image_file.getvalue(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        st.error("The selected file is not a valid image. Please choose a JPG or PNG image.")
    else:
        try:
            model = get_model()
            detections, highest_confidence = detect_frame(
                model,
                frame_bgr,
                confidence_threshold=confidence_threshold,
            )

            annotated_frame = frame_bgr.copy()
            detection_rows = []
            for detection in detections:
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{detection.class_name} {detection.confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                detection_rows.append(
                    {
                        "Class": detection.class_name,
                        "Confidence": detection.confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

            if detections:
                st.success(
                    f"Pest detected: {len(detections)}. "
                    f"Highest confidence: {highest_confidence:.2%}."
                )
                st.table(detection_rows)
            else:
                st.info("No pest detected at this confidence threshold.")

            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detection result", use_container_width=True)
        except Exception as error:
            st.error(f"Unable to process this image: {error}")