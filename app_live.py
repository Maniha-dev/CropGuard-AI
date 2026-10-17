"""CropGuard-AI web app: live browser camera, pest detection, and spray decision."""
import av
import cv2
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import webrtc_streamer

from src.decision.spray_decision import REQUIRED_FRAMES, SprayDecision
from src.detection.detector import load_model
from src.detection.inference import detect_frame
from src.webrtc_config import build_rtc_configuration


RUN_MODEL_EVERY = 3

st.set_page_config(page_title="CropGuard-AI", page_icon="🐛")
st.title("CropGuard-AI: Live Pest Detection")
st.caption("Press START and allow camera access. Simulation only; no hardware is controlled.")
confidence_threshold = st.slider("Box confidence threshold", 0.10, 0.95, 0.60, 0.05)


@st.cache_resource
def get_model():
    return load_model(require_trained=True)


@st.cache_resource
def get_state():
    return {
        "confidence_threshold": 0.60,
        "frame_count": 0,
        "detections": [],
        "decision": "DON'T SPRAY",
        "engine": SprayDecision(),
    }


model = get_model()
state = get_state()
state["confidence_threshold"] = confidence_threshold

try:
    turn_settings = st.secrets.get("turn", {})
except StreamlitSecretNotFoundError:
    turn_settings = {}

try:
    rtc_configuration = build_rtc_configuration(turn_settings)
except ValueError as error:
    st.warning(f"TURN is not configured correctly; using STUN only. {error}")
    rtc_configuration = build_rtc_configuration()


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    state["frame_count"] += 1
    if state["frame_count"] % RUN_MODEL_EVERY == 0:
        detections, highest_confidence = detect_frame(
            model,
            image,
            state["confidence_threshold"],
        )
        state["detections"] = detections
        state["decision"] = state["engine"].update(bool(detections), highest_confidence)

    for detection in state["detections"]:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Pest {detection.confidence:.2f}",
            (x1, max(y1 - 8, 95)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    spray = state["decision"] == "SPRAY"
    cv2.rectangle(image, (0, 0), (image.shape[1], 85), (0, 0, 255) if spray else (0, 170, 0), -1)
    status = "PEST DETECTED" if state["detections"] else "NO PEST"
    cv2.putText(image, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(image, state["decision"], (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    confirmed = state["engine"].consecutive_detections
    cv2.putText(
        image,
        f"Confirmed {confirmed}/{REQUIRED_FRAMES}",
        (12, image.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="cropguard-live",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_configuration,
    async_processing=True,
)