import numpy as np, cv2, streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="HandTalkAI", page_icon="🤟")

@st.cache_resource
def load_all():
    def mk(**kw):
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE, num_hands=1, **kw))
    det = mk(min_hand_detection_confidence=0.5)
    det_low = mk(min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2)
    data = np.load("handtalk_ann.npz")
    return det, det_low, {k: data[k] for k in data.files}

det, det_low, W = load_all()

def extract(img_bgr):
    """Photo -> 63 normalized landmark numbers (same logic as training). Returns (features, preview image)."""
    h, w = img_bgr.shape[:2]
    pad = int(0.35 * max(h, w))
    fill = tuple(int(c) for c in np.median(img_bgr.reshape(-1, 3), axis=0))
    padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
    for d in (det, det_low):
        for im in (img_bgr, padded):
            rgb = np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            res = d.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                continue
            pts = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
            preview = rgb.copy()
            ih, iw = preview.shape[:2]
            for x, y in pts[:, :2] * [iw, ih]:
                cv2.circle(preview, (int(x), int(y)), max(3, iw // 120), (0, 255, 0), -1)
            pts = pts - pts[0]
            scale = np.linalg.norm(pts[:, :2], axis=1).max()
            if scale < 1e-6:
                return None, None
            return (pts / scale).flatten(), preview
    return None, None

def predict(x):
    relu = lambda a: np.maximum(a, 0)
    a = relu(x @ W["d0_W"] + W["d0_b"])
    a = (a - W["bn_mean"]) / np.sqrt(W["bn_var"] + 1e-3) * W["bn_gamma"] + W["bn_beta"]
    a = relu(a @ W["d1_W"] + W["d1_b"])
    a = relu(a @ W["d2_W"] + W["d2_b"])
    z = a @ W["d3_W"] + W["d3_b"]
    e = np.exp(z - z.max())
    return e / e.sum()

def show_result(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read this image."); return
    if max(img.shape[:2]) > 1280:                      # keep it fast
        s = 1280 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s)
    x, preview = extract(img)
    if x is None:
        st.warning("No hand detected. Retake the photo with the whole hand visible and good light.")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); return
    p = predict(x)
    top = p.argsort()[::-1][:3]
    st.image(preview, caption="Detected hand points", use_container_width=True)
    st.markdown(f"## Prediction: **{W['classes'][top[0]]}**  ({p[top[0]]:.0%})")
    if p[top[0]] < 0.5:
        st.info("The model is not sure about this one. Check the other options below.")
    st.write("Top 3:", ", ".join(f"{W['classes'][i]} ({p[i]:.0%})" for i in top))

st.title("🤟 HandTalkAI")
st.write("Sign language (ASL alphabet) recognition from a photo. It uses MediaPipe hand landmarks and a neural network, "
         "so it looks at the hand **shape**, not the skin, background or lighting.")

tab1, tab2 = st.tabs(["Upload a photo", "Use camera"])
with tab1:
    f = st.file_uploader("Choose a photo of a hand sign", type=["jpg", "jpeg", "png", "webp"])
    if f: show_result(f.read())
with tab2:
    c = st.camera_input("Take a photo of your sign")
    if c: show_result(c.getvalue())

st.caption("Limits: J and Z need motion; M, N, S, T look alike; trained on a small number of signers.")