import time
import math
import cv2
import av
import pyautogui
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ---------------- UI ----------------
st.set_page_config(page_title="Hand Gesture Mouse", layout="wide")
st.title("🖐️ Hand Gesture Mouse Control (Local)")
st.caption("Tip: keep 'Enable mouse control' OFF until video starts.")

with st.sidebar:
    st.header("Controls")
    enable_mouse = st.toggle("Enable mouse control", value=False)
    click_threshold = st.slider("Pinch threshold (smaller = more sensitive)",
                                min_value=0.02, max_value=0.15, value=0.06, step=0.005)
    smoothing = st.slider("Pointer smoothing", 0.0, 0.9, 0.25, 0.05)
    click_cooldown_ms = st.slider("Click cooldown (ms)", 50, 800, 250, 10)
    st.markdown("---")
    st.write("**Gestures**")
    st.write("- Move cursor: Index fingertip")
    st.write("- Click: Pinch (Index tip + Thumb tip close)")

# ------------- Video Processor -------------
mp_hands = mp.solutions.hands

class HandMouseProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.prev_sx = None
        self.prev_sy = None
        self.last_click_t = 0.0
        self.screen_w, self.screen_h = pyautogui.size()

        # will be updated from sidebar
        self.enable_mouse = False
        self.click_threshold = 0.06
        self.smoothing = 0.25
        self.click_cooldown_ms = 250

    def _smooth(self, new, old, alpha):
        if old is None:
            return new
        return old * (1 - alpha) + new * alpha

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                img, hand, mp_hands.HAND_CONNECTIONS
            )

            lm = hand.landmark
            # Normalized [0..1] coords from MediaPipe
            idx_x_n, idx_y_n = lm[8].x, lm[8].y        # index tip
            thm_x_n, thm_y_n = lm[4].x, lm[4].y        # thumb tip

            # Visual dots on the video
            cv2.circle(img, (int(idx_x_n * w), int(idx_y_n * h)), 8, (0, 255, 0), -1)
            cv2.circle(img, (int(thm_x_n * w), int(thm_y_n * h)), 8, (0, 140, 255), -1)

            # Distance (normalized) for pinch click
            dist = math.hypot(idx_x_n - thm_x_n, idx_y_n - thm_y_n)
            cv2.putText(img, f"pinch: {dist:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

            # Map to screen coords
            sx = int(idx_x_n * self.screen_w)
            sy = int(idx_y_n * self.screen_h)

            # Smoothing to reduce jitter
            sx = int(self._smooth(sx, self.prev_sx, self.smoothing))
            sy = int(self._smooth(sy, self.prev_sy, self.smoothing))
            self.prev_sx, self.prev_sy = sx, sy

            # Safety: small "dead zone" (top-left corner) to park finger
            #   If finger stays in 0..0.12 of frame both axes, we suppress movement
            in_dead_zone = idx_x_n < 0.12 and idx_y_n < 0.12

            if self.enable_mouse and not in_dead_zone:
                pyautogui.moveTo(sx, sy)

                # Pinch click with cooldown
                now = time.time() * 1000.0
                if dist < self.click_threshold and (now - self.last_click_t) > self.click_cooldown_ms:
                    pyautogui.click()
                    self.last_click_t = now
                    cv2.putText(img, "CLICK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # HUD
            color = (0, 255, 0) if self.enable_mouse and not in_dead_zone else (60, 60, 60)
            cv2.rectangle(img, (0, 0), (220, 90), (0, 0, 0), -1)
            cv2.putText(img, f"MOUSE: {'ON' if self.enable_mouse else 'OFF'}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------- Start WebRTC -------------
webrtc_ctx = webrtc_streamer(
    key="hand-mouse",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandMouseProcessor,
)

# Push sidebar settings into the running video processor
if webrtc_ctx and webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.enable_mouse = enable_mouse
    webrtc_ctx.video_processor.click_threshold = float(click_threshold)
    webrtc_ctx.video_processor.smoothing = float(smoothing)
    webrtc_ctx.video_processor.click_cooldown_ms = int(click_cooldown_ms)

st.info(
    "If the cursor keeps moving and you need to interact with the app, "
    "toggle **Enable mouse control** OFF in the sidebar, or move your finger "
    "to the top-left corner (dead zone)."
)
