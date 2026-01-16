import time
import threading
from collections import deque

import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO

def draw_box(img, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.rectangle(img, (x1, y1 - 22), (x1 + max(80, 10 * len(label)), y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

class FireDetector:
    def __init__(self, mask_history=5):
        self.masks = deque(maxlen=mask_history)
        self.prev_mask = None

    def fire_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 80, 180], dtype=np.uint8)
        upper1 = np.array([50, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 80, 150], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def detect(self, frame, min_area=800):
        mask = self.fire_mask(frame)
        dynamic = 0.0
        if self.prev_mask is not None:
            diff = cv2.absdiff(mask, self.prev_mask)
            dynamic = float(np.sum(diff > 0)) / (mask.shape[0] * mask.shape[1])
        self.prev_mask = mask.copy()
        self.masks.append(mask.copy())

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull) + 1e-6
            solidity = area / hull_area
            if solidity > 0.35:
                boxes.append((x, y, x + w, y + h))
        is_dynamic = dynamic > 0.002
        return boxes, is_dynamic

class UnknownDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    def detect(self, frame, known_boxes, min_area=1200):
        fg = self.bg.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        unknowns = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            candidate = (x, y, x + w, y + h)
            overlap = False
            for kb in known_boxes:
                if iou(candidate, kb) > 0.2:
                    overlap = True
                    break
            if not overlap:
                unknowns.append(candidate)
        return unknowns

class YOLOWrapper:
    def __init__(self, weights="yolov8n.pt", conf=0.35, iou_thres=0.5):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou_thres

    def detect(self, frame):
        res = self.model.predict(source=frame, conf=self.conf, iou=self.iou, verbose=False)
        boxes = []
        labels = []
        scores = []
        if len(res):
            r = res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    boxes.append(tuple(xyxy))
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    sc = float(b.conf[0].item()) if b.conf is not None else 0.0
                    labels.append(self.model.model.names.get(cls_id, str(cls_id)))
                    scores.append(sc)
        return boxes, labels, scores

st.set_page_config(page_title="Realtime Fire + Unknown Detection", layout="wide")
st.title("Realtime Fire + Unknown Object Detection")

with st.sidebar:
    st.markdown("Sources: Webcam (WebRTC), RTSP/HTTP URL, or uploaded video.")
    source = st.selectbox("Input source", ["Webcam (WebRTC)", "RTSP/HTTP URL", "Upload video"])
    conf_th = st.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
    iou_th = st.slider("YOLO IoU", 0.1, 0.9, 0.5, 0.05)
    enable_fire = st.checkbox("Enable fire detection", True)
    enable_unknown = st.checkbox("Enable unknown detection", True)
    max_width = st.slider("Max display width", 480, 1280, 800, 20)

yolo = YOLOWrapper(conf=conf_th, iou_thres=iou_th)
fire_detector = FireDetector()
unknown_detector = UnknownDetector()

def annotate_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    draw = frame_bgr.copy()

    boxes, labels, scores = yolo.detect(frame_bgr)
    for box, lab, sc in zip(boxes, labels, scores):
        draw_box(draw, box, (0, 180, 0), f"{lab} {sc:.2f}")

    fire_boxes = []
    fire_dynamic = False
    if enable_fire:
        fire_boxes, fire_dynamic = fire_detector.detect(frame_bgr)
        for fb in fire_boxes:
            draw_box(draw, fb, (0, 140, 255), "FIRE")

    if enable_unknown:
        unknowns = unknown_detector.detect(frame_bgr, boxes)
        for ub in unknowns:
            draw_box(draw, ub, (40, 40, 220), "UNKNOWN")

    status = []
    if enable_fire and len(fire_boxes) > 0:
        status.append("Fire detected")
    if enable_fire and fire_dynamic and len(fire_boxes) > 0:
        status.append("Fire flickering detected")

    if status:
        txt = " | ".join(status)
        cv2.rectangle(draw, (0, h - 28), (min(w, 460), h), (30, 30, 30), -1)
        cv2.putText(draw, txt, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return draw

class WebRTCProcessor(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self._yolo = YOLOWrapper(conf=conf_th, iou_thres=iou_th)
        self._fire = FireDetector()
        self._unknown = UnknownDetector()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        scale = 800 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)

        boxes, labels, scores = self._yolo.detect(img)
        draw = img.copy()
        for box, lab, sc in zip(boxes, labels, scores):
            draw_box(draw, box, (0, 180, 0), f"{lab} {sc:.2f}")

        if enable_fire:
            fire_boxes, fire_dyn = self._fire.detect(img)
            for fb in fire_boxes:
                draw_box(draw, fb, (0, 140, 255), "FIRE")

        if enable_unknown:
            unknowns = self._unknown.detect(img, boxes)
            for ub in unknowns:
                draw_box(draw, ub, (40, 40, 220), "UNKNOWN")

        return draw

rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if source == "Webcam (WebRTC)":
    st.markdown("Start the stream and allow camera access in the browser.")
    webrtc_streamer(
        key="cv-webcam",
        video_processor_factory=WebRTCProcessor,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
elif source == "RTSP/HTTP URL":
    url = st.text_input("Enter RTSP/HTTP stream URL")
    col1, col2 = st.columns(2)
    run = col1.button("Start")
    stop = col2.button("Stop")

    if "run_rtsp" not in st.session_state:
        st.session_state.run_rtsp = False

    if run:
        st.session_state.run_rtsp = True
    if stop:
        st.session_state.run_rtsp = False

    frame_area = st.empty()
    info = st.empty()

    if st.session_state.run_rtsp and url:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            st.error("Failed to open stream.")
        else:
            fps_t0 = time.time()
            fcnt = 0
            while st.session_state.run_rtsp:
                ok, frame = cap.read()
                if not ok:
                    info.warning("Stream ended or read error.")
                    break
                h, w = frame.shape[:2]
                scale = max_width / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
                out = annotate_frame(frame)
                frame_area.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                fcnt += 1
                if fcnt % 20 == 0:
                    fps = fcnt / (time.time() - fps_t0 + 1e-6)
                    info.info(f"Processed frames: {fcnt} | Approx FPS: {fps:.1f}")
            cap.release()
else:
    up = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if up is not None:
        container = av.open(up)
        stream = container.streams.video[0]
        frame_area = st.empty()
        info = st.empty()
        fps_t0 = time.time()
        fcnt = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                scale = max_width / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
                out = annotate_frame(img)
                frame_area.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                fcnt += 1
                if fcnt % 20 == 0:
                    fps = fcnt / (time.time() - fps_t0 + 1e-6)
                    info.info(f"Processed frames: {fcnt} | Approx FPS: {fps:.1f}")
