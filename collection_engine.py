import base64
from pathlib import Path
import threading

import cv2
import numpy as np

try:
    from cvzone.HandTrackingModule import HandDetector
except Exception:
    HandDetector = None

try:
    import mediapipe as mp
except Exception:
    mp = None


class DataCollectionEngine:
    """Web-friendly data collection pipeline for skeleton, gray, and binary datasets."""

    MODES = ("skeleton", "gray", "binary", "gray_with_drawing")

    def __init__(self, offset=29, default_dataset_dir="collected_data"):
        if HandDetector is not None:
            self.backend = "cvzone"
            self.detector = HandDetector(maxHands=1)
            self.detector_roi = HandDetector(maxHands=1)
        elif mp is not None:
            self.backend = "mediapipe"
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector_roi = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            raise ImportError("Install either cvzone or mediapipe to enable hand detection.")

        self.offset = offset
        self.default_dataset_dir = Path(default_dataset_dir)
        self.lock = threading.Lock()

    @staticmethod
    def _extract_hands(result):
        if isinstance(result, tuple):
            return result[0] or []
        return result or []

    @staticmethod
    def _extract_hand_obj(hand):
        if isinstance(hand, list) and hand:
            hand = hand[0]
        return hand

    def _detect_hands_mediapipe(self, detector, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        h_img, w_img = image.shape[:2]
        hands = []

        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            x_values = []
            y_values = []

            for lm in hand_landmarks.landmark:
                x = int(lm.x * w_img)
                y = int(lm.y * h_img)
                z = int(lm.z * w_img)
                lm_list.append([x, y, z])
                x_values.append(x)
                y_values.append(y)

            x_min = max(min(x_values), 0)
            y_min = max(min(y_values), 0)
            x_max = min(max(x_values), w_img)
            y_max = min(max(y_values), h_img)
            bbox = [x_min, y_min, max(x_max - x_min, 1), max(y_max - y_min, 1)]

            hands.append({"bbox": bbox, "lmList": lm_list})

        return hands

    def _detect_hands(self, image, roi=False):
        detector = self.detector_roi if roi else self.detector

        if self.backend == "cvzone":
            return self._extract_hands(detector.findHands(image, draw=False, flipType=True))

        return self._detect_hands_mediapipe(detector, image)

    @staticmethod
    def _encode_image(image):
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _center_on_canvas(image, fill_value=255):
        if image is None or image.size == 0:
            return None

        if len(image.shape) == 2:
            canvas = np.ones((400, 400), dtype=np.uint8) * fill_value
        else:
            canvas = np.ones((400, 400, image.shape[2]), dtype=np.uint8) * fill_value

        h, w = image.shape[:2]
        h = min(h, 400)
        w = min(w, 400)
        image = image[:h, :w]

        y0 = (400 - h) // 2
        x0 = (400 - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = image
        return canvas

    @staticmethod
    def _normalize_label(label):
        if not label:
            return "A"
        ch = str(label).strip().upper()[:1]
        if ch and "A" <= ch <= "Z":
            return ch
        return "A"

    @classmethod
    def _normalize_modes(cls, modes):
        if not modes:
            return ["skeleton"]

        cleaned = []
        for mode in modes:
            mode_name = str(mode).strip().lower()
            if mode_name in cls.MODES and mode_name not in cleaned:
                cleaned.append(mode_name)

        return cleaned or ["skeleton"]

    @staticmethod
    def _draw_hand_lines(image, pts, color=(0, 255, 0), thickness=2):
        segments = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17), (0, 5), (0, 17),
        ]
        for a, b in segments:
            cv2.line(image, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), color, thickness)

    def _draw_skeleton(self, pts, w, h):
        white = np.ones((400, 400, 3), dtype=np.uint8) * 255

        os = ((400 - w) // 2) - 15
        os1 = ((400 - h) // 2) - 15

        for t in range(0, 4):
            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(5, 8):
            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(9, 12):
            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(13, 16):
            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(17, 20):
            cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)

        cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
        cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
        cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

        for i in range(21):
            cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

        return white

    @staticmethod
    def _count_images(folder):
        if not folder.exists():
            return 0
        return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    @staticmethod
    def _next_index(folder):
        max_idx = -1
        for path in folder.iterdir():
            if not path.is_file():
                continue
            try:
                max_idx = max(max_idx, int(path.stem))
            except ValueError:
                continue
        return max_idx + 1

    def _prepare_previews(self, frame, x, y, w, h, pts):
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x - self.offset)
        y1 = max(0, y - self.offset)
        x2 = min(w_img, x + w + self.offset)
        y2 = min(h_img, y + h + self.offset)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        skeleton = self._draw_skeleton(pts, w, h)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (1, 1), 2)
        gray_canvas = self._center_on_canvas(gray_blur, fill_value=148)

        gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, binary = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_canvas = self._center_on_canvas(binary, fill_value=255)

        roi_drawing = roi.copy()
        self._draw_hand_lines(roi_drawing, pts, color=(0, 255, 0), thickness=2)
        for i in range(21):
            cv2.circle(roi_drawing, (pts[i][0], pts[i][1]), 2, (0, 0, 255), 1)
        gray_draw = cv2.cvtColor(roi_drawing, cv2.COLOR_BGR2GRAY)
        gray_draw_blur = cv2.GaussianBlur(gray_draw, (1, 1), 2)
        gray_draw_canvas = self._center_on_canvas(gray_draw_blur, fill_value=148)

        return {
            "skeleton": skeleton,
            "gray": gray_canvas,
            "binary": binary_canvas,
            "gray_with_drawing": gray_draw_canvas,
        }

    def _save_outputs(self, outputs, dataset_root, label, modes):
        saved = {}

        for mode in modes:
            image = outputs.get(mode)
            if image is None:
                continue

            out_dir = dataset_root / mode / label
            out_dir.mkdir(parents=True, exist_ok=True)
            next_idx = self._next_index(out_dir)
            out_path = out_dir / f"{next_idx}.jpg"
            if cv2.imwrite(str(out_path), image):
                saved[mode] = str(out_path)

        return saved

    def _label_counts(self, dataset_root, label):
        counts = {}
        for mode in self.MODES:
            counts[mode] = self._count_images(dataset_root / mode / label)
        return counts

    def process_frame(self, frame_bgr, label="A", dataset_dir=None, save=False, modes=None):
        with self.lock:
            frame = cv2.flip(frame_bgr, 1)
            hands = self._detect_hands(frame)

            label = self._normalize_label(label)
            save_modes = self._normalize_modes(modes)
            dataset_root = Path(dataset_dir).expanduser() if dataset_dir else self.default_dataset_dir

            if not hands:
                return {
                    "has_hand": False,
                    "label": label,
                    "dataset_dir": str(dataset_root),
                    "counts": self._label_counts(dataset_root, label),
                    "saved_files": {},
                    "previews": {},
                    "error": "No hand detected.",
                }

            hand = self._extract_hand_obj(hands[0])
            bbox = hand.get("bbox") if isinstance(hand, dict) else None
            if not bbox:
                return {
                    "has_hand": False,
                    "label": label,
                    "dataset_dir": str(dataset_root),
                    "counts": self._label_counts(dataset_root, label),
                    "saved_files": {},
                    "previews": {},
                    "error": "Hand bounding box unavailable.",
                }

            x, y, w, h = bbox

            h_img, w_img = frame.shape[:2]
            x1 = max(0, x - self.offset)
            y1 = max(0, y - self.offset)
            x2 = min(w_img, x + w + self.offset)
            y2 = min(h_img, y + h + self.offset)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return {
                    "has_hand": False,
                    "label": label,
                    "dataset_dir": str(dataset_root),
                    "counts": self._label_counts(dataset_root, label),
                    "saved_files": {},
                    "previews": {},
                    "error": "Empty ROI.",
                }

            roi_hands = self._detect_hands(roi, roi=True)
            if not roi_hands:
                return {
                    "has_hand": False,
                    "label": label,
                    "dataset_dir": str(dataset_root),
                    "counts": self._label_counts(dataset_root, label),
                    "saved_files": {},
                    "previews": {},
                    "error": "No hand landmarks detected.",
                }

            roi_hand = self._extract_hand_obj(roi_hands[0])
            pts = roi_hand.get("lmList") if isinstance(roi_hand, dict) else None
            if not pts or len(pts) < 21:
                return {
                    "has_hand": False,
                    "label": label,
                    "dataset_dir": str(dataset_root),
                    "counts": self._label_counts(dataset_root, label),
                    "saved_files": {},
                    "previews": {},
                    "error": "Insufficient landmarks.",
                }

            outputs = self._prepare_previews(frame, x, y, w, h, pts)
            if outputs is None:
                return {
                    "has_hand": False,
                    "label": label,
                    "dataset_dir": str(dataset_root),
                    "counts": self._label_counts(dataset_root, label),
                    "saved_files": {},
                    "previews": {},
                    "error": "Could not create previews.",
                }

            saved_files = {}
            if save:
                saved_files = self._save_outputs(outputs, dataset_root, label, save_modes)

            previews = {mode: self._encode_image(img) for mode, img in outputs.items()}

            return {
                "has_hand": True,
                "label": label,
                "dataset_dir": str(dataset_root),
                "counts": self._label_counts(dataset_root, label),
                "saved_files": saved_files,
                "previews": previews,
                "error": None,
            }
