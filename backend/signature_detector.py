from ultralytics import YOLO
import cv2


class SignatureDetector:

    def __init__(self):

        # Load trained YOLO model
        self.model = YOLO("models/signature_yolo.pt")

    def detect_signatures(self, image):

        original_h, original_w = image.shape[:2]

        # Resize image if too large
        max_size = 1280

        scale = min(
            max_size / original_w,
            max_size / original_h,
            1.0
        )

        if scale != 1.0:

            resized = cv2.resize(
                image,
                (
                    int(original_w * scale),
                    int(original_h * scale)
                )
            )

        else:

            resized = image.copy()

        # Run YOLO
        results = self.model.predict(

            resized,

            conf=0.35,

            iou=0.45,

            verbose=False

        )

        signature_boxes = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < 0.35:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Convert back to original image size

                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

                w = x2 - x1
                h = y2 - y1

                # Ignore tiny detections

                if w < 20 or h < 10:
                    continue

                # Expand box slightly

                padding = 8

                x = max(0, x1 - padding)
                y = max(0, y1 - padding)

                w = min(original_w - x, w + 2 * padding)
                h = min(original_h - y, h + 2 * padding)

                signature_boxes.append(
                    (
                        int(x),
                        int(y),
                        int(w),
                        int(h)
                    )
                )

        return signature_boxes