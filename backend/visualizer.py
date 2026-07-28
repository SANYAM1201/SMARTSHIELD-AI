import cv2


class Visualizer:

    def __init__(self):
        pass


    def draw_boxes(self, image, face_boxes, signature_boxes):

        output = image.copy()

        # Draw Face Boxes (Green)
        for (x, y, w, h) in face_boxes:

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                "Face",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Draw Signature Boxes (Blue)
        for (x, y, w, h) in signature_boxes:

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                output,
                "Signature",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        return output