from .face_detector import FaceDetector
from .signature_detector import SignatureDetector
from .blur import Blur
from .visualizer import Visualizer


# Initialize models only once
face_detector = FaceDetector()
signature_detector = SignatureDetector()
visualizer = Visualizer()


def redact_document(
    image,
    blur_type="gaussian", #default
    blur_strength=99      #default
):

    blur = Blur(
        blur_type=blur_type,
        kernel_size=blur_strength
    )

    # -----------------------------
    # Face Detection
    # -----------------------------
    face_boxes = face_detector.detect_faces(image)

    # -----------------------------
    # Signature Detection
    # -----------------------------
    signature_boxes = signature_detector.detect_signatures(image)

    # -----------------------------
    # Draw Bounding Boxes
    # -----------------------------
    detected_image = visualizer.draw_boxes(
        image.copy(),
        face_boxes,
        signature_boxes
    )

    # -----------------------------
    # Blur Faces
    # -----------------------------
    redacted = blur.blur_regions(
        image.copy(),
        face_boxes
    )

    # -----------------------------
    # Blur Signatures
    # -----------------------------
    redacted = blur.blur_regions(
        redacted,
        signature_boxes
    )

    return (
        redacted,
        detected_image,
        face_boxes,
        signature_boxes
    )