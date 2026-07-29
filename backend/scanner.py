# ==========================================================
# Intelligent Document Scanner & Alignment Tool
# ==========================================================

import cv2
import numpy as np

from .filters import gaussian_blur
from .filters import sobel_edge_detection


# ==========================================================
# Helper Function
# Arrange Corner Points
# ==========================================================

def order_points(points):
    """
    Arrange the detected corner points in the order:
    Top Left, Top Right, Bottom Right, Bottom Left.
    """

    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


# ==========================================================
# Main Document Scanner
# ==========================================================

def scan_document(image):
    """
    Detects the document boundary and performs
    perspective correction on camera-captured images.
    """

    # ======================================================
    # Step 1 : Validate Input
    # ======================================================

    if image is None:
        raise ValueError("Input image is None.")

    original = image.copy()

    # ======================================================
    # Step 2 : Resize Image
    # ======================================================

    target_height = 700

    ratio = image.shape[0] / target_height

    target_width = int(image.shape[1] / ratio)

    image = cv2.resize(
        image,
        (target_width, target_height)
    )

    # ======================================================
    # Step 3 : Apply Gaussian Blur
    # ======================================================

    blurred = gaussian_blur(image)

    # ======================================================
    # Step 4 : Detect Edges using Sobel Operator
    # ======================================================

    edges = sobel_edge_detection(blurred)

    # ======================================================
    # Step 5 : Convert Edge Image to Binary
    # ======================================================

    _, thresh = cv2.threshold(
        edges,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # ======================================================
    # Step 6 : Perform Morphological Closing
    # ======================================================

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ======================================================
    # Step 7 : Find External Contours
    # ======================================================

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document = None

    # ======================================================
    # Step 8 : Detect Document Boundary
    # ======================================================

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            document = approx
            break

    # ======================================================
    # Step 9 : Return Original Image if No Document Found
    # ======================================================

    if document is None:
        return original

    # ======================================================
    # Step 10 : Arrange Corner Points
    # ======================================================

    corners = order_points(
        document.reshape(4, 2)
    )

    corners = corners * ratio

    (tl, tr, br, bl) = corners

    # ======================================================
    # Step 11 : Compute Width and Height
    # ======================================================

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    max_height = int(max(height_left, height_right))

    # ======================================================
    # Step 12 : Define Destination Points
    # ======================================================

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # ======================================================
    # Step 13 : Compute Perspective Transform Matrix
    # ======================================================

    matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        destination
    )

    # ======================================================
    # Step 14 : Warp the Document
    # ======================================================

    scanned = cv2.warpPerspective(
        original,
        matrix,
        (max_width, max_height)
    )

    # ======================================================
    # Step 15 : Return Final Scanned Document
    # ======================================================

    return scanned