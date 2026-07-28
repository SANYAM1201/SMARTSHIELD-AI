# ==========================================================
# Intelligent Document Scanner & Alignment Tool
# scanner.py
# ==========================================================

# --------------------------
# Import Required Libraries
# --------------------------

import cv2
import numpy as np

from .filters import gaussian_blur
from .filters import sobel_edge_detection



def scan_document(img):
    """
    Scan and align a document image.

    Parameters:
        img (numpy.ndarray): Input image in OpenCV (BGR) format.

    Returns:
        numpy.ndarray: Perspective corrected scanned image.
    """

    # ==========================================================
    # STEP 1 : READ INPUT IMAGE
    # ==========================================================

    if img is None:
        raise ValueError("Input image is None.")

    original = img.copy()

    # ==========================================================
    # STEP 2 : RESIZE IMAGE
    # ==========================================================

    # Resize image for faster processing

    height = 700

    ratio = img.shape[0] / height

    width = int(img.shape[1] / ratio)

    img = cv2.resize(img, (width, height))

    # Keep a copy after resizing
    original = img.copy()
    display = img.copy()

    # ==========================================================
    # STEP 3 : APPLY GAUSSIAN BLUR
    # ==========================================================

    blur = gaussian_blur(img)

    # ==========================================================
    # STEP 4 : APPLY SOBEL EDGE DETECTION
    # ==========================================================

    edges = sobel_edge_detection(blur)

    # ==========================================================
    # STEP 5 : APPLY THRESHOLDING
    # ==========================================================

    _, thresh = cv2.threshold(
        edges,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # ==========================================================
    # STEP 6 : MORPHOLOGICAL CLOSING
    # ==========================================================

    # This fills small gaps between broken edges

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ==========================================================
    # STEP 7 : FIND EXTERNAL CONTOURS
    # ==========================================================

    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ==========================================================
    # STEP 8 : SORT CONTOURS
    # ==========================================================

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    # ==========================================================
    # STEP 9 : FIND DOCUMENT CONTOUR
    # ==========================================================

    doc = None

    for cnt in contours:

        # Ignore very small contours

        area = cv2.contourArea(cnt)

        if area < 10000:
            continue

        # Calculate perimeter

        peri = cv2.arcLength(cnt, True)

        # Approximate contour

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        # Document should have four corners

        if len(approx) == 4:

            doc = approx

            break

    # ==========================================================
    # CHECK IF DOCUMENT IS FOUND
    # ==========================================================

    if doc is None:
        # If document isn't detected, return original image
        return original

    # ==========================================================
    # STEP 10 : DRAW DOCUMENT BOUNDARY
    # ==========================================================

    cv2.drawContours(
        display,
        [doc],
        -1,
        (0, 255, 0),
        3
    )

    # ==========================================================
    # STEP 11 : EXTRACT THE FOUR CORNER POINTS
    # ==========================================================

    # Convert contour into four corner points

    pts = doc.reshape(4, 2)

    # ==========================================================
    # STEP 12 : ARRANGE THE CORNER POINTS
    # Order:
    # Top Left
    # Top Right
    # Bottom Right
    # Bottom Left
    # ==========================================================

    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]      # Top Left
    rect[2] = pts[np.argmax(s)]      # Bottom Right

    # Difference of coordinates
    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]   # Top Right
    rect[3] = pts[np.argmax(diff)]   # Bottom Left

    # ==========================================================
    # STEP 13 : CALCULATE WIDTH OF DOCUMENT
    # ==========================================================

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    # ==========================================================
    # STEP 14 : CALCULATE HEIGHT OF DOCUMENT
    # ==========================================================

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    # ==========================================================
    # STEP 15 : DEFINE DESTINATION POINTS
    # ==========================================================

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # ==========================================================
    # STEP 16 : COMPUTE PERSPECTIVE TRANSFORMATION MATRIX
    # ==========================================================

    matrix = cv2.getPerspectiveTransform(rect, dst)

    # ==========================================================
    # STEP 17 : Perspective Transform
    # ==========================================================

    scanned = cv2.warpPerspective(
        img,
        matrix,
        (maxWidth, maxHeight)
    )


    # ==========================================================
    # STEP 18 : RETURN RESULTS
    # ==========================================================

    return scanned