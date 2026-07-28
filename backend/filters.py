import cv2
import numpy as np


def gaussian_blur(img):
    """
    Applies Gaussian Blur to reduce noise.
    """

    blur = cv2.GaussianBlur(img, (7, 7), 0)

    return blur


def sobel_edge_detection(img):
    """
    Detect edges using Sobel Filter.
    Returns a grayscale edge image.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize between 0-255
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    return edges.astype(np.uint8)