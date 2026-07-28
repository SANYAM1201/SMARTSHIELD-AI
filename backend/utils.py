import cv2
import os


def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Cannot find image: {path}")

    return image


def save_image(path, image):

    folder = os.path.dirname(path)

    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder)

    cv2.imwrite(path, image)