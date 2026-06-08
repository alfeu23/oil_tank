import cv2
import numpy as np


def extract_l_channel(image):
    """
    Converts image to LAB color space and extracts the L channel.
    Apply CLAHE for contrast enhancement.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    return l_enhanced


def threshold_l_channel(l_channel):
    """
    Applies Otsu's thresholding and morphological operations.
    """
    # Use Gaussian Blur before thresholding
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise and fill holes
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return closing
