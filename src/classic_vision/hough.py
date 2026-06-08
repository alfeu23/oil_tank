import cv2
import numpy as np

def detect_circles(image, min_dist=20, param1=50, param2=30, min_radius=10, max_radius=50):
    """
    Detects circles in an image using the Circular Hough Transform.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=min_dist,
        param1=param1, 
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    return []

def detect_by_contours(binary_image, min_area=500, max_area=100000, circularity_threshold=0.6):
    """
    Detects circular objects using contour analysis.
    Filters by area and circularity.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            # Circularity = 4 * pi * area / perimeter^2
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > circularity_threshold:
                # Get the minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circles.append([int(x), int(y), int(radius)])
                
    return np.array(circles)

def draw_circles(image, circles):
    """
    Draws detected circles on the image.
    """
    output = image.copy()
    for i in circles:
        # draw the outer circle
        cv2.circle(output, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(output, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
    return output
