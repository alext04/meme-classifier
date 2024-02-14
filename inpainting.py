

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import keras_ocr

# Function to calculate midpoint of a line
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


# Initialize keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Path to the image with text
image_path = 'data/img/01236.png'

# Read the image
image = keras_ocr.tools.read(image_path) 

# Recognize text in the image
predictions = pipeline.recognize([image])

# Create a mask for inpainting
mask = np.zeros(image.shape[:2], dtype="uint8")

# Iterate through predicted text regions and create mask
for box in predictions[0]:
    x0, y0 = box[1][0]
    x1, y1 = box[1][1] 
    x2, y2 = box[1][2]
    x3, y3 = box[1][3]
    
    # Calculate midpoints for line drawing
    x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
    x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)
    
    # Calculate thickness based on line length
    thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    
    # Draw line on mask
    cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)

# Inpaint the text regions
inpainted_image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)


# Save the image without text
cv2.imwrite('curdir/no_caption.jpg', cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

