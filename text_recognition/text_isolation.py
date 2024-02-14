import cv2
import numpy as np

# Load the image
image = cv2.imread('data/img/01327.png')
# Assuming white can have any hue, but low saturation and high value
hue_low = 0
sat_low = 0
val_low = 254 # Start with a high value and adjust as needed

hue_high = 1  # Covering the entire hue range
sat_high = 1  # Keeping saturation low to moderate
val_high = 255  # Maximum brightness

# Convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of your text color in HSV
lower_color = np.array([hue_low, sat_low, val_low])
upper_color = np.array([hue_high, sat_high, val_high])

# Threshold the HSV image to get only the specific colors
mask = cv2.inRange(hsv, lower_color, upper_color)

# Bitwise-AND mask and original image to isolate the text
isolated_text = cv2.bitwise_and(image, image, mask=mask)

# Convert to grayscale and binarize the image for OCR
gray = cv2.cvtColor(isolated_text, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

final_image = cv2.bitwise_not(binary)
# Save or display the preprocessed image
cv2.imwrite('preprocessed_image.jpg', final_image)
# cv2.imshow('Preprocessed Image', final_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


