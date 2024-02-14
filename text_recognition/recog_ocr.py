import cv2
import numpy as np


from PIL import Image
import pytesseract


# Load the image
image = cv2.imread('data/img/01672.png')
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

# final_image = cv2.bitwise_not(binary)
image = binary

scale_factor = 2

# Resize the image
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
dim = (width, height)

# It's often beneficial to sharpen the image after scaling
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

# Create a sharpening kernel
sharpening_kernel = np.array([[-1, -1, -1],
                                         [-1,45, -1],  # Increased center value
                                         [-1, -1, -1]])   # Optional normalization

# Apply the kernel to the image
sharpened_image = cv2.filter2D(resized_image, -1, sharpening_kernel)



# Convert the processed OpenCV image to a PIL Image
final_image = Image.fromarray(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))

# Perform OCR using Tesseract
text = pytesseract.image_to_string(final_image, lang='eng', config='--psm 11')



# text = pytesseract.image_to_string(final_image, lang='eng', config='--psm 11')



print(text)

