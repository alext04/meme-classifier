

import cv2
import numpy as np

from PIL import Image
import pytesseract



image = cv2.imread('preprocessed_image.jpg')



# Define the scale factor; for example, 2 times the original size
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
text = pytesseract.image_to_string(final_image, lang='eng', config='--psm 7')


cv2.imshow('Preprocessed Image',sharpened_image)
cv2.waitKey(0)

print(text)





# import cv2
# import numpy as np

# from PIL import Image
# import pytesseract



# image = cv2.imread('preprocessed_image.jpg')



# text = pytesseract.image_to_string(image, lang='eng', config='--psm 11')


# # cv2.imshow('Preprocessed Image', image)
# # cv2.waitKey(0)

# print(text)
