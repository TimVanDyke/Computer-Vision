import cv2
import sys

# This reads a word from an image and saves each letter as an individual file
# Inspired by https://github.com/lavanya-m-k/character-detection-and-crop-from-an-image-using-opencv-python/blob/master/charDetect.py

# Read Image
img = cv2.imread('./pictures/' + sys.argv[1])

print("Finding letters...")

# Getting name to save new files as
i = sys.argv[1].find(".png")
img_name = sys.argv[1][0:i]

# Create threshold for image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours for each letter in image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

new_img_num = 0

for c in contours:
    # Get bounds for contours
    x,y,w,h = cv2.boundingRect(c)
    # Removes noise. Hardcoded for a test image but this could be improved.
    if w * h > 400 and x != 0 and y != 0:
        # Save each letter
        cv2.imwrite('./pictures/' + img_name + str(new_img_num) + ".png", thresh[y:y+h,x:x+w])
        new_img_num += 1


