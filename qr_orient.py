import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
 
# Load the image
img = cv2.imread("206_NURAENI_XI TKB 4.png")
qcd = cv2.QRCodeDetector()
retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)
# print(points.astype(int))
 
# Was the image there?
if img is None:
  print("Error: File not found")
  exit(0)
 
cv2.imshow('Input Image', img)

img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 3)
 
# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Convert image to binary
_, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
# Find all the contours in the thresholded image
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
 
for i, c in enumerate(contours):
 
  # Calculate the area of each contour
  area = cv2.contourArea(c)
 
  # Ignore contours that are too small or too large
  if area < 3700 or 100000 < area:
    continue
 
  # cv.minAreaRect returns:
  # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
#   print(rect)
 
  # Retrieve the key parameters of the rotated bounding box
  center = (int(rect[0][0]),int(rect[0][1])) 
  width = int(rect[1][0])
  height = int(rect[1][1])
  angle = int(rect[2])
 
  if width < height:
    angle = 90 - angle
  else:
    angle = -angle
         
  label = "  Rotation Angle: " + str(angle) + " degrees"
  textbox = cv2.rectangle(img, (center[0]-35, center[1]-25), (center[0] + 295, center[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (center[0]-50, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
  cv2.drawContours(img,[box],0,(0,0,255),2)

# for s, p in zip(decoded_info, points):
#     img = cv2.putText(img, s, p[0].astype(int),
#                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('Output Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('206_NURAENI_XI TKB 4.png', img)
  
# Save the output image to the current directory
cv2.imwrite("min_area_rec_output.jpg", img)