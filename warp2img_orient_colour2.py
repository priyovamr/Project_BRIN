import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from pyzbar.pyzbar import decode
  
cap1 = cv2.VideoCapture(0)
cap1.set(3,540)
cap1.set(4,320)

cap2 = cv2.VideoCapture(2)
cap2.set(3,540)
cap2.set(4,320)

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
  textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
 
  return angle

def nothing(a):
    pass

init_val = [0,179,0,255,0,50]

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 540, 240)
cv2.createTrackbar("Hue Min", "HSV", init_val[0],179, nothing)
cv2.createTrackbar("Hue Max", "HSV", init_val[1],179, nothing)
cv2.createTrackbar("Sat Min", "HSV", init_val[2],255, nothing)
cv2.createTrackbar("Sat Max", "HSV", init_val[3],255, nothing)
cv2.createTrackbar("Val Min", "HSV", init_val[4],255, nothing)
cv2.createTrackbar("Val Max", "HSV", init_val[5],255, nothing)

while True:
    # success1, img = cap1.read()
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    m_frame1 = cv2.flip(frame1, 1)
    m_frame2 = cv2.flip(frame2, 1)
    r_frame2 = cv2.rotate(m_frame2, cv2.ROTATE_180)
    img = np.concatenate((r_frame2, m_frame1), axis=0)

    imgCopy = img.copy()

    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "HSV")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV")
    v_min = cv2.getTrackbarPos("Val Min", "HSV")
    v_max = cv2.getTrackbarPos("Val Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img, lower, upper)
    hsl_result = cv2.bitwise_and(img, img, mask = mask)

    gray = cv2.cvtColor(hsl_result, cv2.COLOR_BGR2GRAY)
    
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for i, c in enumerate(contours):
    
      area = cv2.contourArea(c)
    
      if area < 3700 or 100000 < area:
        continue
    
      cv2.drawContours(img, contours, i, (0, 0, 255), 2)

      getOrientation(c, img)

    cv2.imshow('HSL', hsl_result)
    cv2.imshow('Output Image', img)
    # cv2.imshow("Kamera 1", m_frame1)
    # cv2.imshow("Kamera 2", m_frame2)
    # cv2.imshow('VERTICAL', Verti)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break