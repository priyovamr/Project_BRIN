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
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
 
  return angle

while True:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    m_frame1 = cv2.flip(frame1, 1)
    m_frame2 = cv2.flip(frame2, 1)
    r_frame2 = cv2.rotate(m_frame2, cv2.ROTATE_180)
    img = np.concatenate((r_frame2, m_frame1), axis=0)

    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        # print(myData)
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        print(pts)

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

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
        cv2.putText(img, label, (center[0]-50, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        # cv2.drawContours(img,[box],0,(0,0,255),2)

        cv2.polylines(img,[pts],True,(255,0,255),5)
        pts2 = barcode.rect
        # print(pts2)
        cv2.putText(img,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)

    cv2.imshow('Output Image', img)
    # cv2.imshow("Kamera 1", m_frame1)
    # cv2.imshow("Kamera 2", m_frame2)
    # cv2.imshow('VERTICAL', Verti)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break