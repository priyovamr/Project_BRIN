import cv2
import numpy as np
  
cap1 = cv2.VideoCapture(0)
cap1.set(3,540)
cap1.set(4,320)

cap2 = cv2.VideoCapture(2)
cap2.set(3,540)
cap2.set(4,320)

while True:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    m_frame1 = cv2.flip(frame1, 1)
    m_frame2 = cv2.flip(frame2, 1)
    Verti = np.concatenate((m_frame2, m_frame1), axis=1)
    # cv2.imshow("Kamera 1", m_frame1)
    # cv2.imshow("Kamera 2", m_frame2)
    cv2.imshow('VERTICAL', Verti)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break