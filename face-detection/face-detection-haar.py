''' Reference Tutorials:
- Getting started with videos in opencv: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
- Face detection using Haar Cascades: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
'''

import numpy as np
import cv2

def face_detection_haar():
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, img = cap.read()

		# Our operations on the frame come here
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Face detection
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		# Draw rectangles
		for (x,y,w,h) in faces:
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			# # Eye detection
			# eyes = eye_cascade.detectMultiScale(roi_gray)
			# for (ex,ey,ew,eh) in eyes:
			# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

		# Display the resulting frame
		cv2.imshow('Face Detection',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


if __name__  == "__main__":
	face_detection_haar()