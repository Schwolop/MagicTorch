import cv
import os
import sys

def GrabAndShow(windowName, captureDevice):
	frame = cv.QueryFrame(captureDevice)
	if frame is None:
		return
	cv.Flip(frame, None, 1) # Mirror the display.
	DetectFaces(frame)
	DetectCorners(frame)
	cv.ShowImage(windowName,frame)

def CreateWindow(windowName):
	cv.NamedWindow(windowName, flags=cv.CV_WINDOW_AUTOSIZE)
	cv.MoveWindow(windowName,0,0)

def CloseWindow(windowName):
	cv.DestroyWindow(windowName)

def InitCapture():
	for camera in EnumerateCameras():
		return cv.CaptureFromCAM(camera) # Return first. Todo, GUI selection.

def ReleaseCapture(captureDevice):
	del captureDevice

def ConvertToGrayscale(image):
	imageSize = cv.GetSize(image)
	
	# Convert to grayscale
	grayscale = cv.CreateImage(imageSize, 8, 1)
	cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
	
	# Equalise histogram
	cv.EqualizeHist(grayscale, grayscale)
	return grayscale

def DetectFaces(image):	
	grayscale = ConvertToGrayscale(image)
	
	# Detect objects
	cascade = cv.Load('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
	faces = cv.HaarDetectObjects(grayscale,cascade,cv.CreateMemStorage(0), 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (50,50))

	for ((x,y,w,h),n) in faces:
		cv.Rectangle(	image, 
						( int(x), int(y) ),
						( int(x + w), int(y + h) ),
						cv.CV_RGB(0, 255, 0), 3, 8, 0
						)

def DetectCorners(image):
	grayscale = ConvertToGrayscale(image)
	dst = cv.CreateMat(image.height, image.width, cv.CV_32FC1)
	cv.CornerHarris(grayscale, dst, 3) # blocksize = 3
	
	for i in range(0,dst.width):
		for j in range(0,dst.height):
			val = cv.Get2D(dst,j,i)
			if val[0] > 10e-6:
				cv.Circle( image, (i,j), 2, cv.RGB(0,255,0), 1, 8, 0 )

def EnumerateCameras():
	cameraList=[]
	for i in range(0,256):
		tempCam = cv.CaptureFromCAM(i)
		if cv.GetCaptureProperty(tempCam,cv.CV_CAP_PROP_FRAME_WIDTH) != 0.0:
			cameraList.append(i)
	return cameraList
	
if __name__ == "__main__":
	print "Press Esc to exit..."
	windowName = "Test"
	CreateWindow(windowName)
	cvCap = InitCapture()
	while(1):
		GrabAndShow(windowName, cvCap)
		key = cv.WaitKey(10)
		if key == 27: #Esc to quit.
			break
	CloseWindow(windowName)
	ReleaseCapture(cvCap)