# Import OpenCV2 for image processing
import cv2
import time
from picamera import PiCamera
from picamera.array import PiRGBArray

# Start capturing video 
camera = PiCamera()
camera.resolution = (480, 320)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = (480, 320))

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('/home/pi/opencv/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = 5

# Initialize sample face image
count = 0

time.sleep(0.1)
# Start looping
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    # Capture video frame
    image = frame.array 

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        print(count)

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count > 100:
        break

    # Stop video
    rawCapture.truncate(0)
