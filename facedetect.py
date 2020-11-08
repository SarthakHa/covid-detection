import cv2
import sys

# Take name of image file as string parameter, output list of start/end coordinates
def detectFaces(imageFile):

    # Get user supplied values
    imagePath = imageFile
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Output list of coordinate tuples (startX, startY, endX, endY)
    coordinates = []
    for (x, y, w, h) in faces:
        coordinates.append((x, y, x + w, y + h))
    return coordinates


#tests - draws rectangles around faces
faces = detectFaces("people.jpg")
image = cv2.imread("people.jpg")
for face in faces:
    cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
cv2.imshow("Faces found", image)
cv2.waitKey(0)
