import cv2
import sys
from PIL import Image

from google.cloud import automl

# TODO(developer): Uncomment and set the following variables
project_id = "noted-will-294917"
model_id = "ICN5340972289921908736"

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
    #print("Found {0} faces".format(len(faces)))

    # Output list of coordinate tuples (startX, startY, endX, endY)
    coordinates = []
    for (x, y, w, h) in faces:
        x1 = max(int(x - .25 * w), 0)
        y1 = max(int(y - .25 * h), 0)
        x2 = min(int(x + 1.25 * w), image.shape[1] - 1)
        y2 = min(int(y + 1.25 * h), image.shape[0] - 1)
        #print (x1, y1, x2, y2)
        coordinates.append((x1, y1, x2, y2))
    return coordinates

#GCloud prediction client
prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = automl.AutoMlClient.model_path(
    project_id, "us-central1", model_id
)

def predictionCall(file_path_image):
    #returns faces detected in image
    faces = detectFaces(file_path_image)

    count = 0
    for face in faces:
        count = count + 1
        im = Image.open(file_path_image)
        xTopLeft = face[0]
        yTopLeft = face[1]
        xBotRight = face[2]
        yBotRight = face[3]
        im1 = im.crop((xTopLeft, yTopLeft, xBotRight, yBotRight))
        img_path = "Predict" + str(count) + ".jpg"
        im1 = im1.save(img_path)

        # Read the file.
        with open(img_path, "rb") as content_file:
            content = content_file.read()

        image = automl.Image(image_bytes=content)
        payload = automl.ExamplePayload(image=image)
        params = {"score_threshold": "0.8"}

        request = automl.PredictRequest(
            name=model_full_id,
            payload=payload,
            params=params
        )

        response = prediction_client.predict(request=request)

        mask = False
        unmask = False
        both = False

        print("Prediction results:")
        for result in response.payload:
            if "UnmaskedPeople" == result.display_name:
                unmask = True
            if "MaskedPeople" == result.display_name:
                mask = True
            #print("Predicted class name: {}".format(result.display_name))
            #print("Predicted class score: {}".format(result.classification.score))
        if mask == True and unmask == True:
            both = True

        if both == True:
            print("Person " + str(count) + ": Error. Both Mask and Unmask detected")
        elif mask == True:
            print("Person " + str(count) + " is wearing mask :)")
        else:
            print("Person " + str(count) + " is NOT wearing a mask :(")

print("Welcome to mask detection. Created to facilitate the enforcement of mandatory mask-wearing rules in public spaces")
print("Time taken to predict is contingent on strength of internet connection. Prediction is done on google cloud AutoML")
print("\nExample 1 (Close Up): ")
predictionCall("CloseUp.jpg")
print("\nExample 2 (Wearing mask): ")
predictionCall("mask.jpg")
print("\nExample 3 (Not Wearing Mask Group): ")
predictionCall("nomask.jpg")



"""
#test - draws rectangles around faces
faces = detectFaces(file_path_image)
image = cv2.imread(file_path_image)
for face in faces:
    cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (255, 0, 255), 2)
    print("{} x {}".format(face[0] - face[2], face[3] - face[1]))
cv2.imshow("Faces found", image)
cv2.waitKey(0)
"""
