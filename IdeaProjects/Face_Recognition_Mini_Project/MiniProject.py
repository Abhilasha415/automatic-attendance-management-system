import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
##################    Data set collection   #################
face_classifier = cv2.CascadeClassifier('C:/Users/Abhilasha kumari/IdeaProjects/Face_Recognition_Mini_Project/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting image into gray scale
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face
cap = cv2.VideoCapture(0) # for video capture in GUI
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200)) # resizeing image into 200x200
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'C:/Users/Abhilasha kumari/IdeaProjects/Sample/Sample'+str(count)+'.jpg' # declaare the path location

        cv2.imwrite(file_name_path,face) # saving image to the declare folder

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face) # displaying image
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) & count==50: # After pressing the enter the GUI window will be closed
        break
cap.release()
cv2.destroyAllWindows() # for close all running window
print('Samples Colletion Completed ')


################  Data set collection  ###############

# Get the training data we previously made
data_path = 'C:/Users/Abhilasha kumari/IdeaProjects/Sample/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
# print(onlyfiles[0])

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    # print(image_path)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print(type(images),images)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

# Let's train our model
print("Dataset Model Training Completed..")


####################  Detection    ###################

face_classifier = cv2.CascadeClassifier('C:/Users/Abhilasha kumari/IdeaProjects/Face_Recognition_Mini_Project/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert gray code to color image
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2) # create a rectange on the face
        roi = img[y:y+h, x:x+w] # crop the image height and width
        roi = cv2.resize(roi, (200,200)) # resizing the data

    return img,roi

cap = cv2.VideoCapture(0) # for video capture in GUI
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # convert into color image
        result = model.predict(face) # redecting face from the trained model

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))


        if confidence > 82: # If matches the result with 82% then it will show the student name
            cv2.putText(image, "Abhilasha", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)

        else: # if not matched it will show Unknown
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except: # if no face then it will show Face not found
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) & 0xFF==ord('q'): # After pressing the enter the GUI window will be closed
        break

cap.release()
cv2.destroyAllWindows() # for close all running window

