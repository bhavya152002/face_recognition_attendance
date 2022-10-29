from importlib.resources import path
import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread



# -------------- image labesl ------------------------

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    faces = []
    Ids = []
    #loading the Ids and the images
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# ----------- train images function ---------------
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    faces, Id = getImagesAndLabels("/home/bhavya/Documents/vscode/python/projects/Face-Recognition-Attendance-System/FRAS/TrainingImage")
    Thread(target = recognizer.train(faces, np.array(Id))).start()
    Thread(target = counter_img("/home/bhavya/Documents/vscode/python/projects/Face-Recognition-Attendance-System/FRAS/TrainingImage")).start()
    recognizer.save("/home/bhavya/Documents/vscode/python/projects/Face-Recognition-Attendance-System/FRAS/TrainingImageLabel"+os.sep+"Trainner.yml")
    print("All Images")

# counter
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1