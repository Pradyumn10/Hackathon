import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
import time
import reference_world as world


#fitmentScore = -100


def face(image,fitmentScore):

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        './haarcascade/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(grayImage)
    countFace = faces.shape[0]

    if len(faces) == 0:
        print("ERROR!! \n No Face Detected")
        print("1. Fitment Score : ", fitmentScore)
        return ["No Face Detected", fitmentScore]

    elif(countFace == 1):
        #print("One Face Detected!! \n Selecting ROI")
        for (x, y, w, h) in faces:
            # x=x-150
            # y=y-200
            # w=w+300
            # h=h+300
            roi_gray = grayImage[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
        cv2.imwrite("./roi.png", roi_color)
        fitmentScore = fitmentScore+30
        print("Passed Criteria 1 : Face Detected")
        print("1. Fitment Score : ", fitmentScore)
        blur(image, roi_gray, roi_color, fitmentScore)
    else:
        print("ERROR!! \n More than one face Detected")
        print("1. Fitment Score : ", fitmentScore)
        return ["No Face or More than one Face Detected", fitmentScore]

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def blur(image, grayImage, colorImage, fitmentScore):
    laplace = cv2.Laplacian(grayImage, cv2.CV_64F).var()
    threshold = 55.0
    if laplace < threshold:
        print("ERROR!! \n Image is Blurred")
        print("2. Fitment Score : ", fitmentScore)
        return ["ERROR!! Image is Blurred", fitmentScore]

    else:
        print("Passed Criteria 2 : No Blur Detected")
        print("2. Fitment Score : ", fitmentScore)
        brightness(image, grayImage, colorImage, fitmentScore)


def brightness(image, grayImage, colorImage, fitmentScore):
    L, A, B = cv2.split(cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB))
    L = L/np.max(L)
    # print(np.mean(L))
    if(np.mean(L) < 0.4):
        print("Error!! Image is Dark")
        print("3.Fitment Score : ", fitmentScore)
        return ["Error!! Image is Dark", fitmentScore]
    elif(np.mean(L) > 0.8):
        print("Error!! Image is Too Bright")
        print("3.Fitment Score : ", fitmentScore)
        return ["Error!! Image is Too Bright", fitmentScore]
    else:
        fitmentScore = fitmentScore+30
        print("Passed Criteria 3 : Image is Normal")
        print("3. Fitment Score : ", fitmentScore)
        eye(image, grayImage, colorImage, fitmentScore)


def eye(image, grayImage, colorImage, fitmentScore):
    eye_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(grayImage)
    # print(grayImage)
    # ,scaleFactor=1.04,minNeighbors=13,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(eyes) == 0:
        print("Error!! No eyes detected")
        print("4.Fitment Score : ", fitmentScore)
        return ["Error!! No eyes detected", fitmentScore]
    elif eyes.shape[0] == 2:
        fitmentScore = fitmentScore+30
        print("Passed Criteria 4 : Detected Eyes")
        print("4. Fitment Score : ", fitmentScore)
        mouth(image, grayImage, colorImage, fitmentScore)
    else:
        print("Error!! Eyes are covered")
        print("4. Fitment Score : ", fitmentScore)
        return ["Error!! Eyes are covered", fitmentScore]


def mouth(image, grayImage, colorImage, fitmentScore):
    mouth_cascade = cv2.CascadeClassifier(
        './haarcascade/haarcascade_smile.xml')
    mouth = mouth_cascade.detectMultiScale(grayImage)
    # ,scaleFactor=1.4,minNeighbors=26,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    countMouth = mouth.shape[0]
    if countMouth == 1:
        fitmentScore = fitmentScore+30
        print("Passed Criteria 5 : Face is not covered")
        print("5. Fitment Score : ", fitmentScore)
        pose(image, grayImage, colorImage, fitmentScore)
    else:
        print("Error!! Face is covered")
        print("5. Fitment Score : ", fitmentScore)
        return ["Error!! Face is covered", fitmentScore]


def pose(image, grayImage, colorImage, fitmentScore):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    #im = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/clear.png')
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0)
    face3Dmodel = world.ref3DModel()
    for face in faces:
        shape = predictor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), face)
        refImgPts = world.ref2dImagePoints(shape)
        height, width, channel = image.shape
        focalLength = 1 * width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)
        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(
            face3Dmodel, refImgPts, cameraMatrix, mdists)
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(
            noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
        # draw nose line
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        # calculating angle
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x = np.arctan2(Qx[2][1], Qx[2][2])
        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] *
                       Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
        z = np.arctan2(Qz[0][0], Qz[1][0])
        gaze = "Looking: "
        if angles[1] < -15:
            fitmentScore = fitmentScore-5
            print("Looking Slightly Left")
            print("6. Fitment Score : ", fitmentScore)
            return ["Looking Slightly Left", fitmentScore]
        elif angles[1] > 15:
            fitmentScore = fitmentScore-5
            print("Looking Slightly Right")
            print("6. Fitment Score : ", fitmentScore)
            return ["Looking Slightly Right", fitmentScore]
        else:
            fitmentScore = fitmentScore+20
            print("Looking Forward")
            print("6. Fitment Score : ", fitmentScore)
            return ["Normal Image", fitmentScore]
if __name__ == "__main__":
    image = cv2.imread('./easy_accept-3.png')
#    #obj = Detection(image)
    face(image,-100)
