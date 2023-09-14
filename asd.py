import cv2
import cvlib as cv
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array
model=load_model(r"D:\ALaa\github\gender detection\modelsmall.h5")

webcam = cv2.VideoCapture(0)   # to open cam 
classes = ['male','female'] # two classes 
color_dict={0:(0,255,0),1:(0,0,255)} # red for female and green for male 
while webcam.isOpened(): # main loop
    ckeck, frame = webcam.read() # to grab frame
    face, conf = cv.detect_face(frame)# to detect face 
    if not ckeck:
        print("failed to grab frame")
        break
    for index, f in enumerate(face): # bounding box
       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        face_crop = np.copy(frame[startY:endY,startX:endX])#crop the detected face region
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing  on image
        face_crop = cv2.resize(face_crop, (64,64))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] 

        # get label with max accuracy
        print(index, "index")
        if conf >= 0.5:
            index = 0
        elif conf < 0.5:
            index = 1
        label = classes[index]
        print(label)

        label = "{}".format(label)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label on rectangle
        cv2.rectangle(frame, (startX,startY), (endX,endY), color_dict[index], 2) # draw rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color_dict[index], 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop app 
    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()