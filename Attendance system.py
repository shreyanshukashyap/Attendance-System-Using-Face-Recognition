import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'Images'
images = []
personNames = []
myList = os.listdir(path)
# print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)

# returns those 128 features of each image passed(HOG encoding)


def faceEncodings(images):
    encodeList = []
    for img in images:
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    # r+ === read append mode
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            # current time and date
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

# reading camera (0 - laptop camera and 1 - external camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # resize the input coming from camera
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    # convert again
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    # finds the faces
    facesCurrentFrame = face_recognition.face_locations(faces)

    encodesCurrentFrame = face_recognition.face_encodings(
        faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        # compares on the basis on encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        # return index calue of min distance
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    # 13 is ascii value for enter key
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
