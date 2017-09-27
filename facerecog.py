import cv2
import sqlite3
import os
from PIL import Image
import numpy as np

database = 'faceBase.db'  # database name
cascade_path = 'data/haarcascades/haarcascade_frontalface_default.xml'  # face cascade xml path
data_folder = 'dataSet'
yml_file_location = 'recognizer/trainingData.yml'


def file_check():
    if not (os.path.isdir("dataSet")):
        os.mkdir("dataSet")
    if not (os.path.isdir("recognizer")):
        os.mkdir("recognizer")


def sql():
    conn = sqlite3.connect(database)
    c = conn.cursor()

    c.execute('CREATE TABLE IF NOT EXISTS faces  (user_id INTEGER PRIMARY KEY, user_name VARCHAR(25) NOT NULL)')
    c.execute('SELECT user_id FROM faces')
    a = len(c.fetchall())  # check is there any given user_id
    u_name = input('Please enter your name: ')
    if a == 0:
        a = 1
    else:
        a += 1

    query = 'INSERT INTO faces (user_id, user_name) VALUES (?, ?)'
    c.execute(query, (a, u_name))
    conn.commit()
    conn.close()

    return a, u_name


def set_creator():
    user_id, user_name = sql()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    count = 1

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.imwrite(data_folder+'/'+'User.'+str(user_id)+"."+str(count)+".jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.waitKey(100)  # time delay in milliseconds
            count += 1

        if count > 20:  # gets 20 pictures
            print('Operation Completed')
            break

        cv2.imshow('Face Data Collecting', frame)

    cap.release()
    cv2.destroyAllWindows()


def get_images_with_id(data_folder):
    image_paths = [os.path.join(data_folder, p) for p in os.listdir(data_folder)]
    faces = []
    ids = []
    for image_path in image_paths:
        face_img = Image.open(image_path).convert('L')  # converted to gray scale image
        face_np = np.array(face_img, 'uint8')  # unsigned integer 8
        user_id = int(os.path.split(image_path)[-1].split('.')[1])
        faces.append(face_np)
        ids.append(user_id)
        cv2.imshow('training', face_np)
        cv2.waitKey(10)
    return ids, faces


def run_trainer():
    recognizer = cv2.face.createLBPHFaceRecognizer()
    ids, faces = get_images_with_id(data_folder)
    recognizer.train(faces, np.array(ids))  # np array must be in int type
    recognizer.save(yml_file_location)
    cv2.destroyAllWindows()


def get_profile(user_id):
    conn = sqlite3.connect(database)
    c = conn.cursor()
    query = 'SELECT user_name FROM faces WHERE user_id='+str(user_id)
    profile = None
    name_info = c.execute(query)
    for row in name_info:
        profile = row
    conn.close()
    return profile


def detect():
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)
    rec = cv2.face.createLBPHFaceRecognizer()
    rec.load(yml_file_location)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            user_id, conf = rec.predict(gray[y:y+h, x:x+w])  # configuration
            profile = get_profile(user_id)
            if profile:
                cv2.putText(frame, profile[0], (x, y+h), font, 1, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run():
    file_check()
    message = """Select from the menu:\n
    \t1-) Add faces to dataset
    \t2-) Train the dataset
    \t3-) Run the face recognition
    \tq for quit
    \t>"""

    while True:
        print(message, end="")
        ans = input()
        if ans == "1":
            set_creator()
        elif ans == "2":
            run_trainer()
        elif ans == "3":
            detect()
        elif ans == "q":
            print("Quitting...")
            break
        else:
            print("Invalid input !!")

if __name__ == "__main__":
    run()
