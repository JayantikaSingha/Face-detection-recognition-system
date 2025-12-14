import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream


print("Loading Face Detector...")
protoPath = "C:/Users/HP/Desktop/face detection/face_detection_model/deploy.prototxt"
modelPath = "C:/Users/HP/Desktop/face detection/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("C:/Users/HP/Desktop/face detection/openface_nn4.small2.v1.t7")


recognizer = pickle.loads(open("C:/Users/HP/Desktop/face detection/output/recognizer.pickle", "rb").read())
le = pickle.loads(open("C:/Users/HP/Desktop/face detection/output/le.pickle", "rb").read())


print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


fps = FPS().start()


detected_names = set()


while True:
    
    frame = vs.read()

    
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.9:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW >= 20 and fH >= 20:
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                
                if name != "unknown" and name not in detected_names:
                    detected_names.add(name)
                    print("Face detected, name =", name)

                
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    fps.update()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
