from flask import Flask, render_template, Response, jsonify
import cv2
from imutils.video import VideoStream
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
whT = 320
confThreshold = 0.7
nmsThreshold = 0.4
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/new")
def new():
    return render_template('i.html')
def gen():
    vs = VideoStream(src=0).start()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "Please wear Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    persons_detected = 0  # Counter for persons detected

    # Process only the detection with the highest confidence for each person
    for output in outputs:
        det = max(output, key=lambda x: x[5])
        scores = det[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]

        if confidence > confThreshold and classNames[classId].lower() == 'person':
            persons_detected += 1
            w, h = int(det[2] * wT), int(det[3] * hT)
            x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
            bbox.append([x, y, w, h])
            classIds.append(classId)
            confs.append(float(confidence))

    # Draw rectangles and put text only for detected persons
    for i in range(persons_detected):
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'Person {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def generate_frames():
    while True:
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        # Ensure there are unconnected layers before attempting to access them
        unconnected_layers = net.getUnconnectedOutLayers()
        if len(unconnected_layers) > 0:
            layers_names = net.getLayerNames()
            output_names = [layers_names[i - 1] for i in unconnected_layers.flatten()]
            outputs = net.forward(output_names)
            find_objects(outputs, img)

            ret, frame = cv2.imencode('.jpg', img)

            data = frame.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        else:
            print("No unconnected layers found.")


if __name__ == '__main__':
    prototxtPath = r"face_detector/deploy.prototxt"
    weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")
    app.run(debug=True, use_reloader=False)  

