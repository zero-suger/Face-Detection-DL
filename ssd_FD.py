# PLEASE DOWNLOAD THESE LINKS

#model structure: https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
#pre-trained weights: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel 

import cv2 
import numpy as np
import argparse

# Initialize detector
detector = cv2.dnn.readNetFromCaffe(
    r"C:\Users\KII_win10pro\Desktop\FD_Face_Detection\pretrained_models\deploy.prototxt",
    r"C:\Users\KII_win10pro\Desktop\FD_Face_Detection\pretrained_models\res10_300x300_ssd_iter_140000.caffemodel"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Face Detection")
    parser.add_argument("--input", type=str, default="webcam", choices=["image", "webcam"], help="Input source: 'image' for image file, 'webcam' for webcam feed")
    parser.add_argument("--path", type=str, default="", help="Path to the input image file (if input is 'image')")
    return parser.parse_args()

def detect_faces(img):
    # Getting detections
    detector.setInput(cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0)))
    detections = detector.forward()

    (h, w) = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_arguments()

    if args.input == "image":
        if args.path == "":
            print("Please provide the path to the input image file.")
        else:
            img = cv2.imread(args.path)
            detect_faces(img)
    elif args.input == "webcam":
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            detect_faces(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
