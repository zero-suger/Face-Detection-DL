# Please install dlib library by running 'pip install mtcnn'

import cv2
import argparse
from mtcnn.mtcnn import MTCNN

def detect_faces(image, face_cascade_path):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model
    classifier = cv2.CascadeClassifier(face_cascade_path)

    # Perform face detection with pretrained model Haar Cascade
    face_rects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    # Draw rectangles around detected faces
    for (x, y, w, h) in face_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def main(args):
    if args.image_path:
        img = cv2.imread(args.image_path)
        if img is None:
            print("Error: Invalid image path")
            return
        result_img = detect_faces(img, args.model_path)
        cv2.imshow("Detected Faces", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam")
                break

            result_frame = detect_faces(frame, args.model_path)
            cv2.imshow("Detected Faces", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Please provide an image path or use the webcam option")

if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", help="Path to the input image file")
    ap.add_argument("-w", "--webcam", action="store_true", help="Use webcam as input")
    ap.add_argument("-m", "--model_path", help="Path to the face detection model")
    args = ap.parse_args()

    
    main(args)
