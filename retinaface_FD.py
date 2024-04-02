# Please install 'pip install retina-face' to use retina face 

import cv2
import argparse
from retinaface import RetinaFace

def detect_faces(image):
    # Detect faces using RetinaFace
    resp = RetinaFace.detect_faces(image)

    for face_name, face_info in resp.items():
        x1, y1, x2, y2 = face_info['facial_area']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection using RetinaFace")
    parser.add_argument("-i", type=str, help="Path to input image")
    parser.add_argument("-w", action="store_true", help="Use webcam for face detection")
    args = parser.parse_args()

    if args.w:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            detect_faces(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif args.i:
        # Use image for face detection
        image = cv2.imread(args.i)
        detect_faces(image)
    else:
        print("Please provide either --image or --webcam option.")
