# code is edited from 'https://github.com/nadeemakhter0602'. (Thank you)

import cv2
import onnxruntime as rt
import os
import numpy as np
import time
import argparse

work_dir = os.path.dirname(os.path.realpath(__file__))

class Detector():

    def __init__(self, model_name, rescale, input_shape, backend='cpu', config=None):
        self.rescale = rescale
        self.input_shape = input_shape
        if backend=='cpu' or backend=='cuda':
            provider = ['CUDAExecutionProvider' if backend=='cuda' else 'CPUExecutionProvider']
            self.model = rt.InferenceSession(os.path.join(work_dir, model_name), providers=provider)
            self.input_name = self.model.get_inputs()[0].name
        elif backend=='cv2':
            self.model = cv2.dnn_DetectionModel(os.path.join(work_dir, model_name), config=config)
            self.model.setInputScale(scale=self.rescale)
            self.model.setInputSize(size=self.input_shape)

    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def detect(self, image):
        image = self.format_yolov5(image)
        
        if isinstance(self.model, cv2.dnn_DetectionModel):
            outputs = self.model.predict(image)
            # last model layer is of shape (1, 25500, 85)
            outputs = outputs[-1][0]
        elif isinstance(self.model, rt.capi.onnxruntime_inference_collection.InferenceSession):
            blob = cv2.dnn.blobFromImage(image, self.rescale, self.input_shape)
            outputs = self.model.run([], {self.input_name : blob})
            # first model layer is of shape (1, 25500, 85)
            outputs = outputs[0][0]
        boxes, confidences = self._detect(outputs, image)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.5)
        boxes = [boxes[i] for i in indexes]
        return boxes

    # only works for outputs of shape (25500, 85)
    def _detect(self, outputs, image):
        image_width, image_height, _ = image.shape
        x_factor = image_width / self.input_shape[0]
        y_factor =  image_height / self.input_shape[1]
        outputs = [row for row in outputs if np.amax(row[5:])>0.5 and row[4]>0.5]
        confidences = [row[4] for row in outputs]
        xywh = [(row[0], row[1], row[2], row[3]) for row in outputs]
        boxes = [np.array([int((x - 0.5 * w) * x_factor), int((y - 0.5 * h) * y_factor), int(w * x_factor), int(h * y_factor)])
        for x, y, w, h in xywh]
        return boxes, confidences
    
    
def draw_bounding_boxes(image, boxes):
    for box in boxes:
        # overlay rectangle boxes and class name 
        cv2.rectangle(image, box, (0, 255, 0), 2)

def main():
    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser(description='Object detection using YOLOv5 model.')

    # Adding arguments for image path, webcam, and model path
    parser.add_argument('--image_path', type=str, help='Path to the input image file.')
    parser.add_argument('--webcam', action='store_true', help='Flag to indicate using webcam for inference.')
    parser.add_argument('--model_path', type=str, help='Path to the YOLOv5 model file.')

    # Parsing the command-line arguments
    args = parser.parse_args()

    if args.webcam:
        detect_webcam(args.model_path)
    elif args.image_path:
        detect_image(args.model_path, args.image_path)
    else:
        print("Please provide either '--image_path' or '--webcam' argument.")

def detect_webcam(model_path):
    # need cudnn and CUDA toolkit installed for cuda as ONNXRuntime backend
    detector_yolov5n6 = Detector(model_path, 1/255.0, (416, 416), backend='cuda')

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        boxes = detector_yolov5n6.detect(frame)

        # Draw bounding boxes on the frame
        draw_bounding_boxes(frame, boxes)

        # Display the frame
        cv2.imshow('YOLOv5 Object Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def detect_image(model_path, image_path):
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image from", image_path)
        return

    # Initialize detector
    detector_yolov5n6 = Detector(model_path, 1/255.0, (416, 416), backend='cuda')

    # Detect objects in the image
    boxes = detector_yolov5n6.detect(image)

    # Draw bounding boxes on the image
    draw_bounding_boxes(image, boxes)

    # Display the image
    cv2.imshow('YOLOv5 Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
