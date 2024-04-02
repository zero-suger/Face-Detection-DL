# Face Detection DL
Face Detection DL is testing all [FD] Face Detection models and methods : DLib, Haar-Cascade(OpenCV), Mediapipe(Google), MTCNN, RetinaFace(Insightface), SCRFD (Insightface), SSD, YOLOv5 (Ultralytics), YuNet, etc with Python and user can use IMAGE or WEBCAM as a input. 

*Note : All models and methods are tested by using 'iVcam' (real-time) camera, with PYTHON version 3.9.7* 

![Merged_image](https://github.com/zero-suger/Face-Detection-DL/assets/63332872/0d041d62-57f7-452d-8f76-a5ea3e1a0a5c)

## Get started

To test all [FD] Methods and Models, please first create conda environment and then install important requirements by running : 

```bash
  conda create -n FD python=3.9.7

  pip install - r requirements.txt
```


## TESTING

1) If you want to test 'DLib' to face detection task, please : 

```bash
  pip install dlib

  python ./dlib_FD.py --image_path '(your image path)' 

  or 
  
  python ./dlib_FD.py --webcam (automatically detect first webcam) 
   
```

2) If you want to test 'Haar-Cascade' to face detection task, please : 

```bash

  python ./Haar_Cascade_FD.py --image_path '(your image path)' --model_path '(path to haarcascade_frontalface.xml file)'

  or 
  
  python ./Haar_Cascade_FD.py --webcam (automatically detect first webcam) 
   
```

3) If you want to test 'Mediapipe' to face detection task, first set 'model_asset_path={your local path}/blaze_face_short_range.tflite', then change 'IMAGE_FILE' to your testing image and enjoy :

```bash
  pip install mediapipe

  python ./mediapipe_FD.py'

```

3) If you want to test 'MTCNN' to face detection task, please : 

```bash
  pip install mtcnn

  pip install tensorflow
  
  python ./mtcnn_FD.py --image_path '(your image path)' 

  or 
  
  python ./mtcnn_FD.py --webcam (automatically detect first webcam)

```

4) If you want to test 'RetinaFace' to face detection task, please : 

```bash
  pip install retina-face
  
  python ./retinaface_FD.py -i '(your image path)' 

  or 
  
  python ./retinaface_FD.py --w (automatically detect first webcam)

```

5) If you want to test 'SCRFD' to face detection task, please : 

```bash
  pip install onnx

  pip install onnxruntime
  
  python ./scrfd_FD.py --image_path '(your image path)' --model_path '{your local}/scrfd_face_detector.onnx'

  or 
  
  python ./scrfd_FD.py --use_webcam (automatically detect first webcam) --model_path '{your local}/scrfd_face_detector.onnx'

```

6) If you want to test 'SSD' to face detection task, download 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' (check ssd_FD.py)

```bash
  download or use : model structure: https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
  download or use : https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.   caffemodel
  
  python ./ssd_FD.py --input "image"/"webcam" (if input is "image" --path '(your testing image)') 

```

7) If you want to test 'YOLO_v5' to face detection task, please : 

```bash
  pip install onnx

  pip install onnxruntime

  pip install ultralytics
  
  python ./yolo_5s_face.py --image_path '(your image path)' --model_path '{your local}/best_yolov5n6.onnx'

  or 
  
  python ./yolo_5s_face.py --webcam (automatically detect first webcam) --model_path '{your local}/best_yolov5n6.onnx'

```

8) If you want to test 'YuNet' to face detection task, please : 

```bash
  
  python ./YuNet_FD.py --input '(your image path)' --model_path '{your local}/face_detection_yunet_2023mar.onnx'

  or 
  
  python ./YuNet_FD.py (automatically detect first webcam) --model '{your local}/face_detection_yunet_2023mar.onnx'

```
