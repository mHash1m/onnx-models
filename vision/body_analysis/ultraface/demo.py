# SPDX-License-Identifier: MIT
import os
import time
from pathlib import Path
import cv2
import onnxruntime as ort
import argparse
import numpy as np
from dependencies.box_utils import predict

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-320 onnx model
face_detector_onnx = "../ultraface/models/version-RFB-640.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])

face_detector = ort.InferenceSession(face_detector_onnx, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_detector = ort.InferenceSession(face_detector_onnx, providers=['CUDAExecutionProvider'])

COLOR = (204, 102, 0)
# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num

# face detection method
def faceDetector(orig_image, threshold = 0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main void

def run_for_image(target):
    image = target
    file_name = Path(image).stem
    orig_image = cv2.imread(image)
    boxes, labels, probs = faceDetector(orig_image)

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), COLOR, 4)
        cv2.putText(orig_image, str("%.2f" % round(probs[i], 2)), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 229, 204), 2)
        # cv2.imshow('', orig_image)
    cv2.imwrite(f"./results/{file_name}_pred.jpg", orig_image)

def run_for_dir(target):
    image_dir = target
    folder_name = os.path.basename(os.path.normpath(image_dir))
    pred_dir = os.path.join("./results", f"{folder_name}_preds")
    try:
        os.mkdir(pred_dir)
    except OSError as error:
        print(str(error)[10:])
    images = os.listdir(image_dir)
    for image in images:
        img_path = os.path.join(image_dir, image)
        orig_image = cv2.imread(img_path)
        boxes, labels, probs = faceDetector(orig_image)
        pred_image = Path(img_path).stem
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), COLOR, 4)
            cv2.putText(orig_image, str("%.2f" % round(probs[i], 2)), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 229, 204), 2)
            # cv2.imshow('', orig_image)
        cv2.imwrite(f"{pred_dir}/{pred_image}_pred.jpg", orig_image)

def run_for_video(target):
    start_time = time.time()
    cap = cv2.VideoCapture(target)
    fps, width, height = (int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = Path(target).stem
    out = cv2.VideoWriter(f"./results/{file_name}_pred.mp4", fourcc, fps, (width, height))
    counter = 1
    while(cap.isOpened()):  
        ret, frame = cap.read()
        if ret == False:
            break
        boxes, labels, probs = faceDetector(frame)

        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR, 4)
            cv2.putText(frame, str("%.2f" % round(probs[i], 2)), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 229, 204), 2)
        out.write(frame)
        # print(f"Frame # {counter} predicted!")
        counter+=1
    print("Video Width = ", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), ", Video Height = ", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Average FPS: ", counter / (time.time() - start_time))
    cap.release()
    cv2.destroyAllWindows()

parser=argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, required=True, help="input type")
parser.add_argument("-tg", "--target", type=str, required=True, help="path to vid/image/dir")
args=parser.parse_args()
task = args.task
target = args.target
try:
    os.mkdir("./results")
except OSError as error:
    print(str(error)[10:])
if task == "video":
    run_for_video(target)
elif task == "dir":
    run_for_dir(target)
elif task == "image":
    run_for_image(target)
else:
    print("UNKNOWN TASK, should be 'video', 'dir', or 'image'")
