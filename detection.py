#!/usr/bin/env python3
# -*-coding: utf-8 -*-

import cv2
import time
import numpy as np
import torch
import torchvision

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def colors_for_labels():
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = [(i * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1]) % 255).tolist() for i in range(len(COCO_INSTANCE_CATEGORY_NAMES))]
    #colors = np.array(range(len(COCO_INSTANCE_CATEGORY_NAMES))) * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    #colors = (colors % 255).numpy().astype("uint8")
    return colors


def drawpred(pred, frame, colors):
    #print(pred[0]['boxes'])
    #print(pred[0]['labels'])
    #print(pred[0]['scores'])
    for box, label, score in zip(pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']):
        if score > 0.9:
            frame = cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), tuple(colors[label]), 2)
            #frame = cv2.putText(frame, label, tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            frame = cv2.putText(frame, COCO_INSTANCE_CATEGORY_NAMES[label], tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return frame


def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().cuda()

    colors = colors_for_labels()

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, _ = frame.shape
        #frame = cv2.resize(frame,(int(width/2), int(height/2)))

        frame_tensor = torch.from_numpy(frame.astype(np.float32).transpose(2, 0, 1)).cuda()/255.0
        #print(frame_tensor)
        #print(torch.rand(3, 300, 400))

        pred = model([frame_tensor])
        #print(drawpred(pred))

        frame = drawpred(pred, frame, colors)
        frame = cv2.resize(frame, (int(width*2), int(height*2)))
        if ret:
            cv2.imshow('frame', frame)
        else:
            time.sleep(2)

        # Display the resulting frame
        if cv2.waitKey(20) == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
