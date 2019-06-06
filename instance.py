#!/usr/bin/env python3
# -*-coding: utf-8 -*-

import cv2
import time
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

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
    colors = [(i * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1]) % 255).astype(np.uint8) for i in range(len(COCO_INSTANCE_CATEGORY_NAMES))]
    #colors = np.array(range(len(COCO_INSTANCE_CATEGORY_NAMES))) * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    #colors = (colors % 255).numpy().astype("uint8")
    return colors


def drawpred(pred, frame, colors):
    blend_rate = 0.5
    all_mask = np.zeros_like(frame)
    for box, label, score, mask in zip(pred['boxes'], pred['labels'], pred['scores'], pred['masks']):
        if score > 0.9:
            #print((mask * 255).cpu().squeeze())
            #mask = cv2.cvtColor((mask * 255).cpu().squeeze().detach().numpy(), cv2.COLOR_GRAY2BGR).astype(np.uint8)
            mask = cv2.cvtColor(mask.cpu().squeeze().detach().numpy(), cv2.COLOR_GRAY2BGR)
            all_mask += (colors[label] * mask).astype(np.uint8)
            #frame[mask.cpu().squeeze().detach().numpy() != 0] = colors[label]
            
            frame = cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), colors[label].tolist(), 2)
            frame = cv2.putText(frame, COCO_INSTANCE_CATEGORY_NAMES[label], tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    frame = cv2.addWeighted(all_mask.astype(np.uint8), blend_rate, frame, blend_rate, 0)

    return frame


def main():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
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
        #print(torch.max(pred['out'], 1)[1].shape)
        #print(pred[0]['boxes'])
        #print(pred[0]['labels'])
        #print(pred[0]['scores'])
        #print(pred[0]['masks'].shape)
        #print(torch.max(pred[0]['masks'], 0)[1].shape)
        #pred_result = torch.max(pred[0]['masks'], 0)[1]

        frame = drawpred(pred[0], frame, colors)
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
