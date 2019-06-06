#!/usr/bin/env python3
# -*-coding: utf-8 -*-

import cv2
import time
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

CATEGORY = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def colors_for_labels():
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = [(i * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1]) % 255).astype(np.uint8) for i in range(len(CATEGORY))]
    #colors = np.array(range(len(COCO_INSTANCE_CATEGORY_NAMES))) * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    #colors = (colors % 255).numpy().astype("uint8")
    return colors


def drawpred(pred, frame, colors):
    blend_rate = 0.5
    pred = pred.cpu().squeeze(dim=0).numpy()
    #cmap = cv2.applyColorMap((pred * 255 /21).astype(np.uint8), cv2.COLORMAP_JET)
    cmap = np.zeros_like(frame)
    for cat in range(len(CATEGORY)):
        cmap[np.where(pred == cat)] = colors[cat]
    frame = cv2.addWeighted(cmap, blend_rate, frame, blend_rate, 0)
    return frame


def main():
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
    model.eval().cuda()

    colors = colors_for_labels()

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, _ = frame.shape
        #frame = cv2.resize(frame,(int(width/2), int(height/2)))

        #frame_tensor = torch.from_numpy(frame.astype(np.float32).transpose(2, 0, 1)).cuda().unsqueeze(dim=0)/255.0
        frame_tensor = torch.from_numpy(frame.astype(np.float32).transpose(2, 0, 1)).cuda()/255.0
        frame_tensor = torchvision.transforms.functional.normalize(frame_tensor, torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])).unsqueeze(dim=0)
        #print(frame_tensor.shape)
        #frame_tensor = (frame_tensor - torch.Tensor([0.485, 0.456, 0.406]))/torch.Tensor(0.229, 0.224, 0.225)
        #print(frame_tensor)
        #print(torch.rand(3, 300, 400))

        pred = model(frame_tensor)
        #print(drawpred(pred))
        #print(torch.max(pred['out'], 1)[1].shape)
        pred_result = torch.max(pred['out'], 1)[1]

        frame = drawpred(pred_result, frame, colors)
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
