#!/usr/bin/env python3
# -*-coding: utf-8 -*-

import cv2
import time
import numpy as np
import torch
import torchvision

COCO_PERSON_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

def colors_for_labels():
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = [(i * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1]) % 255).tolist() for i in range(len(COCO_PERSON_KEYPOINT_NAMES))]
    #colors = np.array(range(len(COCO_INSTANCE_CATEGORY_NAMES))) * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    #colors = (colors % 255).numpy().astype("uint8")
    return colors


def drawpred(pred, frame, colors):
    #print(pred[0]['boxes'])
    #print(pred[0]['labels'])
    #print(pred[0]['scores'])
    for box, label, score, keypoint, keypoint_score in zip(pred[0]['boxes'], pred[0]['labels'], pred[0]['scores'], pred[0]['keypoints'], pred[0]['keypoints_scores']):
        if score > 0.9:
            frame = cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), tuple(colors[label]), 2)
            frame = cv2.putText(frame, 'person', tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            
            if keypoint_score[0] + keypoint_score[2] > 1.8: frame = cv2.line(frame, tuple(keypoint[0][:2]), tuple(keypoint[2][:2]), color=(125, 255, 125), thickness=1) 
            if keypoint_score[0] + keypoint_score[1] > 1.8: frame = cv2.line(frame, tuple(keypoint[0][:2]), tuple(keypoint[1][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[1] + keypoint_score[2] > 1.8: frame = cv2.line(frame, tuple(keypoint[1][:2]), tuple(keypoint[2][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[2] + keypoint_score[4] > 1.8: frame = cv2.line(frame, tuple(keypoint[2][:2]), tuple(keypoint[4][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[1] + keypoint_score[3] > 1.8: frame = cv2.line(frame, tuple(keypoint[1][:2]), tuple(keypoint[3][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[0] + keypoint_score[11] + keypoint_score[12] > 2.7: frame = cv2.line(frame, tuple(keypoint[0][:2]), tuple((keypoint[11][:2] + keypoint[12][:2])/2), color=(125, 255, 125), thickness=1)
            if keypoint_score[8] + keypoint_score[10] > 1.8: frame = cv2.line(frame, tuple(keypoint[8][:2]), tuple(keypoint[10][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[6] + keypoint_score[8] > 1.8: frame = cv2.line(frame, tuple(keypoint[6][:2]), tuple(keypoint[8][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[5] + keypoint_score[6] > 1.8: frame = cv2.line(frame, tuple(keypoint[5][:2]), tuple(keypoint[6][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[5] + keypoint_score[7] > 1.8: frame = cv2.line(frame, tuple(keypoint[5][:2]), tuple(keypoint[7][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[7] + keypoint_score[9] > 1.8: frame = cv2.line(frame, tuple(keypoint[7][:2]), tuple(keypoint[9][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[14] + keypoint_score[16] > 1.8: frame = cv2.line(frame, tuple(keypoint[14][:2]), tuple(keypoint[16][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[12] + keypoint_score[14] > 1.8: frame = cv2.line(frame, tuple(keypoint[12][:2]), tuple(keypoint[14][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[11] + keypoint_score[12] > 1.8: frame = cv2.line(frame, tuple(keypoint[11][:2]), tuple(keypoint[12][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[11] + keypoint_score[13] > 1.8: frame = cv2.line(frame, tuple(keypoint[11][:2]), tuple(keypoint[13][:2]), color=(125, 255, 125), thickness=1)
            if keypoint_score[13] + keypoint_score[15] > 1.8: frame = cv2.line(frame, tuple(keypoint[13][:2]), tuple(keypoint[15][:2]), color=(125, 255, 125), thickness=1)
            #idx = -1
            #for kp, kp_score in zip(keypoint, keypoint_score):
            #    idx += 1
            #    if kp_score > 0.9:
            #        frame = cv2.drawMarker(frame, tuple(kp[:2]), color=colors[idx], markerType=cv2.MARKER_DIAMOND, thickness=1)

    return frame


def main():
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
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
        #print(pred)

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
