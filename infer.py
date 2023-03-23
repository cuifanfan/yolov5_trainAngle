import os
import sys
from pathlib import Path
import cv2
# from PIL import Image, ImageDraw
import torch
import math
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.datasets import letterbox
from utils.plots import Colors
# import numpy as np
# import pdb

CLASS_LABEL = ['corn']
colors = Colors()


class Detector(object):
    def __init__(self, weight_path="weights/best.pt",
                 device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.2,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 input_size=(640, 640)  # resize of input image.
                 ):
        super(Detector, self).__init__()
        self.weight_path = weight_path
        assert self.weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.input_size = input_size
        # Load model
        self.device = select_device(device)
        self.model = attempt_load(self.weight_path, map_location=self.device)  # load FP32 model
        self.model.eval()

    def get_result(self, img):
        """
        :param img:  ndarray, shape is 'h,w,c'
        :return:     list , [ {'position':ndarray:[x1,y1,x2,y2,p],'label':str} ,
                              {...},
                              ....
                            ]
                    or None.
        """
        pointsList = []
        im0 = img
        img = letterbox(img, self.input_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = torch.Tensor(img.copy())
        # Inference
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.float().to(self.device)
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        if det.shape[0] > 0:
            det = det.to("cpu").numpy()
            for e in det:
                w = int(e[2]-e[0])
                h = int(e[3]-e[1])
                x_center = int(e[0]+w)
                y_center = int(e[1]+1)
                x1 = int(e[2]+1)
                y1 = int(e[3])
                x2 = int(e[0])
                y2 = int(e[1])
                pointsList.append([x1,y1])
                pointsList.append([x2,y2])
                pointsList.append([x_center,y_center])
        return pointsList

if __name__ == '__main__':
    path = 'C://Users/simon/Desktop/data/images/(142).jpg'
    model = Detector()

    def gradient(pt1, pt2):
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    def getAngle(pointsList):
        pt1, pt2, pt3 = pointsList[-3:]
        m1 = gradient(pt1, pt2)
        m2 = gradient(pt1, pt3)
        angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
        angD = math.degrees(angR)

        cv2.putText(img, str(angD), (pt2[0] - 40, pt2[1] - 20), cv2.FONT_HERSHEY_COMPLEX,
                    5, (255, 0, 0), 5)

    img = cv2.imread(path)
    result = model.get_result(img)
    print('回归得到的关键点坐标：', result)

    cv2.circle(img, (int(result[0][0]), int(result[0][1])), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (int(result[1][0]), int(result[1][1])), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (int(result[2][0]), int(result[2][1])), 5, (0, 0, 255), cv2.FILLED)
    cv2.line(img, (int(result[0][0]),int(result[0][1])), (int(result[1][0]),int(result[1][1])), (0, 0, 255), 2)
    cv2.line(img, (int(result[0][0]), int(result[0][1])), (int(result[2][0]), int(result[2][1])), (0, 0, 255), 2)

    while True:
        getAngle(result)
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img',(800,800))
        cv2.imshow('img', img)
        cv2.waitKey(0)