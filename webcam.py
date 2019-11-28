#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import time 

def _main_():
    config_path  = "config_voc.json"

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)
    net_h, net_w = 64, 64 
    obj_thresh, nms_thresh = 0.5, 0.45

    #os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    #print(os.environ)
    infer_model = load_model(config['train']['saved_weights_name'])
    cap = cv2.VideoCapture(0)
    images = []

    while True:
        ret, image = cap.read()
        stime = time.time()
        if ret: 
            images += [image]
            batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

            for i in range(len(images)):
                images[i], bbox = draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                cv2.imshow('video with bboxes', images[i])
                try:
                    print(bbox)
                    print("detection var")
                except:
                    print("detection yok")
                    pass
                print('FPS {:.1f}'.format(1 / (time.time() - stime))) 
            images = []
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()        
        
if __name__ == '__main__':
    _main_()
