#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 02:27:36 2021
@author: Safoine EL KHABICH "TTM" <https://www.linkedin.com/in/safoinme/>
"""
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
import os, json, cv2, uuid

class PlateDetector:
    
    def __init__(self):
        #create a predictor
        self._cfg = get_cfg()
        self._predictor = self._makePredictor()
        self._class = MetadataCatalog.get("licences").set(thing_classes=["licence"])
    
    """
    This method initalizes the model and configuration 
    to return the predictor
    """
    def _makePredictor(self):
        self._cfg
        self._cfg.MODEL.DEVICE = "cpu"
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.SOLVER.IMS_PER_BATCH = 2
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self._cfg.MODEL.WEIGHTS = os.path.join("./weights/plate_detector", "model_final.pth")  # path to the model we just trained
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        return DefaultPredictor(self._cfg)
    
    """
    This method takes an opencv image and perfroms instance segmentation
    """
    def predict(self, image):
        return self._predictor(image)
    """
    This method takes the prediction result and return boxes and scores
    """
    def plateBoxes(self, output):
        boxes = output['instances'].pred_boxes.tensor.cpu().numpy().tolist() 
        scores = output['instances'].scores.numpy().tolist()
        if len(scores)>0: 
            Plates = { "Licence Plate "+str(i) : {"score" : scores[i], "boxes" : boxes[i]} for i in range(0, len(scores) ) }
        return (Plates)
    """
    This method takes the image & the prediction result and return image with boxes of plates
    """
    def detectedPlateSaver(self, image, output):
        visual = Visualizer(image[:, :, ::-1],
                   metadata=self._class, 
                   scale=0.5, 
                   
        )
        visual_output = visual.draw_instance_predictions(output["instances"])
        output_image = os.path.join("./images/plate_detector", str(uuid.uuid4())+".jpg")
        cv2.imwrite(output_image,visual_output.get_image()[:, :, ::-1])
        return (output_image)
