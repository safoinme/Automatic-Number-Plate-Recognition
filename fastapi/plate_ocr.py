#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:17:44 2021
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
import os, json, cv2, uuid , statistics

class PlateOCR:
    
    def __init__(self):
        #create a predictor
        self._cfg = get_cfg()
        self._predictor = self._makePredictor()
        self._characters = ["0","1","2","3","4","5","6","7","8","9", "a","b","h","w","d","p","waw","j","m","m"]
        self._class = MetadataCatalog.get("characters").set(thing_classes=self._characters)
    
    """
    This method initalizes the model and configuration 
    to return the predictor
    """
    def _makePredictor(self):
        self._cfg
        self._cfg.MODEL.DEVICE = "cpu"
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.SOLVER.IMS_PER_BATCH = 2
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
        self._cfg.MODEL.WEIGHTS = os.path.join("../weights/plate_ocr", "model_final.pth")  # path to the model we just trained
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        return DefaultPredictor(self._cfg)
    
    """
    This method takes an opencv image and perfroms instance segmentation
    """
    def predict(self, image):
        return self._predictor(image)
    """
    This method Loads plates images only from original images using plateExtractor to extract the exact plate
    """
    def plateLoader(self, image, plates):
        plateImages = []
        for plate in plates.values():
            boxes = plate['boxes']
            plateImages.append(image[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2]), :])
        return (plateImages)
    """
    This method takes the prediction result and return boxes and scores
    """
    def ocrExtractor(self, output):
        boxes = output['instances'].pred_boxes.tensor.cpu().numpy().tolist() 
        scores = output['instances'].scores.numpy().tolist()
        classes = output['instances'].pred_classes.to('cpu').tolist()
        if len(scores)>0:
            characters = { i : {"character": self._characters[classes[i]], "score" : scores[i], "boxes" : boxes[i]} for i in range(0, len(scores) ) }
        return (characters)
    """
    This method takes the image & the prediction result and return image with boxes of plates
    """
    def detectedImagesaver(self, image, output):
        visual = Visualizer(image[:, :, ::-1],
                   metadata=self._class, 
                   scale=0.5, 
                   
        )
        visual_output = visual.draw_instance_predictions(output["instances"])
        output_image = os.path.join("./images/plate_ocr", str(uuid.uuid4())+".jpg")
        cv2.imwrite(output_image,visual_output.get_image()[:, :, ::-1])
        return (output_image)
    """
    This method post-process the prediction output in order to return string of plates in the right order respect Moroccan standards
    """
    def postProcess(self, image, output):
        if len(output.keys())<=0:
            plate_ocr_string = {'plate':image[:-4],'plate_string':''}
        else :
            y_mins = []
            for character in list(output.items()):
                y_mins.append(character[1]['boxes'][1])
            median_y_mins = statistics.median(y_mins)
            top_characters =  dict()
            bottom_characters = dict()
            for key,value in output.items():
                if (value['boxes'][3] <= median_y_mins ):
                    top_characters[key] = value
                else :
                    bottom_characters[key] = value
            sorted_top_characters = sorted(top_characters.items(), key=lambda e: e[1]['boxes'][0])
            sorted_bottom_characters = sorted(bottom_characters.items(), key=lambda e: e[1]['boxes'][0])
            top_plate_ocr = [item[1]['character'] for item in sorted_top_characters]
            bottom_plate_ocr = [item[1]['character'] for item in sorted_bottom_characters]
            plate_ocr = bottom_plate_ocr+top_plate_ocr
            plate_ocr = "".join(str(x) for x in plate_ocr)
            plate_ocr_string = {'plate':image[:-4],'plate_string':plate_ocr}
        return(plate_ocr_string)
