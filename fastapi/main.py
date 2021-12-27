from fastapi import FastAPI ,  File, UploadFile
from fastapi.responses import ORJSONResponse
from plate_detector import PlateDetector
from plate_ocr import PlateOCR
import io
import os, json, cv2, uuid , statistics
from PIL import Image
import numpy as np
import base64
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
#model = get_segmentator()
plate_model = PlateDetector()
ocr_model = PlateOCR()

app = FastAPI(title="MoroccoAI Data Challenge",
              description='''Automatic Number Plate Recognition (ANPR) in Morocco Licensed Vehicles.''',
              version="0.1.0",
              )

#app.mount("/images/plate_detector", StaticFiles(directory="images",html = True), name="images")

@app.post("/platedetector", response_class=ORJSONResponse)
async def get_plate_detection(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = image[:,:,::-1].copy()
    output = plate_model.predict(image)
    plate_boxes = plate_model.plateBoxes(output)
    output_image = plate_model.detectedPlateSaver(image, output)
    return FileResponse(output_image)

"""
@app.post("/plateocr", response_class=ORJSONResponse)
def get_plate_ocr(file: bytes = File(...)):
    image = cv2.imread(io.BytesIO(file))
    image = image[:,:,::-1].copy()
    plate_boxes = get_plate_detection(file)
    output = plate_model.predict(image)
    plate_boxes = plate_model.plateBoxes(output)
    output_image = plate_model.detectedPlateSaver(image, output)
    return [{"plate_boxes": plate_boxes, "output_image": output_image}]
"""
