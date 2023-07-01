import pdf2image
import numpy as np
import layoutparser as lp
import torch
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"})

def get_separator_y(pdf_file):
    
    img = np.asarray(pdf2image.convert_from_bytes(pdf_file)[0])
    img_height = img.shape[0]   
    layout_result = model.detect(img)
    separator_regions = []
    for element in layout_result:
        if element.type == "SeparatorRegion":
            separator_regions.append(element)
    if len(separator_regions) == 0:
        raise HTTPException(status_code=400, detail="No separator region found in the PDF.")
    separator_y = separator_regions[0].block.y_1
    return separator_y/ img_height

@app.get("/")
async def getHello():
    return {"message": "hello world"}

@app.post("/")
async def get_separator_y_endpoint(pdf_file: bytes = File(...)):
    print("got here")
    try:
        separator_y_ratio = get_separator_y(pdf_file)
        return {"y_ratio": separator_y_ratio}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))