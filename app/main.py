from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi  import  UploadFile
from core.config import settings
from pathlib import Path
import uuid
import logging
from core.cv_services import image_classification, object_detection, semantic_segmentation, panoptic_segmentation, image_captioning
from logging.config import dictConfig
from core.config import LogConfig

def get_application():
    # generate folder for tmp data.
    Path("tmp").mkdir(parents=False, exist_ok=True)
    dictConfig(LogConfig().dict())
    logger = logging.getLogger("client")

    _app = FastAPI(title=settings.PROJECT_NAME)

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app, logger


app, logger = get_application()


def save_img(file):
    # save file in temp folder. 
    id = str(uuid.uuid4())
    path_to_img = f"tmp/{id}.{file.filename.split('.')[-1]}"
    with open(path_to_img, "wb") as buffer:
        # save file to filesystem
        buffer.write(file.file.read())

    # log the file storage.
    logger.info("File saved to %s", path_to_img)

    return id


# rest service for image classification
@app.post("/image-classification")
async def img_cla(file: UploadFile):

    id = save_img(file)

    # call service.
    logger.info("Starting Image Classification")
    result = image_classification(f"tmp/{id}.{file.filename.split('.')[-1]}")

    logger.info("Returning the results.")

    return {"result": result}

# rest service for object detection
@app.post("/object-detection")
async def obj_det(file: UploadFile):

    id = save_img(file)
    pass

# rest service for semantic segmentation.
@app.post("/semantic-segmentation")
async def sem_seg(file: UploadFile):

    id = save_img(file)
    pass

# rest service for panoptic segmentation.
@app.post("/pan-segmentation")
async def pan_seg(file: UploadFile):

    id = save_img(file)
    pass

# rest service for keypoint detection.
@app.post("/keypoint-detection")
async def key_det(file: UploadFile):

    id = save_img(file)
    pass

# rest service for image captioning.
@app.post("/image-captioning")
async def img_cap(file: UploadFile):
    id = save_img(file)

    # call service.
    pred = image_captioning(f"tmp/{id}.jpg")
    return pred

