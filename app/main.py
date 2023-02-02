import logging
import uuid
from logging.config import dictConfig
from pathlib import Path

from core.config import LogConfig, settings
from core.cv_services import (
    image_captioning,
    image_classification,
    object_detection,
    panoptic_segmentation,
    semantic_segmentation,
)
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


def get_application():
    # generate folder for tmp data.
    Path("tmp").mkdir(parents=False, exist_ok=True)
    # generate  folder for saving data.
    Path("data").mkdir(parents=False, exist_ok=True)
    # generate folder for the models, it is not necessary for use with docker but allows for faster development.
    Path("models").mkdir(parents=False, exist_ok=True)

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

    # call service.
    logger.info("Starting Object Detection")
    result = object_detection(f"tmp/{id}.{file.filename.split('.')[-1]}", f"data/{id}.jpg")

    logger.info("Returning the results.")

    return FileResponse(result)


# rest service for semantic segmentation.
@app.post("/semantic-segmentation")
async def sem_seg(file: UploadFile):

    id = save_img(file)

    # call service.
    logger.info("Starting Semantic Segmentation")
    result = semantic_segmentation(f"tmp/{id}.{file.filename.split('.')[-1]}", f"data/{id}.jpg")

    logger.info("Returning the results.")

    return FileResponse(result)


# rest service for panoptic segmentation.
@app.post("/pan-segmentation")
async def pan_seg(file: UploadFile):

    id = save_img(file)

    # call service.
    logger.info("Starting Panoptic Segmentation")
    result = panoptic_segmentation(f"tmp/{id}.{file.filename.split('.')[-1]}", f"data/{id}.jpg")

    logger.info("Returning the results.")
    return FileResponse(result)


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
    pred = image_captioning(f"tmp/{id}.{file.filename.split('.')[-1]}")
    return pred


# rest service for document analysis.
@app.post("/document-analysis")
async def doc_ana(file: UploadFile):
    id = save_img(file)
    pass
