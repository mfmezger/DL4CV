from transformers import (
    AutoFeatureExtractor,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    AutoModelForImageClassification,
    DetrFeatureExtractor,
    DetrForObjectDetection,
    DetrForSegmentation,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
)
import cv2
import io
import os, io, os 
import torch
import numpy as np
from PIL import Image
import logging
from logging.config import dictConfig
from core.config import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("client")
cache_dir = "models"


def draw_on_image(results, img, model, score_confidence=0.9, debugging=False):
    """Draws the bounding boxes on the image

    # future: check if speedup is possible / need for profiling.

    :param results: the results from the model
    :type results: dict of tensors
    :param img: the image to draw on
    :type img: cv2.Image
    :param model: the model
    :type model: model
    :param score_confidence: The confidence score that the model should reach to display the predictions, defaults to 0.9
    :type score_confidence: float, optional
    :param debugging: Debugging Mode with more prints, defaults to False
    :type debugging: bool, optional
    :return: the image with the bounding boxes drawn on it, the detection class, and the probability
    :rtype: cv2.Image, list, list
    """
    color = [255, 255, 255]
    # save detection and time stamp
    detection_class = []
    prob = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9
        if score > score_confidence:
            if debugging:
                print(f"Detected {model.config.id2label[label.item()]} with confidence " f"{round(score.item(), 3)} at location {box}")
            # draw bouding box on img.
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                img,
                model.config.id2label[label.item()],
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                color,
                2,
                lineType=cv2.LINE_4,
            )
            detection_class.append(model.config.id2label[label.item()])
            prob.append(round(score.item(), 3))

    return img, detection_class, prob


def image_classification(path_to_img):

    image = Image.open(path_to_img)
    image = image.convert("RGB")

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50", cache_dir=cache_dir)

    blubb = 1



    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50", cache_dir=cache_dir)

    inputs = extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


def object_detection(path_to_image, path_to_save):
    """Run a forward pass of the model on the image

    :param path_to_image: Parth to the image in the temporary folder
    :type path_to_image: str
    """
    image = Image.open(path_to_image)
    image = image.convert("RGB")
    logger.debug("Initializing Models")
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50", cache_dir=cache_dir)
    logger.debug("Initializing Models")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir=cache_dir)

    logger.debug("Models are initialized")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logger.debug("Forward Passcompleted")

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    img = cv2.imread(path_to_image)

    logger.debug("Drawing Bounding Boxes")
    # draw on image.
    img, detection_class, prob = draw_on_image(results, img, model)
    logger.info("Object Detection complete.")
    img = Image.fromarray(img)

    # save image
    img.save(path_to_save)
    logger.debug("Returning Image")
    return path_to_save


def semantic_segmentation(path_to_image, path_to_save):
    """https://huggingface.co/facebook/detr-resnet-50-panoptic"""
    logger.debug("Initialize Model")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    logger.debug("Initialize Model")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    # load the image. The image should be in RGB format and will be converted.
    image = Image.open(path_to_image)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    logger.debug("Size of the Image: %s", image.size)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # argmax the second dimension.
    predicted_class = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

    # upsample the image to the original size with nearest neighbor interpolation.
    # create random colors for the classes.
    colors = np.random.randint(0, 256, size=(predicted_class.max() + 1, 3), dtype="uint8")

    color_map = np.array(colors)
    color_image = Image.fromarray(color_map[predicted_class])
    color_image = color_image.resize(image.size, Image.NEAREST)

    logger.debug(list(logits.shape))

    # save image
    color_image.save(path_to_save)
    logger.debug("Returning Image")
    return path_to_save


def panoptic_segmentation(path_to_image, path_to_save):
    """https://huggingface.co/facebook/detr-resnet-50-panoptic"""
    image = Image.open(path_to_image)
    image = image.convert("RGB")

    logger.debug("Initialize  the models")
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", cache_dir=cache_dir)

    logger.debug("Initialize  the models")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic", cache_dir=cache_dir)

    logger.debug("Start forward pass")
    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)

    logger.debug("Start postprocessing")

    # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
    processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic_segmentation(outputs, processed_sizes)[0]

    # the segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
    # save image
    panoptic_seg.save(path_to_save)

    logger.debug("Returning Image")

    return path_to_save


def document_recognition(path_to_image):
    pass


def keypoint_detection(path_to_image):
    pass


def image_captioning(path_to_image):

    # load the model
    logger.debug("Start Initalizing Models.")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
    logger.debug("Start Initalizing Models.")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
    logger.debug("Start Initalizing Models.")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Device: %s", device)
    model.to(device)

    # maximum length of the return. The model will stop generating if it reaches this length.
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # load the image. The image should be in RGB format and will be converted.
    image = Image.open(path_to_image)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    logger.debug("Image converted!")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    logger.debug("Returning Caption")

    return preds


def main():
    panoptic_segmentation("/Users/marc/Downloads/jeff-griffith-ZqYPM8i60F8-unsplash.jpeg")


if __name__ == "__main__":
    main()
