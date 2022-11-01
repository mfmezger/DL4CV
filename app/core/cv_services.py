from transformers import AutoFeatureExtractor, AutoModelForImageClassification, DetrFeatureExtractor, DetrForObjectDetection, AutoModelForImageSegmentation
import cv2
import torch
from PIL import Image


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

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")

    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


def object_detection(path_to_image):
    """Run a forward pass of the model on the image

    :param path_to_image: Parth to the image in the temporary folder
    :type path_to_image: str
    """
    image = Image.open(path_to_image)
    image = image.convert("RGB")

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
    img = cv2.imread(path_to_image)

    # draw on image.
    img, detection_class, prob = draw_on_image(results, img, model)

    return img


def semantic_segmentation(path_to_image):
    pass


def panoptic_segmentation(path_to_image):
    """https://huggingface.co/facebook/detr-resnet-50-panoptic"""
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")

    model = AutoModelForImageSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)

    # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
    processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]


def document_recognition(path_to_image):
    pass


def keypoint_detection(path_to_image):
    pass


def image_captioning(path_to_image):
    # load the model
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # maximum length of the return. The model will stop generating if it reaches this length.
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # load the image. The image should be in RGB format and will be converted.
    image = Image.open(path_to_image)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds
