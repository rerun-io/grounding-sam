import logging
import os
from pathlib import Path
from typing import Final, Tuple
from urllib.parse import urlparse

import cv2
from PIL import Image
import numpy as np
import requests
import rerun as rr
from rerun.components.rect2d import RectFormat
import torch
import torchvision
from cv2 import Mat
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from tqdm import tqdm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from groundingdino.models import GroundingDINO


CONFIG_PATH: Final = (
    Path(os.path.dirname(__file__))
    / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
MODEL_DIR: Final = Path(os.path.dirname(__file__)) / "model"
MODEL_URLS: Final = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "grounding": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
}


def download_with_progress(url: str, dest: Path) -> None:
    """Download file with tqdm progress bar."""
    chunk_size = 1024 * 1024
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as dest_file:
        with tqdm(
            desc="Downloading model",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for data in resp.iter_content(chunk_size):
                dest_file.write(data)
                progress.update(len(data))


def get_downloaded_model_path(model_name: str) -> Path:
    """Fetch the segment-anything model to a local cache directory."""
    model_url = MODEL_URLS[model_name]

    model_location = MODEL_DIR / model_url.split("/")[-1]
    if not model_location.exists():
        os.makedirs(MODEL_DIR, exist_ok=True)
        download_with_progress(model_url, model_location)

    return model_location


def create_sam(model: str, device: str) -> Sam:
    """Load the segment-anything model, fetching the model-file as necessary."""
    model_path = get_downloaded_model_path(model)

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("Torchvision version: {}".format(torchvision.__version__))
    logging.info("CUDA is available: {}".format(torch.cuda.is_available()))

    logging.info("Building sam from: {}".format(model_path))
    sam = sam_model_registry[model](checkpoint=model_path)
    return sam.to(device=device)


def run_segmentation(
    predictor: SamPredictor, image: Mat, boxes_filt, prompt: str
) -> None:
    """Run segmentation on a single image."""
    rr.log_image("image", image)
    if boxes_filt.shape[0] == 0:
        return
    logging.info("Finding masks")
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    )

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(predictor.device),
        multimask_output=False,
    )

    logging.info("Found {} masks".format(len(masks)))
    # mask_tensor = masks.squeeze().numpy().astype("uint8") * 128
    # rr.log_tensor(f"query_{idx}/mask_tensor", mask_tensor)

    # TODO(jleibs): we could instead draw each mask as a separate image layer, but the current layer-stacking
    # does not produce great results.
    masks_with_ids = list(enumerate(masks.cpu(), start=1))

    # Work-around for https://github.com/rerun-io/rerun/issues/1782
    # Make sure we have an AnnotationInfo present for every class-id used in this image
    # TODO(jleibs): Remove when fix is released
    rr.log_annotation_context(
        "image",
        [rr.AnnotationInfo(id) for id, _ in masks_with_ids],
        timeless=False,
    )

    # Layer all of the masks together, using the id as class-id in the segmentation
    segmentation_img = np.zeros((image.shape[0], image.shape[1]))
    for id, m in masks_with_ids:
        segmentation_img[m.squeeze()] = id

    rr.log_segmentation_image(f"image/{prompt}/masks", segmentation_img)

    rr.log_rects(
        f"image/{prompt}/boxes",
        rects=boxes_filt.numpy(),
        class_ids=[id for id, _ in masks_with_ids],
        rect_format=RectFormat.XYXY,
    )


def is_url(path: str) -> bool:
    """Check if a path is a url or a local file."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def resize_img(img: Mat, max_dimension: int = 512) -> Mat:
    height, width = img.shape[:2]
    # Check if either dimension is larger than the maximum
    if max(height, width) > max_dimension:
        # Calculate the new dimensions while maintaining the aspect ratio
        if height > width:
            new_height = max_dimension
            new_width = int((new_height * width) / height)
        else:
            new_width = max_dimension
            new_height = int((new_width * height) / width)

        # Resize the image
        resized_image = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    return resized_image


def image_to_tensor(image: Mat) -> torch.Tensor:
    """
    Assumes a RGB OpenCV image, this is required for the DINO model
    """
    image_pil = Image.fromarray(image)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # 3, h, w
    return image_tensor


def load_image(image_uri: str) -> Tuple[Mat, torch.Tensor]:
    """Conditionally download an image from URL or load it from disk."""
    logging.info("Loading: {}".format(image_uri))
    if is_url(image_uri):
        response = requests.get(image_uri)
        response.raise_for_status()
        image_data = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_uri, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = image_to_tensor(image)
    return image, image_tensor


def load_grounding_model(
    model_config_path: Path, model_checkpoint_path: Path, device: str
) -> GroundingDINO:
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(
    model: GroundingDINO,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    with_logits: bool = True,
    device: str = "cpu",
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )

        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
