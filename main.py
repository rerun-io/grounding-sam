#!/usr/bin/env python3
"""
Example of using Rerun to log and visualize the out of grounded dino + segment-anything.

See: [segment_anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

Can be used to test mask-generation on one or more images, as well as videos. Images can be local file-paths
or remote urls. Videos must be local file-paths. Can use multiple prompts.
"""


import argparse
import logging
import rerun as rr
import torch
import cv2
from pathlib import Path
from models import CONFIG_PATH, MODEL_URLS, get_downloaded_model_path
from models import load_grounding_model, create_sam, load_image, image_to_tensor
from models import get_grounding_output, run_segmentation, resize_img
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from groundingdino.models import GroundingDINO


def log_images_segmentation(args, model: GroundingDINO, predictor: Sam):
    for n, image_uri in enumerate(args.images):
        rr.set_time_sequence("image", n)
        image, image_tensor = load_image(image_uri)
        predictor.set_image(image)
        for prompt in args.prompts:
            # run grounding dino model
            logging.info(f"Running GroundedDINO with DETECTION PROMPT {prompt}.")
            boxes_filt, pred_phrases = get_grounding_output(
                model, image_tensor, prompt, 0.3, 0.25, device=args.device
            )
            # denormalize boxes (from [0, 1] to image size)
            H, W, _ = image.shape
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            run_segmentation(predictor, image, boxes_filt, prompt)


def log_video_segmentation(args, model: GroundingDINO, predictor: Sam):
    video_path = args.video_path
    assert video_path.exists()
    cap = cv2.VideoCapture(str(video_path))

    prompt = args.prompts

    idx = 0
    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break
        rr.set_time_sequence("frame", idx)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_img(rgb, 512)
        image_tensor = image_to_tensor(rgb)
        predictor.set_image(rgb)
        
        for prompt in args.prompts:    
            # run grounding dino model
            logging.info(f"Running GroundedDINO with DETECTION PROMPT {prompt}.")
            boxes_filt, pred_phrases = get_grounding_output(
                model, image_tensor, prompt, 0.3, 0.25, device=args.device
            )
            # denormalize boxes (from [0, 1] to image size)
            H, W, _ = rgb.shape
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            
            run_segmentation(predictor, rgb, boxes_filt, prompt)

        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IDEA Research Grounded Dino + SAM example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="store",
        default="vit_b",
        choices=MODEL_URLS.keys(),
        help="Which model to use."
        "(See: https://github.com/facebookresearch/segment-anything#model-checkpoints)",
    )
    parser.add_argument(
        "--device",
        action="store",
        default="cpu",
        help="Which torch device to use, e.g. cpu or cuda. "
        "(See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)",
    )

    parser.add_argument(
        "--prompts",
        default=["tires", "windows"],
        nargs="+",
        type=str,
        help="List of prompts to use for bounding box detection.",
    )

    parser.add_argument(
        "images", metavar="N", type=str, nargs="*", help="A list of images to process."
    )

    parser.add_argument(
        "--bbox-threshold",
        default=0.3,
        type=float,
        help="Threshold for a bounding box to be considered.",
    )

    parser.add_argument(
        "--video-path",
        default=None,
        type=Path,
        help="Path to video to run segmentation on",
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "grounded_sam")
    logging.getLogger().addHandler(rr.LoggingHandler("logs"))
    logging.getLogger().setLevel(logging.INFO)

    # load model
    grounded_checkpoint = get_downloaded_model_path("grounding")
    model = load_grounding_model(CONFIG_PATH, grounded_checkpoint, device=args.device)
    sam = create_sam(args.model, args.device)

    predictor = SamPredictor(sam)

    if len(args.images) == 0 and args.video_path is None:
        logging.info("No image provided. Using default.")
        args.images = [
            "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
        ]

    if len(args.images) > 0:
        log_images_segmentation(args, model, predictor)
    elif args.video_path is not None:
        log_video_segmentation(args, model, predictor)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
