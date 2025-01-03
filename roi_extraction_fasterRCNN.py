import cv2
import os
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load the Faster R-CNN model from Detectron2's model zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for this model
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

predictor = DefaultPredictor(cfg)

# Define the path to input directory
input_path = "Data2/4. Bougainvillea"
output_folder = "annotated_outputs"
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input directory
for image_name in os.listdir(input_path):
    if not image_name.endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_path, image_name)
    image = cv2.imread(image_path)

    # Make predictions
    outputs = predictor(image)

    # Visualize the predictions
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the annotated image
    output_path = os.path.join(output_folder, f"annotated_{image_name}")
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

print(f"Annotated images saved to {output_folder}")