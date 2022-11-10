from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because it has a good balance between accuracy and speed.
# User guide: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.

# Assign the loaded detection model to global variable DET_MODEL
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
try:
    # It may fail if no GPU was found
    DET_MODEL = DefaultPredictor(cfg)
except:
    # Load the model for CPU only
    print(
        f"Failed to load Detection model on GPU, "
        "trying with CPU. Message: {exp}."
    )
    cfg.MODEL.DEVICE='cpu'
    DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector (loaded in DET_MODEL model
    variable) over the the image, get the vehicle position in the picture
    and return it.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : list
        List having bounding box coordinates as [left, top, right, bottom].
        Also known as [x1, y1, x2, y2].
    """
      
    # COC Classnames https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda

    # Set default corrdinates to cover the full image
    h, w = img.shape[:2]
    box_coordinates =  [0, 0, w, h]

    # Get instances detected on img using our model
    outputs = DET_MODEL(img)
    instances = outputs["instances"]
    
    # Filter instances corresponding to cars and truckss
    vehicle_instances = instances[(instances.pred_classes == 2) | (instances.pred_classes == 7)]
    if len(vehicle_instances):
      # If there is at least one car or trcuk, get the coordinates for the biggest Box
      biggest_idx_box = vehicle_instances.pred_boxes.area().argmax().item()
      box_coordinates = vehicle_instances.pred_boxes[biggest_idx_box].tensor[0].numpy()   

    return box_coordinates

