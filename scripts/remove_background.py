"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import cv2
from utils import utils, detection
from os import makedirs
from os.path import join, isdir, split, dirname


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    #   2. Load the image
    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task
    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.
    # TODO
    for file in utils.walkdir(data_folder):
      # Load image and correct its dimmension
      image = cv2.imread(join(*file))
      box_coordinates = detection.get_vehicle_coordinates(image)
      crop_img = image[box_coordinates[1]:box_coordinates[3], box_coordinates[0]:box_coordinates[2]]
      output_path = join(output_data_folder, file[0].split(sep=data_folder)[1], file[1])
      # cerate folder if it does not alreadt exist
      if not isdir(dirname(output_path)):
        makedirs(dirname(output_path))
      # save cropped image
      cv2.imwrite(output_path, crop_img)
      print(output_path)





if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
