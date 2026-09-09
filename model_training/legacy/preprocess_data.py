import util
from network_config import SqueezeNetConfig
import os
import json

if __name__ == "__main__":
    config = SqueezeNetConfig()

    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()

    # train
    util.Preprocessor(
        image_directories = secrets['PreProcessData']['crowdHumanImgTrain'], 
        annotations_path = secrets['PreProcessData']['crowdHumanAnnoTrain'], 
        processed_image_dir = secrets['PostProcessData']['imagesTrain'], 
        processed_annotations_dir = secrets['PostProcessData']['annosTrain'],
        target_size = config.IMG_SIZE
    ).preprocess_images()
    
    # val
    util.Preprocessor(
        image_directories = secrets['PreProcessData']['crowdHumanImgVal'], 
        annotations_path = secrets['PreProcessData']['crowdHumanAnnoVal'], 
        processed_image_dir = secrets['PostProcessData']['imagesVal'], 
        processed_annotations_dir = secrets['PostProcessData']['annosVal'],        
        target_size = config.IMG_SIZE
    ).preprocess_images()