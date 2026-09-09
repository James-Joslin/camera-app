import pandas as pd
import numpy as np

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import json
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import json
import os
import cv2

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Data prep
class Preprocessor:
    def __init__(self, image_directories, annotations_path, processed_image_dir, processed_annotations_dir, target_size):
        self.image_directories = image_directories
        self.annotations_path = annotations_path
        self.processed_image_dir = processed_image_dir 
        self.processed_annotations_dir = processed_annotations_dir
        self.target_width, self.target_height = target_size

        if not os.path.exists(self.processed_annotations_dir):
            os.makedirs(self.processed_annotations_dir)

        if not os.path.exists(self.processed_image_dir):
            os.makedirs(self.processed_image_dir)

    def read_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                annotations.append(data)
        return annotations

    def preprocess_images(self):
        annotations = self.read_annotations(self.annotations_path)
        all_fboxes = []

        for annotation in tqdm(annotations):
            image_id = annotation['ID']
            image_file = self.find_image(image_id)
            if image_file:
                image = cv2.imread(image_file)
                processed_image, adjusted_annotations = self.transform(image, annotation)
                # print(processed_image.shape)
                # self.plot_image_with_boxes(processed_image, adjusted_annotations, image_id, './src/python/temp/')
                # print(adjusted_annotations)
                if self.has_valid_contents(adjusted_annotations):
                    all_fboxes.append({
                        'ID': image_id, 
                        'persons': [box['fbox'] for box in adjusted_annotations['persons']],
                        'masks': [box['fbox'] for box in adjusted_annotations['masks']]
                    })
                    self.save_image(image_id, processed_image)
                else:
                    pass

        # Save all fboxes to a new JSON file
        self.save_fboxes_json(all_fboxes, "annotations.json")

    def find_image(self, image_id):
        for directory in self.image_directories:
            for filename in os.listdir(directory):
                if filename.startswith(image_id):
                    return os.path.join(directory, filename)
        return None

    def transform(self, image, annotation):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width, _ = image.shape

        # Resize the image while maintaining aspect ratio
        scale = min(self.target_width / orig_width, self.target_height / orig_height)
        new_width, new_height = int(orig_width * scale), int(orig_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calculate padding to maintain aspect ratio
        pad_x = (self.target_width - new_width) // 2
        pad_y = (self.target_height - new_height) // 2

        # Pad the resized image to the target dimensions
        processed_image = cv2.copyMakeBorder(resized_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if processed_image.shape[1] == self.target_width - 1:
            # Add one column of black pixels to the right side of the image
            processed_image = cv2.copyMakeBorder(processed_image, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Check if the height is off by one pixel
        if processed_image.shape[0] == self.target_height - 1:
            # Add one row of black pixels to the bottom of the image
            processed_image = cv2.copyMakeBorder(processed_image, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        person_boxes = [self.adjust_bbox(box['fbox'], scale, pad_x, pad_y) for box in annotation['gtboxes'] if box['tag'] == 'person']
        mask_boxes = [self.adjust_bbox(box['fbox'], scale, pad_x, pad_y) for box in annotation['gtboxes'] if box['tag'] == 'mask']

        # Update the annotations with adjusted bounding boxes
        adjusted_annotations = {
            'persons': [{'fbox': box} for box in person_boxes],
            'masks': [{'fbox': box} for box in mask_boxes]
        }

        return processed_image, adjusted_annotations

    def adjust_bbox(self, bbox, scale, pad_x, pad_y):
        x, y, w, h = bbox
        x = int(x * scale + pad_x)
        y = int(y * scale + pad_y)
        w = int(w * scale)
        h = int(h * scale)
        return [x, y, w, h]

    def plot_image_with_boxes(self, image, adjusted_annotations, image_id, save_dir):
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Loop through each bounding box and add a rectangle to the plot
        for bbox in adjusted_annotations['persons']:
            fbox = bbox['fbox']
            rect = patches.Rectangle((fbox[0], fbox[1]), fbox[2], fbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Plot 'mask' bounding boxes
        for bbox in adjusted_annotations['masks']:
            fbox = bbox['fbox']
            rect = patches.Rectangle((fbox[0], fbox[1]), fbox[2], fbox[3], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        # Define the path for the plot
        plot_path = os.path.join(save_dir, f"{image_id}.png")

        # Save the figure
        plt.title(f"Image ID: {image_id}")
        plt.axis('off')  # Optional: turn off the axis for a cleaner image
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

    def save_fboxes_json(self, all_fboxes, file_name):
        path = os.path.join(self.processed_annotations_dir, file_name)
        with open(f'{path}', 'w') as file:
            json.dump(all_fboxes, file, indent=4)
        file.close()

    def save_image(self, image_id, processed_image):
        # Convert RGB back to BGR for saving
        processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        # Use the image ID from the annotations as the filename
        new_filename = f"{image_id}.jpg"  # You can add a prefix or suffix if desired
        new_path = os.path.join(self.processed_image_dir, new_filename)

        # Save the processed image
        cv2.imwrite(new_path, processed_image_bgr)

    def has_valid_contents(self, data):
        # Check if 'persons' key exists and has bounding boxes
        persons_exist = 'persons' in data and len(data['persons']) > 0
        
        # Check if 'masks' key exists and has bounding boxes
        masks_exist = 'masks' in data and len(data['masks']) > 0

        # Return True if either 'persons' or 'masks' have bounding boxes
        return persons_exist or masks_exist

# Dataloading
# split not needed, data already loaded into train and val dirs
def split_data(train_size:float, val_size:float, test_size:float, x_data:np.array, y_data):
    
    assert 1 - (train_size + val_size + test_size) < 1e-6
    assert len(x_data) == len(y_data)
    
    train_end_idx = int(train_size * len(x_data))
    val_end_idx = int((train_size + val_size) * len(y_data))

    train_images = x_data[:train_end_idx]
    train_gt = y_data[:train_end_idx]

    val_images = x_data[train_end_idx:val_end_idx]
    val_gt = y_data[train_end_idx:val_end_idx]

    test_images = x_data[val_end_idx:]
    test_gt = y_data[val_end_idx:]
    
    return train_images, train_gt, val_images, val_gt, test_images, test_gt

# loading to data and ground truth tensors
class DatasetBuilder(Dataset):
    def __init__(self, data: str, annotations: str):
        super(DatasetBuilder, self).__init__()
        self.data = data
        with open(os.path.join(annotations.strip(), "annotations.json"), 'r') as file:
            self.ground_truth = json.load(file)
        file.close()
    
    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        line = self.ground_truth[idx]
        id = line['ID']
        image = self.load_image(id)
        # Ensure your annotations are formatted as a tensor or a dictionary that can be processed later
        ground_truth = {'persons': line['persons'], 'masks': line['masks']}
        return image, ground_truth

    def load_image(self, id: str):
        image_dir = os.path.join(self.data, f'{id}.jpg')
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image/255)
        image = np.transpose(image, (2,0,1))
        return torch.tensor(image)

def collate_fn(batch):
    images = []
    annotations = []

    for img, annot in batch:
        images.append(img)  # Add image tensor to list

        # Convert annotations to tensor and add to list
        persons = torch.tensor(annot['persons'], dtype=torch.float32)
        masks = torch.tensor(annot['masks'], dtype=torch.float32)
        annotations.append({'persons': persons, 'masks': masks})

    # Stack all image tensors together
    images = torch.stack(images)

    return images, annotations

# Output and export
def export_to_onnx(onnx_base_path, onnx_name, model:nn.Module, checkpoint, input_size):
    # write brackets model
    model.load_state_dict(
        checkpoint['model_state_dict']
    )
    print(model)
    x = torch.randn(1, 1, input_size, requires_grad=False)
    torch_out = model(x)
    torch.onnx.export(model,                   # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f'{onnx_base_path}{onnx_name}.onnx',   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    print(f'File saved to: {onnx_base_path}{onnx_name}')

# System checks
def check_gpu():
    if torch.cuda.is_available():
        f"GPU is available. Detected {torch.cuda.device_count()} GPU(s)."
        return "cuda"
    else:
        "GPU is not available."
        return "cpu"
    
def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  # For consistent results on the GPU
    torch.backends.cudnn.benchmark = False  # Faster convolutions, but might introduce randomness
    
# Evaluate Functions
def model_summary(model, input_size):
    print("----------------------------------------------------------------")
    print(f"Input Shape:               {str(input_size).ljust(25)}")
    print("----------------------------------------------------------------")
    print("Layer (type)               Output Shape         Param #")
    print("================================================================")
    
    total_params = 0

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params

            # Initialize output_shape variable
            output_shape = "Undefined"

            # Check if output is a tensor
            if torch.is_tensor(output):
                output_shape = str(list(output.shape))

            # Check if output is a tuple of tensors
            elif isinstance(output, tuple):
                output_shape = [str(list(o.shape)) if torch.is_tensor(o) else str(type(o)) for o in output]
                output_shape = output_shape[0]  # Pick first size if there are multiple identical sizes in the tuple

            # Check if output is a list of tensors
            elif isinstance(output, list):
                output_shape = [str(list(o.shape)) for o in output if torch.is_tensor(o)]
                output_shape = ', '.join(output_shape)  # Combine all shapes into a single string

            # Handle other types of output (if any)
            else:
                output_shape = str(type(output))

            # Print layer details
            if len(list(module.named_children())) == 0:  # Only print leaf nodes
                print(f"{module.__class__.__name__.ljust(25)}  {output_shape.ljust(25)} {f'{num_params:,}'}")

        # Register hook if the module is not a container of other modules
        if not isinstance(module, nn.Sequential) and \
        not isinstance(module, nn.ModuleList) and \
        not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)

    print("----------------------------------------------------------------")
    DEVICE = next(model.parameters()).device
    output = model(torch.randn(1, *input_size).to(DEVICE))

    for h in hooks:
        h.remove()

    output_shape = str(list(output.shape)) if torch.is_tensor(output) else str(type(output))
    print("----------------------------------------------------------------")
    print(f"Total params: {total_params:,}")
    print(f"Output Shape: {output_shape.ljust(25)}")
    print(f'Model on: {next(model.parameters()).device}')
    print("----------------------------------------------------------------")

def save_checkpoint(current_epoch, model, optimiser, loss, checkpoint_file='./'):
    torch.save({
        'Epoch' : current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss,
    }, checkpoint_file)
    print(f'Saving model at epoch {current_epoch}')
    
def load_checkpoint(directory):
    if os.path.isfile(directory):
        print("Loading Model Checkpoint")
        checkpoint = torch.load(directory)
        return checkpoint
    else:
        pass

# Model performance logs
class logger(object):
    """docstring for logger."""
    def __init__(self, log_dir):
        super(logger, self).__init__()
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        print(f'Logging run results to {log_dir}')

    def log_hyperparameters(self, parameter_names_values):
        hp_table = [f"| {name} | {value} |" for name, value in parameter_names_values]
        table_str = "| Parameter | Value |\n|-----------|-------|\n" + "\n".join(hp_table)
        self.writer.add_text("Hyperparameters", table_str, 0)

    def write_to_tensorboard(self, epoch, loss):
        """
        Writes metrics to TensorBoard.
        
        Parameters:
            writer (SummaryWriter): TensorBoard SummaryWriter object.
            epoch (int): Current epoch number.
            ep_reward (float): Total reward obtained in the episode.
            epsilon (float): Current epsilon value for epsilon-greedy action selection.
            loss (float): Loss value.
        """
        self.writer.add_scalar('Epoch Average Loss', loss, epoch)
        # self.writer.add_scalar('Epsilon', epsilon, epoch)
        # if loss is not None:
        #     self.writer.add_scalar('Epoch Cumulative Loss', loss, epoch)