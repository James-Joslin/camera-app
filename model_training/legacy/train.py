import json
import util

import torchvision
from torch.utils.data import DataLoader

from network_config import SqueezeNetConfig

from squeezenet_architecture import ObjectDetector, generate_anchors, apply_deltas_to_anchors

import torch
from torch.optim import Adam

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

if __name__ == "__main__":
    # Config
    squeezeNet_config = SqueezeNetConfig()
    
    # Seed
    util.set_seed(squeezeNet_config.SEED)
    
    # Secrets - file directories
    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    
    # Load training data
    train_builder = util.DatasetBuilder(
        secrets["PostProcessData"]["imagesTrain"],
        secrets["PostProcessData"]["annosTrain"]    
    )
    train_loader = DataLoader(train_builder, batch_size=2, shuffle=True, num_workers=2, collate_fn=util.collate_fn)
    
    # Load val data
    val_builder = util.DatasetBuilder(
        secrets["PostProcessData"]["imagesVal"],
        secrets["PostProcessData"]["annosVal"]   
    )
    val_loader = DataLoader(val_builder, batch_size=2, shuffle=True, num_workers=2, collate_fn=util.collate_fn)

    # model
    model = ObjectDetector().to(squeezeNet_config.DEVICE)
    optimiser = Adam(model.parameters(), lr=squeezeNet_config.LR, weight_decay=squeezeNet_config.WEIGHT_DECAY)
    # util.model_summary(model, (3, 360, 640))
    
    # training loop
    for epoch in range(squeezeNet_config.EPOCHS):
        for images, batch_annotations in train_loader:
            images = images.to(squeezeNet_config.DEVICE)  # Move images to the device
            # print(images.shape)
            
            # Initialize a new list to hold dictionaries with tensors moved to the device
            device_annotations = []

            # Iterate through each annotations dictionary in the batch
            for annot in batch_annotations:
                # Move each tensor in the dictionary to the device
                device_annot = {k: v.to(squeezeNet_config.DEVICE) for k, v in annot.items()}
                device_annotations.append(device_annot)

            for item in device_annotations:
                print(item)

            rpn_probs, rpn_bbox_regs = model.forward(images)
            
            anchors = generate_anchors(squeezeNet_config.ANCHOR_BASE_SIZE, squeezeNet_config.ANCHOR_RATIO, squeezeNet_config.ANCHOR_SCALE)
            print(f'Anchors Shape: {anchors.shape}')
            
            # Step 1: Reshape Predicted Bounding Boxes
            decoded_boxes = [apply_deltas_to_anchors(anchors, bbox_reg) for bbox_reg in rpn_bbox_regs]
            print(f'Shape of each batch of decoded boxes: {decoded_boxes[0].shape}')
            for i, boxes in enumerate(decoded_boxes):
                batch_size, _, height, width = boxes.shape
                decoded_boxes[i] = boxes.view(batch_size, anchors.shape[0], 4, height, width)
                decoded_boxes[i] = decoded_boxes[i].permute(0, 1, 3, 4, 2)  # Now shape is [batch_size, num_anchors, height, width, 4]
            
            orig_width, orig_height = squeezeNet_config.IMG_SIZE
            scale_y = orig_width / height  # Adjust for transposed images
            scale_x = orig_height / width
            for i, boxes in enumerate(decoded_boxes):
                boxes[..., 0] *= scale_x  # x
                boxes[..., 1] *= scale_y  # y
                boxes[..., 2] *= scale_x  # width
                boxes[..., 3] *= scale_y  # height
                decoded_boxes[i] = boxes[..., [1, 0, 3, 2]]  # Adjust for transposed images

            total_classification_loss = 0
            total_localization_loss = 0
            for i, (probs, boxes) in enumerate(zip(rpn_probs, decoded_boxes)):
                

                # Filter out low-confidence predictions
                high_confidence_idxs = (probs > squeezeNet_config.CONFIDENCE_THRESHOLD).nonzero(as_tuple=True)
                print(f"Shape of boxes: {boxes.shape}")
                print(f"high_confidence_idxs: {high_confidence_idxs}")
                
                filtered_probs = probs[high_confidence_idxs]
                print(f'probs/scores shape: {filtered_probs.shape}')
                filtered_boxes = boxes[high_confidence_idxs]
                print(f'boxes shape: {filtered_boxes.shape}')
                
                # Ensure all tensors are on the same device
                if filtered_boxes.device != filtered_probs.device:
                    raise ValueError("filtered_boxes and filtered_probs tensors are not on the same device")

                # Check for NaN or infinite values in tensors
                if torch.isnan(filtered_boxes).any() or torch.isinf(filtered_boxes).any():
                    raise ValueError("filtered_boxes tensor contains NaN or infinite values")

                if torch.isnan(filtered_probs).any() or torch.isinf(filtered_probs).any():
                    raise ValueError("filtered_probs tensor contains NaN or infinite values")
                
                # Apply NMS
                keep_idxs = torchvision.ops.nms(filtered_boxes.cpu(), filtered_probs.cpu(), squeezeNet_config.NMS_THRESHOLD)
                final_scores = filtered_probs[keep_idxs]
                final_boxes = filtered_boxes[keep_idxs]
                
                print(f'final boxes (scores/probs): {final_scores.shape}', f'final boxes (reg): {final_boxes.shape}')
                
                # Calculate losses, assuming you have a function to do so
                classification_loss, localization_loss = calculate_losses(final_boxes, final_scores, device_annotations[i])
                total_classification_loss += classification_loss
                total_localization_loss += localization_loss

            # Step 6: Backpropagation and Update
            total_loss = total_classification_loss + total_localization_loss
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

            # Log or print your loss here
            print(f"Epoch {epoch}, Total Loss: {total_loss.item()}")
