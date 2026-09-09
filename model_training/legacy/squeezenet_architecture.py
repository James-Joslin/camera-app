from network_config import SqueezeNetConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SqueezeNet_Config = SqueezeNetConfig()

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))

        for feature, inner_block, layer_block in zip(
            reversed(x[:-1]), reversed(self.inner_blocks[:-1]), reversed(self.layer_blocks[:-1])
        ):
            # Upsample 'last_inner' to match the spatial dimensions of 'inner_lateral'
            inner_lateral = inner_block(feature)
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[2:], mode="nearest")
            
            # Element-wise addition
            last_inner = inner_lateral + inner_top_down
            
            # Pass the combined features through the next layer block and add to results
            results.insert(0, layer_block(last_inner))

        return results

class FireWithBypass(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireWithBypass, self).__init__()
        # Adjust squeeze channels based on the squeeze ratio of 0.75
        adjusted_squeeze_channels = int(squeeze_channels * 0.75)

        self.squeeze = nn.Conv2d(in_channels, adjusted_squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand layers
        self.expand1x1 = nn.Conv2d(adjusted_squeeze_channels, expand_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(adjusted_squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Store the original input for the bypass connection
        identity = x

        x = self.squeeze_activation(self.squeeze(x))
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

        # Add the bypass (identity) connection if the sizes match
        if x.size() == identity.size():
            x += identity
        return x

class CustomSqueezeNet(nn.Module):
    def __init__(self):
        super(CustomSqueezeNet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)  # Adjusted for smaller models
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Fire modules with simple bypass and squeeze ratio adjustments
        self.fire2 = FireWithBypass(64, 16, 64)
        self.fire3 = FireWithBypass(128, 16, 64)
        self.fire4 = FireWithBypass(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire5 = FireWithBypass(256, 32, 128)
        self.fire6 = FireWithBypass(256, 48, 192)
        self.fire7 = FireWithBypass(384, 48, 192)
        self.fire8 = FireWithBypass(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire9 = FireWithBypass(512, 64, 256)
        
        # Final convolution (no softmax)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=1)  # The output channels can be adjusted

        self.fpn = FPN(in_channels_list=[256, 256, 384, 512], out_channels=256)
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        f2 = self.fire2(x)
        f3 = self.fire3(f2)
        f4 = self.fire4(f3)
        x = self.maxpool2(f4)
        f5 = self.fire5(x)
        f6 = self.fire6(f5)
        f7 = self.fire7(f6)
        f8 = self.fire8(f7)
        x = self.maxpool3(f8)
        f9 = self.fire9(x)

        # Collect selected feature maps
        feature_maps = [f4, f5, f7, f9]
        for ele in feature_maps:
            print(ele.shape)

        # Pass feature maps through FPN
        fpn_feature_maps = self.fpn(feature_maps)
        for ele in fpn_feature_maps:
            print(ele.shape)

        return fpn_feature_maps

class RPN(nn.Module):
    def __init__(self, anchor_scales, anchor_ratios, feature_channels, mid_channels=256):
        super(RPN, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        
        # The number of anchors
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        # The RPN has a conv layer followed by two separate layers for classifying and regressing anchors
        self.conv = nn.Conv2d(feature_channels, mid_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, self.num_anchors * 2, 1)  # 2 for objectness score (obj, not obj)
        self.bbox_pred = nn.Conv2d(mid_channels, self.num_anchors * 4, 1)  # 4 for bbox offsets

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize the weights and biases with a specific strategy
        pass

    def forward(self, features):
        # Apply the RPN layers to get the objectness score and bbox predictions
        x = self.conv(features)
        logits = self.cls_logits(x)
        bbox_regs = self.bbox_pred(x)
        return logits, bbox_regs

# Example of initializing the RPN
# feature_channels = 512  # Adjust based on the output channels of your SqueezeNet
# rpn = RPN(anchor_scales=[128, 256, 512], anchor_ratios=[0.5, 1, 2], feature_channels=feature_channels)
    
class ObjectDetector(nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        self.backbone = CustomSqueezeNet()  # Updated backbone
        self.rpn = RPN(anchor_scales=SqueezeNet_Config.ANCHOR_SCALE, anchor_ratios=SqueezeNet_Config.ANCHOR_RATIO, feature_channels=SqueezeNet_Config.RPN_OUT)  # Update channels if needed

    def forward(self, x):
        features = self.backbone(x)
        for i, feature in enumerate(features):
            print(f"Feature map {i} shape:", feature.shape)
        # self.test_rpn_forward()
        # Assume RPN takes a list of feature maps
        rpn_logits = []
        rpn_bbox_regs = []
        for feature in features:
            logits, bbox_regs = self.rpn(feature)
            rpn_logits.append(logits)
            rpn_bbox_regs.append(bbox_regs)
        probabilities = [torch.sigmoid(logit) for logit in rpn_logits]
        
        for i, probs in enumerate(probabilities):
            print(f"Objectness probabilities {i} (after sigmoid) shape (from probabilities list):", probs.shape)
        print("Bounding box regression deltas shape:", bbox_regs.shape)
        
        return probabilities, rpn_bbox_regs

    def test_rpn_forward(self):
        # Create a dummy feature map (e.g., one you might get from your backbone)
        print("testing rpn outputs")
        dummy_feature_map = torch.randn(1, 256, 14, 14)  # Example shape: [batch_size, channels, height, width]
        print("Dummy feature map shape:", dummy_feature_map.shape)
        
        # Instantiate your RPN with the expected number of anchors
        test_rpn = RPN(anchor_scales=[64, 128, 256, 512, 1024, 2048], anchor_ratios=[0.5, 0.75, 1, 1.25, 1.5, 2], feature_channels=256)
        
        # Forward pass through the RPN
        objectness_logits, bbox_regs = test_rpn(dummy_feature_map)
        
        # Print the shape of the outputs
        print("Objectness logits shape:", objectness_logits.shape)
        print("Bounding box regression deltas shape:", bbox_regs.shape)

def generate_anchors(base_size, ratios, scales):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])  # [x1, y1, x2, y2]
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    anchors = torch.from_numpy(anchors).float().to(SqueezeNet_Config.DEVICE)
    return anchors

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    return _mkanchors(ws, hs, x_ctr, y_ctr)

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.array(scales)
    hs = h * np.array(scales)
    
    # Debugging
    # print("w:", w, "h:", h)
    # print("ws:", ws, "hs:", hs)
    # print("Type of ws:", type(ws), "Type of hs:", type(hs))
    
    anchors = [_mkanchors(ws[i], hs[i], x_ctr, y_ctr) for i in range(len(scales))]
    return np.concatenate(anchors, axis=0)

def _whctrs(anchor):
    # Convert anchor (x1, y1, x2, y2) to (width, height, x_center, y_center)
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    # Convert (width, height, x_center, y_center) to (x1, y1, x2, y2)
    ws = np.array([ws]).flatten() if np.isscalar(ws) else ws
    hs = np.array([hs]).flatten() if np.isscalar(hs) else hs

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def apply_deltas_to_anchors(anchors, deltas):
    anchors = anchors.to(deltas.device)
    
    num_anchors = anchors.shape[0]  # The number of anchors you have
    assert num_anchors * 4 == deltas.shape[1], f"Mismatch in number of anchors (expected {num_anchors * 4}, got {deltas.shape[1]})"
    
    # Calculate widths, heights, and centers of the anchors
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    
    # Extract deltas for each anchor
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    
    # Expand widths, heights, and centers to match the shape of deltas
    widths_expanded = widths.view(1, num_anchors, 1, 1).expand_as(dx)
    heights_expanded = heights.view(1, num_anchors, 1, 1).expand_as(dy)
    ctr_x_expanded = ctr_x.view(1, num_anchors, 1, 1).expand_as(dx)
    ctr_y_expanded = ctr_y.view(1, num_anchors, 1, 1).expand_as(dy)
    
    # Apply the deltas to the anchors to get the predicted boxes
    pred_ctr_x = dx * widths_expanded + ctr_x_expanded
    pred_ctr_y = dy * heights_expanded + ctr_y_expanded
    pred_w = torch.exp(dw) * widths_expanded
    pred_h = torch.exp(dh) * heights_expanded
    
    # Calculate the top-left and bottom-right coordinates
    top_left_x = pred_ctr_x - 0.5 * pred_w
    top_left_y = pred_ctr_y - 0.5 * pred_h
    bottom_right_x = pred_ctr_x + 0.5 * pred_w
    bottom_right_y = pred_ctr_y + 0.5 * pred_h
    
    # Initialize pred_boxes tensor
    pred_boxes = torch.zeros_like(deltas)
    
    # Interpolate the predicted box coordinates to match the shape of deltas
    target_shape = deltas.shape[-2:]  # Spatial dimensions of the target shape
    top_left_x = F.interpolate(top_left_x, size=target_shape, mode='bilinear', align_corners=False)
    top_left_y = F.interpolate(top_left_y, size=target_shape, mode='bilinear', align_corners=False)
    bottom_right_x = F.interpolate(bottom_right_x, size=target_shape, mode='bilinear', align_corners=False)
    bottom_right_y = F.interpolate(bottom_right_y, size=target_shape, mode='bilinear', align_corners=False)
    
    # Assign the interpolated values to pred_boxes
    pred_boxes[:, 0::4, :, :] = top_left_x
    pred_boxes[:, 1::4, :, :] = top_left_y
    pred_boxes[:, 2::4, :, :] = bottom_right_x
    pred_boxes[:, 3::4, :, :] = bottom_right_y
    
    return pred_boxes