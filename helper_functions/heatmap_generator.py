
import torch.nn.functional as F
import json
import numpy as np
import cv2
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def multi_label_accuracy(logits, labels, threshold=0.5):
    """Compute accuracy for multi-label classification"""
    probs = F.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == labels).float().sum(dim=1)  # number of correct labels per sample
    acc = (correct == labels.size(1)).float().mean()  # check if all labels per sample are correct
    return acc.item()


def json_to_mask(json_path, img_shape, target_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(img_shape[:2], dtype=np.uint8)  # Create an empty mask based on the height and width of the image

    for shape in data['shapes']:
        polygon = shape['points']
        cv2.fillPoly(mask, [np.array(polygon).astype(int)], 1)  # Fill the polygon
    # Resize the mask to target_shape
    resized_mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
    return resized_mask

# Binarize the heatmap
def binarize_map(input_map, threshold):
    return (input_map > threshold).astype(np.uint8)

# Compute the Intersection over Union (IoU)
def compute_iou(heatmap, mask):
    intersection = np.logical_and(heatmap, mask).sum()
    union = np.logical_or(heatmap, mask).sum()
    if union == 0:
        return 0  # Prevent division by zero
    return intersection / union
