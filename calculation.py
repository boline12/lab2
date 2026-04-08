import numpy as np

def calculate_metrics(pred_mask, true_mask):
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0
    return iou, dice

# FIX: Create dummy data to simulate a 68% IoU
# In a real scenario, these would be your model's output and your Week 7 masks
y_true = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
y_pred = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

iou_val, dice_val = calculate_metrics(y_pred, y_true)
print(f"Calculated IoU: {iou_val:.4f}")
print(f"Calculated Dice: {dice_val:.4f}")