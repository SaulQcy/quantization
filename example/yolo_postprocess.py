import torch
import PIL.Image
import numpy as np
import onnxruntime as ort
import cv2


def nms_py(boxes, scores, iou_threshold):
    # boxes: [N, 4] (x1, y1, x2, y2)
    # scores: [N]
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break

        ious = compute_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

def compute_iou(box1, box2):
    # box1: [1, 4], box2: [N, 4]
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter

    return inter / union



# -----------------------
# Load image and preprocess
# -----------------------
import cv2

img_path = "/home/saul/code/rknn_c/bus.jpg"
img = cv2.imread(img_path)

# 获取原始尺寸 (H, W, C)
orig_shape = img.shape[:2]  # (H, W)

# Resize 到模型输入
img = cv2.resize(img, (640, 640))


x_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
x_np = x_np.transpose(2, 0, 1)  # HWC -> CHW
x_np = np.expand_dims(x_np, axis=0)  # [1, 3, 640, 640]
print("Input shape:", x_np.shape)

# -----------------------
# Load ONNX model and run inference
# -----------------------
session = ort.InferenceSession(
    "/home/saul/code/rknn_python/model/yolov5n.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: x_np})

# -----------------------
# Postprocess YOLO outputs
# -----------------------
conf_thres = 0.5
iou_thres = 0.45
input_shape = (640, 640)
strides = [8, 16, 32]

anchors = [
    [(10, 13), (16, 30), (33, 23)],   # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]

# Decode outputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

all_preds = []
for output, stride, anchor in zip(outputs, strides, anchors):
    bs, na_c, ny, nx = output.shape
    num_anchors = len(anchor)
    nc = na_c // num_anchors  # 85
    output = output.reshape(bs, num_anchors, nc, ny, nx)
    output = output.transpose(0, 1, 3, 4, 2)  # [1, 3, ny, nx, 85]

    grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, ny, nx, 2)

    anchor = np.array(anchor).reshape(1, num_anchors, 1, 1, 2)

    pred = output.copy()
    pred[..., 0:2] = (sigmoid(pred[..., 0:2]) + grid) * stride        # x, y
    pred[..., 2:4] = np.exp(pred[..., 2:4]) * anchor                  # w, h
    pred[..., 4:] = sigmoid(pred[..., 4:])                            # objectness & class scores

    pred = pred.reshape(-1, nc)
    all_preds.append(pred)

pred = np.concatenate(all_preds, axis=0)  # [N, 85]

# -----------------------
# Filter by confidence and apply NMS
# -----------------------
boxes = pred[:, :4]
objectness = pred[:, 4]
class_scores = pred[:, 5:]
class_ids = np.argmax(class_scores, axis=1)
scores = objectness * class_scores[np.arange(len(class_scores)), class_ids]

# Filter by confidence
mask = scores > conf_thres
boxes = boxes[mask]
scores = scores[mask]
class_ids = class_ids[mask]

if boxes.shape[0] == 0:
    print("No detections above threshold.")
    exit()

# Convert xywh -> xyxy
boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

# Scale boxes back to original image size
scale_x = orig_shape[0] / input_shape[0]
scale_y = orig_shape[1] / input_shape[1]
boxes[:, [0, 2]] *= scale_x
boxes[:, [1, 3]] *= scale_y

# NMS using torchvision
keep = nms_py(
    torch.tensor(boxes),
    torch.tensor(scores),
    iou_thres
)

boxes = boxes[keep]
scores = scores[keep]
class_ids = class_ids[keep]

# -----------------------
# Visualization
# -----------------------
img = np.array(PIL.Image.open(img_path).convert("RGB"))

for i in range(len(boxes)):
    x0, y0, x1, y1 = boxes[i].astype(int)
    cls_id = int(class_ids[i])
    score = scores[i]
    label = f"{cls_id}: {score:.2f}"

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(img, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Save or show result
cv2.imwrite("result.png", img)
print("Result saved to result.png")
