import onnxruntime as ort
import cv2
import numpy as np

# load image
img_path = "/home/saul/code/rknn_c/1-FemaleNoGlasses_334.jpg"
img = cv2.imread(img_path)
orig_shape = img.shape[:2]  # (H, W)
img = cv2.resize(img, (256, 256))
x_np = np.array(img).astype(np.float32) / 255.
x_np = x_np.transpose(2, 0, 1)
x_np = np.expand_dims(x_np, axis=0)
n, c, h, w = x_np.shape

# load model
session = ort.InferenceSession(
    "/home/saul/code/rknn_python/model/pfld.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: x_np})
for i in range(len(output)):
    print(output[i].shape)

# postprocess
landmark, euler_angle, main_classes = output
x_np = (x_np * 255.).astype(np.uint8)
x_np = x_np.squeeze()
x_np = x_np.transpose(1, 2, 0)
print(x_np.shape)
# draw landmark
for i in range(19):
    norm_x = landmark[:, 2 * i]
    norm_y = landmark[:, 2 * i + 1]
    px = int(norm_x * (w - 1))
    py = int(norm_y * (h - 1))

    cv2.circle(x_np, (px, py), 1, (0, 255, 0), -1)

cv2.imwrite("result.png", x_np)

