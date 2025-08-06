import torch
import PIL
import PIL.Image
import numpy as np
from rknn.api import RKNN
import argparse
import cv2

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path", type=str, default="/home/saul/proj/PFLD_test_onnx/output/0729_466_dict_sim.onnx")
    p.add_argument("--rknn_path", type=str, default="/home/saul/code/rknn_python/model/pfld.rknn")
    # these settings are dependent with model, should align with the model in traning stage.
    p.add_argument("--mean", type=list, default=[0., 0., 0.])
    p.add_argument("--std", type=list, default=[255.0, 255.0, 255.0])
    p.add_argument("--rgb2bgr", type=bool, default=True)

    p.add_argument("--chip", type=str, default="RV1106")
    return p

parser = get_args()
args = parser.parse_args()
# config, load, build and export RKNN
rknn = RKNN(verbose=True)
mean=args.mean
std=args.std
rknn.config(
    mean_values=[[v for v in mean]],
    std_values=[[v for v in std]],
    quant_img_RGB2BGR=args.rgb2bgr,
    target_platform=args.chip,
    # MMSE is better than normal, but need more time and memory.
    # It set MMSE, the length of dataset should not exceed 100! If not, there will be OOM.
    # quantized_algorithm="mmse",
    quantized_algorithm="normal",
    quantized_method="channel",
    optimization_level=3,
    quantized_dtype="asymmetric_quantized-8",
    custom_string="say_my_name",
)

ret = rknn.load_onnx(model=args.onnx_path)
if ret != 0:
    raise TypeError(f"model load error, {ret}")

ret = rknn.build(do_quantization=True, dataset="./dataset.txt")
if ret != 0:
    raise TypeError(f"rknn build error, ret: {ret}")

ret = rknn.export_rknn(args.rknn_path)
if ret != 0 :
    raise TypeError(f"export RKNN model error, ret: {ret}")

# init rknn runtime
ret = rknn.init_runtime(target=None)
if ret != 0:
    raise Exception(f"rknn init fail: {ret}")

IMAGE_PATH = "/home/saul/code/rknn_python/img_celeba_256/00000.jpg"
img_ori = cv2.imread(IMAGE_PATH)
img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
img_ori = cv2.resize(img_ori, (256, 256), interpolation=cv2.INTER_LINEAR)
img = np.expand_dims(img_ori, 0)

outputs = rknn.inference(inputs=[img])

print(outputs)

rknn.release()