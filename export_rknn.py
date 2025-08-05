import torch
import PIL
import PIL.Image
import numpy as np
from rknn.api import RKNN
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path", type=str, default="/home/saul/proj/PFLD_test_onnx/output/0729_466_dict_sim.onnx")
    p.add_argument("--rknn_path", type=str)
    p.add_argument("--mean", type=list, default=[0, 0, 0])
    p.add_argument("--std", type=list, default=[1, 1, 1])
    p.add_argument("--rgb2bgr", type=bool, default=True)
    p.add_argument("--chip", type=str, default="RV1106")
    return p

parser = get_args()
args = parser.parse_args()
# config, load, build and export RKNN
rknn_runtime = RKNN(verbose=True)
mean=args.mean
std=args.std
rknn_runtime.config(
    mean_values=[[v * 255. for v in mean]],
    std_values=[[v * 255. for v in std]],
    quant_img_RGB2BGR=args.rgb2bgr,
    target_platform=args.chip,
    quantized_algorithm="normal",
    quantized_method="channel",
    optimization_level=3,
    quantized_dtype="asymmetric_quantized-8",
    custom_string="say_my_name",
)

ret = rknn_runtime.load_onnx(model=args.onnx_path)
if ret != 0:
    raise TypeError(f"model load error, {ret}")

ret = rknn_runtime.build(do_quantization=True, dataset="./dataset.txt")
if ret != 0:
    raise TypeError(f"rknn build error, ret: {ret}")

ret = rknn_runtime.export_rknn(args.rknn_path)
if ret != 0 :
    raise TypeError(f"export RKNN model error, ret: {ret}")

# init rknn runtime
ret = rknn_runtime.init_runtime(target=None)
if ret != 0:
    raise Exception(f"rknn init fail: {ret}")

rknn_runtime.release()