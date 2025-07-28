import torch
import PIL
import PIL.Image
import numpy as np
from rknn.api import RKNN

# load original torch model, not torchscript.
model = torch.load("./resnet18_imagenette_model/resnet18.pth")
dummy_input = torch.randn(1, 3, 224, 224)
# export to ONNX
torch.onnx.export(model, dummy_input, "./resnet18_imagenette_model/resnet18.onnx", opset_version=11)

# config, load, build and export RKNN
rknn_runtime = RKNN(verbose=True, verbose_file="./resnet_build.log")
mean=[0.4648, 0.4543, 0.4247]
std=[0.2785, 0.2735, 0.2944]
rknn_runtime.config(
    mean_values=[[v * 255. for v in mean]],
    std_values=[[v * 255. for v in std]],
    quant_img_RGB2BGR=False,
    target_platform="RV1106",
    quantized_algorithm="normal",
    quantized_method="channel",
    optimization_level=3,
    quantized_dtype="asymmetric_quantized-8",
    custom_string="saul_v0.0.1",
)

ret = rknn_runtime.load_onnx(model="./resnet18_imagenette_model/resnet18.onnx")
if ret != 0:
    raise TypeError(f"model load error, {ret}")

ret = rknn_runtime.build(do_quantization=True, dataset="./dataset.txt")
if ret != 0:
    raise TypeError(f"rknn build error, ret: {ret}")

ret = rknn_runtime.export_rknn('./resnet18_imagenette_model/resnet18.rknn')
if ret !=0 :
    raise TypeError(f"export RKNN model error, ret: {ret}")

# init rknn runtime
ret = rknn_runtime.init_runtime(target=None)
if ret != 0:
    raise Exception(f"rknn init fail: {ret}")

# img = PIL.Image.open("dataset/imagenette2-320/train/n02102040/ILSVRC2012_val_00000665.JPEG").resize((224, 224))
# x_np = np.array(img)
# x_np = np.expand_dims(x_np, axis=0)
# # x_tc = torch.from_numpy(x_np).permute(2, 0 ,1).unsqueeze(0)

# outputs = rknn_runtime.inference(inputs=[x_np])
# print(outputs[0].shape)
# y = np.argmax(outputs[0])
# print(np.argmax(y))

rknn_runtime.release()