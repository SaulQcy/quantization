from rknn.api import RKNN
import PIL.Image
import numpy as np


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

ret = rknn_runtime.load_onnx("resnet18_imagenette_model/resnet18.onnx")
if ret != 0:
    raise Exception("load rknn model wrong")

ret = rknn_runtime.build(do_quantization=True, dataset="./dataset.txt")
if ret != 0:
    raise TypeError(f"rknn build error, ret: {ret}")

ret = rknn_runtime.init_runtime(target=None)
if ret != 0:
    raise Exception("RKNN init wrong")

img = PIL.Image.open("dataset/imagenette2-320/train/n02102040/ILSVRC2012_val_00000665.JPEG").resize((224, 224))
x_np = np.array(img)
# from HWC to NHWC
x_np = np.expand_dims(x_np, axis=0)
# from NHWC to NCHW
x_np = np.transpose(x_np, (0, 3, 1, 2))
print(x_np.shape)
# x_tc = torch.from_numpy(x_np).permute(2, 0 ,1).unsqueeze(0)

rknn_runtime.accuracy_analysis(inputs=[x_np], target=None)

