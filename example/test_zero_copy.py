from PIL import Image
import numpy as np

img = Image.open('/home/saul/code/rknn_c/bus.jpg')
img = img.convert('RGB')  # 确保和 C 端一样
x_np = np.array(img)

x_np_flat = x_np.flatten()
print("Python image data (first 1000 values):")
for i in range(1000):
    print(f"{x_np_flat[i]}", end=",")
    if (i + 1) % 20 == 0:
        print()
