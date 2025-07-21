import torchvision
from torchvision.transforms.transforms import Compose, Resize, ToTensor
import torch
import os

tf = Compose([
    ToTensor(), Resize((224, 224))
])
ds = torchvision.datasets.Imagenette(root="./dataset", download=True, size="320px", transform=tf)

model = torchvision.models.resnet18(num_classes=10)
ld = torch.load("./results.pth", weights_only=False)
model.load_state_dict(ld["model"])
torch.save(model, "resnet.pt")
model.eval()
i = 0
acc = 0
num = 0
os.makedirs("./img/", exist_ok=True)
# dataset_normalization = Normalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
mean = torch.tensor([0.4648, 0.4543, 0.4247]).view(3, 1, 1)
std  = torch.tensor([0.2785, 0.2735, 0.2944]).view(3, 1, 1)
for x, y in ds:
    x = (x - mean) / std
    y_p = model(x.unsqueeze(0))
    y_p = torch.argmax(y_p, dim=1).item()
    if y == y_p:
        acc += 1
    num += 1
    if num == 100:
        break
    # print(y)
    # if y == y_p:
    #     torchvision.utils.save_image(x, f"./img/img_{y_p}_{224}_{i}.png")
    #     i += 1
    # if i == 10:
    #     break
print(acc)