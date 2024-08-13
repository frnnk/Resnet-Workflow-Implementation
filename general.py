import torch
import torchvision.models as models

model = models.resnet50(weights="IMAGENET1K_V1")
model_dict = model.state_dict()

print(model_dict.keys())

for name, parameter in model.named_parameters():
    print(name, parameter.shape)

