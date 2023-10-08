import torch

models = torch.hub.list("intel-isl/MiDaS")
print("Available MiDaS models:")
for model_name in models:
    print(model_name)