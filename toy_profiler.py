import torch
import torchvision.models as models
from torch.profiler import profiler, ProfilerActivity, record_function

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)


