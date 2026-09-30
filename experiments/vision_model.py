"""ResNet-18 with the exact 32x32 adaptation used in the experiments."""
from torch import nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Construct the full torchvision model first: this preserves RNG order.
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)
