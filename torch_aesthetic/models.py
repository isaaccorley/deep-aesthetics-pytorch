import torch
import torch.nn as nn
import torchvision


class Backbone(nn.Module):

    def __init__(
        self,
        backbone: str,
        pretrained: bool
    ):
        super().__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.num_features = resnet.fc.in_features
        self.model = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()


class RegressionNetwork(nn.Module):

    def __init__(
        self,
        backbone: str = "resnet18",
        num_attributes: int = 12,
        pretrained: bool = True
    ):
        super().__init__()
        backbone = Backbone(backbone, pretrained)
        self.model = nn.Sequential(
            backbone,
            nn.Linear(
                in_features=backbone.num_features,
                out_features=num_attributes
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
