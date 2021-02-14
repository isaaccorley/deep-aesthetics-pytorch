import torch
import torch.nn as nn
import torchvision


class Backbone(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = torchvision.models.alexnet(pretrained=pretrained)
        self.num_features = backbone.classifier[6].in_features
        backbone.classifier = backbone.classifier[:-1]
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RegressionNetwork(nn.Module):

    def __init__(
        self,
        num_attributes: int = 12,
        pretrained: bool = True
    ):
        super().__init__()
        backbone = Backbone(pretrained=pretrained)
        self.model = nn.Sequential(
            backbone,
            nn.Linear(
                in_features=backbone.num_features,
                out_features=num_attributes
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
