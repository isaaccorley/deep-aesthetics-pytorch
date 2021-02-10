import os
from typing import List, Tuple

import torch
import torchvision.transforms as T
import scipy.io
from PIL import Image


class AADB(torch.utils.data.Dataset):

    attributes = [
        "score",
        "balancing_elements",
        "color_harmony",
        "content",
        "depth_of_field",
        "light",
        "motion_blur",
        "object",
        "repetition",
        "rule_of_thirds",
        "symmetry",
        "vivid_color"
    ]

    splits = {
        "train": {"idx": 0, "file": "imgListTrainRegression_score.txt"},
        "test": {"idx": 1, "file": "imgListTestNewRegression_score.txt"},
        "val": {"idx": 2, "file": "imgListValidationRegression_score.txt"}
    }

    labels_file = "attMat.mat"
    
    def __init__(
        self,
        image_dir: str = "data/aadb/images",
        labels_dir: str = "data/aadb/labels",
        split: str = "train",
        transforms: T.Compose = T.Compose([T.ToTensor()])
    ):
        self.self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.files, self.labels = self.load_split(split)

    def load_split(self, split: str) -> Tuple[List[str], np.ndarray]:
        # Load labels
        labels_path = os.path.join(self.labels_dir, self.labels_file)
        labels = scipy.io.loadmat(labels_path)["dataset"]
        labels = labels[0][self.splits[split]["idx"]]

        # Load file paths
        with open(self.splits[split]["file"], "r") as f:
            files = f.read().strip().splitlines()
            files = [f.split()[0] for f in files]
            files = [os.path.join(self.image_dir, f) for f in files]

        return files, labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = Image.open(self.files[idx]).convert("RGB")
        x = self.transforms(x)
        y = torch.from_numpy(self.labels[idx])
        return x, y
