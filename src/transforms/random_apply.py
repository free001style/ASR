import random

from torchvision.transforms.v2 import Compose


class RandomApply:
    """Custom Random Apply for applying transforms or not applying at all."""

    def __init__(self, transforms: Compose, p: int):
        """Args
        transforms (Compose): transforms to apply.
        p (int): probability to apply all transforms or not apply at all."""
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if random.random() <= self.p:
            return self.transforms(x)
        return x
