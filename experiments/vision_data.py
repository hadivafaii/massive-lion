"""Fixed CIFAR split and device-local crop/flip augmentation."""
from pathlib import Path
import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class CIFAR10Loader:
    """Iterate over GPU-resident CIFAR-10 with online crop/flip augmentation."""

    def __init__(self, images, labels, batch_size, train=False, seed=0):
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(
                f'Expected CIFAR-10 images shaped (N, 3, 32, 32), '
                f'got {tuple(images.shape)}')
        if labels.ndim != 1 or len(labels) != len(images):
            raise ValueError('Expected one CIFAR-10 label per image')
        if images.device != labels.device:
            raise ValueError('CIFAR-10 images and labels must share a device')
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.train = train
        self.seed = seed
        self.epoch = 0
        if train:
            black = images.new_tensor([
                -mean / std for mean, std in zip(MEAN, STD)
            ]).view(1, 3, 1, 1)
            padded = images.new_empty(len(images), 3, 40, 40)
            padded[:] = black
            padded[:, :, 4:36, 4:36] = images
            self.images = padded.unfold(2, 32, 1).unfold(3, 32, 1)
            self.images = self.images.permute(0, 2, 3, 1, 4, 5)

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError(f'epoch must be non-negative, got {epoch}')
        self.epoch = epoch

    def __iter__(self):
        if not self.train:
            for start in range(0, len(self.labels), self.batch_size):
                stop = min(start + self.batch_size, len(self.labels))
                yield self.images[start:stop], self.labels[start:stop]
            return

        generator = torch.Generator(device=self.labels.device)
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(
            len(self.labels), device=self.labels.device, generator=generator)
        offsets = torch.randint(
            9, (2, len(self.labels)),
            device=self.labels.device, generator=generator)
        flips = torch.rand(
            len(self.labels),
            device=self.labels.device, generator=generator) < 0.5
        for start in range(0, len(self.labels), self.batch_size):
            stop = min(start + self.batch_size, len(self.labels))
            indices = order[start:stop]
            x = self.images[
                indices, offsets[0, start:stop], offsets[1, start:stop]]
            flip = flips[start:stop]
            x[flip] = x[flip].flip(-1)
            yield x, self.labels[indices]


def normalize(images):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (images.float() / 255.0 - mean) / std


def split_indices():
    """Use the fixed seed-0 45,000/5,000 split, independent of training seed."""
    order = torch.randperm(50_000, generator=torch.Generator().manual_seed(0))
    return order[:45_000], order[45_000:]


def prepare(root):
    from torchvision.datasets import CIFAR10
    root = Path(root)
    output = root / 'processed'
    if output.exists():
        raise FileExistsError(f'{output} already exists; refusing to replace prepared data')
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)
    images = normalize(torch.from_numpy(train.data).permute(0, 3, 1, 2))
    labels = torch.tensor(train.targets)
    train_idx, val_idx = split_indices()
    splits = {
        'trn': (images[train_idx], labels[train_idx]),
        'vld': (images[val_idx], labels[val_idx]),
        'tst': (normalize(torch.from_numpy(test.data).permute(0, 3, 1, 2)),
                torch.tensor(test.targets)),
    }
    output.mkdir(parents=True)
    for split, (x, y) in splits.items():
        np.save(output / f'x_{split}.npy', x.numpy())
        np.save(output / f'y_{split}.npy', y.numpy())
    np.savez(output / 'split_indices.npz', train=train_idx.numpy(), validation=val_idx.numpy())
    print(f'Prepared CIFAR-10 at {output}')


def load_splits(path, device):
    splits = []
    for split, count in [('trn', 45_000), ('vld', 5_000), ('tst', 10_000)]:
        x = torch.from_numpy(np.load(Path(path) / f'x_{split}.npy'))
        y = torch.from_numpy(np.load(Path(path) / f'y_{split}.npy')).long()
        if x.shape != (count, 3, 32, 32) or y.shape != (count,) or x.dtype != torch.float32:
            raise ValueError(f'Invalid CIFAR-10 {split} tensors')
        splits.append((x.to(device), y.to(device)))
    return splits
