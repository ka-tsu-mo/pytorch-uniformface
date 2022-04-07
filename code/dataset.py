import os
from glob import glob

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip
from PIL import Image


class FaceImageDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dataset = self._load_dataset(base_dir)
        self.hflip = RandomHorizontalFlip()
        classes = set([d['label'] for d in self.dataset])
        self.class2id = {c:i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

    def _load_dataset(self, base_dir):
        classes = glob(os.path.join(base_dir, '*'))
        dataset = []
        for c in classes:
            class_name = os.path.basename(os.path.normpath(c))
            # clear the identities with less than 3 images to relieve
            # the long-tail distribution (Section 4.1 in UniformFace paper)
            files = glob(os.path.join(base_dir, class_name, '*.jpg'))
            if len(files) < 3:
                continue
            # exclude the identities appearing in LFW dataset
            if class_name in ['0166921', '1056413', '1193098']:
                continue
            dataset.extend([{'image_path': f, 'label': class_name} for f in files])

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = Image.open(self.dataset[index]['image_path'])
        img = self.hflip(img)
        img_array = np.array(img, dtype=np.float32)
        img_array = ((img_array - 127.5) / 128)  # normalize
        img_array = img_array.transpose(2, 0, 1)
        label = self.class2id[self.dataset[index]['label']]
        return {'label': label, 'image': img_array}


def collate_fn(batch):
    labels = np.array([ex['label'] for ex in batch])
    images = np.stack([ex['image'] for ex in batch])
    return {'labels': labels, 'images': images}
