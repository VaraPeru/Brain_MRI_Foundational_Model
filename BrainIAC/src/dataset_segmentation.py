import pandas as pd
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Resized, NormalizeIntensityd, RandFlipd, RandRotated, Rand3DElasticd, RandBiasFieldd, RandGaussianNoised, ToTensord
)

def get_segmentation_dataloader(csv_file, img_size, batch_size, num_workers, is_train=True):
    df = pd.read_csv(csv_file)
    items = [
        {'image': row['image_path'], 'label': row['mask_path']} for _, row in df.iterrows()
    ]
    if is_train:
        transforms = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Resized(keys=['image', 'label'], spatial_size=img_size, mode=('trilinear', 'nearest')),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandRotated(keys=['image', 'label'], range_x=0.1, prob=0.5, mode='bilinear'),
            Rand3DElasticd(keys=['image', 'label'], sigma_range=(5,8), magnitude_range=(100,200), prob=0.2),
            RandBiasFieldd(keys=['image'], prob=0.3),
            RandGaussianNoised(keys=['image'], prob=0.2),
            EnsureTyped(keys=['image', 'label']),
            ToTensord(keys=['image', 'label'])
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Resized(keys=['image', 'label'], spatial_size=img_size, mode=('trilinear', 'nearest')),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            EnsureTyped(keys=['image', 'label']),
            ToTensord(keys=['image', 'label'])
        ])
    ds = CacheDataset(data=items, transform=transforms, cache_rate=0.1, num_workers=num_workers)
    return ds 