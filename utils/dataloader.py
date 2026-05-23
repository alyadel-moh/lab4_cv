import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, TEST_PATH, TRAIN_PATH, VAL_PATH, BATCH_SIZE


def get_dataloaders(img_size=None):
    size = img_size if img_size is not None else IMG_SIZE

    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    val_dataset   = datasets.ImageFolder(VAL_PATH,   transform=val_tf)
    test_dataset  = datasets.ImageFolder(TEST_PATH,  transform=val_tf)

    assert train_dataset.classes == val_dataset.classes == test_dataset.classes, \
        "Class mismatch between train/val/test folders!"

    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    print(f"Image size   : {size}×{size}")
    print(f"Train images : {len(train_dataset)}")
    print(f"Val images   : {len(val_dataset)}")
    print(f"Test images  : {len(test_dataset)}")
    print(f"Classes      : {class_names}")

    return train_loader, val_loader, test_loader, class_names