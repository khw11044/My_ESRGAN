import torch 
from PIL import Image 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_Model = True 
CHECKPOINT_GEN = './checkpoint/gen.pth'
CHECKPOINT_DISC = './checkpoint/disc.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16      # 16
LAMBDA_GP = 10
NUM_WORKERS = 0
HIGH_RES = 256      # 128 256
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3 
COLAB=True

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),  # HIGH_RES에 맞게 랜덤 crop해서 훈련
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

lowres_transform = A.Compose(
    [   
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)


highres_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)


test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

