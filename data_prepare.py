import PIL.ImageEnhance
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import cv2


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    def __init__(self, p_vflip=0.5, p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1), isTrain=True):
        self.p_flip = p_flip
        self.p_vflip = p_vflip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
        self.to_tensor = T.ToTensor()
        self.isTrain = isTrain

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), Image.fromarray(mask)

        if np.random.rand() < self.p_flip and self.isTrain:
            image, mask = F.hflip(image), F.hflip(mask)

        if np.random.rand() < self.p_vflip and self.isTrain:
            image, mask = F.vflip(image), F.vflip(mask)

        # color transforms
        if self.color_jitter_params and self.isTrain:
            image = self.color_tf(image)

        contrast = 1.5
        enhanced = PIL.ImageEnhance.Contrast(image)
        image = enhanced.enhance(contrast)

        # transforming to tensor
        image = self.normalize(self.to_tensor(correct_dims(np.array(image))))
        mask = self.to_tensor(correct_dims(np.array(mask)))

        return image, mask


class ImageToImage2D(Dataset):

    def __init__(self, image_list, mask_list, joint_transform: Callable = None, one_hot_mask: int = False,
                 isTrain=True) -> None:
        self.one_hot_mask = one_hot_mask
        self.images_list = image_list
        self.mask_list = mask_list
        self.isTrain = isTrain

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
            self.norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.color_tf = T.ColorJitter(0.1, 0.1, 0.1, 0.1)

    def __len__(self):
        # return len(os.listdir(self.input_path))
        return len(self.images_list)

    def __getitem__(self, idx):
        # read image
        image = cv2.imread(self.images_list[idx])
        # read mask images
        mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)

        img_target_size = (1024, 1024)
        mask_target_size = (256, 256)

        image = cv2.resize(image, img_target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, mask_target_size, interpolation=cv2.INTER_LINEAR)
        mask[mask > 0] = int(255)

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
            # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
            # image, mask = self.joint_transform(image, mask)
            # gray scale input
            # image = torch.repeat_interleave(image, 3, dim=0)
            # end
            image = self.norm(image)
            # image = self.color_tf(image)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # img_padh = 1024 - img_target_size[1]
        # img_padw = 1024 - img_target_size[0]
        # image = torch.nn.functional.pad(image, (0, img_padw, 0, img_padh))
        # mask_padh = 256 - mask_target_size[1]
        # mask_padw = 256 - mask_target_size[0]
        # mask = torch.nn.functional.pad(mask, (0, mask_padw, 0, mask_padh))
        # hf_image = high_frequency_filter(image)

        return image.to(torch.float), mask, img_target_size, mask_target_size, self.images_list[idx].split('/')[-1:][0]


def flatten(tensor):
    N = tensor.size(0)

    return tensor.contiguous().view(N, -1)
