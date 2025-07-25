import cv2
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from segment_anything.modeling import DVPT
from data_prepare import JointTransform2D, ImageToImage2D

parser = argparse.ArgumentParser(description='DVPT')
parser.add_argument('--epochs', default=200, type=int, metavar='epochs',
                    help='number of total epochs to run(default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='batch size (default: 2)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dataset', default=None, type=str, metavar='dataset',
                    help='momentum')
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./train', type=str,
                    help='directory to save')
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()
direc = args.direc
mask_threshold = 0

images = []
masks_list = []
test_img = []
test_mask = []

folders = ['train']
image_folder = 'ISIC_2017/img'
mask_folder = 'ISIC_2017/label'

for folder in folders:
    imgdir = image_folder + '/' + folder
    maskdir = mask_folder + '/' + folder
    imgs = os.listdir(imgdir)
    masks = os.listdir(maskdir)
    for i in tqdm(range(100)):
        images.append(imgdir + '/' + imgs[i])
        masks_list.append(maskdir + '/' + imgs[i][:-4] + '_segmentation.png')

f = ['test']

for folder in f:
    imgs = os.listdir(image_folder + '/' + folder)
    masks = os.listdir(mask_folder + '/' + folder)
    for i in tqdm(range(100)):
        test_img.append(image_folder + '/' + folder + '/' + imgs[i])
        test_mask.append(mask_folder + '/' + folder + '/' + imgs[i][:-4] + '_segmentation.png')

print('load dataset')

tf_train = JointTransform2D()
tf_val = JointTransform2D(isTrain=False)
train_dataset = ImageToImage2D(images, masks_list, isTrain=True)
val_dataset = ImageToImage2D(test_img, test_mask, isTrain=False)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
valloader = DataLoader(val_dataset, 1, shuffle=False)

device = torch.device("cuda")

model = DVPT()
model.train()

print('load model')

sam_checkpoint = torch.load('path-to-sam-b-checkpoint')
model.load_state_dict(sam_checkpoint, strict=False)
model.image_encoder.requires_grad_(False)
model.image_encoder.dense_stem.requires_grad_(True)
model.pe_layer.requires_grad_(False)
for block in model.image_encoder.blocks:
    block.prompt_adapter.requires_grad_(True)

print('load optimizer')

encoder_optimizer = torch.optim.AdamW(model.image_encoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
decoder_optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
prompt_optimizer = torch.optim.AdamW(model.instance_prompt_encoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)

model.encoder_optimizer = encoder_optimizer
model.decoder_optimizer = decoder_optimizer
model.prompt_optimizer = prompt_optimizer
encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.encoder_optimizer, 1000, 1e-6)
decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.decoder_optimizer, 1000, 1e-6)
prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.prompt_optimizer, 1000, 1e-6)

model.to(device)
bestDice = 0

def cal_metrics(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    dice = (2 * intersection + smooth) / (union + smooth)
    iou = dice / (2 - dice)

    return iou, dice

print('start training')

for epoch in range(args.epochs):

    epoch_running_loss = 0
    train_dice = 0
    train_prompt_loss = 0
    train_no_dense_dice = 0
    train_no_sparse_dice = 0
    train_no_prompt_dice = 0

    for batch_idx, (X_batch, y_batch, img_original_size, mask_original_size, _) in tqdm(enumerate(dataloader)):
        
        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        loss = model(X_batch, y_batch, mask_original_size, epoch, True)

        epoch_running_loss += torch.mean(loss).item()
        dice, prompt_loss, no_dense_dice, no_sparse_dice, no_prompt_dice = model.cal_dice_loss(epoch)
        train_dice += (1 - dice.item())

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, current lr:{:.8f}'.format(epoch, args.epochs,
                                                                 epoch_running_loss / (batch_idx + 1),
                                                                 decoder_scheduler.get_lr()[0]))

    print('train dice:{:.4f}, prompt loss:{:.4f}, no dense dice:{:4f}, no sparse dice:{:4f}, no prompt dice:{:4f}'
          .format(train_dice / (batch_idx + 1),
                  train_prompt_loss / (batch_idx + 1), train_no_dense_dice / (batch_idx + 1),
                  train_no_sparse_dice / (batch_idx + 1), train_no_prompt_dice / (batch_idx + 1)))

    if ((epoch) % args.save_freq) == 0:

        validation_IOU = []
        mDice = []
        mAssd = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch, img_original_size, mask_original_size, name) in tqdm(enumerate(valloader)):

                X_batch = Variable(X_batch.to(device='cuda'))
                y_batch = Variable(y_batch.to(device='cuda'))
                y_batch = y_batch[..., : mask_original_size[1], : mask_original_size[0]]

                y_out = model.infer(X_batch, img_original_size)
                y_out = y_out[..., : mask_original_size[1], : mask_original_size[0]]

                tmp2 = y_batch.detach().cpu().numpy()
                tmp = y_out.detach().cpu().numpy()
                tmp = tmp[0][0]
                tmp2 = tmp2[0][0]

                vimask = np.zeros((mask_original_size[1], mask_original_size[0]), dtype=np.uint8)
                gt_mask = np.zeros((mask_original_size[1], mask_original_size[0]), dtype=np.uint8)
                mask_shape = np.full((mask_original_size[1], mask_original_size[0]), True)
                gt_shape = np.full((mask_original_size[1], mask_original_size[0]), True)

                vimask[tmp >= mask_threshold] = 1
                vimask[tmp < mask_threshold] = 0
                gt_mask[tmp2 != 0] = 1
                gt_mask[tmp2 == 0] = 0

                iou, dice = cal_metrics(vimask.flatten(), gt_mask.flatten())
                if dice > 0:
                    validation_IOU.append(iou)
                    mDice.append(dice)

                del X_batch, y_batch, tmp2, y_out

                vimask[vimask == 1] = 255
                gt_mask[gt_mask == 1] = 255

                fulldir = direc + "pred/{}/".format(epoch)
                if not os.path.isdir(fulldir):
                    os.makedirs(fulldir)

                cv2.imwrite(fulldir + name[0], vimask)

        if np.mean(mDice) > bestDice:
            bestDice = np.mean(mDice)
            path_direc = direc + "pred/{}/".format(epoch)
            if not os.path.isdir(path_direc):
                os.makedirs(path_direc)
            torch.save(model.state_dict(), path_direc + args.modelname + ".pth")
        print(np.mean(validation_IOU))
        print(np.mean(mDice))
        print('best dice: {}'.format(bestDice))

    encoder_scheduler.step()
    decoder_scheduler.step()
    prompt_scheduler.step()
