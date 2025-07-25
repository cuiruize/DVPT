import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_prompt_image_encoder import VisualPromptImageEncoder
from .instance_prompt_encoder import InstancePromptEncoder
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer

logger = logging.getLogger(__name__)
from typing import Any, Optional, Tuple


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = bce1(pred, gt)
        # loss = w_neg * bce1(pred, gt)

        return loss


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        # pred = F.softmax(pred, dim=0)
        target = torch.argmax(gt, dim=0)
        loss = self.celoss(pred.unsqueeze(0), target.unsqueeze(0))

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):

        smooth = 1e-5
        intersect = torch.sum(score * target)
        # y_sum = torch.sum(target * target)
        # z_sum = torch.sum(score * score)
        # dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        union = torch.sum(score + target)
        dice = (2 * intersect + smooth) / (union + smooth)
        loss = 1 - dice
        return loss

    def forward(self, pred, gt, smooth=1e-5):
        bs = pred.shape[0]
        #
        pred = torch.sigmoid(pred).view(bs, -1)
        gt = gt.contiguous().view(bs, -1)
        #
        intersect = (pred * gt).sum(-1)
        union = (pred + gt).sum(-1)
        dice = (2 * intersect + smooth) / (union + smooth)
        loss = 1 - dice
        #
        return loss

        # multi-organ
        # pred = F.softmax(pred, dim=0)
        # class_wise_dice = []
        # loss = 0.0
        # for i in range(0, 9):
        #     dice = self._dice_loss(pred[i], gt[i])
        #     class_wise_dice.append(1.0 - dice.item())
        #     loss += dice
        # return loss / 9


class PromptLoss(nn.Module):
    def __init__(self, expectation=0.2, penalty_co=1):
        super(PromptLoss, self).__init__()
        self.penalty_co = penalty_co
        self.expectation = expectation
        self.bias = -1 / (penalty_co * (expectation + 1))

    def forward(self, prompt_pred, contrast_pred, gt):
        prompt_dice = self.dsc(prompt_pred, gt)
        no_dense_dice = self.dsc(contrast_pred[0], gt)
        no_sparse_dice = self.dsc(contrast_pred[1], gt)
        no_prompt_dice = self.dsc(contrast_pred[2], gt)

        p_dense = self.penalty(prompt_dice - no_dense_dice)
        p_sparse = self.penalty(prompt_dice - no_sparse_dice)
        p_prompt = self.penalty(prompt_dice - no_prompt_dice)

        # p_final = torch.mean(0.5 * p_prompt + 0.1 * p_dense + 0.4 * p_sparse)
        p_final = torch.mean(1.5 * p_prompt)
        # print('dense penalty:{:.6f}, sparse penalty:{:.6f}'.format(torch.mean(p_dense).item(), torch.mean(p_sparse).item()))

        return p_final

    def dsc(self, m1, gt, smooth=1e-5):
        bs = m1.shape[0]
        m1 = torch.sigmoid(m1).view(bs, -1)
        gt = gt.contiguous().view(bs, -1)

        intersect = (m1 * gt).sum(-1).sum()
        union = (m1 + gt).sum(-1).sum()
        dice = (2 * intersect + smooth) / (union + smooth)

        return dice

    def penalty(self, dice_diff):
        if dice_diff >= self.expectation:
            p = torch.zeros_like(dice_diff)
        else:
            p = 1 / (self.penalty_co * (dice_diff + 1)) + self.bias
        return p


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


class DVPT(nn.Module):
    def __init__(self, inp_size=1024, encoder_mode=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = 768
        # Sam-b
        self.image_encoder = VisualPromptImageEncoder(
            img_size=inp_size,
            patch_size=16,
            in_chans=3,
            embed_dim=self.embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            out_chans=256,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=[2, 5, 8, 11],
        )
        self.prompt_embed_dim = 256
        self.instance_prompt_encoder = InstancePromptEncoder(
            encoder_depth=4,
            embedding_dim=self.embed_dim,
            prompt_dim=256,
            num_heads=8,
            mask_in_chans=8,
            attention_downsample_rate=2,
            num_tokens_per_propmt=32,
            channel_dim=64 * 64,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # if 'evp' in encoder_mode['name']:
        #     for k, p in self.encoder.named_parameters():
        #         if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
        #             p.requires_grad = False

        # self.loss_mode = loss
        # if self.loss_mode == 'bce':
        #     self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        #
        # elif self.loss_mode == 'bbce':
        #     self.criterionBCE = BBCEWithLogitLoss()
        #
        # elif self.loss_mode == 'iou':
        #     self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        #     self.criterionIOU = IOU()
        self.criterionDICE = DiceLoss()
        self.criterionBBCE = BBCEWithLogitLoss()
        self.criterionCE = CELoss()
        self.critetionPROMPT = PromptLoss()

        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.inp_size = 256
        self.image_embedding_size = inp_size // 16
        self.no_mask_embed = nn.Embedding(1, 256)

    def set_input(self, input, gt_mask, original_size):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.inp_size_w = original_size[0]
        self.inp_size_h = original_size[1]

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    # def set_high_freq_input(self, img, original_size):
    #     hf_img = torch.zeros_like(img)
    #     for i in range(img.shape[0]):
    #         original_image = img[i][..., : original_size[1][i], : original_size[0][i]]
    #         hf_img[i][..., : original_size[1][i], : original_size[0][i]] = self.edge_enhancement(original_image)
    #     hf_img = F.interpolate(hf_img, scale_factor=0.25, mode='bilinear', align_corners=True)
    #     self.hf_input = hf_img.to(self.device)

    # def edge_enhancement(self, x):
    #     # x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
    #     mask = torch.zeros(x.shape).to(x.device)
    #     w, h = x.shape[-2:]
    #     line = 45
    #     mask[:, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 0
    #
    #     fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    #
    #     fft = fft * mask
    #     fr = fft.real
    #     fi = fft.imag
    #
    #     fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    #     inv = torch.fft.ifft2(fft_hires, norm="forward").real
    #
    #     inv = torch.abs(inv)
    #     eq = torchvision.transforms.RandomEqualize(p=0.5)
    #     inv = eq(inv)
    #
    #     return inv

    def train_forward(self):
        features, features_k, img_embed, trans_layer_feat = self.image_encoder(self.input)

        # Embed instance prompts
        sparse_embeddings = self.instance_prompt_encoder(self.input, trans_layer_feat)
        # dense_embeddings = self.instance_prompt_encoder.get_dense_prompt()

        # Embed prompts
        # zero_sparse_embeddings = torch.empty((features.shape[0], 0, self.prompt_embed_dim),
        #                                      device=self.input.device)
        zero_dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            features.shape[0], -1, self.image_embedding_size, self.image_embedding_size
        )

        # Predict masks
        # low_res_masks, iou_predictions = self.mask_decoder(
        #     image_embeddings=features,
        #     # image_pe=self.get_dense_pe(),
        #     image_pe=features_k,
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=False,
        # )

        low_res_masks = self.mask_decoder(
            image_embeddings=features,
            # image_pe=self.get_dense_pe(),
            image_pe=features_k,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=zero_dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution

        # masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)

        self.pred_mask = low_res_masks
        # ablation_masks = torch.cat(
        #     (no_dense_mask.unsqueeze(1), no_sparse_mask.unsqueeze(1), no_prompt_mask.unsqueeze(1)), dim=1)
        # self.ablation_masks = ablation_masks

    def infer(self, input, original_size):
        features, features_k, img_embed, trans_layer_feat = self.image_encoder(input)

        # Predict masks
        # self.set_high_freq_input(input, original_size)
        sparse_embeddings = self.instance_prompt_encoder(input, trans_layer_feat)
        # zero_sparse_embeddings = torch.empty((features.shape[0], 128, self.prompt_embed_dim),
        #                                      device=self.input.device)
        zero_dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            features.shape[0], -1, self.image_embedding_size, self.image_embedding_size
        )
        # dense_embeddings = self.instance_prompt_encoder.get_dense_prompt()
        low_res_masks = self.mask_decoder(
            image_embeddings=features,
            # image_pe=self.get_dense_pe(),
            image_pe=features_k,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=zero_dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        # _, indices = torch.sort(iou_predictions, descending=True, dim=1)
        # masks = self.postprocess_masks(low_res_masks[0][indices[0][0]], self.inp_size, self.inp_size)

        # masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)

        return low_res_masks

    def forward(self, X_batch, y_batch, mask_size, epoch, update=False):
        self.set_input(X_batch, y_batch, mask_size)
        self.optimize_parameters(epoch)

        # gradient accumulation
        # self.train_forward()
        # self.loss = 0
        # for i in range(self.pred_mask.shape[0]):
        #     pred = self.pred_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
        #     gt = self.gt_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
        #     self.loss += (self.criterionDICE(pred, gt) + self.criterionBBCE(pred, gt))
        #
        # # self.loss = self.loss / self.pred_mask.shape[0]
        # self.loss /= 4
        #
        # self.loss.backward()
        # if update:
        #     # self.encoder_optimizer.step()
        #     self.decoder_optimizer.step()
        #     self.prompt_optimizer.step()
        #     # self.encoder_optimizer.zero_grad()
        #     self.decoder_optimizer.zero_grad()
        #     self.prompt_optimizer.zero_grad()
        return self.loss

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.
        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.
        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            size=[self.image_encoder.img_size, self.image_encoder.img_size],
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # self.loss = self.criterionBBCE(self.pred_mask, self.gt_mask)
        # if self.loss_mode == 'iou':
        #     self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
        self.loss = 0
        for i in range(self.pred_mask.shape[0]):
            pred = self.pred_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            gt = self.gt_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            self.loss += (self.criterionDICE(pred, gt) + self.criterionBBCE(pred, gt))

        self.loss = self.loss / self.pred_mask.shape[0]

        self.loss.backward()

    def backward_prompt(self):
        self.prompt_loss = 0
        for i in range(self.pred_mask.shape[0]):
            pred = self.pred_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            gt = self.gt_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            contrast_pred = self.ablation_masks[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            self.prompt_loss += self.critetionPROMPT(pred, contrast_pred, gt)

        if self.prompt_loss != 0:
            self.prompt_loss = self.prompt_loss / self.pred_mask.shape[0]
            self.prompt_loss.backward()

    def optimize_parameters(self, epoch_num):
        self.train_forward()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.prompt_optimizer.zero_grad()
        self.backward_G()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.prompt_optimizer.step()

    @torch.no_grad()
    def cal_dice_loss(self, epoch_num):
        dice = 0
        sparse_dice = 0
        dense_dice = 0
        no_prompt_dice = 0
        prompt_loss = 0
        for i in range(self.pred_mask.shape[0]):
            pred = self.pred_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            gt = self.gt_mask[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            dice += self.criterionDICE(pred, gt)
            # contrast_pred = self.ablation_masks[i][..., : self.inp_size_h[i], : self.inp_size_w[i]]
            #
            # prompt_loss += self.critetionPROMPT(pred, contrast_pred, gt)
            #
            # sparse_dice += self.criterionDICE(contrast_pred[0], gt)
            # dense_dice += self.criterionDICE(contrast_pred[1], gt)
            # no_prompt_dice += self.criterionDICE(contrast_pred[2], gt)
        dice = dice / self.pred_mask.shape[0]
        prompt_loss = prompt_loss / self.pred_mask.shape[0]
        sparse_dice = sparse_dice / self.pred_mask.shape[0]
        dense_dice = dense_dice / self.pred_mask.shape[0]
        no_prompt_dice = no_prompt_dice / self.pred_mask.shape[0]

        return dice, prompt_loss, sparse_dice, dense_dice, no_prompt_dice
