import warnings
from typing import Type

import torch
from torch import nn
from torch.nn.init import trunc_normal_
import math
from segment_anything.ops.modules import MSDeformAttn
from functools import partial


from segment_anything.modeling.transformer import Attention


class FFN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class InstancePromptEncoder(nn.Module):
    def __init__(self,
                 encoder_depth: int,
                 embedding_dim: int,
                 prompt_dim: int,
                 num_heads: int,
                 mask_in_chans: int,
                 attention_downsample_rate: int = 2,
                 num_tokens_per_propmt: int = 16,
                 channel_dim: int = 64 * 64,
                 deform_ratio: float = 1.0
                 ):
        super(InstancePromptEncoder, self).__init__()

        self.tokens_to_feature_attns = nn.ModuleList()
        self.feature_projs = nn.ModuleList()
        self.prompt_aggregations = nn.ModuleList()
        for i in range(encoder_depth):
            attn_layer = Attention(prompt_dim, num_heads=num_heads, downsample_rate=attention_downsample_rate)
            self.tokens_to_feature_attns.append(attn_layer)
            feature_proj = nn.Sequential(
                nn.Linear(embedding_dim, prompt_dim),
                nn.GELU()
            )
            self.feature_projs.append(feature_proj)
            temp = nn.Linear(prompt_dim, prompt_dim)
            self.prompt_aggregations.append(temp)

        self.layer_norm = nn.LayerNorm(prompt_dim)
        self.prompt_final_channels = encoder_depth * num_tokens_per_propmt
        self.prompt_token = nn.Embedding(self.prompt_final_channels, prompt_dim)

        self.prompt_final_ffn = FFN(prompt_dim, 2048, nn.GELU)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_dense_prompt(self):
        return self.dense_prompt

    def forward(self, image_input, encoder_features):
        B, ED, H, W = encoder_features[0].shape

        # self.dense_prompt = self.dense_stem(image_input)

        # test
        tokens = self.prompt_token.weight.expand(B, -1, -1)

        sparse_prompts = torch.zeros_like(tokens)

        for i in range(len(encoder_features)):
            feature = encoder_features[i]
            # deform_input = deform_inputs(feature)
            v = self.layer_norm(self.feature_projs[i](feature.permute(0, 2, 3, 1)).view(B, H*W, -1))
            # k = self.feature_k_projs[i](v)
            k = v
            q = self.layer_norm(tokens[:, 32 * i:32 * (i + 1), :])

            attn = self.tokens_to_feature_attns[i](q=q, k=k, v=v)
            q = self.layer_norm(q + attn)

            sparse_prompts[:, 32 * i:32 * (i + 1), :] = q
            # sparse_prompts[:, :32*(i+1), :] = self.prompt_aggregations[i](sparse_prompts[:, :32*(i+1), :])

        ffn_output = self.prompt_final_ffn(sparse_prompts)
        sparse_prompts = sparse_prompts + ffn_output

        return sparse_prompts

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    # spatial_shapes = torch.as_tensor([(h // 8, w // 8),
    #                                   (h // 16, w // 16),
    #                                   (h // 32, w // 32)],
    #                                  dtype=torch.long, device=x.device)
    # level_start_index = torch.cat((spatial_shapes.new_zeros(
    #     (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    # deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs2

