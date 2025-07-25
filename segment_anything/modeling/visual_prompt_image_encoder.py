import math

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
from torchvision import transforms

from segment_anything.modeling.image_encoder import ImageEncoderViT, Block
from segment_anything.modeling.transformer import Attention
from typing import Type, Tuple


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


class VisualPromptImageEncoder(ImageEncoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = kwargs.get('patch_size')
        self.embed_dim = kwargs.get('embed_dim')
        self.global_index = kwargs.get('global_attn_indexes')

        self.feature_dim = kwargs.get('img_size') // kwargs.get('patch_size')

        self.token_post_process = nn.Sequential(
            nn.ConvTranspose2d(32, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dense_stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # nn.Conv2d(16, 32, kernel_size=1),
        )

        self.blocks = nn.ModuleList()
        for i in range(kwargs.get('depth')):
            block = AdapterBlock(
                dim=self.embed_dim,
                num_heads=kwargs.get('num_heads'),
                mlp_ratio=kwargs.get('mlp_ratio'),
                qkv_bias=kwargs.get('qkv_bias'),
                norm_layer=kwargs.get('norm_layer'),
                act_layer=kwargs.get('act_layer'),
                use_rel_pos=kwargs.get('use_rel_pos'),
                rel_pos_zero_init=kwargs.get('rel_pos_zero_init'),
                window_size=kwargs.get('window_size') if i not in kwargs.get('global_attn_indexes') else 0,
                input_size=(self.img_size // self.patch_size, self.img_size // self.patch_size),
            )
            self.blocks.append(block)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    def embeddings(self, x):
        # ultra sound
        # x = torch.repeat_interleave(x, 3, dim=1)
        # x = self.new_patch_embed(x).permute(0, 2, 3, 1)
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        return x

    def forward(self, x):
        task_token = self.dense_stem(x).permute(0, 2, 3, 1)
        B, H, W, C = task_token.shape
        task_token = task_token.view(B, H*W, C)
        embedding = self.embeddings(x)
        layer_features = []

        for i, blk in enumerate(self.blocks):
            out, task_token = blk(embedding, task_token)
            # out = blk(embedding)
            if i in self.global_index:
                layer_features.append(out.permute(0, 3, 1, 2))

        out = self.neck(out.permute(0, 3, 1, 2))

        task_token = self.token_post_process(task_token.view(B, H, W, C).permute(0, 3, 1, 2))
        out = out + task_token

        out_k = out

        return out, out_k, embedding, layer_features


class PromptAdapter(nn.Module):

    def __init__(self, scale_factor, embed_dim, num_prompt_tokens):
        super().__init__()

        # self.layer_visual_prompt = nn.Parameter(
        #     torch.zeros(1, num_prompt_tokens, embed_dim // scale_factor))

        # self.prompt_key_projection = nn.Linear(embed_dim // scale_factor, embed_dim // scale_factor)
        # self.feature_key_projection = nn.Linear(embed_dim // scale_factor, embed_dim // scale_factor)

        self.q_norm = nn.LayerNorm(embed_dim // scale_factor)
        self.v_norm = nn.LayerNorm(embed_dim // scale_factor)

        self.prompt_tune = nn.Sequential(
            nn.Linear(embed_dim // scale_factor, embed_dim // scale_factor),
            nn.GELU()
        )

        self.tuned_feature_ffn = FFN(embed_dim // scale_factor, 2048, nn.GELU)

        self.layer_norm = nn.LayerNorm(embed_dim // scale_factor)

        self.cross_attn_prompt_to_feature = Attention(embed_dim // scale_factor, num_heads=8, downsample_rate=1)
        self.ffn = FFN(embed_dim // scale_factor, 2048, nn.GELU)
        self.cross_attn_feature_to_prompt = Attention(embed_dim // scale_factor, num_heads=8, downsample_rate=1)

        self.down = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // scale_factor),
            nn.GELU()
        )

        self.up = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim // scale_factor, embed_dim)
        )

        self.apply(self._init_weights)

    def forward(self, embedding, task_specific_token):
        B, H, W, E = embedding.shape
        q = self.q_norm(self.down(embedding.view(B, H * W, E)))
        # feature_k = q

        # v = self.layer_visual_prompt.expand(B, -1, -1)
        v = self.v_norm(self.prompt_tune(task_specific_token))
        prompt_k = v

        # attn1 = self.cross_attn_prompt_to_feature(q=v, k=feature_k, v=q)
        # v = self.norm1(v + attn1)
        # mlp_out = self.ffn(v)
        # v = self.norm2(v + mlp_out)
        attn_out = self.cross_attn_feature_to_prompt(q=q, k=prompt_k, v=v)
        out = self.layer_norm(q + attn_out)
        out = out + self.tuned_feature_ffn(out)

        out = self.up(out).view(B, H, W, E)

        out = out + embedding

        return out, v

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.data)


class AdapterBlock(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scale_factor = 24
        self.num_prompt_tokens = 32
        self.embed_dim = kwargs.get('dim')

        self.prompt_adapter = PromptAdapter(self.scale_factor, self.embed_dim, self.num_prompt_tokens)

    def forward(self, x: torch.Tensor, task_token) -> torch.Tensor:
        x = super().forward(x)
        x, task_token = self.prompt_adapter(x, task_token)

        return x, task_token
