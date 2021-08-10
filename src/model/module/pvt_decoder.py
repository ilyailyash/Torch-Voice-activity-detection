
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from src.model.module.conv2d import ConvBlock
from src.model.module.pvt import Block

# __all__ = [
#     'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
# ]


class DePatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, spec_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        spec_size = to_2tuple(spec_size)
        patch_size = to_2tuple(patch_size)

        self.spec_size = spec_size
        self.patch_size = patch_size
        self.H, self.W = spec_size[0] * patch_size[0], spec_size[1] * patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H * self.patch_size[0], W * self.patch_size[1]

        return x, (H, W)


class DePyramidVisionTransformer(nn.Module):
    def __init__(self, feature_size=224,
                 patch_size=[1, (2,1), 2, (8, 2)],
                 num_classes=1000,
                 embed_dims=[512, 256, 128, 64],
                 num_heads=[8, 4, 2, 1],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 6, 4, 3],
                 sr_ratios=[1, 2, 4, 8],
                 conv_cfg=None,
                 att_vis=False):
        super().__init__()
        feature_size = torch.tensor(feature_size)
        patch_size_t = torch.cumprod(torch.tensor(patch_size), dim=0)

        self.num_classes = num_classes
        self.depths = depths

        #conv
        self.conv = ConvBlock(**conv_cfg)


        # patch_embed
        self.patch_embed1 = DePatchEmbed(spec_size=feature_size, patch_size=patch_size[0], in_chans=embed_dims[0],
                                         embed_dim=embed_dims[1])
        self.patch_embed2 = DePatchEmbed(spec_size=feature_size * patch_size_t[0], patch_size=patch_size[1], in_chans=embed_dims[1]*2,
                                         embed_dim=embed_dims[2])
        self.patch_embed3 = DePatchEmbed(spec_size=feature_size * patch_size_t[1], patch_size=patch_size[2], in_chans=embed_dims[2]*2,
                                         embed_dim=embed_dims[3])
        self.patch_embed4 = DePatchEmbed(spec_size=feature_size * patch_size_t[2], patch_size=patch_size[3], in_chans=embed_dims[3]*2,
                                         embed_dim=conv_cfg["in_channels"] // 2)

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[1]*2))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[2]*2))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[3]*2))
        self.pos_drop3 = nn.Dropout(p=drop_rate)


        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[1]*2, num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], att_vis=att_vis)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[2]*2, num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], att_vis=att_vis)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[3]*2, num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], att_vis=att_vis)
            for i in range(depths[2])])

        self.norm1 = norm_layer(embed_dims[1]*2)
        self.norm2 = norm_layer(embed_dims[2]*2)
        self.norm3 = norm_layer(embed_dims[3]*2)


        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, encoder_output):
        B = encoder_output[0].shape[0]
        attentions = []

        # stage 1
        x, (H, W) = self.patch_embed1(encoder_output[0])
        x = torch.cat((x, encoder_output[1]), dim=-1)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x, mask_weights = blk(x, H, W)
            attentions.append(mask_weights)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        x = torch.cat((x, encoder_output[2]), dim=-1)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x, mask_weights = blk(x, H, W)
            attentions.append(mask_weights)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        x = torch.cat((x, encoder_output[3]), dim=-1)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x, mask_weights = blk(x, H, W)
            attentions.append(mask_weights)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.cat((x, encoder_output[4]), dim=1)
        x = self.conv(x)
        return x, attentions


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict



if __name__ == '__main__':
    f, t = 256, 128
    inp = torch.ones(16, 12, f, t).to(1)

    model = DePyramidVisionTransformer(feature_size=torch.tensor([f, t]), patch_size=[[8,2], [2,2], [2,1], [1,1]],
                                       conv_cfg={"in_channels": 12, "out_channels": 16})
    model.to(1)
    output, attentions = model(inp)

    print(123)