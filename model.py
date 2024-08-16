
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_c=128, embed_dim=256, norm_layer=None):
        super().__init__()
        self.proj = nn.Linear(in_c, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, adj_mask):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.ones_like(attn) * (-1e9)
        _adj_mask = adj_mask.unsqueeze(1)
        _adj_mask = _adj_mask.repeat(1, self.num_heads, 1, 1)
        attn = torch.where(_adj_mask > 0, attn, mask)
        attn = attn.softmax(dim=-1)
        _attn = attn.clone()
        _attn[attn == 1/attn.shape[-1]] = 0
        attn = self.attn_drop(_attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, args):
        x, adj_mask = args
        x = x + self.drop_path(self.attn(self.norm1(x), adj_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, adj_mask


class NlgtEncoder(nn.Module):
    def __init__(self, in_c=128, num_classes=40, embed_dim=256, depth=6,
                 num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
                 representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer=None):


        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(in_c=in_c, embed_dim=embed_dim)
        self.label_token = nn.Parameter(torch.zeros(41, 128))
        self.cls_token = nn.Parameter(torch.zeros(1, 128))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.label_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x, label, adj_mask):
        # [B, N, C] -> [B, N, embed_dim]
        x = self.patch_embed(x)
        res_x = []
        for _x, _l in zip(x, label):
            l_emb = torch.cat([self.cls_token, self.label_token[_l.tolist()][1:]])
            res_x.append(_x + l_emb)
            # res_x.append(torch.cat([_x, l_emb], axis=1))
        x = torch.stack(res_x, dim=0)
        # x = self.patch_embed(x)  # [B, 168, 256]
        # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        x, adj_mask = self.blocks([x, adj_mask])
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def NLGT(emb_dim=256, in_c=256, num_class=40, depth=4, num_heads=8):
    model = NlgtEncoder(  in_c=in_c,
                          drop_ratio=0.1,
                          mlp_ratio=4.0,
                          embed_dim=emb_dim,
                          depth=depth,
                          num_heads=num_heads,
                          representation_size=None,
                          num_classes=num_class)
    return model



