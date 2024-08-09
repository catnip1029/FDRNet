import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
# from .arch_util import to_2tuple, trunc_normal_
from einops import rearrange
import time


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 5, 1, 5//2, groups=hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = self.act(x) # B C H W
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1, 2 * window_size - 1, self.n_heads
            )
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(
            h,
            w,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(
            attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        B, H, W, C = x.shape
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")
        # Using Attn Mask to distinguish different subwindows.
        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )

        if pad_r > 0 or pad_b > 0:
            output = output[:, :H, :W, :].contiguous()

        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array(
                [
                    [i, j]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                ]
            )
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class SwinBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
        input_resolution=None,
    ):
        """ SwinTransformer Block
        """
        super(SwinBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        print(
            "Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path)
        )
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class FirstOctaveTransformer(nn.Module):
    def __init__(
        self,
        dim,
        alpha,
        window_size,
        mlp_ratio=2,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(FirstOctaveTransformer, self).__init__()
        l_dim = int(alpha * dim)
        h_dim = dim - int(alpha * dim)
        head_dim = 30
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2h = WMSA(
            h_dim, h_dim, head_dim=head_dim, window_size=window_size, type="W"
        )
        # self.h2l = nn.Conv2d(
        #     h_dim, l_dim, kernel_size=3, stride=1, padding=1, groups=l_dim
        # )
        self.h2l = WMSA(l_dim, l_dim, head_dim=head_dim, window_size=window_size, type='W')
        self.linear_l = nn.Linear(h_dim, l_dim)
        self.linear = nn.Linear(h_dim, h_dim)
        self.norm_l1 = norm_layer(l_dim)
        self.norm_l2 = norm_layer(l_dim)
        self.norm_h1 = norm_layer(h_dim)
        self.norm_h2 = norm_layer(h_dim)
        self.mlp_l = Mlp(l_dim, int(l_dim * mlp_ratio), l_dim)
        self.mlp_h = Mlp(h_dim, int(h_dim * mlp_ratio), h_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_l = self.downsample(x)
        x_h = Rearrange("b c h w -> b h w c")(x)
        x_l = Rearrange("b c h w -> b h w c")(x_l)
        x_h = self.linear(x_h)
        x_l = self.linear_l(x_l)
        x_h = x_h + self.drop_path(self.h2h(self.norm_h1(x_h)))
        x_h = x_h + self.drop_path(self.mlp_h(self.norm_h2(x_h)))

        x_l = x_l + self.drop_path(self.h2l(self.norm_l1(x_l)))
        x_l = x_l + self.drop_path(self.mlp_l(self.norm_l2(x_l)))
        x_h = Rearrange("b h w c -> b c h w")(x_h)
        x_l = Rearrange("b h w c -> b c h w")(x_l)
        return x_h, x_l


class LastOctaveTransformer(nn.Module):
    def __init__(
        self,
        dim,
        alpha,
        window_size,
        mlp_ratio=2,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(LastOctaveTransformer, self).__init__()
        l_dim = int(alpha * dim)
        h_dim = dim - int(alpha * dim)
        head_dim = 30
        self.upsample = nn.ConvTranspose2d(l_dim, h_dim, 2, 2, 0, bias=False)
        self.l2h = WMSA(
            l_dim, l_dim, head_dim=head_dim, window_size=window_size, type="SW"
        )
        self.linear = nn.Linear(h_dim, h_dim)
        self.h2h = WMSA(
            h_dim, h_dim, head_dim=head_dim, window_size=window_size, type="SW"
        )
        self.norm_l1 = norm_layer(l_dim)
        self.norm_h1 = norm_layer(h_dim)
        self.norm_h2 = norm_layer(h_dim)
        self.mlp_h = Mlp(h_dim, int(h_dim * mlp_ratio), h_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_h, x_l = x
        x_h2h = Rearrange("b c h w -> b h w c")(x_h)
        x_l2h = Rearrange("b c h w -> b h w c")(x_l)
        x_h2h = x_h2h + self.drop_path(self.h2h(self.norm_h1(x_h2h)))
        x_l2h = x_l2h + self.drop_path(self.l2h(self.norm_l1(x_l2h)))
        x_l2h = rearrange(
            self.upsample(rearrange(x_l2h, "b h w c -> b c h w")), "b c h w -> b h w c"
        )
        x = x_h2h + x_l2h
        x = x + self.drop_path(self.mlp_h(self.norm_h2(x)))
        x = Rearrange("b h w c -> b c h w")(x)
        return x


class OctaveTransformer(nn.Module):
    def __init__(
        self,
        dim,
        alpha,
        window_size,
        shift_size,
        mlp_ratio=2,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(OctaveTransformer, self).__init__()
        l_dim = int(alpha * dim)
        h_dim = dim - int(alpha * dim)
        head_dim = 30
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.ConvTranspose2d(l_dim, h_dim, 2, 2, 0, bias=False)
        # self.l2l = nn.Conv2d(
        #     l_dim, l_dim, kernel_size=3, stride=1, padding=1, groups=l_dim
        # )
        self.l2l = WMSA(l_dim, l_dim, head_dim=head_dim, window_size=window_size, type="W" if shift_size == 0 else "SW")
        self.l2h = WMSA(
            l_dim,
            l_dim,
            head_dim=head_dim,
            window_size=window_size,
            type="W" if shift_size == 0 else "SW",
        )
        self.h2h = WMSA(
            h_dim,
            h_dim,
            head_dim=head_dim,
            window_size=window_size,
            type="W" if shift_size == 0 else "SW",
        )
        # print("W") if shift_size == 0 else print("SW")
        # self.h2l = nn.Conv2d(
        #     h_dim, l_dim, kernel_size=3, stride=1, padding=1, groups=l_dim
        # )
        self.h2l = WMSA(l_dim, l_dim, head_dim=head_dim, window_size=window_size, type="W" if shift_size == 0 else "SW")
        self.linear_1 = nn.Linear(h_dim, l_dim)
        self.norm_l1 = norm_layer(l_dim)
        self.norm_l2 = norm_layer(l_dim)
        self.norm_h1 = norm_layer(h_dim)
        self.norm_h2 = norm_layer(h_dim)
        self.mlp_l = Mlp(l_dim, int(l_dim * mlp_ratio), l_dim)
        self.mlp_h = Mlp(h_dim, int(h_dim * mlp_ratio), h_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_h, x_l = x
        x_h2l = self.linear_1(rearrange(self.downsample(x_h), "b c h w -> b h w c"))
        x_h = Rearrange("b c h w -> b h w c")(x_h)
        x_l = Rearrange("b c h w -> b h w c")(x_l)
        x_h2h = x_h + self.drop_path(self.h2h(self.norm_h1(x_h)))
        x_l2l = x_l + self.drop_path(self.l2l(self.norm_l1(x_l)))

        x_l2h = x_l + self.drop_path(self.l2h(self.norm_l1(x_l)))
        x_l2h = rearrange(
            self.upsample(rearrange(x_l2h, "b h w c -> b c h w")), "b c h w -> b h w c"
        )

        x_h2l = x_h2l + self.drop_path(self.h2l(self.norm_l1(x_h2l)))

        x_h = x_h2h + x_l2h
        x_l = x_l2l + x_h2l
        x_h = x_h + self.drop_path(self.mlp_h(self.norm_h2(x_h)))
        x_l = x_l + self.drop_path(self.mlp_l(self.norm_l2(x_l)))

        x_h = Rearrange("b h w c -> b c h w")(x_h)
        x_l = Rearrange("b h w c -> b c h w")(x_l)

        return x_h, x_l


class BasicLayer(nn.Module):
    def __init__(
        self,
        input_resolution,
        dim,
        alpha,
        depth,
        window_size,
        mlp_ratio=2.0,
        drop_path=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.input_resolution = input_resolution
        h_dim = dim - int(alpha * dim)
        # build blocks
        self.blocks = nn.ModuleList()
        first_block = FirstOctaveTransformer(
            dim=dim,
            alpha=alpha,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path[0],
        )
        self.blocks.append(first_block)
        for i_block in range(depth-2):
            block = OctaveTransformer(
                dim=dim,
                alpha=alpha,
                window_size=window_size,
                shift_size=window_size // 2 if (i_block%2) == 0 else 0,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i_block+1]
                if isinstance(drop_path, list)
                else drop_path,
            )
            self.blocks.append(block)
        last_block = LastOctaveTransformer(
            dim=dim,
            alpha=alpha,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path[-1]
        )
        self.blocks.append(last_block)

        # self.dwconv = nn.Conv2d(h_dim, h_dim, kernel_size=5, stride=1,padding=2,groups=h_dim)
        # self.pointwise = nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        shortcut = x
        for block in self.blocks:
            x = block(x)
        # x = self.pointwise(self.dwconv(x))
        x = self.conv(x)
        x = shortcut + x
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

@ARCH_REGISTRY.register()
class OctaveIRRes(nn.Module):
    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        alpha=0.2,
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        upscale=2,
        img_range=1.0,
        upsampler="",
        **kwargs,
    ):
        super(OctaveIRRes, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        octave_dim=180
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio


        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        # build octave trasformer block
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                input_resolution=img_size,
                dim=octave_dim,
                depth=depths[i_layer],
                alpha=alpha,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            )
            self.layers.append(layer)

        self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim*self.num_layers, octave_dim, 1), nn.LeakyReLU(inplace=True))

        # for classical SR
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(octave_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        res = F.interpolate(x, scale_factor=self.upscale, mode="bicubic",align_corners=False)
        x = self.conv_first(x)
        dense = []
        for layer in self.layers:
            x = layer(x)
            dense.append(x)

        x = self.conv_after_body(torch.cat(dense, dim=1))

        x = self.conv_last(self.upsample(self.conv_before_upsample(x))) + res
        x = x / self.img_range + self.mean

        return x


# if __name__ == "__main__":
#     x = torch.randn(1, 48 * 48, 64)
    # y = torch.randn(1, 24*24, 48)
    # ot = LastOctaveTransformer(240, (48, 48), 0.2, 8, 0, 2)
    # x= ot((x, y), (48, 48))
    # print(x.shape)
    # dpr = [x.item() for x in torch.linspace(0, 0.1, 6)]
    # print(dpr)
    # print(dpr[-1])
    # from thop import profile
    # model = OctaveIRRes(upscale=4, img_size=(48, 48),patch_size=1,
    #                window_size=8, img_range=1., depths=[6,6,6,6, 6,6,6],alpha=0.286,
    #                embed_dim=160, mlp_ratio=2, upsampler='pixelshuffle')
    # x = torch.randn((1, 3, 56, 56))
    # x = model(x)
    # print(x.shape)
    # x = torch.randn((1, 3, 1280 // 2, 720 // 2))
    # print('==> Building model..')

    # # dummy_input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, (x,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # basic = BasicLayer(64, (48, 48), 3, 1, 8, 2)
    # x = basic(x, (48, 48))
    # print(x.shape)
    # sb = SwinBlock(192, 192, 32, 8, 0.1, input_resolution=48)
    # x = torch.randn(1, 56*56, 256)
    # bs = BasicLayer(256, 0.25, 4, 8, drop_path=[0.1, 0.1,0.1,0.1])
    # # x = bs(x)
    # print(x.shape)
def measure_inference_speed(model, data, max_iter=100, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    net = OctaveIRRes(upscale=4, img_size=(48, 48),patch_size=1,
                   window_size=16, img_range=1., depths=[6,6,6,6,6,6],alpha=0.223,
                   embed_dim=140, mlp_ratio=2, upsampler='pixelshuffle')
    net = net.to(device)


    data = [torch.rand(1, 3, 320, 180).to(device)]

    fps = measure_inference_speed(net, data)
    print('fps:', fps)
    from thop import profile
    x = torch.randn(1, 3, 320, 180).to(device)
    flops, params = profile(net, inputs=(x,))
    print('FLOPs:{:.2f}G'.format(flops / 1e9))
    print(params)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # print(device)

	# # sample 2
    # # net = UNetSeeInDark()
    # # net = net.to(device)

    # data = [torch.rand(1, 4, 256, 256).to(device)]
    # fps = measure_inference_speed(net, data)
    # print('fps:', fps)

    # gpu : 32ms  cpu: 350ms
    # summary(net, input_size=(1, 4, 400, 400), col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, (4, 400, 400), verbose=True, print_per_layer_stat=True)
    # print(macs, params)
    # net = Network()



# if __name__ == '__main__':
#     from thop import profile
#     x = torch.randn(1, 3, 256, 256)
#     model = Network()
#     # import time
#     # start = time.time()
#     # result = model(x)
#     # print(result)
#     # end = time.time()
#     # t = end - start
#     # print(t)
#     flops, params = profile(model, inputs=(x,))
#     print('FLOPs:{:.2f}G'.format(flops / 1e9))