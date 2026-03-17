from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class LGTExpertScale(nn.Module):
    def __init__(self, visual_width:int, visual_layers: int, visual_head: int, attn_window: int, base_length: int, scale: int):
        super().__init__()
        self.D   = visual_width
        self.s   = int(scale)
        self.Ts  = math.ceil(base_length / self.s)
        self.win = max(1, attn_window // self.s)
        self.attn_window = attn_window

        self.frame_position_embeddings = nn.Embedding(self.Ts, self.D)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head
        )

        half = self.D // 2
        self.gc1 = GraphConvolution(self.D,  half, residual=True)
        self.gc2 = GraphConvolution(half,  half, residual=True)
        self.gc3 = GraphConvolution(self.D,  half, residual=True)
        self.gc4 = GraphConvolution(half,  half, residual=True)
        self.linear = nn.Linear(self.D, self.D)
        self.disAdj = DistanceAdj()
        self.gelu = QuickGELU()

    @staticmethod
    def _build_local_mask(L:int, w:int):
        mask = torch.empty(L, L)
        mask.fill_(float('-inf'))
        for i in range((L + w - 1) // w):
            b, e = i*w, min((i+1)*w, L)
            mask[b:e, b:e] = 0
        return mask

    @staticmethod
    def _adj_cos(x: torch.Tensor, lengths: torch.Tensor):
        # x:[B,Ts,D]  lengths:[B]
        soft = nn.Softmax(1)
        x2   = x @ x.transpose(1,2)      # [B,Ts,Ts]
        xnm  = torch.norm(x, p=2, dim=2, keepdim=True)
        x2   = x2 / (xnm @ xnm.transpose(1,2) + 1e-20)
        out  = torch.zeros_like(x2)
        for i in range(x.size(0)):
            L = int(lengths[i].item())
            adj = F.threshold(x2[i, :L, :L], 0.7, 0.0)
            out[i, :L, :L] = soft(adj)
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):

        device = x.device
        B, Ts, D = x.shape

        pos_ids = torch.arange(Ts, device=device).unsqueeze(0).expand(B, -1)   # [B,Ts]
        pos_emb = self.frame_position_embeddings(pos_ids).permute(1,0,2)    # [Ts,B,D]
        x_in = x.permute(1,0,2) + pos_emb                            # [Ts,B,D]

        Ts = x.size(1)
        win = max(1, self.attn_window // self.s)
        attn_mask = self._build_local_mask(Ts, win).to(x.device)
        x_out, _ = self.temporal((x_in, None), attn_mask=attn_mask)
        x_out = x_out.permute(1,0,2)                        # [B,Ts,D]

        adj  = self._adj_cos(x_out, lengths)             # [B,Ts,Ts]
        disadj = self.disAdj(B, Ts)
        if disadj.device != device: disadj = disadj.to(device)

        x1 = self.gelu(self.gc1(x_out, adj));    x1 = self.gelu(self.gc2(x1, adj))    # [B,Ts,D/2]
        x2 = self.gelu(self.gc3(x_out, disadj)); x2 = self.gelu(self.gc4(x2, disadj))  # [B,Ts,D/2]

        y  = torch.cat([x1, x2], dim=2)               # [B,Ts,D]
        y  = self.linear(y)                               # [B,Ts,D]
        return y


def safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    denom = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    return x / denom

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# ===== modified ResidualAttentionBlock =====
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # 移除 self.attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor, attn_mask: torch.Tensor = None):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        attn_mask = attn_mask.to(x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x, attn_mask: torch.Tensor = None):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask, attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class MoeClipVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 moe_res_alpha: float,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.moe_res_alpha = moe_res_alpha
        self.device = device

        self.gelu = QuickGELU()

        # # self.moe_res_alpha = 0.2  # UCF crime setting
        # self.moe_res_alpha = 0.1   # XD setting

        # ===== experts：s=1,2,3 =====
        self.expert_s1 = LGTExpertScale(visual_width, visual_layers, visual_head,
                                        attn_window=self.attn_window,
                                        base_length=self.visual_length, scale=1)
        self.expert_s2 = LGTExpertScale(visual_width, visual_layers, visual_head,
                                        attn_window=self.attn_window,
                                        base_length=self.visual_length, scale=2)
        self.expert_s3 = LGTExpertScale(visual_width, visual_layers, visual_head,
                                        attn_window=self.attn_window,
                                        base_length=self.visual_length, scale=3)

        self.moe_gate = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(visual_width * 3, visual_width)),
            ("act", QuickGELU()),
            ("fc2", nn.Linear(visual_width, 3))
        ]))
        with torch.no_grad():  # --- gate 前期偏向 s=1 ---
            if hasattr(self.moe_gate[-1], "bias") and self.moe_gate[-1].bias is not None:
                nn.init.zeros_(self.moe_gate[-1].bias)
                self.moe_gate[-1].bias.data[0] = 2.0  # s=1
                self.moe_gate[-1].bias.data[1] = -0.8  # s=2
                self.moe_gate[-1].bias.data[2] = -1.2  # s=3

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output


    @staticmethod
    def _downsample_mean(x: torch.Tensor, lengths: torch.Tensor, scale: int):

        if scale == 1:
            return x, lengths
        B, T, D = x.shape
        device = x.device
        # [B,D,T] for pooling
        x_ch = x.permute(0,2,1)  # [B,D,T]
        Ts = (lengths + scale - 1) // scale

        xs = F.avg_pool1d(x_ch, kernel_size=scale, stride=scale, ceil_mode=True, count_include_pad=False)  # [B,D,ceil(T/scale)]
        xs = xs.permute(0,2,1)  # [B,Ts,D]

        Ts_max = int(Ts.max().item())
        xs = xs[:, :Ts_max, :]
        return xs, Ts

    @staticmethod
    def _upsample_to_T(y: torch.Tensor, scale: int, T: int):

        if scale == 1:
            y_up = y
        else:
            y_up = y.repeat_interleave(scale, dim=1)  # [B,Ts*scale,D]

        B, L, D = y_up.shape
        if L < T:
            pad = y_up.new_zeros(B, T - L, D)
            y_up = torch.cat([y_up, pad], dim=1)
        return y_up[:, :T, :]

    def encode_video(self, images, padding_mask, lengths):

        images = images.to(torch.float)
        device = images.device
        # 统一 lengths 到同设备 & long
        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)

        B, T, D = images.shape

        # --- s=1
        x1_in, len1 = images, lengths          # [B,T,D], [B]
        y1 = self.expert_s1(x1_in, len1)            # [B,T,D]

        # --- s=2
        x2_in, len2 = self._downsample_mean(images, lengths, scale=2)    # [B,T/2,D], [B]
        y2_coarse = self.expert_s2(x2_in, len2)                         # [B,T/2,D]
        y2 = self._upsample_to_T(y2_coarse, scale=2, T=T)        # [B,T,D]

        # --- s=3
        x3_in, len4 = self._downsample_mean(images, lengths, scale=3)    # [B,T/4,D], [B]
        y3_coarse = self.expert_s3(x3_in, len4)                         # [B,T/4,D]
        y3 = self._upsample_to_T(y3_coarse, scale=3, T=T)        # [B,T,D]

        def ln(t):
            # return F.relu(F.layer_norm(t, (t.size(-1),)))
            return F.layer_norm(t, (t.size(-1),))

        y1_stop = y1.detach()
        d2 = y2 - y1_stop  # s=2 residual
        d3 = y3 - y1_stop  # s=4 residual

        # gate 输入 = [LN(y1), LN(d2), LN(d4)]
        gate_in = torch.cat([ln(y1), ln(d2), ln(d3)], dim=-1)  # [B,T,3D]
        gate_log = self.moe_gate(gate_in)  # [B,T,3]

        w23 = torch.softmax(gate_log[..., 1:3], dim=-1)  # [B,T,2]
        alpha = self.moe_res_alpha

        y1_base = y1
        y1_stop = y1.detach()

        delta2 = y2 - y1_stop
        # delta2 = torch.zeros_like(y1)
        delta4 = y3 - y1_stop
        fused = y1_base + alpha * (w23[..., 0:1] * delta2 + w23[..., 1:2] * delta4)  # [B,T,D]

        eff_gate = torch.cat([
            (1.0 - alpha) * torch.ones_like(w23[..., 0:1]),
            alpha * w23[..., 0:1],
            alpha * w23[..., 1:2]
        ], dim=-1)  # [B,T,3]

        self._last_gate = eff_gate.detach()
        self._last_valid = None
        self._last_scales = {"y1": y1.detach(), "y2": y2.detach(), "y3": y3.detach()}

        return fused


    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
    