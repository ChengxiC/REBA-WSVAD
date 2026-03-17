import torch
import torch.nn.functional as F


# bi image-text loss for weakly supervised VAD
class BiDirAlign(torch.nn.Module):
    def __init__(self, visual_width: int, embed_dim: int, attn_tau: float = 0.1, init_tau: float = 0.07):
        super().__init__()
        self.video_proj = torch.nn.Linear(visual_width, embed_dim, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / init_tau)))
        self.attn_tau = attn_tau

    def temporal_pool(self, visual_feats: torch.Tensor, logits1: torch.Tensor, lengths: torch.Tensor):
        """
        visual_feats: [B, T, C] 原始帧/clip特征，C 512
        logits1: [B, T, 1] 片段异常得分（做注意力）
        lengths: [B] 每个视频的特征的实际长度
        return: [B, C] 加权汇聚后的视频向量
        """
        if visual_feats.ndim != 3:
            raise ValueError(f"visual_feats should be [B, T, C], got {visual_feats.shape}")
        B, T, C = visual_feats.shape
        v = []
        for i in range(B):
            L = int(lengths[i].item())
            s = logits1[i, :L, 0]
            w = torch.softmax(s / self.attn_tau, dim=0)
            vi = (w.unsqueeze(1) * visual_feats[i, :L, :]).sum(dim=0)
            v.append(vi)
        return torch.stack(v, dim=0)  # [B, C]

    def forward(self, visual_feats, logits1, lengths, text_features, text_labels):

        device = visual_feats.device
        v_raw = self.temporal_pool(visual_feats, logits1, lengths)  # [B, C] supervised signal
        v = self.video_proj(v_raw)  # [B, D]
        v = F.normalize(v, dim=-1)

        if text_labels.ndim == 2:
            idx = text_labels.argmax(dim=1)  # [B]
        else:
            idx = text_labels.long().view(-1)
        t_pos = text_features[idx]  # [B, D]
        t_pos = F.normalize(t_pos, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i2t = logit_scale * (v @ t_pos.t())  # [B, B]
        logits_t2i = logit_scale * (t_pos @ v.t()) # [B, B]
        targets = torch.arange(v.size(0), device=device)

        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return (loss_i2t + loss_t2i) * 0.5




