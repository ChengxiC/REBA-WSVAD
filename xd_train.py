import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from copy import deepcopy

from model import MoeClipVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option


class BiDirAlign(torch.nn.Module):
    def __init__(self, visual_width: int, embed_dim: int, attn_tau: float = 0.1, init_tau: float = 0.07):
        super().__init__()
        self.video_proj = torch.nn.Linear(visual_width, embed_dim, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / init_tau)))
        self.attn_tau = attn_tau

    def temporal_pool(self, visual_feats: torch.Tensor, logits1: torch.Tensor, lengths: torch.Tensor):

        if visual_feats.ndim != 3:
            raise ValueError(f"visual_feats should be [B, T, C], got {visual_feats.shape}")
        B, T, C = visual_feats.shape
        v = []
        for i in range(B):
            L = int(lengths[i].item())
            s = logits1[i, :L, 0].detach()
            w = torch.softmax(s / self.attn_tau, dim=0)
            vi = (w.unsqueeze(1) * visual_feats[i, :L, :]).sum(dim=0)  # [C]
            v.append(vi)
        return torch.stack(v, dim=0)  # [B, C]

    def forward(self, visual_feats, logits1, lengths, text_features, text_labels):
        device = visual_feats.device
        v_raw = self.temporal_pool(visual_feats, logits1, lengths)  # [B, C]
        v = F.normalize(self.video_proj(v_raw), dim=-1)  # [B, D]

        if text_labels.ndim == 2:
            idx = text_labels.argmax(dim=1)  # [B]
        else:
            idx = text_labels.long().view(-1)

        t_all = F.normalize(text_features.detach(), dim=-1)  # [K, D]

        logit_scale = self.logit_scale.exp().clamp(max=80.0)
        logits_i2t_full = logit_scale * (v @ t_all.t())  # [B, K]
        loss_i2t = F.cross_entropy(logits_i2t_full, idx)

        t_pos = t_all[idx]  # [B, D]
        logits_t2i = logit_scale * (t_pos @ v.t())  # [B, B]
        targets = torch.arange(v.size(0), device=device)
        loss_t2i = F.cross_entropy(logits_t2i, targets)

        return 0.5 * (loss_i2t + loss_t2i)


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    bidir = BiDirAlign(visual_width=args.visual_width, embed_dim=args.embed_dim, attn_tau=0.1, init_tau=0.07).to(device)
    gate_params = [p for n, p in model.named_parameters() if n.startswith("moe_gate.")]
    base_params = [p for n, p in model.named_parameters() if not n.startswith("moe_gate.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params},  # default base lr = args.lr
            {"params": gate_params, "lr": args.lr * args.gatelr_rate},  # gate lr
            {"params": bidir.parameters()},
        ],
        lr=args.lr
    )
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    ##########################################################################
    prompt_text = get_prompt_text(label_map)
    ap_best_global = -1.0
    start_epoch = 0

    log_interval = getattr(args, 'log_interval', 50)
    eval_interval_batches = getattr(args, 'eval_interval_batches', 50)

    for e in range(start_epoch, args.max_epoch):
        model.train()
        loss_total1 = 0.0
        loss_total2 = 0.0

        num_batches = len(train_loader)
        seen_examples = 0
        did_eval_this_epoch = False
        for i, item in enumerate(train_loader):
            epoch_step = i + 1

            visual_feat, text_labels, feat_lengths = item
            bs = visual_feat.size(0)
            seen_examples += bs

            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            align_loss = bidir(visual_feat, logits1, feat_lengths, text_features, text_labels)

            loss = loss1 + loss2 + args.align_coef * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()

            if (epoch_step % log_interval == 0) or (epoch_step == num_batches):
                print(f"epoch: {e + 1} | batch: {epoch_step}/{num_batches} | step: {seen_examples} "
                      f"| loss1: {loss_total1 / epoch_step:.6f} | loss2: {loss_total2 / epoch_step:.6f} "
                      f"| align loss: {align_loss.item():.6f} | total: {loss.item():.6f}")

            if (epoch_step % eval_interval_batches) == 0:
                model.eval()
                with (torch.no_grad()):
                    AUC1, AP1, AUC2, AP2, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                model.train()
                did_eval_this_epoch = True

                if AP2 > ap_best_global:
                    ap_best_global = AP2
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'ap': ap_best_global
                    }
                    torch.save(checkpoint, args.checkpoint_path)

        if (num_batches % eval_interval_batches) != 0 or (not did_eval_this_epoch):
            model.eval()
            with torch.no_grad():
                AUC1, AP1, AUC2, AP2, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
            model.train()
            if AP2 > ap_best_global:
                ap_best_global = AP2
                checkpoint = {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ap': ap_best_global
                }
                torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        scheduler.step()

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    torch.save(checkpoint['model_state_dict'], args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MoeClipVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args.res_alpha,device)
    train(model, train_loader, test_loader, args, label_map, device)

