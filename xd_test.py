import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MoeClipVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
from utils.score_plot import save_score_plot
import xd_option


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    
    model.to(device)
    model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    # dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    # averageMAP = 0
    # for i in range(5):
    #     print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
    #     averageMAP += dmap[i]
    # averageMAP = averageMAP/(i+1)
    # print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1, ROC2, AP2, 0  # averageMAP


def test1(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    element_logits2_stack = []

    # 仅异常视频的
    ano_scores1, ano_scores2, ano_labels = [], [], []
    gt_ptr = 0  # 指向全局 gt 的帧级指针

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

            # 仅异常视频的 AP 收集
            frames_this = int(len_cur) * 16
            gt_slice = gt[gt_ptr:gt_ptr + frames_this]
            score1_slice = np.repeat(prob1.detach().cpu().numpy(), 16)
            score2_slice = np.repeat(prob2.detach().cpu().numpy(), 16)
            if np.any(gt_slice == 1):
                ano_labels.append(gt_slice)
                ano_scores1.append(score1_slice)
                ano_scores2.append(score2_slice)
            gt_ptr += frames_this

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    if len(ano_labels) > 0:
        y_ano = np.concatenate(ano_labels, axis=0)
        s1_ano = np.concatenate(ano_scores1, axis=0)
        s2_ano = np.concatenate(ano_scores2, axis=0)
        AP1_ANO = average_precision_score(y_ano, s1_ano)
        AP2_ANO = average_precision_score(y_ano, s2_ano)
        print("Anomaly AP1:", AP1_ANO, " Anomaly AP2:", AP2_ANO)
    else:
        print("Anomaly AP: N/A（测试集里没有异常视频）")
    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1, ROC2, AP2, 0  # averageMAP


def test2(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, out_dir="plots_xd"):
    model.to(device)
    model.eval()

    element_logits2_stack = []
    ano_scores1, ano_scores2, ano_labels = [], [], []
    gt_ptr = 0

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual, clip_label, clip_length, video_name = item
            visual = visual.squeeze(0).to(device)
            len_cur = int(clip_length)

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            length = len_cur
            lengths = torch.zeros(int(length / maxlen) + 1, dtype=torch.long)
            for j in range(len(lengths)):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length >= maxlen:
                    lengths[j] = maxlen; length -= maxlen
                elif length >= maxlen:
                    lengths[j] = maxlen; length -= maxlen
                else:
                    lengths[j] = length

            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)

            logits1 = logits1.reshape(-1, logits1.shape[-1])  # [T_clip, 1]
            logits2 = logits2.reshape(-1, logits2.shape[-1])  # [T_clip, C]

            prob1 = torch.sigmoid(logits1[:len_cur].squeeze(-1))        # clip级
            prob2 = 1 - logits2[:len_cur].softmax(dim=-1)[:, 0]         # “非normal”概率

            if i == 0:
                ap1, ap2 = prob1, prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            elem2 = logits2[:len_cur].softmax(dim=-1).detach().cpu().numpy()
            elem2 = np.repeat(elem2, 16, axis=0)
            element_logits2_stack.append(elem2)

            frames_this = len_cur * 16
            gt_slice = gt[gt_ptr:gt_ptr + frames_this]

            score1_slice = np.repeat(prob1.detach().cpu().numpy(), 16)
            save_score_plot(score1_slice, gt_slice, video_name, out_dir)

            score2_slice = np.repeat(prob2.detach().cpu().numpy(), 16)
            if np.any(gt_slice == 1):
                ano_labels.append(gt_slice)
                ano_scores1.append(score1_slice)
                ano_scores2.append(score2_slice)

            gt_ptr += frames_this

    ap1 = ap1.cpu().numpy(); ap2 = ap2.cpu().numpy()
    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))
    print("AUC1:", ROC1, " AP1:", AP1)
    print("AUC2:", ROC2, " AP2:", AP2)

    if len(ano_labels) > 0:
        y_ano  = np.concatenate(ano_labels, axis=0)
        s1_ano = np.concatenate(ano_scores1, axis=0)
        s2_ano = np.concatenate(ano_scores2, axis=0)
        print("Anomaly AP1:", average_precision_score(y_ano, s1_ano),
              " Anomaly AP2:", average_precision_score(y_ano, s2_ano))
    else:
        print("Anomaly AP: N/A（测试集中没有异常视频）")

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    avgMAP = np.mean(dmap)
    for k in range(5):
        print(f"mAP@{iou[k]:.1f} = {dmap[k]:.2f}%")
    print(f"average MAP: {avgMAP:.2f}")

    return ROC1, AP1, ROC2, AP2, avgMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = MoeClipVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args.res_alpha, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)   # 40，30 series
    # missing, unexpected = model.load_state_dict(model_param, strict=False)   # 50 series

    test1(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

    # if you want to plot score, use test2, but you have to uncomment the dataloader part.
    # test2(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
