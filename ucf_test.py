import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MoeClipVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
from utils.score_plot import save_score_plot
import ucf_option

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
    ap1 = ap1.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)

    return ROC1, AP1


def test1(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    element_logits2_stack = []

    ano_scores = []  # anomaly-only scores (frame-level)
    ano_labels = []  # anomaly-only labels (frame-level)
    gt_ptr = 0

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
                # ap3 = prob3
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

            frames_this = int(len_cur) * 16

            gt_slice = gt[gt_ptr:gt_ptr + frames_this]

            score_slice = np.repeat(prob1.detach().cpu().numpy(), 16)

            if np.any(gt_slice == 1):
                ano_scores.append(score_slice)
                ano_labels.append(gt_slice)

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
        y_true_ano = np.concatenate(ano_labels, axis=0)
        y_score_ano = np.concatenate(ano_scores, axis=0)
        ROC1_ANO = roc_auc_score(y_true_ano, y_score_ano)
        AP1_ANO = average_precision_score(y_true_ano, y_score_ano)
        print("Ano-AUC1: ", ROC1_ANO, " Ano-AP1: ", AP1_ANO)
    else:
        print("Ano-AUC1: N/A (测试集中未发现异常视频)")

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1


def test2(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, save_dir="avg_ucf"):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    element_logits2_stack = []

    ano_scores, ano_labels = [], []
    gt_ptr = 0

    FRAMES_PER_CLIP = 16

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            ltmp = length
            for j in range(int(length / maxlen) + 1):
                if j == 0 and ltmp < maxlen:
                    lengths[j] = ltmp
                elif j == 0 and ltmp > maxlen:
                    lengths[j] = maxlen; ltmp -= maxlen
                elif ltmp > maxlen:
                    lengths[j] = maxlen; ltmp -= maxlen
                else:
                    lengths[j] = ltmp
            lengths = lengths.to(torch.long)

            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)

            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])

            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))          # frame-classifier
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            if i == 0:
                ap1, ap2 = prob1, prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            elem2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            elem2 = np.repeat(elem2, FRAMES_PER_CLIP, 0)
            element_logits2_stack.append(elem2)

            frames_this = int(len_cur) * FRAMES_PER_CLIP
            gt_slice = gt[gt_ptr:gt_ptr + frames_this]

            try:
                vid_name = item[3]
            except Exception:
                if hasattr(testdataloader.dataset, "video_names"):
                    vid_name = testdataloader.dataset.video_names[i]
                elif hasattr(testdataloader.dataset, "paths"):
                    vid_name = os.path.basename(testdataloader.dataset.paths[i])
                elif hasattr(testdataloader.dataset, "fnames"):
                    vid_name = os.path.basename(testdataloader.dataset.fnames[i])
                else:
                    vid_name = f"video_{i:04d}"

            score_slice = np.repeat(prob1.detach().cpu().numpy(), FRAMES_PER_CLIP)
            save_score_plot(score_slice, gt_slice, vid_name, save_dir)

            if np.any(gt_slice == 1):
                ano_scores.append(score_slice)
                ano_labels.append(gt_slice)

            gt_ptr += frames_this

    ap1 = ap1.cpu().numpy().tolist()
    ap2 = ap2.cpu().numpy().tolist()
    ROC1 = roc_auc_score(gt, np.repeat(ap1, FRAMES_PER_CLIP))
    AP1  = average_precision_score(gt, np.repeat(ap1, FRAMES_PER_CLIP))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, FRAMES_PER_CLIP))
    AP2  = average_precision_score(gt, np.repeat(ap2, FRAMES_PER_CLIP))
    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    if len(ano_labels) > 0:
        y_true_ano  = np.concatenate(ano_labels, axis=0)
        y_score_ano = np.concatenate(ano_scores, axis=0)
        ROC1_ANO = roc_auc_score(y_true_ano, y_score_ano)
        AP1_ANO  = average_precision_score(y_true_ano, y_score_ano)
        print("Ano-AUC1: ", ROC1_ANO, " Ano-AP1: ", AP1_ANO)
    else:
        print("Ano-AUC1: N/A (测试集中未发现异常视频)")

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP /= (i + 1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = MoeClipVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args.moe_res_alpha,device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test1(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

    # if you want to plot score, use test2, but you have to uncomment the dataloader part.
    # test2(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)


