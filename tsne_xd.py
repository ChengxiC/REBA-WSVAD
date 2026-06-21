# t-SNE：类内更紧、类间更分；余弦度量 + 白化PCA + 高early exaggeration + 类中心星标
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from model import MoeClipVad
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
import xd_option as opt  # visual_length / test_list / model_path / classes_num
import re


# 固定调色板
PALETTE_BY_CLASS = {
    'Normal':       '#000000',  # 黑色
    'Fighting':     '#d62728',  # 红
    'Shooting':     '#2ca02c',  # 绿
    'Riot':         '#1f77b4',  # 蓝
    'Abuse':        '#9467bd',  # 紫
    'CarAccident':  '#8c564b',  # 棕
    'Explosion':    '#17becf',  # 青
}

def _color_for(cls_name: str, fallback_idx: int = 0):
    key = str(cls_name).strip().lower()
    pal = {k.strip().lower(): v for k, v in PALETTE_BY_CLASS.items()}
    if key in pal:
        return pal[key]
    return plt.get_cmap("tab20")(fallback_idx % 20)

_VALID_PREFIX = {"A","B1","B2","B4","B5","B6","G"}
def _extract_prefix_set(label_str: str):
    parts = re.split(r'[-_\s,+/|;]+', str(label_str).strip())
    s = set()
    for p in parts:
        up = p.upper()
        if up in _VALID_PREFIX:
            s.add(up)
    return s

def is_multi_label_video(label_str: str) -> bool:
    return len(_extract_prefix_set(label_str)) > 1


# 超参
STRIDE = 2  # 每stride个clip抽取一个
RAND_OFFSET = True
PER_VIDEO_MAX = 100  # 每个视频最多抽取的数量
SIM_DROP_THR = 0.995
LIMIT_PER_CLASS = 500  # 每个类别的点数量

PCA_DIM = 64        # 稍大一些用于白化
PERPLEXITY = 30        # 20~30   # or 45–60，可减少小岛数量、让类团更连贯.
N_ITER = 5000           # 更久训练
EARLY_EXAG = 30.0    # 强化簇分离 （original 24），  如果值较小 （12~16），可以减少早期的簇排斥现象
NO_PROGRESS = 1000
POST_JITTER = 0.06 # 极轻的抖动  0.005 ？
SEED = 5
OUT_PATH = "tsne_xd.pdf"

COARSE_MAP = {
    'A':  'Normal',
    'B1': 'Fighting',
    'B2': 'Shooting',
    'B4': 'Riot',
    'B5': 'Abuse',
    'B6': 'CarAccident',
    'G':  'Explosion',
}
COARSE_ORDER = ['Normal','Fighting','Shooting','Riot','Abuse','CarAccident','Explosion']

def map_to_coarse(raw_label: str) -> str:
    s = str(raw_label).strip()
    prefix = s.split('-')[0].upper()  
    return COARSE_MAP.get(prefix, s)  


@torch.no_grad()
def split_lengths(total_len: int, maxlen: int) -> torch.Tensor:
    parts, remain = [], int(total_len)
    while remain > 0:
        take = min(maxlen, remain)
        parts.append(take); remain -= take
    return torch.tensor(parts, dtype=torch.long)

@torch.no_grad()
def take_valid_snippets(
    fused: torch.Tensor,
    lengths: torch.Tensor,
    stride: int = STRIDE,
    per_video_max: int = PER_VIDEO_MAX,
    sim_drop_thr = SIM_DROP_THR,
    rand_offset: bool = RAND_OFFSET,
) -> torch.Tensor:
    outs = []
    for i in range(fused.size(0)):
        L = int(lengths[i].item())
        if L <= 0: continue
        step = max(1, stride)
        off = np.random.randint(0, min(step, L)) if rand_offset and step > 1 else 0
        idx = np.arange(off, L, step)

        if sim_drop_thr is not None and len(idx) > 1:
            v = fused[i, idx, :].detach().cpu().numpy()
            vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            keep = [0]; last = vn[0]
            for k in range(1, vn.shape[0]):
                if float(np.dot(vn[k], last)) < sim_drop_thr:
                    keep.append(k); last = vn[k]
            idx = idx[keep]

        if per_video_max and len(idx) > per_video_max:
            idx = np.random.choice(idx, per_video_max, replace=False)

        outs.append(fused[i, idx, :])

    return torch.cat(outs, dim=0) if outs else torch.zeros(0, fused.size(-1), device=fused.device)

def build_class_mapping(dataset, put_normal_first=True):
    raw_labels = list(map(str, dataset.df["label"].tolist()))
    coarse_labels = [map_to_coarse(x) for x in raw_labels]

    present = set(coarse_labels)
    classes = [c for c in COARSE_ORDER if c in present]

    label2idx = {c: i for i, c in enumerate(classes)}
    return classes, label2idx

def balance_per_class(X, y, limit, seed: int = SEED):
    if not limit: return X, y
    rng = np.random.RandomState(seed)
    idx_keep = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) > limit:
            idx_c = rng.choice(idx_c, size=limit, replace=False)
        idx_keep.append(idx_c)
    idx_keep = np.concatenate(idx_keep, axis=0)
    return X[idx_keep], y[idx_keep]


# 收集特征
@torch.no_grad()
def collect_clip_features_multiclass(
    model, loader, maxlen, device, label2idx,
    stride=STRIDE, per_video_max=PER_VIDEO_MAX,
    sim_drop_thr=SIM_DROP_THR, rand_offset=RAND_OFFSET,
):
    model.eval().to(device)
    Xs, Ys = [], []

    for batch in loader:
        if len(batch) == 3:
            visual, label, length = batch
        else:
            visual, label, length, _ = batch

        # 统一 label
        if isinstance(label, (list, tuple)) and len(label) == 1:
            label = label[0]
        if torch.is_tensor(label):
            label = label.item()
        # label = str(label)
        # if label not in label2idx:
        #     continue
        # y_idx = label2idx[label]
        label = str(label)
        if is_multi_label_video(label):
            continue
        coarse = map_to_coarse(label)  
        if coarse not in label2idx:
            continue
        y_idx = label2idx[coarse]


        visual = visual.to(device)
        if visual.dim() < 2:
            raise ValueError(f"visual rank<2: {tuple(visual.shape)}")

        C = visual.shape[-1]
        T = int(np.prod(visual.shape[:-1]))  
        visual = visual.reshape(T, C)        # -> [T, C]
        total_len = T

        parts = split_lengths(total_len, maxlen).cpu().tolist()
        start = 0
        for seg_len in parts:
            seg_len = int(seg_len)
            end = start + seg_len
            vis_seg = visual[start:end, :]           # [seg_len, C]
            start = end

            # batch=1 输入
            vis_seg = vis_seg.unsqueeze(0)           # [1, seg_len, C]
            lengths = torch.tensor([seg_len], dtype=torch.long, device=device)  # [1]
            padding_mask = get_batch_mask(lengths, maxlen).to(device)           # [1, maxlen]

            # 编码
            fused = model.encode_video(vis_seg, padding_mask, lengths)      # [1, T', D]

            clips = take_valid_snippets(
                fused, lengths,
                stride=stride,
                per_video_max=per_video_max,
                sim_drop_thr=sim_drop_thr,
                rand_offset=rand_offset,
            )
            if clips.numel() == 0:
                continue

            Xs.append(clips.detach().cpu().numpy())
            Ys.append(np.full((clips.size(0),), y_idx, dtype=np.int64))

    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 1), dtype=np.float32)
    y = np.concatenate(Ys, axis=0) if Ys else np.zeros((0,), dtype=np.int64)
    return X, y


def plot_tsne_layered(Z, y, class_names, out=OUT_PATH, show_centers=False):

    def _color_for(cls_name: str, fallback_idx: int = 0):
        if cls_name in PALETTE_BY_CLASS:
            return PALETTE_BY_CLASS[cls_name]
        return plt.get_cmap("tab20")(fallback_idx % 20)

    plt.figure(figsize=(9.5, 7), dpi=180)

    normal_idx = next((i for i, n in enumerate(class_names) if n.lower() == "normal"), None)

    if normal_idx is not None:
        m0 = (y == normal_idx)
        if np.any(m0):
            c0 = _color_for(class_names[normal_idx], 0)  # 应为 '#1f77b4'
            plt.scatter(
                Z[m0, 0], Z[m0, 1],
                s=8,  # 原来 3 -> 提高点大小
                alpha=0.40,  # 原来 0.12 -> 提高透明度
                lw=0,
                color=c0,
                label=f"{class_names[normal_idx]}",
                zorder=0,
                rasterized=True
            )

    for i, cls in enumerate(class_names):
        if i == normal_idx:
            continue
        mi = (y == i)
        if np.sum(mi) == 0:
            continue
        ci = _color_for(cls, i)
        plt.scatter(
            Z[mi, 0], Z[mi, 1],
            s=10, alpha=0.85, lw=0,
            color=ci, label=f"{cls}",
            zorder=2, rasterized=True
        )

    # 类中心星标（使用同色）
    if show_centers:
        for i, cls in enumerate(class_names):
            mi = (y == i)
            if np.sum(mi) == 0:
                continue
            c = Z[mi].mean(axis=0, keepdims=True)
            ci = _color_for(cls, i)
            plt.scatter(
                c[:, 0], c[:, 1],
                marker='*', s=250, linewidths=0.5,
                edgecolors="white", color=ci,
                zorder=3
            )

    plt.xticks([]); plt.yticks([])
    # plt.title("t-SNE of CLIP-level (snippet) features", fontsize=12)
    plt.legend(fontsize=8, frameon=False, ncol=1, markerscale=1.8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print("Saved:", out)

def run_tsne_and_plot(
    X, y, class_names, out=OUT_PATH,
    zscore=True, pca_dim=PCA_DIM, perplexity=PERPLEXITY, n_iter=N_ITER,
    seed=SEED, post_jitter=POST_JITTER, early_exag=EARLY_EXAG, no_progress=NO_PROGRESS
):
    #  L2 归一化
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    if zscore:
        X = StandardScaler(with_mean=False).fit_transform(X)  
    if pca_dim and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, whiten=True, svd_solver="auto", random_state=seed).fit_transform(X)

    N = X.shape[0]
    if N < 10:
        raise ValueError(f"Too few points for t-SNE: N={N}")
    max_ppl = max(5, min(int((N - 1) / 3), 50))
    perplexity = int(np.clip(perplexity, 5, max_ppl))

    try:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate="auto",
            init="pca", n_iter=n_iter, random_state=seed, verbose=1,
            metric="cosine", early_exaggeration=early_exag,
            n_iter_without_progress=no_progress, method="barnes_hut", angle=0.3
        )
    except TypeError:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate="auto",
            init="pca", n_iter=n_iter, random_state=seed, verbose=1,
            early_exaggeration=early_exag, n_iter_without_progress=no_progress,
            method="barnes_hut", angle=0.3
        )

    Z = tsne.fit_transform(X)

    if post_jitter and post_jitter > 0:
        Z = Z + np.random.randn(*Z.shape) * (post_jitter * Z.std(axis=0, keepdims=True))

    plot_tsne_layered(Z, y, class_names, out=out)


def main():
    args = opt.parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据
    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map={})
    class_names, label2idx = build_class_mapping(test_dataset, put_normal_first=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 模型
    model = MoeClipVad(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix,
        res_alpha=args.res_alpha, device=device
    )
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd, strict=False)

    _ = get_prompt_text({c: c for c in class_names})

    # 收集特征
    X, y = collect_clip_features_multiclass(
        model, test_loader, maxlen=args.visual_length, device=device, label2idx=label2idx,
        stride=STRIDE, per_video_max=PER_VIDEO_MAX, sim_drop_thr=SIM_DROP_THR, rand_offset=RAND_OFFSET
    )
    print(f"[collect] points={len(y)} dim={X.shape[1]} classes={len(class_names)}")

    # 类别均衡下采样
    Xb, yb = balance_per_class(X, y, LIMIT_PER_CLASS, seed=SEED)
    if LIMIT_PER_CLASS:
        print(f"[balance] -> {len(yb)} points after per-class cap={LIMIT_PER_CLASS}")

    # t-SNE + 绘图
    run_tsne_and_plot(
        Xb, yb, class_names, out=OUT_PATH,
        zscore=True, pca_dim=PCA_DIM, perplexity=PERPLEXITY, n_iter=N_ITER,
        seed=SEED, post_jitter=POST_JITTER, early_exag=EARLY_EXAG, no_progress=NO_PROGRESS
    )

if __name__ == "__main__":
    main()
