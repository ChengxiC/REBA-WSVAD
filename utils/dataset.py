import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
# import utils.tools as tools
import utils.tools as tools
import os
from pathlib import Path


# initial dataloader
class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length


# # this is only for plot of curve
# class UCFDataset(data.Dataset):
#     def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
#         self.df = pd.read_csv(file_path)
#         self.clip_dim = clip_dim
#         self.test_mode = test_mode
#         self.label_map = label_map
#         self.normal = normal
#
#         if normal and not test_mode:
#             self.df = self.df[self.df['label'] == 'Normal'].reset_index(drop=True)
#         elif not test_mode:
#             self.df = self.df[self.df['label'] != 'Normal'].reset_index(drop=True)
#
#         # 若 CSV 没有 'video' 列，就从 'path' 推导
#         if 'video' not in self.df.columns:
#             self.df['video'] = self.df['path'].apply(
#                 lambda p: os.path.splitext(os.path.basename(str(p)))[0]
#             )
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index: int):
#         row = self.df.iloc[index]
#         clip_feature = np.load(row['path'])
#         if not self.test_mode:
#             clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
#         else:
#             clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
#
#         clip_feature = torch.tensor(clip_feature)
#         clip_label = row['label']
#         video_name = row['video']
#
#         # 训练保持三元组；测试返回四元组, 多一个 name
#         if self.test_mode:
#             return clip_feature, clip_label, clip_length, video_name
#         else:
#             return clip_feature, clip_label, clip_length


# for ordinary testing and training
class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length


# #### only for plot score ,   for test 3
# class XDDataset(data.Dataset):
#     def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
#         self.df = pd.read_csv(file_path)
#         self.clip_dim = clip_dim
#         self.test_mode = test_mode
#         self.label_map = label_map
#
#         # 统一用 path 的“去扩展名的文件名”作为 video_name
#         # 例如 G:\...\Young...__0.npy -> Young...__0
#         def _derive_name(p):
#             p = str(p).strip().strip('"').strip("'")  # 清理引号/空格
#             return Path(p).stem                # 仅去扩展名，不按 '.' 分割
#         self.df['video_name'] = self.df['path'].map(_derive_name)
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         feat_path = row['path']
#         clip_feature = np.load(feat_path)
#
#         if not self.test_mode:
#             clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
#         else:
#             clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
#
#         clip_feature = torch.tensor(clip_feature)
#         clip_label = row.get('label', 'Unknown')
#         video_name  = row['video_name']
#
#         if self.test_mode:
#             # 测试返回四元组
#             return clip_feature, clip_label, int(clip_length), video_name
#         else:
#             return clip_feature, clip_label, int(clip_length)








