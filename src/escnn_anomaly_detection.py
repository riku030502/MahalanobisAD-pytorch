import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import cv2
from escnn import gspaces
from escnn import nn as escnn_nn

import datasets.mvtec as mvtec

N = 8  # C8 symmetry group, can be changed to C4 or others


def parse_args():
    parser = argparse.ArgumentParser('MahalanobisAD')
    parser.add_argument("--model_name", type=str, default='efficientnet-b4')
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def generate_heatmap_overlay(image_tensor, anomaly_map, mean, std):
    image = denormalize(image_tensor.clone(), mean, std)
    image = torch.clamp(image, 0, 1)
    image_np = np.transpose(image.numpy(), (1, 2, 0)) * 255
    image_np = image_np.astype(np.uint8)

    # アノマリーマップをリサイズ・正規化
    anomaly_map_resized = cv2.resize(anomaly_map, (image_np.shape[1], image_np.shape[0]))
    normalized_anomaly = (255 * (anomaly_map_resized - anomaly_map_resized.min()) / 
                          (anomaly_map_resized.ptp() + 1e-6)).astype(np.uint8)

    # スコアの計算（ここでは最大値を表示）
    anomaly_score = anomaly_map_resized.max()

    # ヒートマップ重ね合わせ
    heatmap = cv2.applyColorMap(normalized_anomaly, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    # スコアをテキストで描画
    score_text = f"Score: {anomaly_score:.2f}"
    cv2.putText(
        overlay, score_text, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
    )

    return overlay

#escnnによる特徴マップの取得
class ESCNNModel(torch.nn.Module):
    def __init__(self, model_name): 
        super().__init__()
        self.r2_act = gspaces.rot2dOnR2(N) # C8 or C4, etc.

        # 入力フィールド：RGBチャンネル
        in_type = escnn_nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        self.input_type = in_type
        self.block1 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, escnn_nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                            kernel_size=5, padding=2, bias=False),
            escnn_nn.ReLU(escnn_nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), inplace=True),
            escnn_nn.PointwiseAvgPoolAntialiased(
                escnn_nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]), sigma=0.6, stride=2
            )
        )
        self.block2 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(self.block1.out_type,
                            escnn_nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                            kernel_size=5, padding=2, bias=False),
            escnn_nn.ReLU(escnn_nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True),
            escnn_nn.PointwiseAvgPoolAntialiased(
                escnn_nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), sigma=0.6, stride=2
            )
        )

    def forward(self, x):
        x = escnn_nn.GeometricTensor(x, self.input_type)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return [x1.tensor, x2.tensor]  # torch.Tensor に変換して返す
# class EfficientNetModified(EfficientNet):
#     def extract_features_spatial(self, inputs):
#         feat_list = []

#         x = self._swish(self._bn0(self._conv_stem(inputs)))
#         feat_list.append(x)

#         x_prev = x
#         for idx, block in enumerate(self._blocks):
#             drop_connect_rate = self._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if (x_prev.shape[1] != x.shape[1] and idx != 0) or idx == (len(self._blocks) - 1):
#                 feat_list.append(x_prev)
#             x_prev = x

#         x = self._swish(self._bn1(self._conv_head(x)))
#         feat_list.append(x)

#         return feat_list  # 各特徴マップ (B, C, H, W)


def main():
    args = parse_args()
    assert args.model_name.startswith('efficientnet-b'), f'Only EfficientNet variants supported, not {args.model_name}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ESCNNModel(args.model_name)  
    model.to(device)
    model.eval()

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

        save_dir = os.path.join(args.save_path, 'heatmap_result', class_name)
        os.makedirs(save_dir, exist_ok=True)

        feature_stats = []
        train_feat_path = os.path.join(args.save_path, 'temp', f'train_{class_name}_{args.model_name}_spatial.pkl')

        if not os.path.exists(train_feat_path):
            print(f"Extracting spatial features for training set of {class_name}")
            all_feats = []

            for x, _, _ in tqdm(train_loader):
                with torch.no_grad():
                    feats = model(x.to(device))
                all_feats.append([f.cpu().numpy() for f in feats])

            num_levels = len(all_feats[0])
            feature_stats = []

            for level in range(num_levels):
                spatial_feats = []
                for batch in all_feats:
                    f = batch[level]  # shape (B, C, H, W)
                    B, C, H, W = f.shape
                    spatial_feats.append(f.transpose(0, 2, 3, 1).reshape(-1, C))
                spatial_feats = np.concatenate(spatial_feats, axis=0)

                mean = np.mean(spatial_feats, axis=0)
                cov = LedoitWolf().fit(spatial_feats).covariance_
                feature_stats.append((mean, np.linalg.inv(cov)))

            with open(train_feat_path, 'wb') as f:
                pickle.dump(feature_stats, f)
        else:
            print(f"Loading spatial train stats from {train_feat_path}")
            with open(train_feat_path, 'rb') as f:
                feature_stats = pickle.load(f)

        for idx, (x, _, _) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            with torch.no_grad():
                feats = model(x)

            score_maps = []

            for level, f in enumerate(feats):
                f = f.squeeze(0).cpu().numpy()  # (C, H, W)
                C, H, W = f.shape
                f = f.reshape(C, -1).T  # (H*W, C)

                mean, inv_cov = feature_stats[level]
                dists = [mahalanobis(pix, mean, inv_cov) for pix in f]
                dists = np.array(dists).reshape(H, W)
                score_maps.append(cv2.resize(dists, (x.shape[3], x.shape[2])))

            total_score_map = np.sum(np.stack(score_maps), axis=0)
            overlay = generate_heatmap_overlay(x.squeeze(0).cpu(), total_score_map, imagenet_mean, imagenet_std)
            cv2.imwrite(os.path.join(save_dir, f'{idx:03d}_overlay.png'), overlay)

if __name__ == '__main__':
    main()