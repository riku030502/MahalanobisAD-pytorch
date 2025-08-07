import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import time

import pickle

from escnn import gspaces
from escnn import nn as enn


# ---------- ESCNNを用いた等変量特徴抽出器 ----------
class EquivariantFeatureExtractor(torch.nn.Module):
    def __init__(self, N=8):  # N-fold 回転対称性
        super().__init__()
        r2_act = gspaces.rot2dOnR2(N)
        in_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
        
        self.input_type = in_type
        self.block1 = enn.R2Conv(in_type,
                                 enn.FieldType(r2_act, 8 * [r2_act.regular_repr]),
                                 kernel_size=7,
                                 padding=3,
                                 stride=2,  # ← ストライド2
                                 bias=False)
        self.relu1 = enn.ReLU(self.block1.out_type, inplace=True)

        self.block2 = enn.R2Conv(self.relu1.out_type,
                                 enn.FieldType(r2_act, 16 * [r2_act.regular_repr]),
                                 kernel_size=5,
                                 padding=2,
                                 stride=2,  # ← ストライド2
                                 bias=False)
        self.relu2 = enn.ReLU(self.block2.out_type, inplace=True)

        self.block3 = enn.R2Conv(self.relu2.out_type,
                                 enn.FieldType(r2_act, 32 * [r2_act.regular_repr]),
                                 kernel_size=3,
                                 padding=1,
                                 stride=2,  # ← ストライド2
                                 bias=False)
        self.relu3 = enn.ReLU(self.block3.out_type, inplace=True)

        self.out_type = self.relu3.out_type

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, H, W]
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.relu1(x)
        x = self.block2(x)
        x = self.relu2(x)
        x = self.block3(x)
        x = self.relu3(x)
        return x.tensor  # [B, C, 28, 28] になるはず
    

# # ---------- CNN特徴抽出器 ----------
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         model = models.resnet18(pretrained=True)
#         self.features = nn.Sequential(*list(model.children())[:6])  # conv1〜layer2

#     def forward(self, x):
#         return self.features(x)  # [B, C, H, W]



# ---------- 既学習確認 ----------
def already_trained(category, save_root="learning"):
    save_dir = os.path.join(save_root, category)
    pca_path = os.path.join(save_dir, "pca_models.pkl")
    mean_path = os.path.join(save_dir, "mean_vectors.pkl")
    return os.path.exists(pca_path) and os.path.exists(mean_path)


# ---------- 正常画像読み込み ----------
def load_images_from_folder(folder, img_size=(224, 224), max_images=300):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith((".png", ".jpg")):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                if len(images) >= max_images:
                    break
    return images


# ---------- 特徴抽出 ----------
def extract_features(images, model, device):
    model.eval()
    features_by_position = {}
    for img in tqdm(images, desc="特徴抽出中"):
        img_resized = cv2.resize(img, (224, 224))
        tensor = transforms.ToTensor()(img_resized).to(device)  # shape: [1, 1, H, W]


        with torch.no_grad():
            feat = model(tensor)[0].cpu().numpy()

        C, H, W = feat.shape
        for i in range(H):
            for j in range(W):
                key = (i, j)
                vec = feat[:, i, j]
                features_by_position.setdefault(key, []).append(vec)

    return features_by_position


# ---------- PCA学習 ----------
def train_pca_model(features_by_position, n_components=0.95):
    pca_models = {}
    mean_vectors = {}
    total_positions = len(features_by_position)
    #print(f"PCA学習開始: 全{total_positions}位置で処理を実施")

    for i, (pos, features) in enumerate(features_by_position.items()):
        #print(f"[{i+1}/{total_positions}] 位置 {pos} の特徴数: {len(features)}")
        try:
            X = np.array(features)
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(X_centered)
            pca_models[pos] = pca
            mean_vectors[pos] = mean
            #print(f"  => PCA学習成功")
        except Exception as e:
            print(f"  => エラー発生: {e}")
    
    #print("PCA学習終了")
    return pca_models, mean_vectors


def get_all_categories(data_root):
    """
    MVTec形式の各カテゴリを列挙する
    例: data_root = "/path/to/mvtec_anomaly_detection"
    """
    return sorted([
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)) and
           os.path.exists(os.path.join(data_root, name, "train", "good"))
    ])

def save_learning_results(pca_models, mean_vectors, features_by_position, category, save_root):
    save_dir = os.path.join(save_root, category)
    os.makedirs(save_dir, exist_ok=True)

    # PCAモデル保存
    with open(os.path.join(save_dir, "pca_models.pkl"), "wb") as f:
        pickle.dump(pca_models, f)

    # 平均ベクトル保存
    with open(os.path.join(save_dir, "mean_vectors.pkl"), "wb") as f:
        pickle.dump(mean_vectors, f)

    # 特徴マップ保存
    npz_path = os.path.join(save_dir, "features_by_position.npz")
    np.savez_compressed(npz_path, **{
        f"{i}_{j}": np.array(vecs)
        for (i, j), vecs in features_by_position.items()
    })

    print(f"\n学習結果を保存しました → {save_dir}")


if __name__ == "__main__":
    # MVTecのルート
    data_root = "/home/zin/kobayashi_ws/src/MahalanobisAD-pytorch/data/mvtec_anomaly_detection"
    save_root = "/home/zin/kobayashi_ws/src/MahalanobisAD-pytorch/src/learning"
    overwrite = False

    categories = get_all_categories(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EquivariantFeatureExtractor(N=8).to(device)

    for category in categories:
        print(f"\n--- カテゴリ: {category} ---")

        if not overwrite and already_trained(category, save_root):
            print(" → 既に学習済み。スキップします。")
            continue

        # 正常画像のパス
        image_folder = os.path.join(data_root, category, "train", "good")
        images = load_images_from_folder(image_folder)
        print(f"  - 画像読み込み数: {len(images)}")
        features_by_position = extract_features(images, model, device)

        print("  - PCA学習中...")
        pca_models, mean_vectors = train_pca_model(features_by_position)

        save_learning_results(pca_models, mean_vectors, features_by_position, category, save_root)
        print(f"  - {category} の学習結果を保存しました。")