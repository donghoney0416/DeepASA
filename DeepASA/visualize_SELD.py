import numpy as np
import os
import sys
import torch
from IPython import embed
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
from scipy.signal import spectrogram
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from models.utils.SELD_evaluation_metrics import *
from sklearn.decomposition import PCA

CLASS_LABEL = [
    'female speech',        # 0
    'male speech',          # 1
    'clapping',             # 2
    'telephone',            # 3
    'laughter',             # 4
    'domestic sound',       # 5
    'walk & footsteps',     # 6
    'door',                 # 7
    'music',                # 8
    'musical instrument',   # 9
    'water tap',            # 10
    'bell',                 # 11
    'knock'                 # 12
]

def calculate_loss(cls_out, doa_out, cls_label, doa_label):
    cls_loss = torch.nn.functional.cross_entropy(cls_out, cls_label)
    doa_loss = torch.nn.functional.mse_loss(doa_out, doa_label)
    return cls_loss + doa_loss

def vis_loss_fn(cls_out, doa_out, cls_label, doa_label):
    B, S, C = cls_out.shape
    _, _, T, _ = doa_out.shape

    # All possible permutations of S sources
    permutations = list(itertools.permutations(range(S)))
    total_loss = 0.
    best_permutations = []
    for b in range(B):
        min_loss = float('inf')  # Initialize with a very large number
        best_perm = None
        # Compute loss for each permutation
        for perm in permutations:
            permuted_cls_out = cls_out[b, list(perm), :]
            permuted_doa_out = doa_out[b, list(perm), :, :]

            loss = calculate_loss(permuted_cls_out, permuted_doa_out, cls_label[b], doa_label[b])
            # Track the minimum loss over all permutations
            if loss < min_loss:
                min_loss = loss
                best_perm = perm

        total_loss += min_loss
        best_permutations.append(best_perm)
    total_loss = total_loss / B

    return total_loss, best_permutations 

def get_class_name(cls_label, s):
    """cls_label 텐서에서 클래스 이름을 반환"""
    label_idx = cls_label[0, s].item()  # 텐서에서 숫자 값 추출
    
    if label_idx == -1:
        return "silence"
    elif 0 <= label_idx < len(CLASS_LABEL):
        return CLASS_LABEL[label_idx]
    else:
        return "Unknown"  # 범위를 벗어난 경우 예외 처리
    
def visualize_att_map(attn_map, path, name):
    for head in range(attn_map.shape[1]):
        plt.figure(figsize=(5, 5))
        attn = attn_map[0, head].cpu().detach().numpy()
        # normalize the attention map
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        plt.imshow(attn, cmap='viridis')
        # plt.title(f'Head {head}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{path}/attention_head_{head}_{name}.png")
        plt.close()


def visualize_sed(onoff_out, pp_onoff_out, doa_out, doa_label, delta, max_idx, min_idx, cls_out, cls_label, path):
    x = np.linspace(0, 4, 40)
    threshold = 0.5
    B, S, T = onoff_out.shape                   # [1, 5, 40]
    frame_scores = torch.sigmoid(onoff_out)
    sed_out = (frame_scores > threshold).float()
    true_events = torch.norm(doa_label, dim=-1) > 0
    onoff_out = onoff_out.detach().cpu().numpy()
    pp_onoff_out = pp_onoff_out.detach().cpu().numpy()
    doa_onoff_out = doa_out.norm(dim=-1).detach().cpu().numpy()
    doa_label = doa_label.detach().cpu().numpy()

    est_cls_out = cls_out.argmax(dim=-1)
    confidence_cls_out = cls_out.softmax(dim=-1)

    fig, axs = plt.subplots(S, 1, figsize=(10, 20))
    for s in range(S):
        axs[s].plot(x, frame_scores[0, s].detach().cpu().numpy(), label='Frame-level scores', color='blue')                         # Frame score (0~1)
        axs[s].plot(x, doa_onoff_out[0, s], label='DoA: Predicted On/Off', color='brown', linestyle='dashdot')                      # predicted DoA (on/off)
        axs[s].plot(x, sed_out[0, s].detach().cpu().numpy(), label='SED: Predicted Events', color='red', linestyle='dashdot')        # predicted SED (binary)
        axs[s].plot(x, true_events[0, s].detach().cpu().numpy(), label="True Events", color="green", linestyle="dashed")            # ground truth SED (binary)
        axs[s].plot(x, delta[0, s].detach().cpu().numpy(), label='Delta', color='purple', linestyle='dashed')
        axs[s].axhline(y=threshold, color="black", linestyle="--", label="Threshold (0.5)")  # 임계값
        ## proposed on/offset
        axs[s].plot(x, pp_onoff_out[0, s], label='Proposed On/Off', color='orange', linestyle='dotted')
        # Peak 위치 표시
        # max_idx[s]와 min_idx[s]는 리스트 형태이므로 텐서로 변환 후 scatter() 적용
        if len(max_idx[0][s]) > 0:
            axs[s].scatter(x[max_idx[0][s].cpu().numpy()], delta[0, s, max_idx[0][s]].detach().cpu().numpy(), color='purple', marker='o', s=100, label='Local Maxima')
        if len(min_idx[0][s]) > 0:
            axs[s].scatter(x[min_idx[0][s].cpu().numpy()], delta[0, s, min_idx[0][s]].detach().cpu().numpy(), color='purple', marker='s', s=100, label='Local Minima')

        # 라벨 및 저장
        axs[s].set_xlabel("Time (s)")
        axs[s].set_ylabel("Confidence Score")
        axs[s].set_ylim([-1, 2])
        axs[s].set_title(f"Frame-level Scores and SED Events (Source {s+1}, Pred {get_class_name(est_cls_out, s)}, Conf {confidence_cls_out[0, s].max(dim=-1)[0].item()}, GT {get_class_name(cls_label, s)})")
        axs[s].legend()
        axs[s].grid()

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def visualize_out(yr_hat, sed_out, doa_out, sed_label, doa_label, onoff_out, path):
    # change dtype bfloat16 to float32
    yr_hat, sed_out, doa_out, sed_label, onoff_out, doa_label = yr_hat.float(), sed_out.float(), doa_out.float(), sed_label.float(), onoff_out, doa_label.float()
    B, S, T, C = sed_out.shape
    yr_hat = yr_hat.detach().cpu().numpy()
    sed_out = sed_out.detach().cpu().numpy()
    sed_label = sed_label.detach().cpu().numpy()
    doa_out = doa_out.detach().cpu().numpy()
    doa_label = doa_label.detach().cpu().numpy()
    onoff_out = onoff_out.detach().cpu().numpy()

    fig, axs = plt.subplots(S, 5, figsize=(30, 20))  # source별 subplot 생성
    
    for s in range(S):
        # 1. Sound Event Detection (SED) subplot
        axs[s, 0].imshow(sed_out[0, s, :, :].T, aspect='auto', origin='lower', cmap='gray', interpolation='nearest')
        axs[s, 0].set_title(f'Source {s+1} SED (Prediction)')
        axs[s, 0].set_xlabel('Time stamps')
        axs[s, 0].set_ylabel('Classes (C)')

        axs[s, 1].imshow(sed_label[0, s, :, :].T, aspect='auto', origin='lower', cmap='gray', interpolation='nearest')
        axs[s, 1].set_title(f'Source {s+1} SED (Ground Truth)')
        axs[s, 1].set_xlabel('Time stamps')
        axs[s, 1].set_ylabel('Classes (C)')
    
        # 2. DoA subplot (각 source의 x, y, z 축 비교)
        # for dim, axis in enumerate(['X', 'Y', 'Z']):
        #     axs[s, 2].plot(onoff_out[0, s, :, 0] * doa_out[0, s, :, dim], label=f'Pred {axis}', linestyle='--')
        # axs[s, 2].set_title(f'Source {s+1} DoA (Prediction)')
        # axs[s, 2].legend(loc='upper right')
        # axs[s, 2].set_xlabel('Time stamps')
        # axs[s, 2].set_ylabel('DoA Output')
        # axs[s, 2].set_ylim([-1, 1])

        # for dim, axis in enumerate(['X', 'Y', 'Z']):
        #     axs[s, 3].plot(doa_label[0, s, :, dim], label=f'GT {axis}', alpha=0.7)
        # axs[s, 3].set_title(f'Source {s+1} DoA (Ground Truth)')
        # axs[s, 3].legend(loc='upper right')
        # axs[s, 3].set_xlabel('Time stamps')
        # axs[s, 3].set_ylabel('DoA Output')
        # axs[s, 3].set_ylim([-1, 1])
        # 2. DoA subplot (azimuth/elevation 기준)
        x_pred, y_pred, z_pred = doa_out[0, s, :, 0], doa_out[0, s, :, 1], doa_out[0, s, :, 2]
        ele_pred = np.rad2deg(np.arcsin(z_pred))
        azi_pred = np.rad2deg(np.arctan2(y_pred, x_pred))

        x_gt, y_gt, z_gt = doa_label[0, s, :, 0], doa_label[0, s, :, 1], doa_label[0, s, :, 2]
        ele_gt = np.rad2deg(np.arcsin(z_gt))
        azi_gt = np.rad2deg(np.arctan2(y_gt, x_gt))

        # Prediction
        axs[s, 2].plot(onoff_out[0, s, :, 0] * azi_pred, label='Pred Azimuth', linestyle='--')
        axs[s, 2].plot(onoff_out[0, s, :, 0] * ele_pred, label='Pred Elevation', linestyle='--')
        axs[s, 2].set_title(f'Source {s+1} DoA (Prediction)')
        axs[s, 2].legend(loc='upper right')
        axs[s, 2].set_xlabel('Time stamps')
        axs[s, 2].set_ylabel('Degrees')
        axs[s, 2].set_ylim([-100, 100])  # 예시 범위, 필요시 조정

        # Ground Truth
        axs[s, 3].plot(azi_gt, label='GT Azimuth', alpha=0.7)
        axs[s, 3].plot(ele_gt, label='GT Elevation', alpha=0.7)
        axs[s, 3].set_title(f'Source {s+1} DoA (Ground Truth)')
        axs[s, 3].legend(loc='upper right')
        axs[s, 3].set_xlabel('Time stamps')
        axs[s, 3].set_ylabel('Degrees')
        axs[s, 3].set_ylim([-100, 100])

        # 3. Spectrogram subplot (yr_hat의 절댓값을 사용하여 시각화)
        f_yr_hat, t_yr_hat, Sxx_yr_hat = spectrogram(yr_hat[0, s, :], fs=640, window='hann', nperseg=640, noverlap=320)
        pcm_yr_hat = axs[s, 4].pcolormesh(t_yr_hat, f_yr_hat, 10 * np.log10(Sxx_yr_hat + 1e-8), shading='gouraud', vmin=-60, vmax=0)
        axs[s, 4].set_title(f'Source {s+1} Spectrogram (Prediction)')
        axs[s, 4].set_xlabel('Time')
        axs[s, 4].set_ylabel('Frequency')
        fig.colorbar(pcm_yr_hat, ax=axs[s, 4])

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'plot.png'), dpi=400)
    plt.close()


def save_er_histogram(er_values, filename="er_histogram.png"):
    """
    Error Rate (ER) 값으로 히스토그램을 그리고 PNG 파일로 저장하는 함수
    
    Args:
    er_values (torch.Tensor): Error Rate 값들이 들어있는 텐서
    filename (str): 저장할 PNG 파일 이름 (기본값: "er_histogram.png")
    """
    # Convert the tensor to a numpy array for plotting
    er_values_np = er_values.cpu().numpy()

    # Create a histogram with bins from 0 to 1, with a step size of 0.05
    bins = [i * 0.05 for i in range(21)]  # 0, 0.05, 0.10, ..., 1.00
    plt.hist(er_values_np, bins=bins, edgecolor='black')

    # Add title and labels
    plt.title('Error Rate (ER) Histogram')
    plt.xlabel('Error Rate')
    plt.ylabel('Number of Samples')

    # Save the plot as a PNG file
    plt.savefig(filename)
    plt.close()


def save_le_histogram(le_values, filename="le_histogram.png"):
    """
    Localization Error (LE) 값으로 히스토그램을 그리고 PNG 파일로 저장하는 함수
    
    Args:
    le_values (torch.Tensor): Localization Error 값들이 들어있는 텐서
    filename (str): 저장할 PNG 파일 이름 (기본값: "le_histogram.png")
    """
    # Convert the tensor to a numpy array for plotting
    le_values_np = le_values.cpu().numpy()

    # Create a histogram with bins from 0 to 180, with a step size of 10
    plt.hist(le_values_np, bins=range(0, 181, 5), edgecolor='black')

    # Add title and labels
    plt.title('Localization Error (LE) Histogram')
    plt.xlabel('Localization Error (degree)')
    plt.ylabel('Number of Samples')

    # Save the plot as a PNG file
    plt.savefig(filename)
    plt.close()


def save_confusion_matrix(cls_true, cls_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(cls_true.flatten(), cls_pred.flatten(), normalize='true')
    # New index order
    # new_order = [0, 11, 9, 13, 1, 7, 6, 5, 8, 10, 3, 2, 12, 4]
    # cm = cm[np.ix_(new_order, new_order)]
    # Save confusion matrix as a heatmap
    plt.figure(figsize=(20, 14))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    label = ['silence'] + CLASS_LABEL
    plt.xticks(np.arange(len(label)) + 0.5, label, rotation=90)
    plt.yticks(np.arange(len(label)) + 0.5, label, rotation=0)
    plt.tight_layout()
    plt.rcParams.update({'font.size': 13})
    plt.savefig(filename)
    plt.close()


def plot_pca_from_feature_tensor(feature_tensor, save_path="pca_single_layer.png", cmap='Spectral'):
    """
    feature_tensor: numpy array or torch.Tensor of shape (B, F, C)
    save_path: file path to save the PCA plot
    """
    if hasattr(feature_tensor, 'detach'):
        feature_tensor = feature_tensor.detach().cpu().numpy()

    B, F, C = feature_tensor.shape
    BF, F, C = feature_tensor.shape
    features_reshaped = feature_tensor.reshape(-1, C)  # shape: [B*F, C]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_reshaped)  # shape: [B*F, 2]
    # 각 sample의 frequency index (0 ~ F-1)를 반복해서 만들어줌
    freq_bins = np.tile(np.arange(F), BF)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=freq_bins, cmap=cmap, s=10)
    plt.title("PCA of Frequency-wise Feature (B,F,C)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Frequency bin")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✔] PCA plot saved to {save_path}")


def plot_pca_per_class(feature_tensor, label_tensor, class_names=CLASS_LABEL, save_path="./pca_all_classes.png", cmap='Spectral'):
    """
    feature_tensor: (N, F, C) numpy array or torch.Tensor
    label_tensor: (N,) numpy array or torch.Tensor of class indices
    class_names: list of 13 class names (or index strings)
    save_path: path to save the resulting image
    """
    B, Q, C = feature_tensor.shape
    label_tensor = label_tensor.squeeze(0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N_class = len(class_names)
    rows = (N_class + 3) // 4  # 최대 4개 열
    cols = min(N_class, 4)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    for i, cls in enumerate(range(N_class)):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]

        cls_mask = label_tensor == cls
        cls_feat = feature_tensor[cls_mask]  # shape: [Nc, F, C]
        if len(cls_feat) == 0:
            ax.set_title(f"{class_names[cls]} (empty)")
            ax.axis('off')
            continue

        features_reshaped = feature_tensor.reshape(-1, N_class)  # shape: [B*F, C]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_reshaped)  # shape: [B*F, 2]
        # 각 sample의 frequency index (0 ~ F-1)를 반복해서 만들어줌
        freq_bins = np.tile(np.arange(Q), B)
        sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=freq_bins, cmap=cmap, s=10)
        ax.set_title(f"{class_names[cls]}")
        ax.set_xticks([])
        ax.set_yticks([])

    # 공통 colorbar
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
    cbar.set_label("Frequency bin", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✔] PCA per class plot saved to {save_path}")


def plot_tsne(features, labels, output_path, selected_classes, perplexity=30, random_state=42):
    # Remove silence
    valid_indices = labels != -1
    filtered_features = features[valid_indices]
    filtered_labels = labels[valid_indices]
    # 선택된 클래스에 해당하는 데이터 필터링
    indices = np.isin(labels, selected_classes)
    filtered_features = features[indices]
    filtered_labels = labels[indices]
    # T-SNE 적용
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(filtered_features)
    # 각 클래스의 중심 계산
    class_centers = {}
    for label in selected_classes:
        idx = filtered_labels == label
        class_center = tsne_results[idx].mean(axis=0)  # 각 클래스의 중심 계산
        class_centers[label] = class_center
    # 시각화 및 저장
    plt.figure(figsize=(10, 8))
    # 다양한 색상 설정 (클래스마다 다른 색)
    colors = plt.cm.get_cmap('tab20', len(selected_classes))  # 10가지 색상 팔레트
    for i, label in enumerate(selected_classes):
        idx = filtered_labels == label
        plt.scatter(
            tsne_results[idx, 0],
            tsne_results[idx, 1],
            label=f'Class {label}',
            color=colors(i),
            alpha=0.7
        )
        # 클래스 중심 표시
        center = class_centers[label]
        plt.scatter(
            center[0],
            center[1],
            color=colors(i),  # 클래스 색상과 동일
            marker='X',
            s=200,  # 중심 강조
            edgecolor='black',  # 테두리 추가
            linewidth=1.5,
            label=f'Class {label} Center'
        )
    # 그래프 제목 및 축 설정
    plt.title("T-SNE with Class Centers", fontsize=16)
    plt.xlabel("T-SNE Dimension 1", fontsize=12)
    plt.ylabel("T-SNE Dimension 2", fontsize=12)
    # Legend를 그래프 밖으로 이동
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
        borderaxespad=0.
    )
    plt.tight_layout()
    # 그래프 저장
    plt.savefig(output_path, dpi=400, bbox_inches='tight')


def plot_tsne_with_zoom(features, labels, output_dir, selected_classes, zoom_factor=0.1, perplexity=30, random_state=42):
    """
    T-SNE 시각화 및 각 클래스 중심 주변을 확대하여 저장.
    
    Args:
        features (np.ndarray): T-SNE 입력 특성 (N, D).
        labels (np.ndarray): 클래스 라벨 (N,).
        output_dir (str): 그래프 저장 경로.
        selected_classes (list): 시각화할 클래스 목록.
        zoom_factor (float): 중심 확대 범위 (비율).
        perplexity (int): T-SNE perplexity 값.
        random_state (int): 랜덤 시드.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Silence(-1) 제거
    valid_indices = labels != -1
    filtered_features = features[valid_indices]
    filtered_labels = labels[valid_indices]

    # 선택된 클래스 필터링
    indices = np.isin(filtered_labels, selected_classes)
    filtered_features = filtered_features[indices]
    filtered_labels = filtered_labels[indices]

    # T-SNE 적용
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(filtered_features)

    # 클래스 중심 계산
    class_centers = {}
    for label in selected_classes:
        idx = filtered_labels == label
        class_center = tsne_results[idx].mean(axis=0)
        class_centers[label] = class_center

    # 전체 T-SNE 그래프 시각화
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', len(selected_classes))
    for i, label in enumerate(selected_classes):
        idx = filtered_labels == label
        plt.scatter(
            tsne_results[idx, 0],
            tsne_results[idx, 1],
            label=f'Class {label}',
            color=colors(i),
            alpha=0.7
        )
        # 클래스 중심 표시
        center = class_centers[label]
        plt.scatter(
            center[0],
            center[1],
            color=colors(i),
            marker='X',
            s=200,
            edgecolor='black',
            linewidth=1.5,
            label=f'Class {label} Center'
        )
    plt.title("T-SNE with Class Centers", fontsize=16)
    plt.xlabel("T-SNE Dimension 1", fontsize=12)
    plt.ylabel("T-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_overall.png"), dpi=400, bbox_inches='tight')
    plt.close()

    # 각 클래스 중심 주변 확대 저장
    for label, center in class_centers.items():
        plt.figure(figsize=(8, 8))
        idx = filtered_labels == label
        plt.scatter(
            tsne_results[idx, 0],
            tsne_results[idx, 1],
            label=f'Class {label}',
            color=colors(selected_classes.index(label)),
            alpha=0.7
        )
        plt.scatter(
            center[0],
            center[1],
            color=colors(selected_classes.index(label)),
            marker='X',
            s=200,
            edgecolor='black',
            linewidth=1.5,
            label=f'Class {label} Center'
        )

        # 확대 범위 설정
        zoom_x_min = center[0] - zoom_factor * (tsne_results[:, 0].max() - tsne_results[:, 0].min())
        zoom_x_max = center[0] + zoom_factor * (tsne_results[:, 0].max() - tsne_results[:, 0].min())
        zoom_y_min = center[1] - zoom_factor * (tsne_results[:, 1].max() - tsne_results[:, 1].min())
        zoom_y_max = center[1] + zoom_factor * (tsne_results[:, 1].max() - tsne_results[:, 1].min())
        plt.xlim(zoom_x_min, zoom_x_max)
        plt.ylim(zoom_y_min, zoom_y_max)

        plt.title(f"T-SNE Zoom on Class {label}", fontsize=16)
        plt.xlabel("T-SNE Dimension 1", fontsize=12)
        plt.ylabel("T-SNE Dimension 2", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"tsne_zoom_class_{label}.png"), dpi=400, bbox_inches='tight')
        plt.close()


def plot_onoff_histogram(features, onoff_label, filename):
    """
    Features와 onoff_label 데이터를 기반으로 onset/offset에 대한 anomaly score 히스토그램을 생성합니다.
    두 히스토그램이 겹치는 영역은 반투명하게 강조됩니다.
    
    Args:
    features (torch.Tensor or np.ndarray): (S, T) 형태의 feature 데이터 (0~1 값).
    onoff_label (torch.Tensor or np.ndarray): (S, T) 형태의 라벨 데이터 (0 또는 1).
    filename (str): 저장할 파일 이름.
    """
    # Convert to numpy if inputs are torch tensors
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(onoff_label, torch.Tensor):
        onoff_label = onoff_label.cpu().numpy()

    # Flatten features and labels for histogram
    features_flat = features.flatten()
    onoff_label_flat = onoff_label.flatten()

    # Define bins for histogram (0 to 1 with step 0.02)
    bins = np.arange(0, 1.02, 0.02)

    # Calculate histograms for offset (label=0) and onset (label=1)
    offset_values = features_flat[onoff_label_flat == 0]
    onset_values = features_flat[onoff_label_flat == 1]

    offset_hist, _ = np.histogram(offset_values, bins=bins)
    onset_hist, _ = np.histogram(onset_values, bins=bins)

    # Normalize histograms to get proportions
    offset_hist_normalized = offset_hist / (offset_hist.sum() + 1e-8)  # Avoid division by zero
    onset_hist_normalized = onset_hist / (onset_hist.sum() + 1e-8)

    # Find intersection (minimum of both normalized histograms)
    intersection = np.minimum(offset_hist_normalized, onset_hist_normalized)

    # Plot normalized histograms
    plt.figure(figsize=(12, 6))
    plt.bar(bins[:-1], offset_hist_normalized, width=0.02, color='orange', alpha=0.6, label='Offset (Label 0)', align='edge')
    plt.bar(bins[:-1], onset_hist_normalized, width=0.02, color='navy', alpha=0.6, label='Onset (Label 1)', align='edge')

    # Highlight intersection area
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, intersection, width=(bins[1] - bins[0]), color='purple', alpha=0.4, label='False Positive or Negative')

    # Title and labels
    plt.title("Onset/Offset Feature Value Histogram (Normalized)")
    plt.xlabel("Feature Value (0-1)")
    plt.ylabel("Proportion")
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.75)

    # Save plot
    plt.savefig(filename)
    plt.close()


plt.rcParams.update({'font.size': 10})


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


