#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#
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
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

plt.rcParams.update({'font.size': 10})

CLASSLABEL = [
    'domestic sound',       # 0
    'male speech',          # 1
    'musical instrument',   # 2
    'clapping',             # 3
    'music',                # 4
    'laghter',              # 5
    'walk & footsteps',     # 6
    'knock',                # 7
    'water tap',            # 8
    'door',                 # 9
    'female speech',        # 10
    'bell',                 # 11
    'telephone',            # 12
]

def calculate_loss(cls_out, doa_out, cls_label, doa_label):
    cls_loss = torch.nn.functional.cross_entropy(cls_out, cls_label)
    doa_loss = torch.nn.functional.mse_loss(doa_out, doa_label)
    return cls_loss + doa_loss

def loss_fn(cls_out, doa_out, cls_label, doa_label):
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


def save_ASA_confusion_matrix(cls_true, cls_pred, filename="confusion_matrix.png"):
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
    labels = ['silence'] + CLASSLABEL
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=90)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_out(yr_hat, sed_out, doa_out, yr, sed_label, doa_label, onoff_out, path):
    # change dtype bfloat16 to float32
    yr_hat, sed_out, doa_out, yr, sed_label, doa_label = yr_hat.float(), sed_out.float(), doa_out.float(), yr.float(), sed_label.float(), doa_label.float()
    B, S, T, C = sed_out.shape
    yr = yr.detach().cpu().numpy()
    yr_hat = yr_hat.detach().cpu().numpy()
    sed_out = sed_out.detach().cpu().numpy()
    sed_label = sed_label.detach().cpu().numpy()
    doa_out = doa_out.detach().cpu().numpy()
    doa_label = doa_label.detach().cpu().numpy()
    onoff_out = onoff_out.detach().cpu().numpy()

    fig, axs = plt.subplots(S, 6, figsize=(30, 24))  # source별 subplot 생성
    
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
        for dim, axis in enumerate(['X', 'Y', 'Z']):
            # axs[s, 2].plot(onoff_out[0, s, :] * doa_out[0, s, :, dim], label=f'Pred {axis}', linestyle='--')
            axs[s, 2].plot(doa_out[0, s, :, dim], label=f'Pred {axis}', linestyle='--')
        axs[s, 2].set_title(f'Source {s+1} DoA (Prediction)')
        axs[s, 2].legend(loc='upper right')
        axs[s, 2].set_xlabel('Time stamps')
        axs[s, 2].set_ylabel('DoA Output')
        axs[s, 2].set_ylim([-1, 1])

        for dim, axis in enumerate(['X', 'Y', 'Z']):
            axs[s, 3].plot(doa_label[0, s, :, dim], label=f'GT {axis}', alpha=0.7)
        axs[s, 3].set_title(f'Source {s+1} DoA (Ground Truth)')
        axs[s, 3].legend(loc='upper right')
        axs[s, 3].set_xlabel('Time stamps')
        axs[s, 3].set_ylabel('DoA Output')
        axs[s, 3].set_ylim([-1, 1])

        # 3. Spectrogram subplot (yr_hat의 절댓값을 사용하여 시각화)
        f_yr_hat, t_yr_hat, Sxx_yr_hat = spectrogram(yr_hat[0, s, :], fs=640, window='hann', nperseg=640, noverlap=320)
        pcm_yr_hat = axs[s, 4].pcolormesh(t_yr_hat, f_yr_hat, 10 * np.log10(Sxx_yr_hat + 1e-8), shading='gouraud', vmin=-60, vmax=0)
        axs[s, 4].set_title(f'Source {s+1} Spectrogram (Prediction)')
        axs[s, 4].set_xlabel('Time')
        axs[s, 4].set_ylabel('Frequency')
        fig.colorbar(pcm_yr_hat, ax=axs[s, 4])

        f_yr, t_yr, Sxx_yr = spectrogram(yr[0, s, :], fs=640, window='hann', nperseg=640, noverlap=320)
        pcm_yr = axs[s, 5].pcolormesh(t_yr, f_yr, 10 * np.log10(Sxx_yr + 1e-8), shading='gouraud', vmin=-60, vmax=0)
        axs[s, 5].set_title(f'Source {s+1} Spectrogram (Ground Truth)')
        axs[s, 5].set_xlabel('Time')
        axs[s, 5].set_ylabel('Frequency')
        fig.colorbar(pcm_yr_hat, ax=axs[s, 5])

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def scatter_plot(sdr_data, path):
    for class_name, data in sdr_data.items():
        input_sdr = data['input']
        output_sdr = data['output']
        
        plt.figure()
        plt.scatter(input_sdr, output_sdr, alpha=0.6, edgecolors='k')
        plt.title(f'SI-SDR Scatter Plot for {CLASSLABEL[class_name]}', fontdict={'fontsize': 12})
        plt.xlabel('SI-SDR (input)')
        plt.ylabel('SI-SDR (output)')
        ax = plt.gca()  # 현재 축을 가져오기
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        plt.grid(True)
        
        # 현재 위치에 파일 저장
        save_path = os.path.join(path, f'{class_name}_SI_SDR_scatter.png')
        plt.savefig(save_path, dpi=400)
        plt.close()  # 그래프를 닫아 메모리 절약


def scatter_plot_cls(cls_data, path):

    num_bins = 16

    for class_name, data in cls_data.items():
        sdr_imp = np.array(data['imp'])
        pred = np.array(data['pred'])

        total_sdr = np.mean(sdr_imp)
        total_correct = np.sum(pred == class_name)
        total_accuracy = total_correct / len(pred)

        imp_min, imp_max = -40, 40
        bin_edges = np.linspace(imp_min, imp_max, num_bins + 1)
        y_ticks = np.arange(0, 1.1, 0.2)

        accuracies = []
        bin_labels = []
        totals = []

        for b in range(num_bins):
            start = bin_edges[b]
            end = bin_edges[b + 1]

            in_bin = (sdr_imp >= start) & (sdr_imp < end)
            total = np.sum(in_bin)
            totals.append(total)

            if total > 0:
                correct = np.sum(pred[in_bin] == class_name)
                accuracy = correct / total
            else:
                accuracy = 0.0

            accuracies.append(accuracy)
            bin_labels.append(f'{start:.2f} - {end:.2f}')

        # ticks = np.arange(imp_min, imp_max + 1, 9)
        bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.figure()
        bars = plt.bar(bin_center, accuracies, width=(bin_edges[1] - bin_edges[0]) * 0.8, color='skyblue')
        for x, bar, total in zip(range(num_bins), bars, totals):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, 0.01, f'{total}', ha='center', va='bottom', fontsize=8, color='black')
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='black')
        plt.ylim(0, 1.1)
        plt.text(
            imp_min + 0.5, 1.05,
            f'Mean SI-SDRi: {total_sdr:.2f} dB\nTotal Accuracy: {total_accuracy:.2%}',
            fontsize=9, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
        plt.title(f'Accuracy per SI-SDR Improvement for {CLASSLABEL[int(class_name)]}', fontdict={'fontsize': 12})
        plt.xlabel('SI-SDR Improvement (dB)')
        plt.ylabel('Accuracy')
        plt.xticks(bin_edges, [f"{bc:.1f}" for bc in bin_edges], fontsize=8, rotation=45)
        plt.yticks(y_ticks, [f"{yt:.1f}" for yt in y_ticks], fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(path, f'{class_name}_accuracy_per_sisdri.png')
        plt.savefig(save_path, dpi=400)
        plt.close()


def box_plot(sdr_data, path):
    sdr_improvements = {}
    for class_index, data in sdr_data.items():
        si_sdr_input = data['input']
        si_sdr_output = data['output']
        si_sdri = [output - input for input, output in zip(si_sdr_input, si_sdr_output)]
        sdr_improvements[class_index] = si_sdri

    # 박스플롯 생성
    plt.figure(figsize=(10, 8))
    # plt.figure()
    plt.boxplot(
        [sdr_improvements[class_index] for class_index in sorted(sdr_improvements.keys())],
        labels=[f'{CLASSLABEL[class_index]}' for class_index in sorted(sdr_improvements.keys())]
    )
    plt.title('SI-SDR Improvement (SI-SDRi) per Class')
    plt.xlabel('Class')
    plt.ylabel('SI-SDR Improvement (dB)')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(-22, 45)
    plt.tight_layout()
    # 현재 위치에 파일 저장
    save_path = os.path.join(path, f'SI_SDRi_boxplot.png')
    plt.savefig(save_path, dpi=400)
    plt.close()  # 그래프를 닫아 메모리 절약


def violin_plot(nspk_data, path):
    df = pd.DataFrame(nspk_data)
    # violin plot 생성
    plt.figure(figsize=(6, 8))
    ax = sns.violinplot(x='n_spk', y='si_sdri', data=df, inner='point', palette='Set2')

    # 중앙값을 계산하고 각 위치에 강조 표시
    for i, n_spk_value in enumerate(sorted(df['n_spk'].unique())):  # 각 n_spk 그룹에 대해 중앙값 계산
        median_value = df[df['n_spk'] == n_spk_value]['si_sdri'].median()
        ax.scatter(i, median_value, color='black', s=100, marker='o', zorder=3)  # 중앙값을 강조하여 점으로 표시
        # 중앙값 위에 값을 표기
        ax.text(i, median_value + 1, f'{median_value:.2f}', color='black', ha='center', va='bottom', fontsize=10)

    # 축 및 제목 설정
    plt.xlabel('J (Number of speakers)')
    plt.ylabel('SI-SDRi')
    plt.title('SI-SDRi distribution by number of speakers')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(-40, 50)

    # Figure 저장
    plt.savefig(path, dpi=400)
    plt.close()


def save_confusion_matrix_nspk(n_spk, n_spk_hat, path):
    # 텐서를 numpy 배열로 변환
    n_spk_np = np.array(n_spk)
    n_spk_hat_np = np.array(n_spk_hat)
    conf_mat = confusion_matrix(n_spk_np, n_spk_hat_np, labels=[1, 2, 3, 4, 5], normalize='true')
    # confusion matrix를 데이터프레임으로 변환 (시각화 용이하게 하기 위해)
    conf_mat_df = pd.DataFrame(conf_mat, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

    # heatmap으로 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, annot_kws={'size': 14})
    plt.xlabel('Predicted Number of Speakers (n_spk_hat)', fontsize=14)
    plt.ylabel('Actual Number of Speakers (n_spk)', fontsize=14)
    plt.title('Confusion Matrix for Predicted vs Actual Number of Speakers', fontsize=16)
    # Figure 저장
    plt.savefig(path, dpi=400)
    plt.close()


def plot_le_histogram(errors, path):
    """
    errors: list or numpy array of LE values in degrees (range 0~180)
    save_path: file path to save the histogram image
    """
    errors = np.array(errors)
    bins = np.arange(0, 185, 5)  # [0, 5, ..., 180]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, edgecolor='black', color='skyblue')
    plt.xticks(bins, rotation=45)
    plt.xlim(0, 180)
    plt.ylim(0, 700)
    plt.xlabel("Localization Error (degrees)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Histogram of Localization Errors", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 저장
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


