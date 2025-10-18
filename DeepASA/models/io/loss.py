from torch import Tensor
import torch
from torch import nn

from torch.nn import functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio as sa_sdr
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate as permutate
from torch.nn.functional import cross_entropy as ce
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from collections import Counter
import math
from torch.nn import Parameter

from typing import *

class AudioClassificationLoss(nn.Module):
    def __init__(self, num_classes=13, label_smoothing=0.0, keep_batch=False):
        super(AudioClassificationLoss, self).__init__()
        self.silence_target_prob = 1/num_classes
        self.label_smoothing = label_smoothing
        self.keep_batch = keep_batch

    def forward(self, output, target):
        """
        output: (batch_size, num_classes) - Model's predicted probabilities/logits.
        target: (batch_size,) - Ground truth labels, with -1 indicating silence (OOD).
        """
        return audioclassification_loss(output, target, self.silence_target_prob, self.label_smoothing, self.keep_batch)


def audioclassification_loss_(output, target, silence_target_prob=1/13, label_smoothing=0.0, keep_batch=False):
    B, C = output.size()
    loss = [] if keep_batch else 0.
    for batch_idx in range(B):
        # target이 0 ~ 12인 경우 일반 Cross-Entropy Loss 사용
        if target[batch_idx] != -1:
            loss_ = torch.nn.functional.cross_entropy(output[batch_idx].unsqueeze(0), target[batch_idx].unsqueeze(0), label_smoothing=label_smoothing)
        else:
            # target이 -1인 경우 Silence Loss 사용
            silence_output = output[batch_idx]
            silence_target = torch.full_like(silence_output, silence_target_prob)
            loss_ = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(silence_output, dim=0), silence_target, reduction='batchmean')
        if keep_batch:
            loss.append(loss_)
        else:
            loss = loss + loss_
    loss = torch.stack(loss, dim=0) if keep_batch else loss / B
    return loss

def audioclassification_loss(output, target, silence_target_prob=1/13, label_smoothing=0.0, keep_batch=False):
    """
    Args:
        output: shape (B, C) - 배치 크기 B, 클래스 수 C
        target: shape (B,)   - 각각의 샘플에 대한 클래스 라벨 (0~12),
                              Silence 표시는 -1
        silence_target_prob: Silence 일 때 사용하는 목표 확률 (균등분포 비율)
        label_smoothing: Cross Entropy 시의 라벨 스무딩 정도
        keep_batch: True면 모든 샘플별 loss를 (B,)로 반환, False면 평균값(스칼라) 반환
    """
    B, C = output.shape

    # 1) Silence와 일반 클래스 인덱스를 분리
    normal_idx = (target != -1).nonzero(as_tuple=True)[0]   # shape [N1]
    silence_idx = (target == -1).nonzero(as_tuple=True)[0]  # shape [N2]

    # 2) 일반 클래스 샘플에 대한 Cross Entropy
    #    reduction='none'로 각 샘플별 loss (shape [N1]) 반환
    if normal_idx.numel() > 0:
        normal_output = output[normal_idx]               # shape [N1, C]
        normal_target = target[normal_idx]               # shape [N1]
        normal_loss = F.cross_entropy(
            normal_output,
            normal_target,
            reduction='none',
            label_smoothing=label_smoothing
        )  # shape [N1]
    else:
        # 일반 클래스가 하나도 없으면 빈 텐서
        normal_loss = torch.zeros(0, device=output.device)

    # 3) Silence 샘플에 대한 KL-Divergence
    #    reduction='none'로 각 샘플별 loss (shape [N2]) 얻기
    if silence_idx.numel() > 0:
        silence_output = output[silence_idx]             # shape [N2, C]
        silence_target = torch.full_like(silence_output, silence_target_prob)  # shape [N2, C]
        log_probs = F.log_softmax(silence_output, dim=1) # shape [N2, C]
        # kl_div(reduction='none') => shape [N2, C]
        # 샘플마다 합(sum(dim=1)) -> shape [N2]
        silence_loss = F.kl_div(log_probs, silence_target, reduction='none').sum(dim=1)
    else:
        silence_loss = torch.zeros(0, device=output.device)

    # 4) 최종 Loss를 (B,) 형태로 모아두기
    #    - normal 샘플 위치에는 normal_loss
    #    - silence 샘플 위치에는 silence_loss
    final_loss = torch.zeros(B, device=output.device)
    final_loss[normal_idx] = normal_loss
    final_loss[silence_idx] = silence_loss

    # 5) keep_batch 옵션에 따라 반환
    #    - True면 (B,) 반환
    #    - False면 전체 평균(스칼라) 반환
    return final_loss if keep_batch else final_loss.mean()


class MSELoss(nn.Module):
    def __init__(self, keep_batch=False):
        super(MSELoss, self).__init__()
        self.keep_batch = keep_batch
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        mse = self.mse(output, target)
        return mse.mean(dim=list(range(mse.dim()))[1:]) if self.keep_batch else mse.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, keep_batch=False, Mask=False):
        super(BCEWithLogitsLoss, self).__init__()
        self.keep_batch = keep_batch
        self.Mask = Mask
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        bce = self.bce(output, target)
        # 2) Weighted loss - OnOff
        if self.Mask:
            transition_mask = F.pad(torch.abs(target[:, 1:] - target[:, :-1]), (1, 0)) + F.pad(torch.abs(target[:, 1:] - target[:, :-1]), (0, 1))
            bce = bce * (1 + 1 * transition_mask)
        return bce.mean(dim=list(range(bce.dim()))[1:]) if self.keep_batch else bce.mean()


def onoffset_loss(predicted_coords, true_coords):
    true_distances = torch.norm(true_coords, dim=-1)  # [B, S, T]
    onset_matrix = (true_distances > 0.5).float()  # 0.5보다 크면 1, 아니면 0
    predicted_distances = torch.norm(predicted_coords, dim=-1)  # [B, S, T]
    return (predicted_distances - onset_matrix).pow(2).mean()
    # return torch.nn.functional.mse_loss(predicted_distances, onset_matrix)


def soft_boundary_loss(predicted_coords, true_coords):
    # Sigmoid로 soft boundary를 적용
    predicted_prob = torch.sigmoid(torch.norm(predicted_coords, dim=-1) - 0.5)  # Centered at 0.5
    true_labels = (torch.norm(true_coords, dim=-1) > 0.5).float()  # Binary labels
    # BCE Loss로 확률적으로 학습
    return torch.nn.functional.binary_cross_entropy_with_logits(predicted_prob, true_labels)


def onoffset_loss_tanh(predicted_coords, true_coords):
    true_distances = torch.norm(true_coords, dim=-1)  # [B, S, T]
    onset_matrix = (true_distances > 0.5).float()  # Onset/Offset 기준
    predicted_coords = torch.norm(predicted_coords, dim=-1)  # [B, S, T]
    # Tanh로 부드러운 구분
    diff = predicted_coords - 0.5
    tanh_loss = torch.tanh(diff) - 2*(onset_matrix - 0.5)
    return tanh_loss.pow(2).mean()


def onoffset_loss_sigmoid(predicted_coords, true_coords):
    true_distances = torch.norm(true_coords, dim=-1)  # [B, S, T]
    onset_matrix = (true_distances > 0.5).float()  # Onset/Offset 기준
    predicted_coords = torch.norm(predicted_coords, dim=-1)  # [B, S, T]
    # Sigmoid로 부드러운 구분
    sigmoid_loss = torch.sigmoid(predicted_coords) - onset_matrix
    return sigmoid_loss.pow(2).mean()


def cossim_loss(predicted_coords, true_coords, keep_batch=False):
    # Calculate active mask based on true_coords distance
    true_distances = torch.norm(true_coords, dim=-1)
    active_mask = true_distances > 0.5
    # Normalize the vectors
    predicted_coords_normalized = torch.nn.functional.normalize(predicted_coords, dim=-1)
    true_coords_normalized = torch.nn.functional.normalize(true_coords, dim=-1)    
    # Calculate the cosine similarity
    cos_sim = torch.sum(predicted_coords_normalized * true_coords_normalized, dim=-1)
    cos_sim[~active_mask] = 0.0
    cos_sim = torch.mean(cos_sim, dim=list(range(cos_sim.dim()))[1:]) if keep_batch else torch.mean(cos_sim)
    return 1 - cos_sim


class BCELoss(nn.Module):
    def __init__(self, num_classes=2, embedding_size=128):
        super(BCELoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label, mode):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.to(x.device)))    # cos(theta)
        if mode == 'train':
            bce_loss = F.cross_entropy(cosine, label)
        else:
            bce_loss = F.cross_entropy(cosine, label)
        return bce_loss, cosine
    

class WeightedBCELoss(nn.Module):
    def __init__(self, num_classes=2, embedding_size=128, boundary_weight=5.0, continuous_weight=1.0, smooth_window=3):
        super(WeightedBCELoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.boundary_weight = boundary_weight
        self.continuous_weight = continuous_weight
        self.smooth_window = smooth_window
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def create_weight_map(self, label):
        diff = torch.abs(label[1:] - label[:-1])  # Detect changes between adjacent timesteps
        boundary_map = F.pad(diff, (1, 0))  # Pad to match input shape (batch_size, seq_len)
        kernel = torch.arange(-self.smooth_window, self.smooth_window + 1, device=label.device).abs()
        kernel = (self.smooth_window - kernel + 1).float().clamp(min=0)  # Linear smoothing kernel
        kernel /= kernel.sum()  # Normalize kernel
        smooth_map = F.conv1d(
            boundary_map.unsqueeze(0).float(),  # (batch_size, 1, seq_len)
            kernel.view(1, 1, -1),  # (1, 1, kernel_size)
            padding=self.smooth_window
        ).squeeze(1)
        weight_map = smooth_map * (self.boundary_weight - self.continuous_weight) + self.continuous_weight
        return weight_map

    def forward(self, x, label, mode):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.to(x.device)))
        if mode == 'train':
            weight_map = self.create_weight_map(label)
            bce_loss = (F.cross_entropy(x, label, reduction='none') * weight_map).mean()
        else:
            bce_loss = F.cross_entropy(cosine, label)
        return bce_loss, cosine
    

class CELoss(nn.Module):
    def __init__(self, num_classes=13, embedding_size=128, silence_target_prob=1/13):
        super(CELoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.silence_target_prob = silence_target_prob

    def forward(self, x, label):
        valid_label = label >= 0  # Mask for non-silence labels
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.to(x.device)))    # cos(theta)
        if valid_label.any():
            arcface_loss = F.cross_entropy(cosine[valid_label], label[valid_label])
        else:
            arcface_loss = 0.0
        # KL Divergence for silence labels (-1)
        silence_labels = ~valid_label  # Mask for silence labels
        silence_loss = 0.0
        if silence_labels.any():
            silence_output = cosine[silence_labels]  # Cosine values for silence labels
            silence_target = torch.full_like(silence_output, self.silence_target_prob)  # Uniform target distribution
            silence_loss = F.kl_div(F.log_softmax(silence_output, dim=1), silence_target, reduction='batchmean')
        return arcface_loss + silence_loss, cosine


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes=2, embedding_size=128, s=64, m=0.7, silence_target_prob=1/2):
        super(ArcFaceLoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))

        self.th = torch.cos(torch.tensor(math.pi - m))
        self.mm = torch.sin(torch.tensor(math.pi - m)) * m

    def forward(self, x, label, mode):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.to(x.device)))    # cos(theta)
        if mode == 'train1':
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))                           # sin(theta)
            phi = cosine * self.cos_m - sine * self.sin_m                           # cos(theta + m)
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)        # cos(theta + m) if cos(theta) > cos(pi - m) else cos(theta - m)
            # One-hot encoding
            one_hot = torch.zeros_like(cosine)                                      # one-hot encoding
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        else:
            output = cosine * self.s
        arcface_loss = F.cross_entropy(output/30, label)
        return arcface_loss, output
    

class SubCenterArcFaceLoss(nn.Module):
    def __init__(self, num_classes=13, embedding_size=128, num_subcenters=2, s=64, m=0.7, silence_target_prob=1/13):
        super(SubCenterArcFaceLoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.num_subcenters = num_subcenters
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(self.out_features * num_subcenters, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))

        self.th = torch.cos(torch.tensor(math.pi - m))
        self.mm = torch.sin(torch.tensor(math.pi - m)) * m
        self.silence_target_prob = silence_target_prob

    def forward(self, x, label, mode):
        # Sub-center ArcFace loss for valid labels
        valid_label = label >= 0  # Mask for non-silence labels
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).float()
        cosine = cosine.view(-1, self.out_features, self.num_subcenters)  # Shape: (B, num_classes, num_subcenters)
        cosine, _ = torch.max(cosine, dim=2)  # Shape: (B, num_classes)
        # order = [4, 1, 0, 5, 7, 6, 10, 2, 12, 9, 3, 8, 11]
        order = [10, 1, 3, 12, 5, 0, 6, 9, 4, 2, 8, 11, 7]
        cosine = cosine[:, order]
        if mode == 'train':
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).to(x.device).float()
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            # One-hot encoding for valid labels
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, label[valid_label].view(-1, 1), 1.0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        else:
            output = cosine * self.s
        # Cross-Entropy Loss for non-silence labels
        if valid_label.any():
            arcface_loss = F.cross_entropy(output[valid_label], label[valid_label])
        else:
            arcface_loss = 0.0
        # KL Divergence for silence labels (-1)
        silence_labels = ~valid_label  # Mask for silence labels
        silence_loss = 0.0
        if silence_labels.any():
            silence_output = cosine[silence_labels]  # Cosine values for silence labels
            silence_target = torch.full_like(silence_output, self.silence_target_prob)  # Uniform target distribution
            silence_loss = F.kl_div(F.log_softmax(silence_output, dim=1), silence_target, reduction='batchmean')
        return arcface_loss + silence_loss, output


class SelectiveSubCenterArcFaceLoss(nn.Module):
    def __init__(self, num_classes=13, embedding_size=128, num_subcenters=2, s=64, m=0.7, silence_target_prob=1/13):
        super(SelectiveSubCenterArcFaceLoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.num_subcenters = num_subcenters
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(self.out_features * num_subcenters, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        # self.gating = Parameter(torch.Tensor(self.out_features, self.num_subcenters))
        # nn.init.xavier_uniform_(self.gating)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))

        self.th = torch.cos(torch.tensor(math.pi - m))
        self.mm = torch.sin(torch.tensor(math.pi - m)) * m
        self.silence_target_prob = silence_target_prob

    def forward(self, x, label, mode):
        # Sub-center ArcFace loss for valid labels
        valid_label = label >= 0  # Mask for non-silence labels
        gating = nn.functional.sigmoid(self.weight.to(x.device).mean(dim=1).reshape(-1, self.num_subcenters))
        # gating = nn.functional.sigmoid(self.gating)
        weight = self.weight.to(x.device) * gating.reshape(-1, 1)
        cosine = F.linear(F.normalize(x), F.normalize(weight.to(x.device))).float()
        cosine = cosine.view(-1, self.out_features, self.num_subcenters)  # Shape: (B, num_classes, num_subcenters)
        # Subcenter Selection
        mask = (gating > 0.5).unsqueeze(0).expand(cosine.size())
        cosine = cosine * mask
        valid_subcenters = mask.sum(dim=2).clamp(min=1)
        cosine = cosine.sum(dim=2) / valid_subcenters
        # order = [4, 1, 0, 5, 7, 6, 10, 2, 12, 9, 3, 8, 11]
        # cosine = cosine[:, order]
        if mode == 'train':
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).to(x.device).float()
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            # One-hot encoding for valid labels
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, label[valid_label].view(-1, 1), 1.0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        else:
            output = cosine * self.s
        # Cross-Entropy Loss for non-silence labels
        if valid_label.any():
            arcface_loss = F.cross_entropy(output[valid_label], label[valid_label])
        else:
            arcface_loss = 0.0
        # KL Divergence for silence labels (-1)
        silence_labels = ~valid_label  # Mask for silence labels
        silence_loss = 0.0
        if silence_labels.any():
            silence_output = cosine[silence_labels]  # Cosine values for silence labels
            silence_target = torch.full_like(silence_output, self.silence_target_prob)  # Uniform target distribution
            silence_loss = F.kl_div(F.log_softmax(silence_output, dim=1), silence_target, reduction='batchmean')
        return arcface_loss + silence_loss, output


def estimate_active_sources(mixed_audio, separated_sources, threshold_db=-30):
    """
    Estimate the number of active sources from separated sources.

    Args:
    - mixed_audio (torch.Tensor): The mixed audio signal, shape (B, T).
    - separated_sources (torch.Tensor): The separated audio signals, shape (B, N, T).
    - threshold_db (float): The dB threshold to determine active/inactive sources.

    Returns:
    - num_active_sources (int): Estimated number of active sources.
    """
    # Calculate the power of the mixed audio
    mixed_power = torch.mean(mixed_audio ** 2)
    num_active_sources = 0
    
    for src_index in range(separated_sources.shape[1]):
        sources = separated_sources[:, src_index]
        # Calculate the power of the separated source
        source_power = torch.mean(sources ** 2)
        # Calculate the SNR in dB
        snr_db = 10 * torch.log10(source_power / mixed_power)
        # Check if the source is active
        if snr_db >= threshold_db:
            num_active_sources += 1
    
    return num_active_sources


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = torch.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = torch.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = torch.clip(dist, -1, 1)
    dist = torch.arccos(dist) * 180 / torch.pi
    return dist


def neg_si_sdr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_si_sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)

def neg_sdr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    sdr_val = sdr(preds=preds, target=target)
    return -torch.mean(sdr_val.view(batch_size, -1), dim=1)

def neg_snr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_snr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    snr_val = snr(preds=preds, target=target)
    return -torch.mean(snr_val.view(batch_size, -1), dim=1)

def neg_sa_sdr(preds: Tensor, target: Tensor, scale_invariant: bool = False) -> Tensor:
    batch_size = target.shape[0]
    sa_sdr_val = sa_sdr(preds=preds, target=target, scale_invariant=scale_invariant)
    return -torch.mean(sa_sdr_val.view(batch_size, -1), dim=1)

def sdr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    eps = 1e-10
    batch_size = target.shape[0]
    distortion = preds - target
    sdr_val = 10 * torch.log10(torch.sum(target**2, dim=-1) / (torch.sum(distortion**2, dim=-1) + eps))
    return sdr_val.view(batch_size, -1)


def fuss_loss(mixed, preds, target):
    batch_size = target.shape[0]
    dist = preds - target
    loss = 0.
    for idx in range(batch_size):
        if torch.max(target[idx]).item() != 0: 
            loss += 10*torch.log10(torch.sum(dist[idx]**2, dim=-1) + 0.001*torch.sum(target[idx]**2, dim=-1))
        else: 
            loss += 10*torch.log10(torch.sum(preds[idx]**2, dim=-1) + 0.001*torch.sum(mixed[idx]**2, dim=-1))
    return loss


def cbir_loss(preds, target):
    loss = 0.
    for idx in range(target.shape[0]):
        if torch.max(target[idx]).item() != 0: 
            loss = -torch.mean(si_sdr(preds[idx], target[idx]))
    return loss


def l1_loss(preds, target):
    return torch.mean(torch.abs(preds - target), dim=-1)

def mse_loss(preds, target):
    return torch.mean((preds - target)**2, dim=-1)

def ce_loss(preds, target):
    return ce(preds, target)

def bce_logits_loss(preds, target):
    return bce_logits(preds, target)


def si_sdri(mixed: Tensor, preds: Tensor, target: Tensor) -> Tensor:
    # SI-SDR improvement
    batch_size = target.shape[0]
    si_sdr_before = si_sdr(mixed, target)
    si_sdr_after = si_sdr(preds, target)
    si_sdri = si_sdr_after - si_sdr_before
    return si_sdri.view(batch_size, -1)

def sdri(mixed: Tensor, preds: Tensor, target: Tensor) -> Tensor:
    # SDR improvement
    batch_size = target.shape[0]
    sdr_before = sdr(mixed, target)
    sdr_after = sdr(preds, target)
    sdri = sdr_after - sdr_before
    return sdri.view(batch_size, -1)


class Loss(nn.Module):
    is_scale_invariant_loss: bool
    name: str
    def __init__(self, loss_func: Callable, pit: bool, loss_func_kwargs: Dict[str, Any] = dict()):
        super().__init__()

        self.loss_func = loss_func
        self.pit = pit
        self.loss_func_kwargs = loss_func_kwargs
        self.is_scale_invariant_loss = {
            neg_sa_sdr: True if 'scale_invariant' in loss_func_kwargs and loss_func_kwargs['scale_invariant'] == True else False,
            neg_si_sdr: True,
            neg_snr: False,
            neg_sdr: False,
            fuss_loss: False,
            l1_loss: False,
            cbir_loss: False,
            mse_loss: False,
        }[loss_func]
        self.name = loss_func.__name__

    def forward(self, preds: Tensor, target: Tensor, reorder: bool = None, reduce_batch: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        perms = None
        if self.pit:
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode='permutation-wise', **self.loss_func_kwargs)
            if reorder:
                preds = permutate(preds, perm=perms)
        else:
            losses = self.loss_func(preds=preds, target=target, **self.loss_func_kwargs)

        return losses.mean() if reduce_batch else losses, perms, preds

    def extra_repr(self) -> str:
        kwargs = ""
        for k, v in self.loss_func_kwargs.items():
            kwargs += f'{k}={v},'

        return f"loss_func={self.loss_func.__name__}({kwargs}), pit={self.pit}"
    

class Loss2(nn.Module):
    is_scale_invariant_loss: bool
    name: str
    def __init__(self, loss_func: Callable, pit: bool, loss_func_kwargs: Dict[str, Any] = dict()):
        super().__init__()

        self.loss_func = loss_func
        self.pit = pit
        self.loss_func_kwargs = loss_func_kwargs
        self.is_scale_invariant_loss = {
            neg_sa_sdr: True if 'scale_invariant' in loss_func_kwargs and loss_func_kwargs['scale_invariant'] == True else False,
            neg_si_sdr: True,
            neg_snr: False,
            fuss_loss: False,
            l1_loss: False,
            ce_loss: False,
            bce_logits_loss: False,
        }[loss_func]
        self.name = loss_func.__name__

    def forward(self, preds: Tensor, target: Tensor, reorder: bool = None, reduce_batch: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        perms = None
        if self.pit:
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode='speaker-wise', **self.loss_func_kwargs)
            if reorder:
                preds = permutate(preds, perm=perms)
        else:
            losses = self.loss_func(preds=preds, target=target, **self.loss_func_kwargs)

        return losses.mean() if reduce_batch else losses, perms, preds

    def extra_repr(self) -> str:
        kwargs = ""
        for k, v in self.loss_func_kwargs.items():
            kwargs += f'{k}={v},'

        return f"loss_func={self.loss_func.__name__}({kwargs}), pit={self.pit}"