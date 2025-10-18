import torch
import torch.nn as nn
import numpy as np
from scipy.signal import find_peaks
import torch.nn.functional as F

class_params = {
    0: {"tau": 4, "threshold": 0.5, "height": 0.1, "prominence": 0.1},       # Female speech
    1: {"tau": 4, "threshold": 0.5, "height": 0.1, "prominence": 0.1},       # Male speech
    2: {"tau": 2, "threshold": 0.5, "height": 0.1, "prominence": 0.1},          # Clapping
    3: {"tau": 8, "threshold": 0.5, "height": 0.1, "prominence": 0.1},          # Telephone
    4: {"tau": 2, "threshold": 0.5, "height": 0.1, "prominence": 0.1},          # Laughter
    5: {"tau": 8, "threshold": 0.25, "height": 0.1, "prominence": 0.1},         # Domestic sound
    6: {"tau": 2, "threshold": 0.25, "height": 0.1, "prominence": 0.1},         # Walk and footsteps
    7: {"tau": 2, "threshold": 0.25, "height": 0.05, "prominence": 0.05},       # Door
    8: {"tau": 8, "threshold": 0.25, "height": 0.1, "prominence": 0.1},         # Music
    9: {"tau": 8, "threshold": 0.25, "height": 0.1, "prominence": 0.1},         # Musical instrument
    10: {"tau": 8, "threshold": 0.25, "height": 0.1, "prominence": 0.1},        # Water tap
    11: {"tau": 8, "threshold": 0.5, "height": 0.1, "prominence": 0.1},         # Telephone
    12: {"tau": 2, "threshold": 0.25, "height": 0.05, "prominence": 0.05},       # Knock
}

def get_class_params(pred_classes):
    """
    pred_classes: (B, S) 형태의 torch.Tensor
    각 샘플과 사운드 소스별로 class_params에서 해당하는 값을 가져와 반환
    """
    B, S = pred_classes.shape
    tau = torch.zeros((B, S))
    threshold = torch.zeros((B, S))
    height = torch.zeros((B, S))
    prominence = torch.zeros((B, S))
    for b in range(B):
        for s in range(S):
            cls_idx = pred_classes[b, s].item()  # 정수 인덱스 가져오기
            params = class_params.get(cls_idx, {"tau": 0.0, "threshold": 0.0, "height": 0.0, "prominence": 0.0})
            tau[b, s] = params["tau"]
            threshold[b, s] = params["threshold"]
            height[b, s] = params["height"]
            prominence[b, s] = params["prominence"]

    return tau, threshold, height, prominence

# def get_sebb_mask(onoff_out, cls_out, tau=2, height=0.1, distance=1, prominence=0.1, width=1):
#     """
#     입력: frame-level scores (B, S, T) 
#     출력: binary mask (B, S, T) (0 또는 1)
#     """
#     B, S, T = onoff_out.shape
#     pred_cls = torch.argmax(cls_out, dim=-1)  # (B, S)
#     tau, threshold, height, prominence = get_class_params(pred_cls)
#     # tau, height, distance, prominence, width = int(tau), float(height), int(distance), float(prominence), int(width)
#     # left_avg: 해당 time frame으로부터 previous tau/2 frame까지의 평균
#     left_avg = F.pad(onoff_out, (tau, 0), 'replicate')  # padding
#     left_avg = F.avg_pool1d(left_avg.view(B*S, 1, -1), kernel_size=tau, stride=1, padding=0).view(B, S, T+1)
#     # right_avg: 해당 time frame으로부터 next tau/2 frame까지의 평균
#     right_avg = F.pad(onoff_out, (0, tau), 'replicate')  # padding
#     right_avg = F.avg_pool1d(right_avg.view(B*S, 1, -1), kernel_size=tau, stride=1, padding=0).view(B, S, T+1)
#     delta = (right_avg - left_avg)[:, :, :-1]  # (B, S, T) shape
#     # find peaks
#     maxima_idx = [[None]* S for _ in range(B)]
#     minima_idx = [[None]* S for _ in range(B)]
#     for b in range(B):
#         for s in range(S):
#             delta_np = delta[b, s].float().clone().cpu().numpy()  # NumPy 변환
#             # 로컬 최대값 찾기
#             max_idx, _ = find_peaks(delta_np, height=height[b, s].item(), distance=distance, prominence=prominence[b, s].item(), width=width)
#             maxima_idx[b][s] = torch.tensor(max_idx, dtype=torch.long)
#             # 로컬 최소값 찾기
#             min_idx, _ = find_peaks(-delta_np, height=height[b, s].item(), distance=distance, prominence=prominence[b, s].item(), width=width)
#             minima_idx[b][s] = torch.tensor(min_idx, dtype=torch.long)

#     return delta, maxima_idx, minima_idx, threshold
def get_sebb_mask(onoff_out, cls_out):
    """
    입력: frame-level scores (B, S, T) 
    출력: binary mask (B, S, T) (0 또는 1)
    """
    B, S, T = onoff_out.shape
    pred_cls = torch.argmax(cls_out, dim=-1)  # (B, S)
    # tau, threshold, height, prominence = get_class_params(pred_cls)  # (B, S) 형태의 파라미터 반환
    # `left_avg`와 `right_avg`를 `tau[b, s]` 별로 개별적으로 계
    delta = torch.zeros((B, S, T), device=onoff_out.device)
    for b in range(B):
        for s in range(S):
            # t = int(tau[b, s].item())  # 개별 tau 값 가져오기
            t = 2
            if t > 1:
                left_padded = F.pad(onoff_out[b:b+1, s:s+1], (t, 0), mode='replicate')
                left_avg = F.avg_pool1d(left_padded, kernel_size=t, stride=1)
            if t > 1:
                right_padded = F.pad(onoff_out[b:b+1, s:s+1], (0, t), mode='replicate')
                right_avg = F.avg_pool1d(right_padded, kernel_size=t, stride=1)
            delta[b:b+1, s:s+1] = (right_avg - left_avg)[:, :, :-1]  # (T,) shape
    maxima_idx = [[None] * S for _ in range(B)]
    minima_idx = [[None] * S for _ in range(B)]
    # Find local maxima and minima
    delta_np = delta.cpu().numpy()  # NumPy 변환 (한 번만)
    for b in range(B):
        for s in range(S):
            d = delta_np[b, s]  # (T,) shape
            # 로컬 최대값 찾기
            # max_idx, _ = find_peaks(d, height=height[b, s].item(), distance=1, prominence=prominence[b, s].item(), width=1)
            max_idx, _ = find_peaks(d)
            maxima_idx[b][s] = torch.tensor(max_idx, dtype=torch.long, device=onoff_out.device)
            # 로컬 최소값 찾기
            # min_idx, _ = find_peaks(-d, height=height[b, s].item(), distance=1, prominence=prominence[b, s].item(), width=1)
            min_idx, _ = find_peaks(-d)
            minima_idx[b][s] = torch.tensor(min_idx, dtype=torch.long, device=onoff_out.device)
    threshold = 0.5

    return delta, maxima_idx, minima_idx, threshold

def post_processing(onoff_out, delta, max_idx, min_idx, threshold):
    # B, S, C = cls_out.shape
    B, S, T = onoff_out.shape
    frame_scores = torch.sigmoid(onoff_out)  # sigmoid 적용
    pp_onoff_out = torch.zeros_like(onoff_out)  # 초기값: 모두 silence (0)
    # threshold = float(threshold)

    for b in range(B):
        for s in range(S):
            scores = frame_scores[b, s]  # (T,) shape
            deltas = delta[b, s]  # (T,) shape
            # 모든 frame이 0.6 미만이면 silence 유지
            # if torch.all(scores < threshold[b, s].item()):)
                # continue  # onoff_out2[b, s]는 이미 0 (silence)
            # elif torch.all(scores > threshold[b, s].item()):
                # pp_onoff_out[b, s] = 1
            # elif len(max_idx[b][s]) == 0 and len(min_idx[b][s]) == 0:
                # pp_onoff_out[b, s] = onoff_out[b, s] > threshold[b, s].item()
            # else:
                # Silence가 아닌 경우: local extrema 기준으로 on/off 설정
            onset_indices = max_idx[b][s].clone().cpu().numpy() if len(max_idx[b][s]) > 0 else []
            offset_indices = min_idx[b][s].clone().cpu().numpy() if len(min_idx[b][s]) > 0 else []
            combined_events = sorted([(idx, 'onset') for idx in onset_indices] + [(idx, 'offset') for idx in offset_indices])
            # Onset과 Offset을 기준으로 on/off 설정
            if len(combined_events) > 0:
                first_event_idx, first_event_type = combined_events[0]  # 가장 처음 등장하는 이벤트
                if first_event_type == 'onset':
                    pp_onoff_out[b, s, :first_event_idx] = 0
                else:
                    pp_onoff_out[b, s, :first_event_idx] = 1
                # Onset/Offset 이후 frame을 설정
                for idx, event_type in combined_events:
                    if event_type == 'onset':
                        active = 1  # 이후 frame을 active(1)로 설정
                    elif event_type == 'offset':
                        active = 0  # 이후 frame을 inactive(0)로 설정
                    pp_onoff_out[b, s, idx:] = active  # onset/offset 이후 적용

    return pp_onoff_out


class SEBB(nn.Module):
    def __init__(self, n_cls=13):
        super(SEBB, self).__init__()
        # Sound Class별 Parameter (학습 가능)
        self.tau = nn.Parameter(torch.randn(n_cls))  # (n_cls,)
        self.height = nn.Parameter(torch.randn(n_cls))
        self.distance = nn.Parameter(torch.randn(n_cls))
        self.prominence = nn.Parameter(torch.randn(n_cls))
        self.width = nn.Parameter(torch.randn(n_cls))
        self.threshold = nn.Parameter(torch.randn(n_cls))

    def forward(self, onoff_out, cls_out):
        """
        Args:
            onoff_out: (B, S, T) - Frame-level score
            cls_out: (B, S, C) - Class probabilities

        Returns:
            sed_out: (B, S, T, C) - Post-processed SED output
        """
        B, S, T = onoff_out.shape
        C = cls_out.shape[-1]  # Number of sound classes (n_cls)
        # 각 source별로 가장 확률이 높은 class 선택
        cls_idx = torch.argmax(cls_out, dim=-1)  # (B, S)
        # 각 source별로 해당 class의 parameter 가져오기
        # tau_s = F.relu(self.tau[cls_idx])  # (B, S)
        # height_s = F.relu(self.height[cls_idx])  # ReLU 적용
        # distance_s = F.relu(self.distance[cls_idx])
        # prominence_s = F.relu(self.prominence[cls_idx])
        # width_s = F.relu(self.width[cls_idx])
        # threshold_s = F.sigmoid(self.threshold[cls_idx])
        tau_s, height_s, distance_s, prominence_s, width_s, threshold_s = F.relu(self.tau[0]), F.relu(self.height[0]), F.relu(self.distance[0]), F.relu(self.prominence[0]), F.relu(self.width[0]), F.sigmoid(self.threshold[0])
        # SEBB mask 생성 (각 source별 parameter 적용)
        delta, max_idx, min_idx = get_sebb_mask(onoff_out, cls_out, tau=torch.round(tau_s)+1, height=height_s, distance=torch.round(distance_s)+1, prominence=prominence_s, width=torch.round(width_s)+1)
        pp_onoff_out = post_processing(cls_out, onoff_out, delta, max_idx, min_idx, threshold_s)
        # SED output
        cls_binary = torch.zeros_like(cls_out).scatter_(-1, cls_idx.unsqueeze(-1), 1)  # (B, S, C)
        sed_out = pp_onoff_out.unsqueeze(-1) * cls_binary.unsqueeze(2)  # (B, S, T, C)

        return sed_out
