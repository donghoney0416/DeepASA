import torch
from torch import nn

from torch import Tensor
from typing import *

paras_16k = {
    'n_fft': 512,
    'n_hop': 256,
    'win_len': 512,
}

paras_8k = {
    'n_fft': 256,
    'n_hop': 128,
    'win_len': 256,
}


class STFT(nn.Module):
    def __init__(self, n_fft: int, n_hop: int, win_len: Optional[int] = None, win: str = 'hann_window') -> None:
        super().__init__()
        self.n_fft, self.n_hop, self.win_len = n_fft, n_hop, win_len if win_len is not None else n_fft
        self.repr = str((n_fft, n_hop, win, win_len))

        assert win == 'hann_window', win
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

    def forward(self, X: Tensor, original_len: int = None, inverse=False) -> Any:
        """istft
        Args:
            X: complex [..., F, T]
            original_len: original length
            inverse: stft or istft
        """
        if not inverse:
            return self.stft(x)
        else:
            return self.istft(x, original_len=original_len)

    def stft(self, x: Tensor) -> Tuple[Tensor, int]:
        """stft
        Args:
            x: [..., time]

        Returns:
            the complex STFT domain representation of shape [..., freq, time] and the original length of the time domain waveform
        """
        shape = list(x.shape)
        x = x.reshape(-1, shape[-1])
        if x.is_cuda:
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):  # use float32 for stft & istft
                X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, return_complex=True)
        else:
            X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, return_complex=True)
        F, T = X.shape[-2:]
        X = X.reshape(shape=shape[:-1] + [F, T])
        return X, shape[-1]

    def istft(self, X: Tensor, original_len: int = None) -> Tensor:
        """istft
        Args:
            X: complex [..., F, T]
            original_len: returned by stft

        Returns:
            the complex STFT domain representation of shape [..., freq, time] and the original length of the time domain waveform
        """
        shape = list(X.shape)
        X = X.reshape(-1, *shape[-2:])
        if X.is_cuda:
            with torch.autocast(device_type=X.device.type, dtype=torch.float32):  # use float32 for stft & istft
                x = torch.istft(X, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, length=original_len)
        else:
            x = torch.istft(X, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window, length=original_len)
        x = x.reshape(shape=shape[:-2] + [original_len])
        return x

    def extra_repr(self) -> str:
        return self.repr


class GaborSTFT(nn.Module):
    def __init__(self, n_fft: int, n_hop: int, num_frames: int, win_len: Optional[int] = None) -> None:
        """
        Gabor Transform 기반 STFT

        Args:
            n_fft (int): FFT 크기
            n_hop (int): Hop 크기 (STFT 시간 해상도)
            num_frames (int): 전체 time frame 수
            win_len (int, optional): 윈도우 길이 (기본적으로 n_fft 사용)
        """
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.win_len = win_len if win_len is not None else n_fft
        self.num_frames = num_frames

        # Learnable standard deviation (mu and sigma) for each time frame
        self.window_conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=n_fft, stride=n_hop, padding=n_fft // 2, bias=True)
        self.mu_linear = nn.Linear(64, 1, bias=True)
        self.sigma_linear = nn.Linear(64, 1, bias=True)
        self.window = torch.hann_window(self.n_fft)
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.window_conv.weight, 1e-4)
        torch.nn.init.constant_(self.window_conv.bias, 1e-4)
        torch.nn.init.constant_(self.mu_linear.bias, 1e-4)
        torch.nn.init.constant_(self.mu_linear.weight, 1e-4)
        torch.nn.init.constant_(self.sigma_linear.bias, self.win_len / 6)
        torch.nn.init.constant_(self.sigma_linear.weight, 1e-4)

    def gabor_window(self, mu: torch.Tensor, sigma: torch.Tensor, n_fft: int) -> torch.Tensor:
        """
        주어진 sigma를 사용하여 Gabor 윈도우 생성
        Args:
            sigma (torch.Tensor): Gaussian 창의 표준편차 (각 time frame마다 learnable)
            n_fft (int): FFT 크기
        
        Returns:
            torch.Tensor: 생성된 Gabor 윈도우
        """
        t = torch.arange(-n_fft // 2, n_fft // 2, dtype=torch.float32, device=sigma.device)
        gabor_windows = torch.exp(-((t[None, None, :] - mu[:, :, None])**2) / (2 * sigma[:, :, None]**2))
        gabor_windows = gabor_windows / gabor_windows.norm(dim=-1, keepdim=True)  # Normalize
        return gabor_windows

    def forward(self, X: torch.Tensor, original_len: int = None, inverse=False) -> Any:
        """
        STFT 또는 ISTFT 수행

        Args:
            X (torch.Tensor): 입력 신호 (STFT의 경우 time domain, ISTFT의 경우 frequency domain)
            original_len (int, optional): 원래 신호 길이
            inverse (bool, optional): ISTFT 수행 여부 (True일 경우 ISTFT)
        
        Returns:
            torch.Tensor: 변환된 신호
        """
        if not inverse:
            return self.stft(X)
        else:
            return self.istft(X, original_len=original_len)

    def stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        shape = list(x.shape)
        x = x.reshape(-1, shape[-1])
        # mu and sigma are learnable parameters
        win_feat = self.window_conv(x.unsqueeze(1)).permute(0, 2, 1).contiguous()       # (B, T, win_len)
        mu = torch.sigmoid(self.mu_linear(win_feat)) * self.n_fft - self.n_fft // 2     # (B, T, 1)
        sigma = torch.nn.functional.softplus(self.sigma_linear(win_feat))               # (B, T, 1)
        # Calculate Gabor window
        gabor_win = self.gabor_window(mu.squeeze(-1), sigma.squeeze(-1), self.n_fft)                        # B, T, win_len
        # Apply Gabor window to each frame
        x = torch.nn.functional.pad(x, (self.n_fft // 2, self.n_fft // 2), mode='constant', value=0)
        frames = x.unfold(dimension=1, size=self.n_fft, step=self.n_hop)       # (B, T, n_fft)
        frames = frames * gabor_win                                                 # (B, T, n_fft)
        X = torch.fft.fft(frames, dim=-1)                                   # (B, T, F)
        X = X[:, :, :self.n_fft // 2 + 1].permute(0, 2, 1)                  # (B, F, T)
        F, T = X.shape[-2:]
        X = X.reshape(shape[:-1] + [F, T])
        return X, shape[-1]

    def istft(self, X: Tensor, original_len: int = None) -> Tensor:
        shape = list(X.shape)
        X = X.reshape(-1, *shape[-2:])
        if X.is_cuda:
            with torch.autocast(device_type=X.device.type, dtype=torch.float32):  # use float32 for stft & istft
                x = torch.istft(X, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window.to(X.device), length=original_len)
        else:
            x = torch.istft(X, n_fft=self.n_fft, hop_length=self.n_hop, win_length=self.win_len, window=self.window.to(X.device), length=original_len)
        x = x.reshape(shape=shape[:-2] + [original_len])
        return x

    def extra_repr(self) -> str:
        return f'GaborSTFT(n_fft={self.n_fft}, n_hop={self.n_hop}, num_frames={self.num_frames}, win_len={self.win_len})'


if __name__ == '__main__':
    x = torch.randn((1, 1, 8000 * 4))
    stft = STFT(**paras_8k)
    X, ol = stft.stft(x)
    x_p = stft.istft(X, ol)
    print(x.shape, x_p.shape, X.shape)
    print(torch.allclose(x, x_p, rtol=1e-1))
    print()
