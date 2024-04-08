import numpy as np
import torch
import torch as th
from utils import FourierTransform
from auraloss.freq import MultiResolutionSTFTLoss as MRSTFT
from scipy.signal import find_peaks

class Loss(th.nn.Module):
    def __init__(self, mask_beginning=0):
        '''
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning

    def forward(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data = data[..., self.mask_beginning:]
        target = target[..., self.mask_beginning:]
        return self._loss(data, target)

    def _loss(self, data, target):
        pass


class L2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.mean((data - target).pow(2))


class AmplitudeLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data), self._transform(target)
        data = th.sum(data**2, dim=-1) ** 0.5
        target = th.sum(target**2, dim=-1) ** 0.5
        return th.mean(th.abs(data - target))


class PhaseLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, ignore_below=0.1):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.ignore_below = ignore_below
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data).view(-1, 2), self._transform(target).view(-1, 2)
        target_energy = th.sum(th.abs(target), dim=-1)
        pred_energy = th.sum(th.abs(data.detach()), dim=-1)
        target_mask = target_energy > self.ignore_below * th.mean(target_energy)
        pred_mask = pred_energy > self.ignore_below * th.mean(target_energy)
        indices = th.nonzero(target_mask * pred_mask).view(-1)
        data, target = th.index_select(data, 0, indices), th.index_select(target, 0, indices)

        data_angles, target_angles = th.atan2(data[:, 0], data[:, 1]), th.atan2(target[:, 0], target[:, 1])
        loss = th.abs(data_angles - target_angles)
        loss = np.pi - th.abs(loss - np.pi)
        return th.mean(loss)


class PhaseDiffLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, ignore_below=0.1,low_frequency = 0,high_frequency = 0,find_peaks = False):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.ignore_below = ignore_below
        self.fft = FourierTransform(sample_rate=sample_rate)
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.find_peaks = find_peaks

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''

        data_L = data[:, 0, :].unsqueeze(1)
        data_R = data[:, 1, :].unsqueeze(1)

        target_L = target[:, 0, :].unsqueeze(1)
        target_R = target[:, 1, :].unsqueeze(1)


        if self.high_frequency > 0:
            data_L, data_R = self._transform(data_L)[:,self.low_frequency:self.high_frequency,:,:].view(-1, 2), self._transform(data_R)[:,self.low_frequency:self.high_frequency,:,:].view(-1, 2)
            target_L, target_R = self._transform(target_L)[:,self.low_frequency:self.high_frequency,:,:].view(-1, 2), self._transform(target_R)[:,self.low_frequency:self.high_frequency,:,:].view(-1, 2)
        else:
            data_L, data_R = self._transform(data_L).view(-1, 2), self._transform(data_R).view(-1, 2)
            target_L, target_R = self._transform(target_L).view(-1, 2), self._transform(target_R).view(-1, 2)

        data_L_energy = th.sum(th.abs(data_L), dim=-1)
        data_R_energy = th.sum(th.abs(data_R.detach()), dim=-1)
        target_L_energy = th.sum(th.abs(target_L), dim=-1)
        target_R_energy = th.sum(th.abs(target_R.detach()), dim=-1)


        if(self.find_peaks):
            data_L_peaks, _ = find_peaks(data_L_energy.numpy().astype(float), height=th.mean(data_L_energy).item() * self.ignore_below)  # threshold 是你设定的阈值
            data_R_peaks, _ = find_peaks(data_R_energy.numpy(), height=th.mean(data_R_energy).item() * self.ignore_below)  # threshold 是你设定的阈值
            target_L_peaks, _ = find_peaks(target_L_energy.numpy(), height=th.mean(target_L_energy).item() * self.ignore_below)  # threshold 是你设定的阈值
            target_R_peaks, _ = find_peaks(target_R_energy.numpy(), height=th.mean(target_R_energy).item() * self.ignore_below)  # threshold 是你设定的阈值
            # 创建掩码
            data_L_mask = th.zeros_like(data_L_energy, dtype=th.bool)
            data_L_mask[torch.tensor(data_L_peaks)] = 1
            data_R_mask = th.zeros_like(data_R_energy, dtype=th.bool)
            data_R_mask[torch.tensor(data_R_peaks)] = 1
            target_L_mask = th.zeros_like( target_L_energy, dtype=th.bool)
            target_L_mask[torch.tensor(target_L_peaks)] = 1
            target_R_mask = th.zeros_like(target_R_energy, dtype=th.bool)
            target_R_mask[torch.tensor(target_R_peaks)] = 1
        else:
            # 根据阈值进行选择
            data_L_mask = data_L_energy > self.ignore_below * th.mean(data_L_energy)
            data_R_mask = data_R_energy > self.ignore_below * th.mean(data_R_energy)
            target_L_mask = target_L_energy > self.ignore_below * th.mean(target_L_energy)
            target_R_mask = target_R_energy > self.ignore_below * th.mean(target_R_energy)


        indices = th.nonzero(data_L_mask * data_R_mask * target_L_mask * target_R_mask).view(-1)


        # 重定位索引计算左右声道的相位差
        data_L, data_R = th.index_select(data_L, 0, indices), th.index_select(data_R, 0, indices)
        data_L = data_L[:, 0] + 1j * data_L[:, 1]
        data_R = data_R[:, 0] + 1j * data_R[:, 1]
        data_diff = th.angle(data_L / data_R)

        epsilon = 1e-8
        target_L, target_R = th.index_select(target_L, 0, indices), th.index_select(target_R, 0, indices)
        target_L = target_L[:, 0] + 1j * target_L[:, 1]
        target_R = target_R[:, 0]+epsilon + 1j * target_R[:, 1]
        target_diff = th.angle(target_L / target_R)

        # 相位差异diff的范围是[-pi，pi]
        diff = data_diff-target_diff

        return th.mean(torch.abs(diff))

class AmpDiffLoss(Loss):
    # 双耳声道的幅度差异这里体现为比值
    def __init__(self, sample_rate, mask_beginning=0, ignore_below=0.1):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.ignore_below = ignore_below
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data_L = data[:, 0, :].unsqueeze(1)
        data_R = data[:, 1, :].unsqueeze(1)

        data_L, data_R = self._transform(data_L).view(-1, 2), self._transform(data_R).view(-1, 2)

        target_L = target[:, 0, :].unsqueeze(1)
        target_R = target[:, 1, :].unsqueeze(1)

        target_L, target_R = self._transform(target_L).view(-1, 2), self._transform(target_R).view(-1, 2)
        # ignore low energy components for numerical stability
        target_L_energy = th.sum(th.abs(target_L), dim=-1)
        target_R_energy = th.sum(th.abs(target_R.detach()), dim=-1)
        target_L_mask = target_L_energy > self.ignore_below * th.mean(target_L_energy)
        target_R_mask = target_R_energy > self.ignore_below * th.mean(target_R_energy)
        indices = th.nonzero(target_L_mask * target_R_mask).view(-1)

        data_L, data_R = th.index_select(data_L, 0, indices), th.index_select(data_R, 0, indices)
        data_L = data_L[:, 0] + 1j * data_L[:, 1]
        data_R = data_R[:, 0] + 1j * data_R[:, 1]
        data_diff = th.abs(data_L / data_R)

        target_L, target_R = th.index_select(target_L, 0, indices), th.index_select(target_R, 0, indices)
        target_L = target_L[:, 0] + 1j * target_L[:, 1]
        target_R = target_R[:, 0] + 1j * target_R[:, 1]
        target_diff = th.abs(target_L / target_R)

        diff = th.log10(data_diff) - th.log10(target_diff)

        return th.mean(torch.abs(diff))


class MrstftLoss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''

        mrstft = MRSTFT()
        mrstft_loss = mrstft(data,target)
        return mrstft_loss

class  StdLoss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.std(data - target)

