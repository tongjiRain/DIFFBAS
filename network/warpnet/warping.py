import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TimeWarperFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, input, warpfield):
        '''
        :param ctx: autograd context
        :param input: input signal (B x 2 x T)
        :param warpfield: the corresponding warpfield (B x 2 x T)
        :return: the warped signal (B x 2 x T)
        '''
        ctx.save_for_backward(input, warpfield)
        idx_left = warpfield.floor().type(th.long)
        idx_right = th.clamp(warpfield.ceil().type(th.long), max=input.shape[-1]-1)
        alpha = warpfield - warpfield.floor()
        output = (1 - alpha) * th.gather(input, 2, idx_left) + alpha * th.gather(input, 2, idx_right)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, warpfield = ctx.saved_tensors
        idx_left = warpfield.floor().type(th.long)
        idx_right = th.clamp(warpfield.ceil().type(th.long), max=input.shape[-1]-1)
        grad_warpfield = th.gather(input, 2, idx_right) - th.gather(input, 2, idx_left)
        grad_warpfield = grad_output * grad_warpfield
        # input gradient
        grad_input = th.zeros(input.shape, device=input.device)
        alpha = warpfield - warpfield.floor()
        grad_input = grad_input.scatter_add(2, idx_left, grad_output * (1 - alpha)) + \
                     grad_input.scatter_add(2, idx_right, grad_output * alpha)

        return grad_input, grad_warpfield


class TimeWarper(nn.Module):

    def __init__(self):
        super().__init__()
        self.warper = TimeWarperFunction().apply

    def _to_absolute_positions(self, warpfield, seq_length):
        # translate warpfield from relative warp indices to absolute indices ([1...T] + warpfield)
        temp_range = th.arange(seq_length, dtype=th.float)
        temp_range = temp_range.cuda() if warpfield.is_cuda else temp_range
        return th.clamp(warpfield + temp_range[None, None, :], min=0, max=seq_length-1)

    def forward(self, input, warpfield):
        '''
        :param input: audio signal to be warped (B x 2 x T)
        :param warpfield: the corresponding warpfield (B x 2 x T)
        :return: the warped signal (B x 2 x T)
        '''
        warpfield = self._to_absolute_positions(warpfield, input.shape[-1])
        warped = self.warper(input, warpfield)
        return warped


class MonotoneTimeWarper(TimeWarper):

    def forward(self, input, warpfield):
        '''
        :param input: audio signal to be warped (B x 2 x T)
        :param warpfield: the corresponding warpfield (B x 2 x T)
        :return: the warped signal (B x 2 x T), ensured to be monotonous
        '''
        warpfield = self._to_absolute_positions(warpfield, input.shape[-1])
        warpfield = th.cummax(warpfield, dim=-1)[0]
        warped = self.warper(input, warpfield)
        return warped


class GeometricTimeWarper(TimeWarper):

    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def displacements2warpfield(self, displacements, seq_length):
        distance = th.sum(displacements**2, dim=2) ** 0.5
        distance = F.interpolate(distance, size=seq_length)
        warpfield = -distance / 343.0 * self.sampling_rate
        return warpfield

    def forward(self, input, displacements):
        '''
        :param input: audio signal to be warped (B x 2 x T)
        :param displacements: sequence of 3D displacement vectors for geometric warping (B x 3 x T)
        :return: the warped signal (B x 2 x T)
        '''
        warpfield = self.displacements2warpfield(displacements, input.shape[-1])
        warped = super().forward(input, warpfield)
        return warped
