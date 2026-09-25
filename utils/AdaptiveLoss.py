import torch
import torch.nn as nn


class AdaptivePatchMixerLoss(nn.Module):

    def __init__(
        self,
        lambda_mse=1.0,
        lambda_grad=0.1,
        lambda_freq=0.05
    ):
        super().__init__()

        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.lambda_freq = lambda_freq

        self.mse = nn.MSELoss()

    # ========================================================
    # Gradient Loss
    # ========================================================

    def gradient_loss(
        self,
        pred,
        true
    ):

        pred_diff = (
            pred[:, 1:, :]
            -
            pred[:, :-1, :]
        )

        true_diff = (
            true[:, 1:, :]
            -
            true[:, :-1, :]
        )

        return torch.mean(
            torch.abs(
                pred_diff - true_diff
            )
        )

    # ========================================================
    # Frequency Loss
    # ========================================================

    def frequency_loss(
        self,
        pred,
        true
    ):

        pred_fft = torch.fft.rfft(
            pred,
            dim=1
        )

        true_fft = torch.fft.rfft(
            true,
            dim=1
        )

        pred_mag = torch.abs(
            pred_fft
        )

        true_mag = torch.abs(
            true_fft
        )

        return torch.mean(
            torch.abs(
                pred_mag - true_mag
            )
        )

    # ========================================================
    # Total Loss
    # ========================================================

    def forward(
        self,
        pred,
        true
    ):

        loss_mse = self.mse(
            pred,
            true
        )

        loss_grad = self.gradient_loss(
            pred,
            true
        )

        loss_freq = self.frequency_loss(
            pred,
            true
        )

        total_loss = (

            self.lambda_mse
            * loss_mse

            +

            self.lambda_grad
            * loss_grad

            +

            self.lambda_freq
            * loss_freq
        )

        return total_loss