import torch
import torch.nn as nn

def wmse_loss(predictions, targets, alpha=0):
    _p = torch.as_tensor(predictions)
    _t = torch.as_tensor(targets)
    n_steps = predictions.shape[-2]
    if alpha == 0:
        loss = (_p - _t) ** 2
        return torch.mean(loss) / n_steps
    weights = torch.exp(-alpha * torch.arange(n_steps, device=_p.device, dtype=_p.dtype))
    weights = weights / weights.sum()
    weights = weights.view(*[1]*(_p.dim()-2), -1, 1)
    loss = weights * (_p - _t) ** 2
    return torch.mean(loss)

class WMSELoss(nn.Module):
    r"""Weighted Mean Squared Error Loss

    At step i, the loss is defined as: w_i(x_i-\hat{x}_i)^2,
    where w_i is the weight for step i.

    Currently, an exponential weighting is used; let v_i = exp(-alpha*i),
    then w_i = v_i / sum_j v_j.

    When alpha=0, this reduces to the standard MSE loss.
    Note that alpha can be both positive and negative, favoring early or late steps
    """
    def __init__(self, alpha=0) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float64))

    def __str__(self):
        return super().__str__() + f"(alpha={self.alpha.item()})"

    def forward(self, predictions, targets):
        return wmse_loss(predictions, targets, alpha=self.alpha.item())

def _vpt_loss(predictions, targets):
    _p = torch.as_tensor(predictions)
    _t = torch.as_tensor(targets)

    # Per-step error
    std = torch.std(_t, dim=-2, keepdim=True)
    std = torch.clamp(std, min=1e-8)  # Avoid division by zero
    E_ik = (_p - _t) ** 2 / (std ** 2)
    E_k = torch.sqrt(torch.mean(E_ik, dim=-1))

    return E_k

def vpt_loss(predictions, targets, gamma=0.1):
    """Exact version of VPT loss (not differentiable)"""
    with torch.no_grad():
        E_k = _vpt_loss(predictions, targets)
        E_k = torch.cat([E_k, torch.full((E_k.shape[0], 1), float('inf'))], dim=-1)
        msk = E_k < gamma
        vpt = torch.argmin(msk.float(), dim=-1)
        avg_vpt = torch.mean(vpt.float())
    return vpt, avg_vpt

class VPTLoss(nn.Module):
    r"""Valid Prediction Time Loss

    The Valid Prediction Time is the time until the prediction error exceeds a threshold.

    Specifically, at step k, for each dimension i, the error is

    E_{k,i}=(x_{k,i} - \hat{x}_{k,i})^2 / std(x_i)^2,

    where std(x_i) is the standard deviation of the single trajectory in dimension i.
    The total error at step k, E_k, is the RMSE of E_{k,i} over all dimensions i.
    The VPT is defined as the largest step index k such that E_k < gamma.

    For training, we estimate k by softmax, average the VPT over trajectories,
    and minimize the loss defined as 1/VPT.
    """
    def __init__(self, gamma=0.1, scl=10.0) -> None:
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float64))
        self.register_buffer("scl", torch.tensor(scl, dtype=torch.float64))
        assert self.gamma > 0, "VPTLoss gamma must be positive."
        assert self.scl > 0, "VPTLoss scl must be positive."

    def __str__(self):
        return super().__str__() + f"(gamma={self.gamma.item()})"

    def forward(self, predictions, targets):
        E_k = _vpt_loss(predictions, targets)
        E_k = torch.cat([E_k, torch.full((E_k.shape[0], 1), float('inf'))], dim=-1)
        B, T = E_k.shape

        # Soft probability that step k is still valid
        s = torch.sigmoid(self.scl * 100 * (self.gamma - E_k))  # (B, T), near 1 if valid

        # Survival up to step k: product_{j<k} s_j
        # Build shifted s with leading 1 so cumprod aligns:
        s_shifted = torch.cat(
            [torch.ones(B, 1, device=E_k.device, dtype=E_k.dtype), s[:, :-1]],
            dim=-1,
        )                                           # (B, T)
        survival = torch.cumprod(s_shifted, dim=-1) # (B, T)

        # Hazard: first failure at k ≈ (1 - s_k) * survival_{k}
        q = (1.0 - s) * survival                    # (B, T)

        # Expected first failure index
        time_idx = torch.arange(T, device=E_k.device, dtype=E_k.dtype)  # (T,)
        expected_vpt = (q * time_idx).sum(dim=-1)    # (B,)
        avg_vpt = expected_vpt.mean()                # scalar

        loss = 1.0 / (avg_vpt + 1e-8)

        return loss

#: Mapping of loss names to loss classes.
LOSS_MAP = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
    "vpt": VPTLoss,
    "wmse": WMSELoss,
}
