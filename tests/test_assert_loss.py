import numpy as np
import pytest
import torch

from dymad.losses import VPTLoss, WMSELoss, vpt_loss, wmse_loss


def _wmse_loss_ref(predictions, targets, alpha):
    n_steps = predictions.shape[-2]
    weights = np.exp(-alpha * np.arange(n_steps))
    weights = weights / weights.sum()
    weights = weights.reshape(1, -1, 1)
    loss = weights * (predictions - targets) ** 2
    return loss.mean()


eps = 1e-15
batch_size = 3
n_steps = 5
n_features = 2
predictions = torch.randn(batch_size, n_steps, n_features, dtype=torch.float64)
targets = torch.randn(batch_size, n_steps, n_features, dtype=torch.float64)


@pytest.mark.parametrize("alpha", [-0.5, 0, 0.5])
def test_wmse_loss(alpha):
    loss = WMSELoss(alpha=alpha)
    loss_cls = loss(predictions, targets)
    loss_fun = wmse_loss(predictions, targets, alpha=alpha)
    loss_ref = _wmse_loss_ref(predictions.numpy(), targets.numpy(), alpha=alpha)

    assert np.abs(loss_cls.item() - loss_ref) < eps, f"WMSELoss alpha={alpha}"
    assert np.abs(loss_fun.item() - loss_cls.item()) == 0, f"wmse_loss alpha={alpha}"


def test_vpt_loss():
    STP = 2
    GMM = 0.08
    ERR = 0.05

    loss = VPTLoss(gamma=GMM, scl=1.0)

    # Trivial case 1 - all steps correct
    pred = targets.clone()
    loss_cls = loss(pred, targets)  # Softmax version
    loss_fun = vpt_loss(pred, targets, gamma=GMM)[1]  # Exact version
    loss_ref = 0.20020132218245484

    assert np.abs(loss_cls.item() - loss_ref) < eps, "VPTLoss softmax, all correct"
    assert loss_fun.item() == float(n_steps), "vpt_loss exact, all correct"

    # Trivial case 2 - all steps wrong
    pred = targets + 100.0
    loss_cls = loss(pred, targets)  # Softmax version
    loss_fun = vpt_loss(pred, targets, gamma=GMM)[1]  # Exact version
    loss_ref = 1e8

    assert np.abs(loss_cls.item() - loss_ref) < eps, "VPTLoss softmax, all wrong"
    assert loss_fun.item() == 0.0, "vpt_loss exact, all wrong"

    # Non-trivial case
    std = torch.std(targets, dim=-2, keepdim=True)
    std = torch.clamp(std, min=1e-8)
    pred1 = targets.clone() + ERR * std
    pred1[:, STP : STP + 2, :] += ERR * std
    pred2 = targets.clone() + ERR * std
    pred2[:, STP + 1 :, :] += ERR * std
    preds = torch.cat([pred1, pred2], dim=0)
    trgts = targets.repeat(2, 1, 1)

    loss_cls = loss(preds, trgts)  # Softmax version
    loss_fun = vpt_loss(preds, trgts, gamma=GMM)[1]  # Exact version

    # Reference
    loss_ref = 0.41382479387515353

    assert np.abs(loss_cls.item() - loss_ref) < eps, "VPTLoss softmax, non-trivial"
    assert loss_fun.item() == STP + 0.5, "vpt_loss exact, non-trivial"
