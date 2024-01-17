import pytest

from torch.nn import BCELoss

from src.losses import get_losses
from src.config import LossConfig


@pytest.fixture
def loss_cfg():
    return LossConfig(
        name='bce',
        weight=0.8,
        loss_fn='torch.nn.BCELoss',
        loss_kwargs={}
    )


def test_get_loss_res_len(loss_cfg: LossConfig):
    assert len(get_losses([loss_cfg])) == 1


def test_get_loss_kwargs(loss_cfg: LossConfig):
    loss_cfg.loss_kwargs = {
        'reduction': 'none',
    }

    assert get_losses([loss_cfg])[0].loss.reduction == 'none'
