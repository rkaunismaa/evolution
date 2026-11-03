import math
from typing import Dict

import torch
from torch import Tensor

from .config import Config


class FoodCapacityField:
    """Per-cell multiplier on max_food, giving the world rich and poor regions of land.

    The field is a sum of `n_modes` plane waves with integer wavevectors, so it tiles seamlessly
    across the world's edges and averages exactly 1 over the world. Wavelengths are roughly
    cfg.food_capacity_scale cells. If cfg.food_capacity_change_steps > 0, each wave's phase drifts at
    its own speed, so regions slowly grow, shrink, merge and split, and the landscape becomes
    unrecognisable after about that many steps. The field is a pure function of time, so a checkpoint
    only needs the wave parameters (see state_dict).

    With cfg.food_types > 1 there is one independent field per food type (the multiplier is
    [food_types, S, S]), so each food has its own rich and poor land.

    Evaluating it is one small matrix product: cos(A + B) = cos A cos B - sin A sin B, with A
    depending only on x and B only on y.
    """

    n_modes = 24
    STD_PER_VARIATION = 0.375   # matches the contrast of the earlier noise-based map

    def __init__(self, cfg: Config, pad: int, device: torch.device):
        self.cfg = cfg
        self.pad = pad
        self.device = device
        size = cfg.size
        shape = (getattr(cfg, 'food_types', 1), self.n_modes)   # wave parameters are [types, modes]

        radius = size / cfg.food_capacity_scale * (0.6 + 0.8 * torch.rand(shape))
        angle = 2 * math.pi * torch.rand(shape)
        kx = torch.round(radius * torch.cos(angle))
        ky = torch.round(radius * torch.sin(angle))
        zero = (kx == 0) & (ky == 0)   # a zero wavevector would just shift the mean
        kx[zero] = 1.0
        self.kx, self.ky = kx, ky
        self.amp = torch.where(torch.rand(shape) < 0.5, -1.0, 1.0)
        self.phase = 2 * math.pi * torch.rand(shape)
        change = getattr(cfg, 'food_capacity_change_steps', 0)
        if change > 0:
            sign = torch.where(torch.rand(shape) < 0.5, -1.0, 1.0)
            self.omega = sign * (2 * math.pi / change) * (0.5 + torch.rand(shape))
        else:
            self.omega = torch.zeros(shape)
        # scale so the waves' sum has standard deviation STD_PER_VARIATION; the multiplier
        # 1 + variation * field then keeps ~99% of land within [1 - variation, 1 + variation]
        self.norm = torch.ones(shape[0])
        self.norm = self._raw(0).std(dim=(1, 2)).cpu() / self.STD_PER_VARIATION   # one per type

        side = size + 2 * pad
        self.multiplier = torch.ones((shape[0], side, side), device=device)   # padding stays at 1
        self.update(0)

    @property
    def moving(self) -> bool:
        return bool((self.omega != 0).any())

    def _raw(self, time: int) -> Tensor:
        """Sum of the waves at `time` over the central grid, divided by self.norm. Shape [types, size, size]."""
        size, dev = self.cfg.size, self.device
        coords = torch.arange(size, device=dev, dtype=torch.float64) * (2 * math.pi / size)
        kx, ky = self.kx.to(dev, torch.float64), self.ky.to(dev, torch.float64)
        amp, phase = self.amp.to(dev, torch.float64), (self.phase + self.omega * time).to(dev, torch.float64)
        a = kx[..., None] * coords + phase[..., None]   # [t, n, x]
        b = ky[..., None] * coords                      # [t, n, y]
        # angles in float64 (phases grow with time), the [size, size] product in float32 (fast on GPUs)
        u = torch.cat([amp[..., None] * torch.cos(a), -amp[..., None] * torch.sin(a)], dim=1).float()   # [t, 2n, x]
        v = torch.cat([torch.cos(b), torch.sin(b)], dim=1).float()                                      # [t, 2n, y]
        return (v.transpose(1, 2) @ u) / self.norm.to(dev)[:, None, None]                              # [t, y, x]

    def update(self, time: int):
        """Recompute the multiplier in place for `time` (the tensor itself is reused)."""
        variation = self.cfg.food_capacity_variation
        p = self.pad
        central = self.multiplier[:, p:-p, p:-p] if p > 0 else self.multiplier
        central.copy_((1.0 + variation * self._raw(time)).clamp_(min=0.0))

    def state_dict(self) -> Dict[str, Tensor | float]:
        return {'kx': self.kx, 'ky': self.ky, 'amp': self.amp, 'phase': self.phase,
                'omega': self.omega, 'norm': self.norm}

    def load_state_dict(self, state: Dict, time: int):
        # older saves have a single field: [modes] parameters and a float norm
        as_2d = lambda x: x.unsqueeze(0) if x.dim() == 1 else x
        self.kx, self.ky, self.amp = as_2d(state['kx']), as_2d(state['ky']), as_2d(state['amp'])
        self.phase, self.omega = as_2d(state['phase']), as_2d(state['omega'])
        self.norm = torch.as_tensor(state['norm'], dtype=torch.float32).reshape(-1)
        self.update(time)
