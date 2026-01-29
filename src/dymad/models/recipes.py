import numpy as np
import torch
from typing import Union, Tuple

from dymad.io import DynData
from dymad.models.helpers import build_processor, fzu_selector, get_dims
from dymad.models.model_base import ComposedDynamics
from dymad.models.prediction import predict_continuous_fenc
from dymad.modules import FlexLinear, make_krr
from dymad.numerics import Manifold

class CD_LDM(ComposedDynamics):
    """Latent Dynamics Model (LDM) class."""

    @classmethod
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Dimensions
        dims = get_dims(model_config, data_meta)

        # Autoencoder
        suffix = "_ctrl" if dims['u'] > 0 else "_auto"
        enc_type += suffix

        # Features in the dynamics
        # As is

        # Processor in the dynamics
        processor_net = build_processor(model_config, dims, dtype, device, ifgnn=ifgnn)

        # Prediction options
        input_order = model_config.get('input_order', 'cubic')
        prd_type = model_config.get('predictor_type', 'ode')

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order


class CD_SDM(ComposedDynamics):
    """Sequential Dynamics Model (SDM) class."""

    @classmethod
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Dimensions
        dims = get_dims(model_config, data_meta)
        dims['s'] = dims['z']
        dims['r'] = dims['z'] // dims['seq']

        # Autoencoder
        assert 'raw' in enc_type, "SDM model needs raw-type encoder."
        suffix = "_ctrl" if dims['u'] > 0 else "_auto"
        enc_type += suffix

        # Features in the dynamics
        # As is

        # Processor in the dynamics
        prc_type = model_config.get('processor_type', 'seq_std')
        model_config['processor_type'] = prc_type
        processor_net = build_processor(model_config, dims, dtype, device, ifgnn=ifgnn)

        # Prediction
        input_order = None
        prd_type = model_config.get('predictor_type', 'ode')
        if prd_type == 'exp':
            assert dims['u'] == 0, "SDM with control cannot use predict_discrete_exp."

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order

    def dynamics(self, z, w):
        """Customized dynamics for SDM.

        The input is (..., seq_len*z_dim) for z_{1:T};
        Dynamics returns (..., z_dim) for z_{T+1};
        then we concatenate (..., seq_len*z_dim) for z_{2:T+1} for the final output.
        """
        z_next = super().dynamics(z, w)
        _dim = z_next.shape[-1]
        ztmp = torch.cat([z[..., _dim:], z_next], dim=-1)
        return ztmp


class CD_LFM(ComposedDynamics):
    """Linear Feature Model (LFM) class."""

    def __init__(self, encoder, dynamics, decoder, predict=None, model_config=None, dims=None):
        super().__init__(encoder, dynamics, decoder, predict, model_config, dims)
        self.koopman_dimension = dims['z']

    @classmethod
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Options
        const_term = model_config.get('const_term', True)

        # Dimensions
        dims = get_dims(model_config, data_meta)
        dims['e'] = dims['x']
        dims['z'] = model_config.get('koopman_dimension')

        blin_term = 'blin' in fzu_type
        if dims['u'] > 0:
            if blin_term:
                if const_term:
                    dyn_dim = dims['z'] * (dims['u'] + 1) + dims['u']
                else:
                    dyn_dim = dims['z'] * (dims['u'] + 1)
            else:
                dyn_dim = dims['z'] + dims['u']
        else:
            dyn_dim = dims['z']
        dims['s'] = dyn_dim

        # Autoencoder
        assert 'auto' in enc_type, "LFM model needs state-only encoder."

        # Features in the dynamics
        fzu_type = fzu_selector(fzu_type, dims['u'], const_term)

        # Processor in the dynamics
        processor_net = FlexLinear(dims['s'], dims['z'], bias=False, dtype=dtype, device=device)

        # Prediction options
        input_order = model_config.get('input_order', 'cubic')
        prd_type = model_config.get('predictor_type', 'ode')

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order


class CD_KM(ComposedDynamics):
    """Kernel Machine (KM) class."""

    @classmethod
    def build_core(cls, types, model_config, data_meta, dtype, device, ifgnn=False):
        enc_type, fzu_type, dec_type = types

        # Options
        const_term = model_config.get('const_term', True)

        # Dimensions
        dims = get_dims(model_config, data_meta)
        dims['z'] = model_config.get('kernel_dimension')

        # Autoencoder
        assert 'auto' in enc_type, "KM model needs state-only encoder."

        # Features in the dynamics
        fzu_type = fzu_selector(fzu_type, dims['u'], const_term)

        # Processor in the dynamics
        opts = {
            'type'       : model_config.get('type', 'share'),
            'kernel'     : model_config.get('kernel', None),
            'ridge_init' : model_config.get('ridge_init', 1e-10),
            'jitter'     : model_config.get('jitter', 1e-12),
            'dtype'      : dtype,
            'device'     : device
        }
        processor_net = make_krr(**opts)

        # Prediction options
        input_order = model_config.get('input_order', 'cubic')
        prd_type = model_config.get('predictor_type', 'ode')

        return dims, (enc_type, fzu_type, dec_type, prd_type), processor_net, input_order

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        self.processor_net.set_train_data(inp, out)
        residual = self.processor_net.fit()
        return self.processor_net._alphas, residual

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        KRR relies on data, and when initialized some parameters are placeholders.
        Here we first update the shapes of those parameters to match the checkpoint,
        then call the standard load_state_dict to load values and do checks.
        """
        with torch.no_grad():
            for name, p in self.named_parameters(recurse=True):
                if name in state_dict:
                    saved = state_dict[name]
                    if p.shape != saved.shape:
                        # keep the same Parameter object (so it's still registered,
                        # and any optimizer state tied to id(p) can be preserved)
                        p.set_(torch.empty_like(saved))

        # Then do standard loading and checks
        return super().load_state_dict(state_dict, strict=strict)


class CD_KMSK(CD_KM):
    """Kernel Machine (KM) class with skip connection."""

    def __init__(self, encoder, dynamics, decoder, predict=None, model_config=None, dims=None):
        super().__init__(encoder, dynamics, decoder, predict, model_config, dims)
        self.kernel_dimension = dims['z']

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.processor_net.set_train_data(inp, out-inp[..., :self.kernel_dimension])
        residual = self.processor_net.fit()
        return self.processor_net._alphas, residual


M_KEYS = ['data', 'd', 'K', 'g', 'T', 'iforit', 'extT']
class CD_KMM(CD_KM):
    """
    KM with Manifold constraints.

    The model is based on Geometrically constrained KRR,
    The prediction uses the normal correction scheme.

    See more in Huang, He, Harlim & Li ICLR2025.
    """
    GRAPH = False
    CONT  = True

    def __init__(self, encoder, dynamics, decoder, predict=None, model_config=None, dims=None):
        super().__init__(encoder, dynamics, decoder, predict, model_config, dims)

        self._man_opts = model_config.get('manifold', {})

        # Register buffers for Manifold parameters
        self.register_buffer(f"_m_data", torch.empty(0, dtype=torch.float64))
        self.register_buffer(f"_m_d", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_K", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_g", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_T", torch.empty(0, dtype=torch.int64))
        self.register_buffer(f"_m_iforit", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(f"_m_extT", torch.empty(0, dtype=torch.float64))

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit the kernel dynamics using input-output pairs.
        """
        # Build manifold from input data
        # This is a Numpy object, and we register buffers to reload it later
        self._manifold = Manifold(inp[:,:self.n_total_state_features], **self._man_opts)
        self._manifold.precompute()
        ts = self._manifold.to_tensors()
        for _k, _v in ts.items():
            setattr(self, f"_m_{_k}", _v)

        # Fit KRR with the manifold constraint
        self.processor_net.set_train_data(inp, out)
        self.processor_net.set_manifold(self._manifold)
        residual = self.processor_net.fit()
        return self.processor_net._alphas, residual

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_continuous_fenc(self, x0, ts, w, **kwargs)

    def fenc_step(self, z: torch.Tensor, w: DynData, dt: float) -> torch.Tensor:
        """
        First-order Euler step with Normal Correction.
        """
        dz = self.dynamics(z, w) * dt
        dn = self._manifold._estimate_normal(z.detach().cpu().numpy(), dz.detach().cpu().numpy())
        return z + dz + torch.as_tensor(dn, dtype=self.dtype, device=z.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        KMM relies on the Numpy-based object Manifold, and
        the defining parameters of the latter are registered as buffers in KMM.
        When self is initialized these buffers are placeholders.
        Here we first update the shapes of those buffers to match the checkpoint,
        then call the standard load_state_dict to load values and do checks.
        In the end we reconstruct the Manifold object from the loaded buffers,
        and set this object in appropriate locations.
        """
        with torch.no_grad():
            for name, p in self.named_buffers(recurse=True):
                if name in state_dict:
                    saved = state_dict[name]
                    if p.shape != saved.shape:
                        p.set_(torch.empty_like(saved))

        res = super().load_state_dict(state_dict, strict=strict)

        t = {_k : getattr(self, f"_m_{_k}") for _k in M_KEYS}
        self._manifold = Manifold.from_tensors(t)
        self.processor_net._manifold = self._manifold
        self.processor_net.kernel._manifold = self._manifold

        return res
