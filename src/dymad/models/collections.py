from dataclasses import dataclass
from typing import Dict

from dymad.models.helpers import build_model
from dymad.models.recipes import CD_KM, CD_KMM, CD_KMSK, CD_LDM, CD_LFM, CD_SDM

@dataclass
class PredefinedModel:
    """
    Predefined model specification for easy building of common models.

    Args:
        CONT (bool): Whether the model is continuous-time
        encoder (str): Encoder type
        feature (str): Feature type
        dynamics (str): Dynamics type
        decoder (str): Decoder type
        model_cls (object): Model class, expected to be subclass of ComposedDynamics
          with a `build_core` class method.
    """
    CONT: bool
    encoder: str
    feature: str
    dynamics: str
    decoder: str
    model_cls: object

    def __post_init__(self):
        # Determine if graph compatible
        # Three possibilities: autoencoder (graph/node), dynamics (graph)
        self.GRAPH = \
            'graph' in self.encoder or 'node' in self.encoder or \
            'graph' in self.decoder or 'node' in self.decoder or \
            'graph' in self.dynamics

    def __call__(
            self,
            model_config: Dict, data_meta: Dict,
            dtype=None, device=None):
        """
        Build the model based on the predefined specification.
        Essentially a wrapper of :func:`~dymad.models.helpers.build_model`.

        This interface is designed such that the predefined model
        can be directly called to build the model as if initialization.

        For example, when `LDM` is instantiated from `PredefinedModel`,
        calling `LDM(model_config, data_meta, dtype, device)` would behave like
        instantiating a class by `__init__`, but actually invokes this `__call__` method.
        """
        model_spec = [
            self.CONT,
            self.encoder,
            self.feature,
            self.dynamics,
            self.decoder,
            self.model_cls
        ]
        return build_model(model_spec, model_config, data_meta, dtype, device)

#                       CONT,  encoder,    feature, dynamics,       decoder, model_cls
LDM   = PredefinedModel(True,  "smpl",     "none",  "direct",       "auto",  CD_LDM)
"""Latent dynamics model (LDM), continuous-time."""
DLDM  = PredefinedModel(False, "smpl",     "none",  "direct",       "auto",  CD_LDM)
"""LDM, discrete-time."""
GLDM  = PredefinedModel(True,  "graph",    "none",  "direct",       "graph", CD_LDM)
"""LDM with graph autoencoder, continuous-time."""
DGLDM = PredefinedModel(False, "graph",    "none",  "direct",       "graph", CD_LDM)
"""LDM with graph autoencoder, discrete-time."""
LDMG  = PredefinedModel(True,  "node",     "none",  "graph_direct", "node",  CD_LDM)
"""LDM with graph dynamics, continuous-time."""
DLDMG = PredefinedModel(False, "node",     "none",  "graph_direct", "node",  CD_LDM)
"""LDM with graph dynamics, discrete-time."""

DSDM  = PredefinedModel(False, "raw",      "none",  "direct",       "auto",  CD_SDM)
"""Sequential dynamics model (SDM), always discrete-time."""
DSDMG = PredefinedModel(False, "node_raw", "none",  "graph_direct", "node",  CD_SDM)
"""SDM with graph dynamics, discrete-time."""

#                       CONT,  encoder,      feature,      dynamics, decoder, model_cls
KBF   = PredefinedModel(True,  "smpl_auto",  "blin",       "direct", "auto",  CD_LFM)
"""Koopman bilinear form (KBF), continuous-time."""
DKBF  = PredefinedModel(False, "smpl_auto",  "blin",       "direct", "auto",  CD_LFM)
"""KBF, discrete-time."""
GKBF  = PredefinedModel(True,  "graph_auto", "graph_blin", "direct", "graph", CD_LFM)
"""KBF with graph autoencoder, continuous-time."""
DGKBF = PredefinedModel(False, "graph_auto", "graph_blin", "direct", "graph", CD_LFM)
"""KBF with graph autoencoder, discrete-time."""

LTI   = PredefinedModel(True,  "smpl_auto",  "cat",        "direct", "auto",  CD_LFM)
"""Linear time-invariant (LTI), continuous-time."""
DLTI  = PredefinedModel(False, "smpl_auto",  "cat",        "direct", "auto",  CD_LFM)
"""LTI, discrete-time."""
GLTI  = PredefinedModel(True,  "graph_auto", "graph_cat",  "direct", "graph", CD_LFM)
"""LTI with graph autoencoder, continuous-time."""
DGLTI = PredefinedModel(False, "graph_auto", "graph_cat",  "direct", "graph", CD_LFM)
"""LTI with graph autoencoder, discrete-time."""

#                        CONT,  encoder,      feature,      dynamics, decoder, model_cls
KM     = PredefinedModel(True,  "smpl_auto",  "blin",       "direct", "auto",  CD_KM)
"""Kernel machine (KM), continuous-time."""
KMM    = PredefinedModel(True,  "smpl_auto",  "blin",       "direct", "auto",  CD_KMM)
"""Kernel machine on manifold (KMM), continuous-time."""
DKM    = PredefinedModel(False, "smpl_auto",  "blin",       "direct", "auto",  CD_KM)
"""KM, discrete-time."""
GKM    = PredefinedModel(True,  "graph_auto", "graph_blin", "direct", "graph", CD_KM)
"""KM with graph autoencoder, continuous-time."""
DGKM   = PredefinedModel(False, "graph_auto", "graph_blin", "direct", "graph", CD_KM)
"""KM with graph autoencoder, discrete-time."""
DKMSK  = PredefinedModel(False, "smpl_auto",  "blin",       "skip",   "auto",  CD_KMSK)
"""Kernel machine with skip-connection (KMSK), continuous-time."""
DGKMSK = PredefinedModel(False, "graph_auto", "graph_blin", "skip",   "graph", CD_KMSK)
"""KMSK with graph autoencoder, discrete-time."""
