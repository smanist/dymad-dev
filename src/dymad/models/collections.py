from dataclasses import dataclass

from dymad.models.helpers import build_model
from dymad.models.model_spec import (
    DecoderKind,
    DecoderSpec,
    DynamicsKind,
    DynamicsSpec,
    EncoderKind,
    EncoderSpec,
    FeatureKind,
    FeatureSpec,
    GraphMode,
    MemorySpec,
    ModelSpec,
    RecipeKind,
    RecipeSpec,
    RolloutSpec,
    TimeDomain,
)
from dymad.models.recipes import CD_KM, CD_KMM, CD_KMSK, CD_LDM, CD_LFM, CD_SDM


def _spec(
    *,
    recipe_kind: RecipeKind,
    model_cls: object,
    time_domain: TimeDomain,
    graph_mode: GraphMode,
    encoder: EncoderKind,
    feature: FeatureKind,
    dynamics: DynamicsKind,
    decoder: DecoderKind,
    rollout: RolloutSpec,
    memory: MemorySpec | None = None,
    name: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        recipe=RecipeSpec(kind=recipe_kind, model_cls=model_cls),
        time_domain=time_domain,
        graph_mode=graph_mode,
        encoder=EncoderSpec(kind=encoder),
        feature=FeatureSpec(kind=feature),
        dynamics=DynamicsSpec(kind=dynamics),
        decoder=DecoderSpec(kind=decoder),
        rollout=rollout,
        memory=memory,
        name=name,
    )


@dataclass
class PredefinedModel:
    """Compatibility wrapper around one typed predefined model spec."""

    model_spec: ModelSpec

    def __post_init__(self):
        self.GRAPH = self.model_spec.graph_mode != "none"
        self.CONT = self.model_spec.continuous_time

    def __call__(self, model_config: dict, data_meta: dict, dtype=None, device=None):
        return build_model(
            self.model_spec,
            model_config,
            data_meta,
            dtype,
            device,
        )

    def typed_spec(self) -> ModelSpec:
        return self.model_spec


DEFAULT_CONT_ROLLOUT = RolloutSpec(
    family="default",
    default_predictor="continuous",
    allowed_predictors=("continuous", "continuous_np", "continuous_exp"),
    supports_control_inputs=True,
)
DEFAULT_DISC_ROLLOUT = RolloutSpec(
    family="default",
    default_predictor="discrete",
    allowed_predictors=("discrete", "discrete_exp"),
    supports_control_inputs=True,
)
LTI_CONT_ROLLOUT = RolloutSpec(
    family="lti",
    default_predictor="continuous",
    allowed_predictors=("continuous", "continuous_np", "continuous_exp"),
    supports_control_inputs=True,
)
LTI_DISC_ROLLOUT = RolloutSpec(
    family="lti",
    default_predictor="discrete",
    allowed_predictors=("discrete", "discrete_exp"),
    supports_control_inputs=True,
)
KMM_ROLLOUT = RolloutSpec(
    family="kmm",
    default_predictor="continuous_fenc",
    allowed_predictors=("continuous_fenc",),
    supports_control_inputs=True,
)
LTI_MEMORY_SPEC = MemorySpec(
    family="concat-latent-control",
    latent_state="cat",
    requires_delay_window=True,
)


LDM = PredefinedModel(
    _spec(
        recipe_kind="ldm",
        model_cls=CD_LDM,
        time_domain="continuous",
        graph_mode="none",
        encoder="smpl",
        feature="none",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="LDM",
    )
)
"""Latent dynamics model (LDM), continuous-time."""
DLDM = PredefinedModel(
    _spec(
        recipe_kind="ldm",
        model_cls=CD_LDM,
        time_domain="discrete",
        graph_mode="none",
        encoder="smpl",
        feature="none",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DLDM",
    )
)
"""LDM, discrete-time."""
GLDM = PredefinedModel(
    _spec(
        recipe_kind="ldm",
        model_cls=CD_LDM,
        time_domain="continuous",
        graph_mode="graph",
        encoder="graph",
        feature="none",
        dynamics="direct",
        decoder="graph",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="GLDM",
    )
)
"""LDM with graph autoencoder, continuous-time."""
DGLDM = PredefinedModel(
    _spec(
        recipe_kind="ldm",
        model_cls=CD_LDM,
        time_domain="discrete",
        graph_mode="graph",
        encoder="graph",
        feature="none",
        dynamics="direct",
        decoder="graph",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DGLDM",
    )
)
"""LDM with graph autoencoder, discrete-time."""
LDMG = PredefinedModel(
    _spec(
        recipe_kind="ldm",
        model_cls=CD_LDM,
        time_domain="continuous",
        graph_mode="node",
        encoder="node",
        feature="none",
        dynamics="graph_direct",
        decoder="node",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="LDMG",
    )
)
"""LDM with graph dynamics, continuous-time."""
DLDMG = PredefinedModel(
    _spec(
        recipe_kind="ldm",
        model_cls=CD_LDM,
        time_domain="discrete",
        graph_mode="node",
        encoder="node",
        feature="none",
        dynamics="graph_direct",
        decoder="node",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DLDMG",
    )
)
"""LDM with graph dynamics, discrete-time."""

DSDM = PredefinedModel(
    _spec(
        recipe_kind="sdm",
        model_cls=CD_SDM,
        time_domain="discrete",
        graph_mode="none",
        encoder="raw",
        feature="none",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DSDM",
    )
)
"""Sequential dynamics model (SDM), always discrete-time."""
DSDMG = PredefinedModel(
    _spec(
        recipe_kind="sdm",
        model_cls=CD_SDM,
        time_domain="discrete",
        graph_mode="node",
        encoder="node_raw",
        feature="none",
        dynamics="graph_direct",
        decoder="node",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DSDMG",
    )
)
"""SDM with graph dynamics, discrete-time."""

KBF = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="continuous",
        graph_mode="none",
        encoder="smpl_auto",
        feature="blin",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="KBF",
    )
)
"""Koopman bilinear form (KBF), continuous-time."""
DKBF = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="discrete",
        graph_mode="none",
        encoder="smpl_auto",
        feature="blin",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DKBF",
    )
)
"""KBF, discrete-time."""
GKBF = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="continuous",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_blin",
        dynamics="direct",
        decoder="graph",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="GKBF",
    )
)
"""KBF with graph autoencoder, continuous-time."""
DGKBF = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="discrete",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_blin",
        dynamics="direct",
        decoder="graph",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DGKBF",
    )
)
"""KBF with graph autoencoder, discrete-time."""

LTI = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="continuous",
        graph_mode="none",
        encoder="smpl_auto",
        feature="cat",
        dynamics="direct",
        decoder="auto",
        rollout=LTI_CONT_ROLLOUT,
        memory=LTI_MEMORY_SPEC,
        name="LTI",
    )
)
"""Linear time-invariant (LTI), continuous-time."""
DLTI = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="discrete",
        graph_mode="none",
        encoder="smpl_auto",
        feature="cat",
        dynamics="direct",
        decoder="auto",
        rollout=LTI_DISC_ROLLOUT,
        memory=LTI_MEMORY_SPEC,
        name="DLTI",
    )
)
"""LTI, discrete-time."""
GLTI = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="continuous",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_cat",
        dynamics="direct",
        decoder="graph",
        rollout=LTI_CONT_ROLLOUT,
        memory=LTI_MEMORY_SPEC,
        name="GLTI",
    )
)
"""LTI with graph autoencoder, continuous-time."""
DGLTI = PredefinedModel(
    _spec(
        recipe_kind="lfm",
        model_cls=CD_LFM,
        time_domain="discrete",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_cat",
        dynamics="direct",
        decoder="graph",
        rollout=LTI_DISC_ROLLOUT,
        memory=LTI_MEMORY_SPEC,
        name="DGLTI",
    )
)
"""LTI with graph autoencoder, discrete-time."""

KM = PredefinedModel(
    _spec(
        recipe_kind="km",
        model_cls=CD_KM,
        time_domain="continuous",
        graph_mode="none",
        encoder="smpl_auto",
        feature="blin",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="KM",
    )
)
"""Kernel machine (KM), continuous-time."""
KMM = PredefinedModel(
    _spec(
        recipe_kind="kmm",
        model_cls=CD_KMM,
        time_domain="continuous",
        graph_mode="none",
        encoder="smpl_auto",
        feature="blin",
        dynamics="direct",
        decoder="auto",
        rollout=KMM_ROLLOUT,
        name="KMM",
    )
)
"""Kernel machine on manifold (KMM), continuous-time."""
DKM = PredefinedModel(
    _spec(
        recipe_kind="km",
        model_cls=CD_KM,
        time_domain="discrete",
        graph_mode="none",
        encoder="smpl_auto",
        feature="blin",
        dynamics="direct",
        decoder="auto",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DKM",
    )
)
"""KM, discrete-time."""
GKM = PredefinedModel(
    _spec(
        recipe_kind="km",
        model_cls=CD_KM,
        time_domain="continuous",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_blin",
        dynamics="direct",
        decoder="graph",
        rollout=DEFAULT_CONT_ROLLOUT,
        name="GKM",
    )
)
"""KM with graph autoencoder, continuous-time."""
DGKM = PredefinedModel(
    _spec(
        recipe_kind="km",
        model_cls=CD_KM,
        time_domain="discrete",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_blin",
        dynamics="direct",
        decoder="graph",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DGKM",
    )
)
"""KM with graph autoencoder, discrete-time."""
DKMSK = PredefinedModel(
    _spec(
        recipe_kind="kmsk",
        model_cls=CD_KMSK,
        time_domain="discrete",
        graph_mode="none",
        encoder="smpl_auto",
        feature="blin",
        dynamics="skip",
        decoder="auto",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DKMSK",
    )
)
"""Kernel machine with skip-connection (KMSK), discrete-time."""
DGKMSK = PredefinedModel(
    _spec(
        recipe_kind="kmsk",
        model_cls=CD_KMSK,
        time_domain="discrete",
        graph_mode="graph",
        encoder="graph_auto",
        feature="graph_blin",
        dynamics="skip",
        decoder="graph",
        rollout=DEFAULT_DISC_ROLLOUT,
        name="DGKMSK",
    )
)
"""KMSK with graph autoencoder, discrete-time."""
