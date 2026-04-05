import copy
import logging
from typing import Dict

from dymad.models.components import ENC_MAP, DEC_MAP, FZU_MAP, DYN_MAP, LIN_MAP
from dymad.models.model_spec import ModelSpec, ModelSpecValidationError, ResolvedModelSpec
from dymad.models.rollout_engine import select_rollout_engine
from dymad.modules import make_autoencoder, make_network

logger = logging.getLogger(__name__)

def get_dims(model_config, data_meta):
    """
    Determine dimensions used in the model based on configuration and metadata.

    This is a generic guess and can be overridden by specific model classes.
    """
    # Basic dimensions
    dim_x  = data_meta.get('n_total_state_features')
    dim_u  = data_meta.get('n_total_control_features')
    dim_e  = dim_x + dim_u                      # Input dim to encoder
    l_seq  = data_meta.get('delay') + 1         # Sequence/time-delay length per step

    dim_h  = model_config.get('hidden_dimension', 64)
    n_enc  = model_config.get('encoder_layers', 2)
    n_dec  = model_config.get('decoder_layers', 2)
    n_prc  = model_config.get('processor_layers', 2)

    # Derived dimensions - default options
    dim_z = model_config.get('latent_dimension', None)
    if dim_z is None:
        dim_z = dim_h if n_enc > 0 else dim_e   # Latent dimension
    dim_r = dim_s = dim_z                       # Feature and processor output dimension
    dims = {
        'x'  : dim_x,
        'u'  : dim_u,
        'e'  : dim_e,
        'z'  : dim_z,
        's'  : dim_s,
        'r'  : dim_r,
        'h'  : dim_h,
        'enc': n_enc,
        'dec': n_dec,
        'prc': n_prc,
        'seq': l_seq
    }
    return dims


def build_autoencoder(model_config, dims, dtype, device, ifgnn = False):
    # Determine other options for MLP layers
    opts = {
        'activation'     : model_config.get('activation', 'prelu'),
        'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
        'bias_init'      : model_config.get('bias_init', 'zeros'),
        'gain'           : model_config.get('gain', 1.0),
        'end_activation' : model_config.get('end_activation', True),
        'dtype'          : dtype,
        'device'         : device
    }
    if ifgnn:
        opts['gcl']      = model_config.get('gcl', 'sage')
        opts['gcl_opts'] = model_config.get('gcl_opts', {})
    aec_type = model_config.get('autoencoder_type', 'smp')

    # Build encoder/decoder networks
    if aec_type[:3] in ["gnn", "mlp"]:
        pref = ''
    else:
        pref = "gnn_" if ifgnn else "mlp_"
    encoder_net, decoder_net = make_autoencoder(
        ae_type    = pref+aec_type,
        input_dim  = dims['e'],
        hidden_dim = dims['h'],
        latent_dim = dims['z'],
        enc_depth  = dims['enc'],
        dec_depth  = dims['dec'],
        output_dim = dims['x'],
        seq_len    = dims['seq'],
        **opts
    )

    return encoder_net, decoder_net


def build_processor(model_config, dims, dtype, device, ifgnn = False):
    # Processor in the dynamics
    opts = {
        'activation'     : model_config.get('activation', 'prelu'),
        'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
        'bias_init'      : model_config.get('bias_init', 'zeros'),
        'gain'           : model_config.get('gain', 1.0),
        'end_activation' : model_config.get('end_activation', True),
        'dtype'          : dtype,
        'device'         : device
    }
    if ifgnn:
        opts['gcl']      = model_config.get('gcl', 'sage')
        opts['gcl_opts'] = model_config.get('gcl_opts', {})

    prc_type = model_config.get('processor_type', None)
    if prc_type is None:
        # Default processor type
        prc_type = 'gnn_smp' if ifgnn else 'mlp_smp'
    else:
        if prc_type[:3] in ["gnn", "mlp", "seq"]:
            pref = ''
        else:
            pref = "gnn_" if ifgnn else "mlp_"
        prc_type = pref + prc_type
    processor_net = make_network(
        nn_type    = prc_type,
        input_dim  = dims['s'],
        hidden_dim = dims['h'],
        output_dim = dims['r'],
        n_layers   = dims['prc'],
        seq_len    = dims['seq'],
        **opts
    )

    return processor_net


def fzu_selector(fzu_type, n_total_control_features, const_term):
    _type = fzu_type
    if n_total_control_features > 0:
        if fzu_type in ["blin", "graph_blin"]:
            if const_term:
                _type += '_with_const'  # Encoder with control, bilinear with const
            else:
                _type += '_no_const'    # Encoder with control, bilinear without const
    else:
        _type = "none"                  # Encoder without control
    assert _type in FZU_MAP, f"Unknown zu_cat type {_type}."
    return _type


def build_model(
        model_spec: ModelSpec,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
    """
    Build a model from a typed :class:`~dymad.models.model_spec.ModelSpec`.
    """
    if not isinstance(model_spec, ModelSpec):
        raise ModelSpecValidationError(
            f"build_model expects ModelSpec, got {type(model_spec).__name__}."
        )

    from dymad.models.recipes import resolve_recipe

    recipe_resolution = resolve_recipe(model_spec, model_config, data_meta, dtype, device)
    rollout_engine = select_rollout_engine(model_spec, model_config, recipe_resolution.dims)
    resolved = ResolvedModelSpec(
        model_spec=model_spec,
        dims=recipe_resolution.dims,
        encoder_key=recipe_resolution.encoder_key,
        feature_key=recipe_resolution.feature_key,
        dynamics_key=recipe_resolution.dynamics_key,
        decoder_key=recipe_resolution.decoder_key,
        predictor_key=rollout_engine.source.split(":", 1)[1],
        predictor=rollout_engine.predictor,
        input_order=recipe_resolution.input_order,
        processor_net=recipe_resolution.processor_net,
        graph_mode=model_spec.graph_mode,
        linear_mode="graph" if model_spec.graph_mode != "none" else "smpl",
        continuous_time=model_spec.continuous_time,
    )
    graph_ae = resolved.encoder_key.startswith("graph")

    # Autoencoder
    encoder_net, decoder_net = build_autoencoder(
        model_config,
        resolved.dims,
        dtype,
        device,
        ifgnn=graph_ae,
    )

    predict = resolved.predictor

    # The full model
    model = model_spec.model_cls(
        encoder=ENC_MAP[resolved.encoder_key],
        dynamics=(FZU_MAP[resolved.feature_key], DYN_MAP[resolved.dynamics_key]),
        decoder=DEC_MAP[resolved.decoder_key],
        predict=(predict, resolved.input_order),
        model_config=copy.deepcopy(model_config),
        dims=copy.deepcopy(resolved.dims),
    )
    model.CONT = resolved.continuous_time
    model.GRAPH = resolved.graph_mode != "none"
    lin_eval, lin_feat = LIN_MAP[resolved.linear_mode]
    model.encoder_net = encoder_net
    model.processor_net = resolved.processor_net
    model.decoder_net = decoder_net
    model._linear_eval = lin_eval
    model._linear_features = lin_feat
    model.dtype = dtype
    model.device = device
    model.model_spec = model_spec
    model.resolved_model_spec = resolved

    logger.info(f"Built model: {model_spec.model_cls.__name__}")
    logger.info(
        f"- Encoder: {resolved.encoder_key}, Dynamics: {resolved.dynamics_key}, "
        f"Decoder: {resolved.decoder_key}, Features: {resolved.feature_key}"
    )
    logger.info(f"- Using predictor: {predict.__name__}, input order: {resolved.input_order}")
    logger.info(f"- If graph model: {model.GRAPH}, continuous-time: {model.CONT}")

    return model


def build_model_from_spec(
        model_spec: ModelSpec,
        model_config: Dict, data_meta: Dict,
        dtype=None, device=None):
    """Compatibility alias for the typed-only build path."""
    return build_model(model_spec, model_config, data_meta, dtype, device)
