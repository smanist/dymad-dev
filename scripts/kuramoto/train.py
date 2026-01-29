import matplotlib.pyplot as plt
import numpy as np
import textwrap as tw
import torch

from dymad.io import load_model
from dymad.models import ComposedDynamics, ENC_MAP, DEC_MAP, get_dims, predict_discrete
from dymad.modules import make_network
from dymad.training import NODETrainer
from dymad.utils import adj_to_edge, plot_multi_trajs


class DSDMSKG(ComposedDynamics):
    GRAPH = True
    CONT = False

    def __init__(self, model_config, data_meta, dtype=None, device=None):
        super().__init__()

        # Dimensions
        dims = get_dims(model_config, data_meta)
        self.dim_x = dims['x'] // dims['seq']
        self.seq_len = dims['seq']

        # Common options
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
        }

        # Autoencoder as in DyMAD: used in predict, do nothing
        self.encoder_net = None
        self.decoder_net = None
        self._encoder = ENC_MAP['iden']
        self._decoder = DEC_MAP['iden']

        # Components of dynamics
        # "Encoder": sequentially applied to each time step
        self.step_encoder = make_network(
            "seq_mlp_smp",
            input_dim  = dims['e'],
            hidden_dim = dims['h'],
            output_dim = dims['z'],
            n_layers   = dims['enc'],
            seq_len    = dims['seq'],
            **opts
        )

        # "Decoder": map only the last step to original space
        self.step_decoder = make_network(
            "mlp_smp",
            input_dim  = dims['r'],
            hidden_dim = dims['h'],
            output_dim = self.dim_x,
            n_layers   = dims['dec'],
            **opts
        )

        # Processor in the dynamics
        opts['gcl']      = model_config.get('gcl', 'sage')
        opts['gcl_opts'] = model_config.get('gcl_opts', {})
        self.processor_net = make_network(
            "gnn_smp",
            input_dim  = dims['s'],
            hidden_dim = dims['h'],
            output_dim = dims['r'],
            n_layers   = dims['prc'],
            **opts
        )

        # Prediction options
        self.input_order = None
        self._predict = predict_discrete

    def dynamics(self, z, w):   # z is x here
        ss = self.step_encoder(w.g(z), w.ug)
        dz = self.processor_net(ss, w.ei, w.ew, w.ea)
        dx = self.step_decoder(w.g(dz))
        x_next = w.xg[..., -self.dim_x:] + dx
        x_pred = torch.cat([w.xg[..., self.dim_x:], x_next], dim=-1)
        return w.G(x_pred)

    def diagnostic_info(self) -> str:
        """
        Return diagnostic information about the model.

        Returns:
            str: String with model details
        """
        ind = "          "
        fin = lambda net: tw.indent(f"{net}", ind)
        return f"Model parameters: {sum(p.numel() for p in self.parameters())}\n" + \
               f"Encoder:  \n{fin(self.step_encoder)}\n" + \
               f"Processor: \n{fin(self.processor_net)}\n" + \
               f"Decoder:  \n{fin(self.step_decoder)}\n" + \
               f"Prediction: {self._predict.__name__}\n" + \
               f"Continuous-time: {self.CONT}, Graph-compatible: {self.GRAPH}, " + \
               f"Sequence length: {self.seq_len}\n"

config_path = 'kur_seq.yaml'
data_path = './data/data_n4_s5_k4_s5'

mdl_sdm = {
    "activation" : "prelu",
    "weight_init" : "xavier_uniform",
    "gain" : 0.01,
    }

trn_nd = {
    "n_epochs": 1000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [2, 3, 5, 7],
    "sweep_epoch_step": 1000,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    }

dat_opt = {
    "path": data_path + '_train.npz',
}

cfgs = [
    ('sdm_skip', DSDMSKG, NODETrainer, {"data": dat_opt, "model": mdl_sdm, "training" : trn_nd}),
    ]

IDX = [0]
labels = [cfgs[i][0] for i in IDX]

iftrn = 0
ifprd = 1

if iftrn:
    for _i in IDX:
        mdl, MDL, Trainer, opt = cfgs[_i]
        opt['model']['name'] = mdl
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    dat = np.load(data_path + '_test.npz', allow_pickle=True)
    x_data = dat['x']
    t_data = np.arange(x_data.shape[1]) * 0.01
    u_data = dat['u']
    ei_data, ew_data = adj_to_edge(dat['adj'])

    res = [x_data]
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        model, prd_func = load_model(MDL, f'{mdl}.pt')
        _d = model.seq_len - 1
        ei = [_e[_d:] for _e in ei_data]
        ew = [_e[_d:] for _e in ew_data]

        with torch.no_grad():
            pred = prd_func(x_data, t_data[_d:], u=u_data, ei=ei, ew=ew)
            res.append(pred)

    plot_multi_trajs(
        np.array(res), t_data, "LTI",
        us=u_data, labels=['Truth']+labels, ifclose=False,
        xidx=[0, 1, 2, 3, 4], uidx=[0])

plt.show()
