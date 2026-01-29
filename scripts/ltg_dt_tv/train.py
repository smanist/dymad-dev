import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model, visualize_model
from dymad.models import DLDMG
from dymad.training import NODETrainer
from dymad.utils import plot_summary, plot_trajectory

mdl_kb = {
    "name" : 'kura_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "processor_layers" : 1,
    "hidden_dimension" : 2,
    "gcl": "gcnv",
    "gcl_opts": {"bias": False},
    "activation" : "none",
    "weight_init" : "xavier_uniform"}

trn_nd = {
    "n_epochs": 1000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 4, 8],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    }

config_path = 'config.yaml'
data_path = './data/data_n2_s3_k4_s10.pkl'
cfgs = [
    ('dldmg', DLDMG, NODETrainer, {"data": {"path": data_path}, "model": mdl_kb, "training" : trn_nd}),
    ]

IDX = [0]
labels = [cfgs[i][0] for i in IDX]

iftrn = 1
ifplt = 1
ifprd = 1

if iftrn:
    for _i in IDX:
        mdl, MDL, Trainer, opt = cfgs[_i]
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    npz_files = ['kura_model']
    npzs = plot_summary(npz_files, labels = labels, ifclose=False)

if ifprd:
    data_path = './data/data_n2_s3_k4_s20.pkl'
    data = np.load(data_path, allow_pickle=True)
    tdx = 10
    x_data = data['x'][tdx]
    t_data = np.arange(0, x_data.shape[0])
    ei_data = data['ei'][tdx]
    ew_data = data['ew'][tdx]

    res = [x_data]
    for _i in IDX:
        with torch.no_grad():
            mdl, MDL, Trainer, opt = cfgs[_i]
            model, prd_func = load_model(MDL, 'kura_model.pt')
            pred = prd_func(x_data, t_data, ei=ei_data, ew=ew_data)
            res.append(pred)

        model_graph = visualize_model(
            model=model, prd_func=prd_func,
            ref_data={'t': t_data, 'x': x_data, 'ei': ei_data, 'ew': ew_data},
            depth=1, ifsave=f'kura_model/kura_model')

    plot_trajectory(
        np.array(res), t_data, "LTGV",
        labels=['Truth']+labels, ifclose=False)

plt.show()
