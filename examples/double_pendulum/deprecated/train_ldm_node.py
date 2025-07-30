# from pathlib import Path
# import sys, logging
# project_root = Path().resolve().parent.parent
# sys.path.insert(0, str(project_root))

# logging.basicConfig(
#     filename='train_ldm_node.log',  
#     filemode='w',  
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# )  

# from src.training.node_trainer import NODETrainer

# if __name__ == "__main__":
#     config_path = 'config_ldm_node.yaml'
    
#     logging.info(f"Training LDM with NODE approach")
#     logging.info(f"Config: {config_path}")
    
#     # Create trainer - TrainerBase handles all the setup
#     trainer = NODETrainer(config_path)
    
#     logging.info(f"Model: {trainer.model_name}")
#     logging.info(f"Latent dimension: {trainer.config['model']['latent_dimension']}")
#     logging.info(f"Architecture: {trainer.config['model']['encoder_layers']}-{trainer.config['model']['processor_layers']}-{trainer.config['model']['decoder_layers']}")
#     logging.info(f"Training epochs: {trainer.config['training']['n_epochs']}")
#     logging.info(f"ODE method: {trainer.ode_method}")
#     logging.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters())}")
#     logging.info(f"Device: {trainer.device}")
    
#     # Train the model
#     trainer.train()

import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

logging.basicConfig(level=logging.INFO)

B= 128
N = 501
t_grid = np.linspace(0, 5, N)

def f(t, x, u):
    dx = np.zeros_like(x)
    # x[0] = theta1, x[1] = theta2, x[2] = theta1_dot, x[3] = theta2_dot
    
    # Physical parameters (normalized)
    m1, m2 = 1.0, 1.0  # masses
    l1, l2 = 1.0, 1.0  # lengths  
    g = 9.81            # gravity
    
    # State variables
    th1, th2, w1, w2 = x[0], x[1], x[2], x[3]
    
    # Position derivatives
    dx[0] = w1  # theta1_dot
    dx[1] = w2  # theta2_dot
    
    # Angle difference
    delta = th2 - th1
    
    # Common terms
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    
    # Denominator for both equations
    denom = l1 * (2*m1 + m2 - m2*np.cos(2*delta))
    
    # Numerator for theta1_ddot
    num1 = (-m2*l1*w1**2*sin_delta*cos_delta 
            + m2*g*np.sin(th2)*cos_delta 
            + m2*l2*w2**2*sin_delta 
            - (m1+m2)*g*np.sin(th1))
    
    # Numerator for theta2_ddot  
    num2 = (-m2*l2*w2**2*sin_delta*cos_delta 
            + (m1+m2)*(g*np.sin(th1)*cos_delta - l1*w1**2*sin_delta - g*np.sin(th2)))
    
    # Angular acceleration
    dx[2] = num1 / denom                    # theta1_ddot
    dx[3] = num2 / (l2 * denom / l1)       # theta2_ddot

    # Double spring-mass system dynamics
    # x[0]: displacement of m1, x[1]: displacement of m2
    # x[2]: velocity of m1, x[3]: velocity of m2
    # m1 = 1.0  # mass 1
    # m2 = 1.0  # mass 2
    # k1 = 5.0  # spring constant for first spring (wall to m1)
    # k2 = 5.0  # spring constant for second spring (m1 to m2)
    # c1 = 0.5  # damping coefficient for m1
    # c2 = 0.5  # damping coefficient for m2

    # # Damped double spring-mass system
    # dx[2] = (-k1 * x[0] + k2 * (x[1] - x[0]) - c1 * x[2] + c2 * (x[3] - x[2])) / m1
    # dx[3] = (-k2 * (x[1] - x[0]) - c2 * (x[3] - x[2])) / m2

    # logging.info(f"dx: {dx}")

    inp= np.zeros_like(x)

    return dx

g = lambda t, x, u: x


ifdat = 0
ifchk = 0
iftrn = 1
ifplt = 0
ifprd = 0


if ifdat:
    os.makedirs('./data', exist_ok=True)
    sampler = TrajectorySampler(f, g, config='double_pendulum_data.yaml')
    # ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    ts, xs, _, ys = sampler.sample(t_grid, batch=B)
    us= np.zeros_like(xs)  # Assuming no control input for this example
    np.savez_compressed('./data/dp_data.npz', t=ts, x=ys, u=us)

if ifchk:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    data = np.load('./data/dp_data.npz')
    t = data['t']
    x = data['x']

    for ax_idx, ax in enumerate(axs.flat):
        idx = np.random.randint(0, x.shape[0])
        for i in range(x.shape[2]):
            ax.plot(t[idx], x[idx, :, i], label=f'x[{i}]')
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.set_title(f'Trajectory #{idx}')
        ax.legend()
    plt.tight_layout()
    plt.show()


if iftrn:
    cases={'model' : LDM, 'trainer': NODETrainer, 'config': 'dp_ldm_node.yaml'}
    Model= cases['model']
    Trainer = cases['trainer']
    config_path = cases['config']
    setup_logging(config_path, mode='info', prefix='./logs')
    logging.info(f'Starting Training : {Model.__name__} with {Trainer.__name__} using config {config_path}')
    trainer= Trainer(config_path,Model)
    trainer.train()
    

