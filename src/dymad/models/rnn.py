import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Union
import logging

from dymad.data import DynData, DynGeoData
from dymad.models import ModelBase
from dymad.utils import make_autoencoder, MLP, predict_continuous, predict_discrete, predict_graph_continuous, predict_graph_discrete

logger = logging.getLogger(__name__)

class RNN(ModelBase):

    GRAPH = False
    
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(RNN, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.proc_latent_dimension = model_config.get('proc_latent_dimension', 32)
        self.time_delay = model_config.get('time_delay', 1)
        self.hidden_size = self.latent_dimension

        self.reconstruction_weight = model_config.get('reconstruction_weight', .3)

        self.input_order = model_config.get('input_order', 'cubic')

        enc_depth = model_config.get('encoder_layers', 2)
        proc_depth = model_config.get('processor_layers', 1)
        dec_depth = model_config.get('decoder_layers', 2)
        

        # Determine dimensions
        enc_out_dim = self.latent_dimension if enc_depth > 0 else self.n_total_features
        dec_inp_dim = self.latent_dimension if dec_depth > 0 else self.n_total_features

        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True)
        }
        aec_type = model_config.get('autoencoder_type', 'smp')

        self.encoder_net,self.decoder_net = make_autoencoder(
            type="mlp_"+aec_type,
            input_dim=self.n_total_features,
            latent_dim=self.latent_dimension,
            hidden_dim=enc_out_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

        self.dynamics_net = nn.RNN(
            input_size=enc_out_dim,  
            hidden_size=self.hidden_size, 
            num_layers=proc_depth,
            batch_first=True
        )

        if self.n_total_control_features == 0:
            self.encoder = self._encoder_auto
        else:
            self.encoder = self._encoder_ctrl
            
    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        """


        Args:
            w (DynData): Raw features with w.x containing time-delayed features

        Returns:
            torch.Tensor: Latent representation
        """
        # Concatenate state and control features
        inputs= torch.cat([w.x, w.u], dim=1)
        z = self.encoder_net(inputs)
        return z
    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        """

        Args:
            w (DynData): Raw features with w.x containing time-delayed features

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder_net(w.x)
   
    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map from latent space back to state space.

        Args:
            z_seq (torch.Tensor): [batch_size, time_delay, latent_dimension]
            decoder_net(): input dim= [batch_size, latent_dimension*timedelay]
            w (DynData): Raw features (for additional context if needed)

        Returns:
            torch.Tensor: Reconstructed state
        """
        return self.decoder_net(z)
            
    def dynamics(self, z_seq: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics (next state in sequence).

        Args:
            z (torch.Tensor): Current latent state
            w (DynData): Raw features (for control inputs if needed)

        Returns:
            torch.Tensor: Next latent state
        """

        output ,z_next = self.dynamics_net(z_seq)

        return output,z_next
    

    def forward(self, w: DynData, **kwargs) -> torch.Tensor:
        """
        Perform a single forward pass through the RNN.
        
        Args:
            w: DynData with:
            - w.x shape [batch, time_delay, state_features]
            - w.u shape [batch, time_delay, control_features] (if controlled)
        
        Returns:
            torch.Tensor: Predicted next state [batch, state_features]
        """
        device = w.x.device
        batch_size = w.x.shape[0]

        z_seq = torch.zeros(batch_size, self.time_delay, self.latent_dimension, device=device)
        
        for t in range(self.time_delay):
            if self.n_total_control_features > 0 and w.u is not None:
                x_input = w.x[:, t, :]
                u_input = w.u[:, t, :]
            else:
                x_input = w.x[:, t, :]
                u_input = None
            z_seq[:, t, :] = self.encoder(DynData(x=x_input, u=u_input))


        out, h_t = self.dynamics(z_seq, w)

        z_next = out[:, -1, :]

        x_next = self.decoder(z_next, w)

        return out, z_next, x_next

        

    
    
    def predict(self, x0: torch.Tensor, w: Union[DynData, DynGeoData], ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predict trajectory starting from initial states x0.
        
        Args:
            x0: Initial state sequence [batch_size, time_delay, state_features]
            w: DynData object containing control inputs for the entire trajectory
                w.u should have shape [batch_size, len(ts), control_features]
            ts: Timestamps to predict at
            
        Returns:
            Predicted trajectory [batch_size, len(ts), state_features]
        """
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)

        device = x0.device
        nsteps = len(ts)
        batch_size = x0.shape[0]

        if w.x is not None and w.x.dim() == 2:
            w.x = w.x.unsqueeze(0)

        if w.u is not None and w.u.dim() == 2:
            w.u = w.u.unsqueeze(0)


        x_traj = torch.zeros(batch_size, nsteps, self.n_total_state_features, device=device)
        x_traj[:, :self.time_delay, :] = x0
        
        for i in range(nsteps - self.time_delay):
            x_window = x_traj[:, i:i+self.time_delay, :]
            
            u_window = None
            if w.u is not None:
                u_window = w.u[:, i:i+self.time_delay, :]
            
        
            input_data = DynData(x=x_window, u=u_window)

            _, _, x_next = self.forward(input_data)
            
            x_traj[:, i + self.time_delay, :] = x_next

        x_traj=x_traj.squeeze(0) if x_traj.dim() == 3 else x_traj

        return x_traj
