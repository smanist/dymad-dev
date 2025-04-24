import numpy as np
import torch
from ..utils.plot import plot_trajectory

def _loss(truth,pred,weakDynParam,lossFn):
    # TODO : check this for various trajectory lengths
    
    # Compute weak form loss and decoder loss
    # For one complete trajectory
    # xTrue should have (nSteps,nStates)
    # pred should include w and wDot in (nSteps,nEmbedStates), and xHat in (nSteps,nStates)
    _C,_D = weakDynParam['C'], weakDynParam['D']
    _N, _dN, _K = weakDynParam['N'], weakDynParam['dN'], weakDynParam['K']
    _w, _wDot, _xHat = pred
    # Assemble matrices 
    _wR = _w.unfold(0,_N,_dN).permute(0,2,1)
    _wDotR = _wDot.unfold(0,_N,_dN).permute(0,2,1)
    _K = np.min([_K,len(_wR)])
    _Cexp = _C.unsqueeze(0).expand(_K, -1, -1)
    _Dexp = _D.unsqueeze(0).expand(_K, -1, -1)
    # Compute weak form loss
    _truth = torch.bmm(_Cexp,_wR).view(-1,_w.shape[1])
    _pred = torch.bmm(_Dexp,_wDotR).view(-1,_wDot.shape[1])
    _wfLoss = lossFn(_pred,_truth)
    # Compute decoder loss
    _deLoss = lossFn(truth,_xHat)
    return _wfLoss+_deLoss

def weak_form_loss(batch,predBatch,dataMeta,criterion):
    _wBatch,_wDotBatch,_xHatBatch = predBatch
    return torch.stack([_loss(_xTrue,[_w,_wDot,_xHat],dataMeta['weakDynParam'],criterion) 
                             for _xTrue,_w,_wDot,_xHat in 
                             zip(batch[...,:dataMeta['n_state_features']],_wBatch,_wDotBatch,_xHatBatch)]).mean()

# TODO : this can move to a base loss.py as this is a common function, and add kwargs
def prediction_rmse(model,truth,ts,data_meta,method='dopri5',plot=False):
    x_truth = truth[:, :data_meta['n_state_features']].detach().cpu().numpy()
    x0, u0 = truth[0, :data_meta['n_state_features']], truth[0,-data_meta['n_control_features']:]
    x_pred = model.predict(x0, u0, ts, method=method).detach().cpu().numpy()
    rmse = np.sqrt(np.mean((x_pred - x_truth)**2))
    if plot:
        plot_trajectory(np.array([x_pred, x_truth]), ts)
    return rmse