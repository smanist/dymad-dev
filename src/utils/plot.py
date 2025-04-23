import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(traj,ts,nRowsCols=None):
    _nTraj,_,_nStates = traj.shape
    assert _nTraj == 2
    _rmse = np.linalg.norm(traj[0]-traj[1])/(traj.shape[1]-1)**0.5
    if len(ts) > traj.shape[1]: ts = ts[:traj.shape[1]]
    if not nRowsCols:
        _nRows,_nCols = _nStates,1
        figSize = (5,_nRows*1.5)
    else:
        _nRows,_nCols = nRowsCols
        figSize = (3*_nRows,3*_nCols)
    _, ax = plt.subplots(_nRows,_nCols,figsize=figSize,sharex=True)
    ax = ax.flatten()
    for _n in range(_nStates):
        ax[_n].plot(ts,traj[0,:,_n],'-r',label='Prediction')
        ax[_n].plot(ts,traj[1,:,_n],'--k',label='Truth')
        ax[_n].set_xlim([0,ts[-1]])
        ax[_n].set_ylabel(f'$x_{_n+1}$')
        ax[_n].legend(loc='upper right')
    ax[0].set_title(f'Prediction RMSE {_rmse:1.3f}')
    ax[-1].set_xlabel('Time steps')
    plt.savefig(f'prediction.png',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close()

def plot_hist(hist,epoch,model_name):
    hist = np.array(hist)
    plt.plot(figsize=(5,5))
    plt.semilogy(hist[:epoch,0],'--k')
    plt.semilogy(hist[:epoch,1],'-r')
    plt.semilogy(hist[:epoch,2],'-b')
    plt.xlim([0,epoch+1])
    plt.xlabel('n_epochs')
    plt.ylabel('Loss')
    plt.savefig(f'./{model_name}_hist.png',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close()