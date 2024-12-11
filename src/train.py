import copy
import pickle
import jax
import jax.numpy as jn
import matplotlib.pyplot as plt
import numpy as np
import optax

def fit_model(case, loss_group, data, numEpoch, maxIter, printInt, reset=True, reset_stop=-1):
    loss = loss_group['loss']
    frst = loss_group['reset']
    keys = loss_group['keys']

    has_reset = frst is not None and reset
    if reset_stop < 0:
        reset_stop = numEpoch

    def _fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
        def step(params, opt_state, batch):
            grads, losses = loss(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, losses

        if has_reset:
            params = frst(params, data.data_train)
        opt_state = optimizer.init(params)

        best, opti = np.inf, None
        hist = {_k:[] for _k in keys}
        flag = 1
        for j in range(numEpoch):
            print(f"Epoch {j}")
            data.shuffle()

            if has_reset and j < reset_stop:
                params = frst(params, data.data_train)
                opt_state = optimizer.init(params)

            tmp = {_k:[] for _k in keys}
            for i in range(data.Nb):
                batch = data[i]
                params, opt_state, losses = step(params, opt_state, batch)
                for _k in keys:
                    tmp[_k].append(losses[_k])
                if losses['L'] < best:
                    best = losses['L']
                    opti = copy.deepcopy(params)
                if i % printInt == 0:
                    ss = f"    step {i}"
                    for _k in keys:
                        ss += f", {_k}:{losses[_k]:5.4e}"
                    print(ss)

                if maxIter > 0:
                    if data.Nb*j + i > maxIter:
                        flag = -1
                        break

            for _k in keys:
                hist[_k].append(jn.array(tmp[_k]))
            print(f"    Saving {case} {j}")
            pickle.dump(opti, open(f'{case}_mdl.pkl', 'wb'))
            pickle.dump(hist, open(f'{case}_hst.pkl', 'wb'))

            if flag < 0:
                break

        return opti, hist
    return _fit
