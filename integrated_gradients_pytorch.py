import numpy as np
import torch
import torch.nn as nn
import math
import itertools

def axis_repeat(tens, times):# need to make general
    inds = list([i]*times for i in range(tens.shape[0]))
    inds = list(itertools.chain.from_iterable(inds))
    return tens[inds]
def axis_tile(tens, reps, axis=0):
    assert int(reps)==reps, 'reps must be an integer.'
    reps = int(reps)
    shape = [1]*len(tens.shape)
    shape[axis] = reps
    out = tens.repeat(*shape)#*tuple(X[i].shape[1:]))
    return out

def integrated_gradients(model, model_kwargs, index, X, baseline, steps, use_batches=True, batch_size=None, return_grads=False, return_preds=False, verbose=False, unpack=True):
    """
    A Pytorch update for the Integrated Gradients algorithm found here: https://github.com/ankurtaly/Integrated-Gradients
    
    Input can be a tensor or a list of tensors (for multi-input models). If you are running a large model and want to predict for a large dataset, set 'use_batches' to true (default) and specify a batch size. 
    
    To do: 
    1) Add prediction output for result verification
    2) add default base options (noise, zeros, sampling)
    
    Inputs
    ------
    
    model : a pytorch model
        The model for which to caclulate attributions. Input for the model needs to be tensors or a list of tensors. If other inputs are needed, they need to be preloaded with a wrapper or parital function.
    model_kwargs : dict
        An optional dictionary of kwargs for model.
    index : int
        The index of the target output node.
    X : torch.Tensor, numpy.ndarray, list(torch.Tensor), list(numpy.ndarray)
        Input data for which to calculate attributions.
    baseline : torch.Tensor, numpy.ndarray, list(torch.Tensor), list(numpy.ndarray)
        Reference data for the IG algorithm. Choosing a good baseline should be done with domain knowledge and testing. See the discussion in the original article for further suggestions: https://arxiv.org/abs/1703.01365  
        For a TLDR, they generally recommend using zero-inputs (zero embeddings, an all black image, and etc). If you are unsure if zero-values represent an appropriate reference for your work, you can just input a bunch of random points and the algorithm will average the result. This is my prefered "general" method.
    steps : int
        The number of steps to calculate over. More steps will be more accurate, but will also require more calculations.
    use_batches : bool <optional>
        A flag to use batches when calculating gradients. Default is True. (Not currently implemented)
    batch_size : int <optional>
        The batch size for calculating gradients. Only applicable if 'use_batches' is set to True. A value of None is mapped to the 0th dimension of X. Default is None. (Not currently implemented)
    return_grads : bool <optional>
        A flag to return grad results in addition to the integrated gradient calculations (probably no reason to need this). Default is False. (Not curently implemented)
    return_preds : bool <optional>
        A flag to return prediction results in addition to the integrated gradient calculations for sanity checks (not implemented yet). Default is False. (Not curently implemented)
    unpack : bool <optional>
        A flag to indicate that model inputs should be unpacked from the input list. Set False if the input for the model is a single list. Default is True.
    """



    def check_args(model, X, baseline, steps, use_batches, batch_size):#check devices used
        def check_list(X, name):
            if(type(X) == torch.Tensor):
                X = [X]
            elif(type(X) == np.ndarray):
                X = [torch.tensor(X)]
            elif(type(X) != list):
                text = '{name} must be a tensor, numpy array, or a list of tensors/numpy arrays'.format(name=name)
                raise TypeError(text)
            for i,x in enumerate(X):
                if(type(x) == np.ndarray):
                    X[i] = torch.tensor(x)
                elif(type(x) != torch.Tensor):
                    text = 'Element {i} of {name} must be either a numpy array or a tensor. It appears to be of type {tp}.'.format(i=i, name=name,tp=type(x))
                    raise TypeError(text)
                if(len(x.shape)==1):
                    X[i] = x.view(1,-1)
            return X
        X = check_list(X, 'X')
        baseline = check_list(baseline, 'baseline')
        assert len(X) == len(baseline), 'X and baseline have different lengths.'

        device = X[0].device
        first = True
        for i in range(len(X)):
            assert X[i].shape[1:] == baseline[i].shape[1:], 'X and baseline do not have matching dimensions'
            if(X[i].device != device):
                if(first):
                    first = False
                    print('Not all inputs are on the same device. Sending all to the device of X[0], {dev}.'.format(dev=device))
                X[i] = X[i].to(device)
            if(baseline[i].device != device):
                if(first):
                    first = False
                    print('Not all inputs are on the same device. Sending all to the device of X[0], {dev}.'.format(dev=device))
                baseline[i] = baseline[i].to(device)
    
        if(batch_size==None):
            batch_size = X[0].shape[0]
        return(X, baseline, batch_size, device)

    
    
    
    def get_grads(model, model_kwargs, unpack, index, X, use_batches, batch_size, verbose, return_preds):
        for x in X:
            x.requires_grad = True
        if(verbose):
            print('Calculating gradients.')
            
        if(use_batches):
            grads = [[] for i in range(len(X))]
            ratio = X[0].shape[0]/batch_size
            loops = math.floor(ratio)
            rem = math.ceil((ratio-loops)*batch_size)
            if(verbose):
                sb = status_bar(loops+1)
                sb.start()
            
            for i in range(loops+1):
                if(i<loops):
                    X_ = []
                    for tens in X:
                        new_tens = tens[i*batch_size:(i+1)*batch_size].data
                        X_.append(new_tens)
                elif(rem!=0):
                    X_ = []
                    for tens in X:
                        new_tens = tens[i*batch_size:i*batch_size+rem].data
                        X_.append(new_tens)
                else:
                    continue
                for x_ in X_:
                    x_.requires_grad = True
                if(unpack):
                    Y = model(*X_, **model_kwargs)[:, index].sum()
                else:
                    Y = model(X_, **model_kwargs)[:, index].sum()
                Y.backward()
                del Y
                for j,ls in enumerate(grads):
                    ls.append(X_[j].grad)
            grads = [torch.cat(ls, dim=0) for ls in grads]
        else:
            if(unpack):
                Y = model(*X, **model_kwargs)[:, index].sum()
            else:
                Y = model(X, **model_kwargs)[:, index].sum()
            Y.backward()
            del Y
            grads = [x.grad for x in X]
        return(grads)
    
    
    
    
    
    #start IG
    X, baseline, batch_size, device = check_args(model, X, baseline, steps, use_batches, batch_size)
    paths = []
    target = []
    base = []
    if(verbose):
        print('Setting up input matrix.')
    for i in range(len(X)):
        target.append(axis_repeat(X[i], baseline[i].shape[0]))
        base.append(axis_tile(baseline[i], X[i].shape[0]))
        paths.append(axis_repeat(base[-1], steps+1) + axis_repeat(target[-1]-base[-1], steps+1) * axis_tile(torch.linspace(0, 1, steps+1).to(device).view(-1,*tuple([1]*len(base[-1].shape[1:]))), target[-1].shape[0]))
    
    grads_ = get_grads(model=model, model_kwargs=model_kwargs, unpack=unpack, index=index, X=paths, use_batches=use_batches, batch_size=batch_size, return_preds=return_preds, verbose=verbose)
    
    if(verbose):
        print('Integrating gradients.')
    mask1 = torch.ones(steps+1)
    mask1[0] = 0
    mask1 = mask1.eq(1)
    mask1 = axis_tile(mask1.view(-1,1), base[0].shape[0]).view(-1)
    mask2 = torch.ones(steps+1)
    mask2[-1] = 0
    mask2 = mask2.eq(1)
    mask2 = axis_tile(mask2.view(-1,1), base[0].shape[0]).view(-1)
    out = []
    for i in range(len(grads_)):
        grads = grads_[i][mask1] + grads_[i][mask2]
        avg_grads = grads.view(-1, steps, *tuple(grads.shape[1:])).mean(dim=1)
        ig = (target[i]-base[i])*avg_grads
        out.append(ig.view(-1, baseline[i].shape[0], *tuple(ig.shape[1:])).mean(dim=1))
    if(return_grads):
        out = (out, grads_)
    return(out)

