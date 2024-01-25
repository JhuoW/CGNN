import yaml
import torch
import os.path as osp
import numpy as np
import random

def set_random_seed(seed):		
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def config2args(args, dataset_name):
    config_path = osp.join('config', f'{args.model}.yaml')
    with open(config_path, "r") as file:
        hyperparams = yaml.safe_load(file)
    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            setattr(args, name, value)
            # raise ValueError(f"Trying to set non existing parameter: {name}")

    return args
    
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)