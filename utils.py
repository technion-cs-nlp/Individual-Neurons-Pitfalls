import os
import pickle
import numpy as np
import torch
import consts
from model import PosTaggerWholeVector


def save_obj(obj, file_name, device, name, data_name, ablation=False):
    path = os.path.join('pickles', 'ablation' if ablation else '',
                        data_name, device.type + '_' + name + file_name + '.pkl')
    with open(path, 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_name, device, name,data_name, model_type, ablation=False):
    path = os.path.join('pickles','ablation' if ablation else '',
                        data_name, model_type, device.type + '_' + name + file_name + '.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def sort_neurons_by_avg_weights(saved_model_path:str, last_layer):
    model = PosTaggerWholeVector(last_layer)
    model.load_state_dict(torch.load(saved_model_path))
    weights = model.fc1.weight
    sorted_weights = weights.abs().mean(dim=0).sort(descending=True).indices
    return sorted_weights.tolist()


def sort_neurons_by_random():
    return torch.randperm(consts.BERT_OUTPUT_DIM).tolist()


def sort_neurons_by_bayes_mi(res_file_path):
    neurons = []
    with open(res_file_path,'r') as f:
        for line in f.readlines():
            if line.startswith('added neuron'):
                neurons.append(int(line.split()[-1]))
    return neurons


def sort_neurons_by_clusters(ranking_path):
    with open(ranking_path,'rb') as f:
        ranking = pickle.load(f)
    return ranking


def divide_zero(num, denom):
    return num / denom if denom else 0


def lnscale(neurons_list, upper_bound:float, lower_bound=0):
    lnsp = np.logspace(np.log(upper_bound), np.log(1 / 1000 if lower_bound == 0 else lower_bound), 768, base=np.e)
    scores = np.array([lnsp[neurons_list.index(i)] if i in neurons_list else 0 for i in range(768)], dtype=np.float32)
    return scores