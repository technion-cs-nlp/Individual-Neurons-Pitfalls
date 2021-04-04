import os
import pickle
import torch
import consts
from model import PosTaggerWholeVector
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

def save_obj(obj, file_name, device, name, data_name, ablation=False):
    path = os.path.join('pickles', 'ablation' if ablation else '',
                        data_name, device.type + '_' + name + file_name + '.pkl')
    with open(path, 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name, device, name,data_name, ablation=False):
    path = os.path.join('pickles','ablation' if ablation else '',
                        data_name, device.type + '_' + name + file_name + '.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def move_557_to_tail(func):
    def inner1(*args, **kwargs):
        neuron_list = func(*args, **kwargs)
        if type(neuron_list) != list:
            neuron_list = neuron_list.tolist()
        neuron_list.append(neuron_list.pop(neuron_list.index(557)))
        return neuron_list
    return inner1


def sort_neurons_by_avg_weights(saved_model_path:str):
    model = PosTaggerWholeVector()
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


def merge_sort(saved_model_path, bayes_res_file_path, k=40):
    by_avg = sort_neurons_by_avg_weights(saved_model_path).tolist()
    by_bayes = sort_neurons_by_bayes_mi(bayes_res_file_path)
    merged = by_bayes[:k]
    for neuron in by_avg:
        if neuron not in merged:
            merged.append(neuron)
    return merged


def sort_zigzag(saved_model_path, bayes_res_file_path):
    by_avg = sort_neurons_by_avg_weights(saved_model_path).tolist()
    by_bayes = sort_neurons_by_bayes_mi(bayes_res_file_path)
    zigzag = []
    flag = True
    while by_bayes and by_avg:
        neuron = by_bayes.pop(0) if flag else by_avg.pop(0)
        if not neuron in zigzag:
            zigzag.append(neuron)
            flag = not flag
    return zigzag

def sort_complex_zigzag(saved_model_path, bayes_res_file_path):
    by_avg = sort_neurons_by_avg_weights(saved_model_path).tolist()
    by_bayes = sort_neurons_by_bayes_mi(bayes_res_file_path)
    zigzag = []
    k = 5
    while k > 0:
        i = k
        while i > 0:
            neuron = by_bayes.pop(0)
            if not neuron in zigzag:
                zigzag.append(neuron)
                i -= 1
        neuron = by_avg.pop(0)
        while neuron in zigzag:
            neuron = by_avg.pop(0)
        zigzag.append(neuron)
        k -= 1
    for neuron in by_avg:
        if neuron not in zigzag:
            zigzag.append(neuron)
    return zigzag

def sort_neurons_by_LMhead_avg_weights(head_path):
    head = BertOnlyMLMHead(BertConfig(vocab_size=consts.VOCAB_SIZE))
    head.load_state_dict(torch.load(head_path))
    head_weights = head.predictions.decoder.weight
    sorted_weights = head_weights.abs().mean(dim=0).sort(descending=True).indices
    return sorted_weights

