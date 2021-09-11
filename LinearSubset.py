import pickle

from train_and_test import train, test
from DataHandler import DataSubset, UMDataHandler
import consts
from model import PosTaggerWholeVector
import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from rankingNeurons import get_top_neurons
from linearCorrelationAnalysis import LCA
from tqdm import tqdm as progressbar
import logging
logging.basicConfig(level=logging.INFO)
import utils
from argparse import ArgumentParser
import sys

def get_ranking(args):
    func, path = args[0], args[1]
    if path != None:
        if len(args) == 3:
            rank = func(path, args[2])
        else:
            rank = func(path)
    else:
        rank = func()
    return rank



if __name__ == "__main__":
    torch.manual_seed(consts.SEED)
    data_name = 'UM'
    parser = ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-language', type=str)
    parser.add_argument('-attribute', type=str)
    parser.add_argument('-layer', type=int)
    parser.add_argument('-ranking', type=str)
    parser.add_argument('--control', default=False, action='store_true')
    args = parser.parse_args()
    model_type = args.model
    language = args.language
    attribute = args.attribute
    layer = args.layer
    ranking = args.ranking
    control = args.control
    small_dataset = False
    control_str = '_control' if control else ''
    small_dataset_str = '_small' if small_dataset else ''
    res_file_dir = Path('results', data_name, model_type, language, args.attribute,'layer '+str(layer))
    if not res_file_dir.exists():
        # res_file_dir.mkdir(parents=True, exist_ok=True)
        sys.exit('WRONG SETTING')
    linear_model_path = Path('pickles', data_name, model_type, language, attribute,
                             'best_model_whole_vector_layer_' + str(layer) + control_str + small_dataset_str)
    bayes_res_path = Path(res_file_dir, 'bayes by bayes mi'+control_str)
    worst_bayes_res_path = Path(res_file_dir, 'bayes by worst mi'+control_str)
    cluster_ranking_path = Path('pickles', 'UM', model_type, language, attribute, str(layer), 'cluster_ranking.pkl')

    res_file_name = 'linear by ' + args.ranking + control_str
    with open(Path(res_file_dir,res_file_name),'w+') as f:
        sys.stdout = f
        print('model: ', model_type)
        print('layer: ', layer)
        print('control: ', control)
        print('small: ', small_dataset)
        print('language: ', language)
        print('attribute: ', attribute)
        print('ranking: ', ranking)
        train_path = Path('pickles', data_name, model_type, language, 'train_parsed.pkl')
        test_path = Path('pickles', data_name, model_type, language, 'test_parsed.pkl')
        print('creating dataset')
        data_model = UMDataHandler if data_name=='UM' else DataSubset
        train_data_handler = data_model(train_path, data_name=data_name, model_type=model_type, layer=layer, control=control,
                                        small_dataset=small_dataset, language=language, attribute=attribute)
        train_data_handler.create_dicts()
        print('extracting features')
        train_data_handler.get_features()
        test_data_handler = data_model(test_path, data_name=data_name, model_type=model_type, layer=layer, control=control,
                                       small_dataset=small_dataset, language=language, attribute=attribute)
        test_data_handler.create_dicts()
        test_data_handler.get_features()
        # model_path = 'pickles/UD/best_model_whole_vector_layer_'+str(layer)+control_str+small_dataset_str
        # model_path = 'pickles/PENN TO UD/best_model_whole_vector_layer_'+str(layer)+control_str+small_dataset_str
        # for i in range(1, len(neurons_list)):
        label_to_idx_path = Path('pickles', data_name, model_type, language, attribute, 'label_to_idx.pkl')
        with open(label_to_idx_path, 'rb') as f:
            label_to_idx = pickle.load(f)
        num_labels = len(label_to_idx)
        ranking_params = {'top avg': (utils.sort_neurons_by_avg_weights, linear_model_path, num_labels),
                          'bottom avg': (utils.sort_neurons_by_avg_weights, linear_model_path, num_labels),
                          'bayes mi': (utils.sort_neurons_by_bayes_mi, bayes_res_path),
                          'worst mi': (utils.sort_neurons_by_bayes_mi, worst_bayes_res_path),
                          'random': (utils.sort_neurons_by_random, None),
                          'top cluster': (utils.sort_neurons_by_clusters, cluster_ranking_path),
                          'bottom cluster': (utils.sort_neurons_by_clusters, cluster_ranking_path)}
        try:
            neurons_list = get_ranking(ranking_params[ranking])
        except FileNotFoundError:
            sys.exit('WRONG SETTING')
        if ranking == 'bottom avg' or ranking == 'bottom cluster':
            neurons_list = list(reversed(neurons_list))

        for i in progressbar(range(1, consts.SUBSET_SIZE + 1)):
            print('using %d neurons'%(i))
            train_data_loader = DataLoader(train_data_handler.create_dataset(neurons_list[:i]), batch_size=consts.BATCH_SIZE)
            classifier = train(train_data_loader, model_name='subset', lambda1=0.001, lambda2=0.01, verbose=False)
            test_data_loader = DataLoader(test_data_handler.create_dataset(neurons_list[:i]), batch_size=consts.BATCH_SIZE)
            test(classifier, test_data_loader)
