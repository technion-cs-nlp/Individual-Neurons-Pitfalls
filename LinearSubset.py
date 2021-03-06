import pickle

from train_and_test import train, test
from dataHandler import UMDataHandler
import consts
import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm as progressbar
import logging
import utils
from argparse import ArgumentParser
import sys

logging.basicConfig(level=logging.INFO)


def get_ranking(args):
    func, path = args[0], args[1]
    if path is not None:
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
    res_file_dir = Path('results', data_name, model_type, language, args.attribute, 'layer ' + str(layer))
    if not res_file_dir.exists():
        sys.exit('WRONG SETTING')
    linear_model_path = Path('pickles', data_name, model_type, language, attribute,
                             'best_model_whole_vector_layer_' + str(layer) + control_str)
    bayes_res_path = Path(res_file_dir, 'gaussian by ttb gaussian' + control_str)
    worst_bayes_res_path = Path(res_file_dir, 'gaussian by btt gaussian' + control_str)
    cluster_ranking_path = Path('pickles', 'UM', model_type, language, attribute, str(layer), 'probeless_ranking.pkl')

    res_file_name = 'linear by ' + args.ranking + control_str
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
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
        data_model = UMDataHandler
        train_data_handler = data_model(train_path, data_name=data_name, model_type=model_type, layer=layer,
                                        control=control,
                                        language=language, attribute=attribute)
        train_data_handler.create_dicts()
        # train_data_handler.get_features()
        test_data_handler = data_model(test_path, data_name=data_name, model_type=model_type, layer=layer,
                                       control=control,
                                       language=language, attribute=attribute)
        test_data_handler.create_dicts()
        # test_data_handler.get_features()
        label_to_idx_path = Path('pickles', data_name, model_type, language, attribute, 'label_to_idx.pkl')
        with open(label_to_idx_path, 'rb') as g:
            label_to_idx = pickle.load(g)
        num_labels = len(label_to_idx)
        ranking_params = {'ttb linear': (utils.sort_neurons_by_avg_weights, linear_model_path, num_labels),
                          'btt linear': (utils.sort_neurons_by_avg_weights, linear_model_path, num_labels),
                          'ttb gaussian': (utils.sort_neurons_by_bayes_mi, bayes_res_path),
                          'btt gaussian': (utils.sort_neurons_by_bayes_mi, worst_bayes_res_path),
                          'random': (utils.sort_neurons_by_random, None),
                          'ttb probeless': (utils.sort_neurons_by_clusters, cluster_ranking_path),
                          'btt probeless': (utils.sort_neurons_by_clusters, cluster_ranking_path)}
        try:
            neurons_list = get_ranking(ranking_params[ranking])
        except FileNotFoundError:
            sys.exit('WRONG SETTING')
        if ranking == 'btt linear' or ranking == 'btt probeless':
            neurons_list = list(reversed(neurons_list))

        for i in progressbar(range(1, consts.SUBSET_SIZE + 1)):
            print('using %d neurons' % (i))
            train_data_loader = DataLoader(train_data_handler.create_dataset(neurons_list[:i]),
                                           batch_size=consts.BATCH_SIZE)
            classifier = train(train_data_loader, model_name='subset', lambda1=0.001, lambda2=0.01, verbose=False)
            test_data_loader = DataLoader(test_data_handler.create_dataset(neurons_list[:i]),
                                          batch_size=consts.BATCH_SIZE)
            test(classifier, test_data_loader)
