import numpy as np

import consts
import utils
from models import BertFromMiddle
import torch
from tqdm import tqdm as progressbar
from dataHandler import DataHandler
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
import time
from argparse import ArgumentParser
import sys
import pickle
import copy


def get_features(data_path, data_name, model_type, language, layer):
    data_handler = DataHandler(data_path, data_name=data_name, model_type=model_type,
                               layer=layer, control=False, ablation=True, language=language)
    data_handler.create_dicts()
    data_handler.get_features()


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


def collate_fn(batch):
    sentences = [item[0] for item in batch]
    features = [item[1] for item in batch]
    return [sentences, features]


def intervene(data_name, set_type, model_type, language, layer, neurons_list, attribute, ranking='', step=0,
              alpha=1, translation=False, scaled=''):
    alpha_str = str(np.ceil(alpha.max())) if scaled else alpha
    set_name = set_type + '_'
    model = BertFromMiddle(model_type, layer)
    skipped = []
    dump_path = Path('pickles', data_name, model_type, language)
    features_path = Path(dump_path, set_name + 'features_layer_' + str(layer))
    with open(features_path, 'rb') as g:
        set_features = pickle.load(g)
    sent_path = Path(dump_path, set_name + 'sentences.pkl')
    with open(sent_path, 'rb') as g:
        set_sentences = pickle.load(g)
    skipped_path = Path(dump_path, set_name + 'skipped_sentences.pkl')
    if skipped_path.exists():
        with open(skipped_path, 'rb') as g:
            skipped = pickle.load(g)
    words_per_attribute_path = Path(dump_path, set_name + 'words_per_attribute.pkl')
    with open(words_per_attribute_path, 'rb') as g:
        words_per_att = pickle.load(g)
    if translation:
        values_avg_path = Path(dump_path, attribute, str(layer), 'avg_embeddings_by_label.pkl')
        with open(values_avg_path, 'rb') as g:
            values_avg = pickle.load(g)
    num_sentences = len(set_features)
    set_features_list = [set_features[i] for i in range(num_sentences) if i not in skipped]
    set_sentences = [set_sentences[i] for i in range(num_sentences) if i not in skipped]
    dataset = [(sentence, features) for sentence, features in zip(set_sentences, set_features_list)]
    batch_size = consts.ABLATION_BATCH_SIZE
    set_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    accs, losses = [], []
    missing_neurons = set(range(consts.BERT_OUTPUT_DIM)) - set(neurons_list)
    missing_num = len(missing_neurons)
    max_ablated = consts.BERT_OUTPUT_DIM
    if missing_num > 0:
        neurons_list = neurons_list + list(missing_neurons)
        max_ablated = consts.BERT_OUTPUT_DIM - missing_num
    decoded_outputs, decoded_tokens = {}, {}
    for num_ablated in progressbar(range(0, max_ablated, step)):
    # for num_ablated in progressbar(range(50, max_ablated, step)):  # for debugging
        print(f'neuron {format(neurons_list[num_ablated])}')
        counters = dict.fromkeys(['total_loss', 'total_correct', 'total_tokens', 'relevant_correct',
                                  'total_relevant', 'total_correct_relevant',
                                  'total_irrelevant', 'total_correct_irrelevant'], 0)
        used_neurons = consts.BERT_OUTPUT_DIM - num_ablated
        print('using ', used_neurons, ' neurons')
        neurons_to_ablate = [neurons_list[:num_ablated]]
        start_time = time.time()
        sentence_idx = 0
        decoded_outputs[num_ablated] = []
        decoded_tokens[num_ablated] = []
        for sentences_and_features in set_dataloader:
            sentences = sentences_and_features[0]
            relevant_indices = []
            features = sentences_and_features[1]
            mod_features = copy.deepcopy(features)
            for feat in mod_features:
                # place an empty set in case no words have the attribute
                relevant_indices.append(set())
                if words_per_att[sentence_idx].get(attribute):
                    if translation:
                        indices_per_label = words_per_att[sentence_idx][attribute]
                        possible_labels = list(values_avg.keys())
                        for label, indices in indices_per_label.items():
                            # ignore labels that have been filtered in parsing (less than 100 examples in some set)
                            if label not in values_avg.keys():
                                continue
                            relevant_indices[-1].update(indices)
                            # if we update feat[indices] directly, changes accumulate
                            relevant_words_features = feat[indices]
                            coef = alpha[tuple(neurons_to_ablate)] if scaled else alpha
                            relevant_words_features[:, neurons_to_ablate] -= coef * values_avg[label][
                                tuple(neurons_to_ablate)]
                            next_label = possible_labels[(possible_labels.index(label) + 1) % len(possible_labels)]
                            relevant_words_features[:, neurons_to_ablate] += coef * values_avg[next_label][
                                tuple(neurons_to_ablate)]
                            feat[indices] = relevant_words_features
                    else:
                        rel_ind = [idx for idxs in words_per_att[sentence_idx][attribute].values() for idx in idxs]
                        relevant_indices[-1] = set(rel_ind)
                        relevant_words_features = feat[rel_ind]
                        relevant_words_features[:, neurons_to_ablate] = 0.
                        feat[rel_ind] = relevant_words_features
                sentence_idx += 1
            res = model(sentences, mod_features, relevant_indices)
            counters['total_loss'] += res['loss']
            counters['total_correct'] += res['correct_all']
            counters['total_tokens'] += res['num_all']
            counters['total_correct_relevant'] += res['correct_relevant']
            counters['total_relevant'] += res['num_relevant']
            counters['total_correct_irrelevant'] += res['correct_irrelevant']
            counters['total_irrelevant'] += res['num_irrelevant']
            # decoded_outputs[num_ablated].extend(res['pred_sentences'])
            decoded_tokens[num_ablated].extend(res['pred_tokens'])
        end = time.time()
        print('time for iteration: {} seconds'.format(end - start_time))
        loss, acc = counters['total_loss'] / len(set_dataloader), \
                    counters['total_correct'] / counters['total_tokens']
        print('loss: ', loss)
        print('accuracy: ', acc)
        relevant_acc = utils.divide_zero(counters['total_correct_relevant'], counters['total_relevant'])
        print('relevant words accuracy: ', relevant_acc)
        irrelevant_acc = utils.divide_zero(counters['total_correct_irrelevant'], counters['total_irrelevant'])
        print('irrelevant words accuracy: ', irrelevant_acc)
        # wrong_relevant = counters['total_relevant'] - counters['total_correct_relevant']
        losses.append(loss)
        accs.append(acc)
    outputs_dir = Path('pickles', 'UM', model_type, language, attribute, str(layer), set_type)
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True, exist_ok=True)
    translation_str = '_translation' if translation else ''
    scaling_str = '_scaled' if scaled else ''
    with open(Path(outputs_dir,
                   f'ablation_token_outputs_by_{ranking}{translation_str}_{step}_{alpha_str}{scaling_str}.pkl'),
              'wb+') as g:
        pickle.dump(decoded_tokens, g)


if __name__ == "__main__":
    torch.manual_seed(consts.SEED)
    data_name = 'UM'
    parser = ArgumentParser()
    parser.add_argument('-set', type=str, help='data set to analyze results on, can be dev or test, default is test. ')
    parser.add_argument('-model', type=str)
    parser.add_argument('-language', type=str)
    parser.add_argument('-attribute', type=str)
    parser.add_argument('-layer', type=int)
    parser.add_argument('-ranking', type=str)
    parser.add_argument('-step', type=int, default=10, help='step size between number of modified neurons (k), '
                                                            'default is 10')
    parser.add_argument('-beta', type=int, default=8, help='value of beta, default is 8')
    parser.add_argument('--translation', default=False, action='store_true',
                        help='if set to true, apply the translation method rather than ablation')
    parser.add_argument('--scaled', default=False, action='store_true',
                        help='if set to true, use a scaled coefficients vector (alpha) instead of a constant '
                             'coefficient for all neurons')
    args = parser.parse_args()
    set_type = args.set
    if set_type is None:
        set_type = 'test'
    model_type = args.model
    language = args.language
    attribute = args.attribute
    layer = args.layer
    ranking = args.ranking
    step = args.step
    alpha = args.beta
    translation = args.translation
    scaled = args.scaled
    translation_str = '_translation' if translation else ''
    alpha_str = str(alpha)
    scaling_str = '_scaled' if scaled else ''
    datas_path = consts.dev_paths if set_type == 'dev' else consts.test_paths
    data_path = datas_path[language]
    res_file_dir = Path('results', data_name, model_type, language, attribute, 'layer ' + str(layer))
    if not res_file_dir.exists():
        sys.exit('WRONG SETTING')
    get_features(data_path, data_name, model_type, language, layer)
    linear_model_path = Path('pickles', data_name, model_type, language, attribute,
                             'best_model_whole_vector_layer_' + str(layer))
    bayes_res_path = Path(res_file_dir, 'gaussian by ttb gaussian')
    worst_bayes_res_path = Path(res_file_dir, 'gaussian by btt gaussian')
    cluster_ranking_path = Path('pickles', 'UM', model_type, language, attribute, str(layer), 'probeless_ranking.pkl')
    label_to_idx_path = Path('pickles', data_name, model_type, language, attribute, 'label_to_idx.pkl')
    with open(label_to_idx_path, 'rb') as f:
        label_to_idx = pickle.load(f)
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
        if ranking == 'random':
            tmp = get_ranking((ranking_params['ttb linear']))
    except FileNotFoundError:
        sys.exit('WRONG SETTING')
    if ranking == 'btt linear' or ranking == 'btt probeless':
        neurons_list = list(reversed(neurons_list))
    if scaled:
        alpha = utils.lnscale(neurons_list, alpha)
    res_file_name = f'by {ranking}{translation_str}_{step}_{alpha_str}{scaling_str}_{set_type}'
    # TODO for debug
    # res_file_name += '_tmp'
    # ##############################
    ablation_res_dir = Path(res_file_dir, 'ablation by attr')
    if not ablation_res_dir.exists():
        ablation_res_dir.mkdir()
    with open(Path(ablation_res_dir, res_file_name), 'w+') as f:
        sys.stdout = f
        print('model: ', model_type)
        print('layer: ', layer)
        print('language: ', language)
        print('attribute: ', attribute)
        print('ranking: ', ranking)
        print('step: ', step)
        print('beta: ', alpha)
        print('translation:', translation)
        print('scaled:', scaled)
        intervene(data_name, set_type, model_type, language, layer, neurons_list, attribute=attribute, ranking=ranking,
                  step=step, alpha=alpha, translation=translation, scaled=scaled)
