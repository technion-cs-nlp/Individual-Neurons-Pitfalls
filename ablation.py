import numpy as np

import consts
import utils
from model import MLM, BertFromMiddle
import torch
from tqdm import tqdm as progressbar
from DataHandler import DataHandler
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
import time
from argparse import ArgumentParser
import sys
import pickle
import copy


def get_bert_features(data_path, data_name, model_type, language, layer):
    data_handler = DataHandler(data_path, data_name=data_name, model_type=model_type,
                               layer=layer, control=False, small_dataset=False, ablation=True, language=language)
    data_handler.create_dicts()
    data_handler.get_features()

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

def collate_fn(batch):
    sentences = [item[0] for item in batch]
    features = [item[1] for item in batch]
    return [sentences, features]

def ablate(data_name, set_type, model_type, language, layer, neurons_list, attribute = '', one_by_one=False, ranking='', step=0,
           alpha=1, intervention=False, scaling=''):
    alpha_str = str(np.ceil(alpha.max())) if scaling else alpha
    set_name = set_type + '_'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertFromMiddle(model_type, layer)
    skipped = []
    if data_name == 'UM':
        dump_path = Path('pickles', data_name, model_type, language)
        features_path = Path(dump_path, set_name + 'features_layer_'+str(layer))
        with open(features_path, 'rb') as f:
            set_features = pickle.load(f)
        sent_path = Path(dump_path, set_name+'sentences.pkl')
        # sent_path = Path(dump_path,set_name+'sentences.pkl')
        with open(sent_path, 'rb') as f:
            set_sentences = pickle.load(f)
        skipped_path = Path(dump_path, set_name+'skipped_sentences.pkl')
        if skipped_path.exists():
            with open(skipped_path,'rb') as f:
                skipped = pickle.load(f)
        if attribute != '':
            words_per_attribute_path = Path(dump_path, set_name + 'words_per_attribute.pkl')
            with open(words_per_attribute_path,'rb') as f:
                words_per_att = pickle.load(f)
            lemmas_path = Path(dump_path, set_name + 'lemmas.pkl')
            with open(lemmas_path, 'rb') as f:
                lemmas = pickle.load(f)
            if intervention:
                values_avg_path = Path(dump_path, attribute, str(layer), 'avg_embeddings_by_label.pkl')
                with open(values_avg_path,'rb') as f:
                    values_avg = pickle.load(f)
            # parsed_data_path = Path(dump_path, set_name+'parsed.pkl')
            # with open(parsed_data_path,'rb') as f:
            #     parsed_data = pickle.load(f)
    else:
        set_features = utils.load_obj('features_layer_' + str(layer),
                                    device, 'train_', data_name, ablation=True, model_type=model_type)
        set_sentences = utils.load_obj('clean_sentences', device, 'train_', data_name, ablation=True, model_type=model_type)
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
    decoded_outputs, decoded_tokens, lemmas_ranks = {}, {}, {}
    for num_ablated in progressbar(range(0, max_ablated, step)): #TODO
    # for num_ablated in progressbar(range(50, max_ablated, step)):
        print('neuron {}'.format(neurons_list[num_ablated]))
        counters = dict.fromkeys(['total_loss', 'total_correct', 'total_tokens', 'relevant_correct',
                                  'total_relevant', 'total_correct_relevant',
                                  'total_irrelevant','total_correct_irrelevant'], 0)
        if one_by_one:
            neuron = num_ablated
            print('using neuron ', neuron)
            neurons_to_ablate = list(set(range(consts.BERT_OUTPUT_DIM)) - {neuron})
        else:
            used_neurons = consts.BERT_OUTPUT_DIM - num_ablated
            print('using ', used_neurons, ' neurons')
            neurons_to_ablate = [neurons_list[:num_ablated]]
        start_time = time.time()
        sentence_idx = 0
        decoded_outputs[num_ablated] = []
        decoded_tokens[num_ablated] = []
        lemmas_ranks[num_ablated] = []
        for sentences_and_features in set_dataloader:
            sentences = sentences_and_features[0]
            # features = torch.cat(sentences_and_features[1])
            relevant_indices = []
            features = sentences_and_features[1]
            mod_features = copy.deepcopy(features)
            for f in mod_features:
                # place an empty set in case no words have the attribute
                relevant_indices.append(set())
                if words_per_att[sentence_idx].get(attribute):
                    if intervention:
                        indices_per_label = words_per_att[sentence_idx][attribute]
                        possible_labels = list(values_avg.keys())
                        for label, indices in indices_per_label.items():
                            # ignore labels that have been filtered in parsing (less than 100 examples in some set)
                            if label not in values_avg.keys():
                                continue
                            relevant_indices[-1].update(indices)
                            relevant_words_features = f[indices]
                            coef = alpha[tuple(neurons_to_ablate)] if scaling else alpha
                            relevant_words_features[:, neurons_to_ablate] -= coef * values_avg[label][tuple(neurons_to_ablate)]
                            next_label = possible_labels[(possible_labels.index(label) + 1) % len(possible_labels)]
                            relevant_words_features[:, neurons_to_ablate] += coef * values_avg[next_label][tuple(neurons_to_ablate)]
                            f[indices] = relevant_words_features
                    else:
                        rel_ind = [idx for idxs in words_per_att[sentence_idx][attribute].values() for idx in idxs]
                        relevant_indices[-1]= set(rel_ind)
                        relevant_words_features = f[rel_ind]
                        relevant_words_features[:, neurons_to_ablate] = 0.
                        f[rel_ind] = relevant_words_features
                sentence_idx+=1
            batch_lemmas = lemmas[sentence_idx - len(sentences):sentence_idx]
            res = model(sentences, mod_features, relevant_indices, batch_lemmas)
            # loss, correct_preds, num_tokens, correct_relevant, num_relevant = res
            counters['total_loss'] += res['loss']
            counters['total_correct'] += res['correct_all']
            counters['total_tokens'] += res['num_all']
            counters['total_correct_relevant'] += res['correct_relevant']
            counters['total_relevant'] += res['num_relevant']
            counters['total_correct_irrelevant'] += res['correct_irrelevant']
            counters['total_irrelevant'] += res['num_irrelevant']
            # counters['lemma_preds'] += res['lemma_preds']
            # decoded_outputs[num_ablated].extend(res['pred_sentences'])
            decoded_tokens[num_ablated].extend(res['pred_tokens'])
            lemmas_ranks[num_ablated].extend(res['lemmas_ranks'])
        end = time.time()
        print('time for iteration: {} seconds'.format(end-start_time))
        loss, acc = counters['total_loss'] / len(set_dataloader),\
                    counters['total_correct'] / counters['total_tokens']
        print('loss: ', loss)
        print('accuracy: ', acc)
        relevant_acc = utils.divide_zero(counters['total_correct_relevant'], counters['total_relevant'])
        print('relevant words accuracy: ', relevant_acc)
        irrelevant_acc = utils.divide_zero(counters['total_correct_irrelevant'], counters['total_irrelevant'])
        print('irrelevant words accuracy: ', irrelevant_acc)
        # wrong_relevant = counters['total_relevant'] - counters['total_correct_relevant']
        # print('lemma predictions: ',counters['lemma_preds'])
        # print('lemma predictions ratio: ', counters['lemma_preds'] / wrong_relevant)
        losses.append(loss)
        accs.append(acc)
    if one_by_one:
        print('sorted by loss:')
        print(sorted(range(len(losses)), key=losses.__getitem__))
        print('sorted by acc:')
        print(sorted(range(len(accs)), key=accs.__getitem__, reverse=True))
        print('accs:')
        print(accs)
    outputs_dir = Path('pickles', 'UM', model_type, language, attribute, str(layer))
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True, exist_ok=True)
    intervention_str = '_intervention' if intervention else ''
    scaling_str = scaling
    with open(Path(outputs_dir,f'ablation_token_outputs_by_{ranking}{intervention_str}_{step}_{alpha_str}_{scaling_str}+{set_type}.pkl'),'wb+') as f:
        pickle.dump(decoded_tokens, f)
    with open(Path(outputs_dir,f'ablation_lemmas_ranks_by_{ranking}{intervention_str}_{step}_{alpha_str}_{scaling_str}+{set_type}.pkl'),'wb+') as f:
        pickle.dump(lemmas_ranks,f)

if __name__ == "__main__":
    torch.manual_seed(consts.SEED)

    data_name = 'UM'
    parser = ArgumentParser()
    parser.add_argument('-set', type=str)
    parser.add_argument('-model', type=str)
    parser.add_argument('-language', type=str)
    parser.add_argument('-attribute', type=str)
    parser.add_argument('-layer', type=int)
    parser.add_argument('-ranking', type=str)
    parser.add_argument('-step', type=int, default=1)
    parser.add_argument('-alpha', type=int, default=1)
    parser.add_argument('--intervention', default=False, action='store_true')
    parser.add_argument('-scaling', type=str)
    args = parser.parse_args()
    set_type = args.set
    model_type = args.model
    language = args.language
    attribute = args.attribute
    layer = args.layer
    ranking = args.ranking
    step = args.step
    alpha = args.alpha
    intervention = args.intervention
    scaling = args.scaling
    control = False
    control_str = '_control' if control else ''
    small_dataset = False
    small_dataset_str = '_small' if small_dataset else ''
    intervention_str = '_intervention' if intervention else ''
    alpha_str = str(alpha)
    scaling_str = scaling
    datas_path = consts.dev_paths if set_type == 'dev' else consts.test_paths
    data_path = datas_path[language]
    res_file_dir = Path('results', data_name, model_type, language, attribute, 'layer ' + str(layer))
    if not res_file_dir.exists():
        sys.exit('WRONG SETTING')
    get_bert_features(data_path, data_name, model_type, language, layer)
    linear_model_path = Path('pickles', data_name, model_type, language, attribute,
                             'best_model_whole_vector_layer_' + str(layer) + control_str + small_dataset_str)
    bayes_res_path = Path(res_file_dir, 'bayes by bayes mi'+control_str)
    worst_bayes_res_path = Path(res_file_dir, 'bayes by worst mi'+control_str)
    cluster_ranking_path = Path('pickles', 'UM', model_type, language, attribute, str(layer), 'cluster_ranking.pkl')
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
        if ranking == 'random':
            tmp = get_ranking((ranking_params['top avg']))
    except FileNotFoundError:
        sys.exit('WRONG SETTING')
    if ranking == 'bottom avg' or ranking == 'bottom cluster':
        neurons_list = list(reversed(neurons_list))
    means_path = Path('pickles', 'UM', model_type, language, attribute, str(layer), 'avg_embeddings_by_label.pkl')
    scaling_params = {'top avg': (utils.linear_score, linear_model_path),
                      'bayes mi': (utils.bayes_score, bayes_res_path),
                      'top cluster': (utils.cluster_score, means_path)}
    if scaling:
        if scaling == 'ranking':
            scores = get_ranking(scaling_params[ranking])
            alpha = utils.scaling(scores, alpha)
        elif scaling == 'lnspace':
            alpha = utils.lnscale(neurons_list, alpha)
    sparsed = '' if step == 1 else 'sparsed '
    res_file_name = f'by {ranking}{intervention_str}_{step}_{alpha_str}_{scaling_str}_{set_type}'
    # #TODO for debug
    # res_file_name += '_tmp'
    # ##############################
    ablation_res_dir = Path(res_file_dir,'ablation by attr')
    if not ablation_res_dir.exists():
        ablation_res_dir.mkdir()
    with open(Path(ablation_res_dir, res_file_name), 'w+') as f: ###############TODO
        sys.stdout = f
        print('model: ', model_type)
        print('layer: ', layer)
        print('control: ', control)
        print('small: ', small_dataset)
        print('language: ', language)
        print('attribute: ', attribute)
        print('ranking: ', ranking)
        print('step: ', step)
        print('alpha: ', alpha)
        print('intervention:', intervention)
        ablate(data_name, set_type, model_type, language, layer, neurons_list, attribute=attribute, ranking=ranking,
               step=step, alpha=alpha, intervention=intervention, scaling=scaling)



    # model_path = 'pickles/UD/best_model_whole_vector_layer_' + str(layer)
    # neurons_list = utils.sort_neurons_by_avg_weights(model_path).tolist()
    # neurons_list = list(reversed(utils.sort_neurons_by_avg_weights(model_path)))
    # bayes_res_path = Path('results','UD','UPOS', 'layer '+str(layer), 'bayes by bayes mi')
    # bayes_res_path = Path('results','UD','UPOS', 'layer '+str(layer), 'bayes by worst mi')
    # neurons_list = list(reversed(utils.sort_neurons_by_bayes_mi(bayes_res_path)))
    # neurons_list = list(reversed(utils.sort_neurons_by_random()))
    # neurons_list = list(reversed(utils.merge_sort(model_path, bayes_res_path, k=40)))
    # neurons_list = list(reversed(utils.sort_zigzag(model_path, bayes_res_path)))
    # neurons_list = list(reversed(utils.sort_complex_zigzag(model_path, bayes_res_path)))
    # head_path = Path('pickles','ablation','OnlyMLMHead')
    # neurons_list = list(reversed(utils.sort_neurons_by_LMhead_avg_weights(head_path)))
    # neurons_list = list((utils.sort_neurons_by_LMhead_avg_weights(head_path)))
    # neurons_list = list(reversed(constants.by_loss[layer]))
    # neurons_list = constants.by_loss[layer]

    # ablate(layer, data_name, neurons_list, one_by_one=True)
