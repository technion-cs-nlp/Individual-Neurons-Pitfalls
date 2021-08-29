import pickle
import numpy as np
import utils
from pathlib import Path
import pandas as pd
from itertools import combinations, product
import seaborn as sns
import matplotlib.pyplot as plt

def obtain_ranking(lan, att, layer, ranking):
    if ranking == 'top avg':
        linear_model_path = Path('pickles', 'UM', lan, att, 'best_model_whole_vector_layer_' + str(layer))
        neurons = utils.sort_neurons_by_avg_weights(linear_model_path.__str__())
    if ranking == 'bayes mi':
        res_file_dir = Path('results', 'UM', lan, att, 'layer ' + str(layer))
        bayes_res_path = Path(res_file_dir, 'bayes by bayes mi')
        neurons = utils.sort_neurons_by_bayes_mi(bayes_res_path)
    if ranking == 'top cluster':
        cluster_ranking_path = Path('pickles', 'UM', lan, att, str(layer), 'cluster_ranking.pkl')
        neurons = utils.sort_neurons_by_clusters(cluster_ranking_path)
    return neurons

def get_all_rankings(num_neurons: int = 768):
    res_root = Path('results', 'UM')
    languages = [name.name for name in res_root.glob('*') if name.is_dir()]
    # languages = ['eng']
    attributes = set([att.name for lan in languages for att in Path(res_root, lan).glob('*') if att.is_dir()])
    layers = [2, 7, 12]
    rankings = ['top avg', 'bayes mi', 'top cluster']
    # cols = pd.MultiIndex.from_product([languages, attributes, layers])
    # rows = pd.MultiIndex.from_product([rankings])
    # df = pd.DataFrame(index=rows, columns=cols).sort_index().sort_index(axis=1)
    all_rankings = dict()
    for lan in languages:
        print(lan)
        all_rankings[lan] = dict()
        for att in attributes:
            if not Path('pickles', 'UM', lan, att).exists():
                continue
            all_rankings[lan][att] = dict()
            for layer in layers:
                all_rankings[lan][att][layer] = dict()
                for ranking in rankings:
                    all_rankings[lan][att][layer][ranking] = obtain_ranking(lan, att, layer, ranking)[:num_neurons]
    with open(Path('pickles', 'UM', 'all_rankings.pkl'),'wb+') as f:
        pickle.dump(all_rankings, f)

def analyze_overlaps(num_neurons:int=768):
    with open(Path('pickles', 'UM', 'all_rankings.pkl'), 'rb') as f:
        all_rankings:dict = pickle.load(f)
    top_neurons = {(lan, att, layer, ranking): values[:num_neurons] for lan, lan_dict in all_rankings.items() for att, att_dict in lan_dict.items()
                   for layer, layer_dict in att_dict.items() for ranking, values in layer_dict.items()}
    top_neurons = pd.DataFrame(top_neurons)
    idx = pd.IndexSlice
    languages, attributes, layers, rankings = top_neurons.axes[1].levels
    for att in attributes:
        for layer in layers:
            for ranking in rankings:
                neurons_per_language = top_neurons.loc[:, idx[languages, [att], [layer], [ranking]]]
                if neurons_per_language.shape[1] <= 1:
                    continue
                relevant_languages = [v[0] for v in neurons_per_language.columns.values]
                for lan_a, lan_b in combinations(relevant_languages, 2):
                    overlap = np.intersect1d(neurons_per_language[lan_a], neurons_per_language[lan_b], assume_unique=True)
                    if len(overlap) >= 50:
                        print(f'{att} layer {layer} by {ranking}, {lan_a}, {lan_b}: {len(overlap)}')
    for lan in languages:
        for layer in layers:
            for ranking in rankings:
                neurons_per_attribute = top_neurons.loc[:, idx[[lan], attributes, [layer], [ranking]]]
                if neurons_per_attribute.shape[1] <= 1:
                    continue
                relevant_attributes = [v[1] for v in neurons_per_attribute.columns.values]
                for att_a, att_b in combinations(relevant_attributes, 2):
                    overlap = np.intersect1d(neurons_per_attribute[lan][att_a], neurons_per_attribute[lan][att_b], assume_unique=True)
                    if len(overlap) >= 40:
                        print(f'{lan} layer {layer} by {ranking}, {att_a}, {att_b}: {len(overlap)}')
    for lan in languages:
        relevant_attributes = set([v[0] for v in top_neurons.loc[:, idx[lan]].columns.values])
        for att in attributes:
            if att not in relevant_attributes:
                continue
            for layer in layers:
                neurons_per_attribute = top_neurons.loc[:, idx[[lan], [att], [layer], rankings]]
                if neurons_per_attribute.shape[1] <= 1:
                    continue
                relevant_rankings = rankings
                for rank_a, rank_b in combinations(relevant_rankings, 2):
                    overlap = np.intersect1d(neurons_per_attribute[lan][att][layer][rank_a], neurons_per_attribute[lan][att][layer][rank_b], assume_unique=True)
                    if len(overlap) >= 50:
                        print(f'{lan} {att} layer {layer}, {rank_a}, {rank_b}: {len(overlap)}')
    # print(top_neurons)

def rename_att(att):
    new_att = 'Gender' if att.startswith('Gender') else 'POS' if att.startswith('Part') else att
    return new_att

def old_att_name(t):
    old_name = 'Gender and Noun Class' if t[1] == 'Gender' else 'Part of Speech' if t[1] == 'POS' else t[1]
    return (t[0], old_name)

def plot_heatmap(num_neurons):
    with open(Path('pickles', 'UM', 'all_rankings.pkl'), 'rb') as f:
        all_rankings: dict = pickle.load(f)
    top_neurons = {(rename_att(att), lan, layer, ranking): values[:num_neurons] for lan, lan_dict in all_rankings.items() for
                   att, att_dict in lan_dict.items()
                   for layer, layer_dict in att_dict.items() for ranking, values in layer_dict.items()}
    top_neurons = pd.DataFrame(top_neurons)
    attributes, languages, layers, rankings = top_neurons.axes[1].levels
    indices = []
    languages = ['eng','spa','fra','fin','bul','rus','hin','ara','tur']
    idx = pd.IndexSlice
    for att in attributes:
        for lan in languages:
            if att not in set([a for a,_,_,_ in top_neurons.loc[:, idx[:,[lan]]].columns.values]):
                continue
            indices.append(f'{att}, {lan}')
    # indices.sort(key=lambda x:x[1])
    matrix = pd.DataFrame(index=indices, columns=indices)
    for layer in layers:
        rankings_overlap = {}
        for r_1, r_2 in combinations(rankings, 2):
            diag = []
            labels = []
            for i_1, i_2 in product(indices, repeat=2):
                # if i_1 == i_2:
                #     continue
                att_1, lan_1 = i_1.split(', ')
                att_2, lan_2 = i_2.split(', ')
                l_1 = top_neurons[(att_1, lan_1, layer, r_1)].values[:x]
                l_2 = top_neurons[(att_2, lan_2, layer, r_2)].values[:x]
                matrix[i_1][i_2] = len(np.intersect1d(l_1, l_2, assume_unique=True))
                if i_1 == i_2:
                    diag.append(len(np.intersect1d(l_1, l_2, assume_unique=True)))
                    labels.append(i_1)

            matrix = matrix.astype(float)
            # h = sns.heatmap(matrix, vmin=0, vmax=75, xticklabels=True, yticklabels=True)
            # h.set_xticklabels(h.get_xmajorticklabels(), fontsize=6)
            # h.set_yticklabels(h.get_ymajorticklabels(), fontsize=6)
            # h.set_facecolor('grey')
            # title = f'layer {str(layer)} by {r_1}, {r_2}'
            # plt.title(title)
            # plt.tight_layout()
            save_dir = Path('results', 'overlaps')
            if not save_dir.exists():
                save_dir.mkdir()
            # plt.savefig(Path(save_dir, title))
            # plt.close()
            # plt.show()
            rankings_overlap[(r_1, r_2)] = diag
        all_three = []
        for i in indices:
            att, lan = i.split(', ')
            l_1 = top_neurons[(att, lan, layer, rankings[0])].values[:x]
            l_2 = top_neurons[(att, lan, layer, rankings[1])].values[:x]
            l_3 = top_neurons[(att, lan, layer, rankings[2])].values[:x]
            intersection = np.intersect1d(l_1, l_2, assume_unique=True)
            intersection = np.intersect1d(intersection, l_3, assume_unique=True)
            all_three.append(len(intersection))
        rankings_overlap['all'] = all_three
        plot_bar(rankings_overlap, labels, 8, layer, save_dir)

def plot_bar(data: dict, settings, num_to_show, layer, save_dir):
    random_2 = 13
    random_3 = 1.7
    order = np.random.permutation(len(settings))
    fig, ax = plt.subplots()
    names = list(data.keys())
    data_table = [vals for vals in data.values()]
    data_table = [[vals[i] for i in order] for vals in data_table]
    data_table = [vals[:num_to_show] for vals in data_table]
    labels = [settings[i] for i in order]
    labels = labels[:num_to_show]
    X = np.arange(num_to_show)
    width = 0.2
    ax.bar(X - width * 3 / 2, data_table[0], width=width, label=names[0])
    ax.bar(X - width * 1 / 2, data_table[1], width=width, label=names[1])
    ax.bar(X + width * 1 / 2, data_table[2], width=width, label=names[2])
    ax.bar(X + width * 3 / 2, data_table[3], width=width, label=names[3])
    plt.axhline(random_2, color='gray', linestyle='dashed')
    plt.axhline(random_3, color='black', linestyle='dashed')
    title = f'rankings overlap layer {layer}'
    ax.set_title(title)
    ax.set_ylabel('overlap')
    ax.set_xticks(X)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig(Path(save_dir, title))
    plt.close()


if __name__ == '__main__':
    x = 100
    # get_all_rankings()
    # analyze_overlaps(x)
    plot_heatmap(x)