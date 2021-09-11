import pickle
import numpy as np
import utils
from pathlib import Path
import pandas as pd
from itertools import combinations, product
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def obtain_ranking(model_type, lan, att, layer, ranking):
    pkl_path = Path('pickles', 'UM', model_type, lan, att)
    res_path = Path('results', 'UM', model_type, lan, att)
    try:
        if ranking == 'top avg':
            label_to_idx_path = Path(pkl_path, 'label_to_idx.pkl')
            with open(label_to_idx_path, 'rb') as f:
                label_to_idx = pickle.load(f)
            num_labels = len(label_to_idx)
            linear_model_path = Path(pkl_path, 'best_model_whole_vector_layer_' + str(layer))
            neurons = utils.sort_neurons_by_avg_weights(linear_model_path.__str__(), num_labels)
        if ranking == 'bayes mi':
            res_file_dir = Path(res_path, 'layer ' + str(layer))
            bayes_res_path = Path(res_file_dir, 'bayes by bayes mi')
            neurons = utils.sort_neurons_by_bayes_mi(bayes_res_path)
        if ranking == 'top cluster':
            cluster_ranking_path = Path(pkl_path, str(layer), 'cluster_ranking.pkl')
            neurons = utils.sort_neurons_by_clusters(cluster_ranking_path)
    except:
        return None
    return neurons

def get_all_rankings(model_type, num_neurons: int = 768):
    res_root = Path('results', 'UM', model_type)
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
            if not Path('pickles', 'UM', model_type, lan, att).exists():
                continue
            all_rankings[lan][att] = dict()
            for layer in layers:
                all_rankings[lan][att][layer] = dict()
                for ranking in rankings:
                    all_rankings[lan][att][layer][ranking] = obtain_ranking(model_type, lan, att, layer, ranking)
    with open(Path('pickles', 'UM', model_type, 'all_rankings.pkl'),'wb+') as f:
        pickle.dump(all_rankings, f)

def analyze_overlaps(model_type, num_neurons:int=768):
    with open(Path('pickles', 'UM', model_type, 'all_rankings.pkl'), 'rb') as f:
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

def plot_heatmap(model_type, num_neurons):
    with open(Path('pickles', 'UM', model_type, 'all_rankings.pkl'), 'rb') as f:
        all_rankings: dict = pickle.load(f)
    top_neurons = {(rename_att(att), lan, layer, ranking): values[:num_neurons] if values else None for lan, lan_dict in all_rankings.items() for
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
    labels = []
    for layer in layers:
        rankings_overlap = {}
        for r_1, r_2 in combinations(rankings, 2):
        # for r in rankings:
            # if r == 'bayes mi':
            #     continue
            diag = []
            labels = []
            for i_1, i_2 in product(indices, repeat=2):
                # if i_1 == i_2:
                #     continue
                att_1, lan_1 = i_1.split(', ')
                att_2, lan_2 = i_2.split(', ')
                l_1 = top_neurons[(att_1, lan_1, layer, r_1)].values[:x]
                l_2 = top_neurons[(att_2, lan_2, layer, r_2)].values[:x]
                # l_1 = top_neurons[(att_1, lan_1, layer, r)].values[:x]
                # l_2 = top_neurons[(att_2, lan_2, layer, r)].values[:x]
                matrix[i_1][i_2] = len(np.intersect1d(l_1, l_2, assume_unique=True))
                if i_1 == i_2:
                    diag.append(len(np.intersect1d(l_1, l_2, assume_unique=True)))
                    labels.append(i_1)

            matrix = matrix.astype(float)
            # wo_lan = [label[:label.index(',')] for label in matrix.columns.values]
            # matrix.columns.values = wo_lan
            # matrix.index.values = wo_lan
            cmap = sns.diverging_palette(0, 255, as_cmap=True)
            divnorm = TwoSlopeNorm(vcenter=13, vmin=0, vmax=75)
            h = sns.heatmap(matrix, vmin=0, vmax=75, xticklabels=True, yticklabels=True, cmap=cmap, norm=divnorm)
            # sizes = [6, 0, 7, 0, 0, 0, 8, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 6, 0, 0, 0, 0, 10, 0, 0, 0, 0,
            #          0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 7, 0, 6, 6, 0, 0, 0, 9, 0, 0, 0, 0, 0, 7, 0]
            # for x_tick, y_tick, size in zip(h.get_xmajorticklabels(), h.get_ymajorticklabels(), sizes):
            #     x_tick.set_size(size)
            #     y_tick.set_size(size)
            h.set_xticklabels(h.get_xmajorticklabels(), fontsize=6)
            h.set_yticklabels(h.get_ymajorticklabels(), fontsize=6)
            h.set_facecolor('grey')
            title = f'layer {str(layer)} by {r_1}, {r_2}'
            # title = f'layer {str(layer)} by {r} by att'
            # plt.title(title)
            plt.tight_layout()
            save_dir = Path('results', 'overlaps', model_type)
            if not save_dir.exists():
                save_dir.mkdir()
            # plt.savefig(Path(save_dir, title))
            plt.close()
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
        above_rand = [overlap for overlap in all_three if overlap > 2]
        print(f'layer {layer} above rand: {len(above_rand)}' )
        rankings_overlap['all'] = all_three
        plot_bar(rankings_overlap, labels, 8, layer, save_dir)

def plot_bar(data: dict, settings, num_to_show, layer, save_dir):
    random_2 = 13
    random_3 = 1.7
    order = np.random.permutation(len(settings))
    order = [settings.index('POS, fra'), settings.index('Number, fin'), settings.index('Gender, hin'), settings.index('Voice, rus'),
             settings.index('Number, rus'), settings.index('POS, ara'), settings.index('POS, bul'), settings.index('Aspect, rus')]
    fig, ax = plt.subplots()
    names = list(data.keys())
    names = ['Gaussian, Linear', 'Gaussian, Cluster', 'Linear, Cluster', 'All']
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
    plt.yticks(list(plt.yticks()[0]) + [13, 1.7], list(plt.yticks()[0]) + ['rand-2', 'rand-3'])
    title = f'rankings overlap layer {layer}'
    # ax.set_title(title)
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
    # get_all_rankings('xlm')
    # analyze_overlaps(x)
    plot_heatmap('bert', x)