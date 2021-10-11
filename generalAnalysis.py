import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm as progressbar
import matplotlib.pyplot as plt
import itertools
from scipy.stats import wilcoxon
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

order = ['bayes by bayes mi', 'linear by bayes mi', 'bayes by top avg', 'linear by top avg',
         'bayes by top cluster', 'linear by top cluster', 'bayes by random', 'linear by random',
         'bayes by worst mi', 'linear by worst mi', 'bayes by bottom avg', 'linear by bottom avg',
         'bayes by bottom cluster', 'linear by bottom cluster']

def load_res(model_type, lan, att, layer, ablation: bool, max_num=None):
    if ablation:
        dump_file = Path('results', 'UM', model_type, lan, att, 'layer ' + str(layer), 'ablation by attr', 'results.pkl')
    else:
        dump_file = Path('results', 'UM', model_type, lan, att, 'layer ' + str(layer), 'test_acc_results.pkl')
    if not dump_file.exists():
        raise Exception(f'{lan} {att} {layer} does not exist')
    with open(dump_file, 'rb') as f:
        res = pickle.load(f)
    return res

def create_empty_df(model_type, ablation: bool):
    root_path = Path('results','UM', model_type)
    lans = {p.name for p in root_path.glob('*') if p.is_dir()}
    layers = [2, 7, 12]
    max_nums = [] if ablation else [10, 50, 150]
    all_atts, all_mets, all_rankings, all_abl = set(), set(), set(), set()
    for lan in lans:
        lan_path = Path(root_path, lan)
        atts = [p.name for p in lan_path.glob('*') if p.is_dir()]
        all_atts.update(atts)
        for att in atts:
            for layer in layers:
                if ablation:
                    curr_res = load_res(model_type, lan, att, layer, ablation)
                    metrics = curr_res.keys()
                    all_mets.update(metrics)
                    for metric in metrics:
                        rankings = curr_res[metric].keys()
                        all_rankings.update(rankings)
                        for ranking in rankings:
                            all_abl.update({abl for abl, r in curr_res[metric][ranking]})
                else:
                    curr_res = load_res(model_type, lan, att, layer, ablation, max_nums[0])
                    rankings = curr_res.keys()
                    all_rankings.update(rankings)
    col_mi = pd.MultiIndex.from_product([lans, all_atts, layers, all_rankings],
                                    names=['language', 'attribute', 'layer', 'ranking'])
    rows = [all_mets, all_abl] if ablation else max_nums
    rows_names = ['metric', 'ablated'] if ablation else 'max num'
    row_mi = pd.MultiIndex.from_product(rows, names=rows_names) if ablation else max_nums
    df = pd.DataFrame(index=row_mi, columns=col_mi).sort_index().sort_index(axis=1)
    return df

def fill_df(model_type, df: pd.DataFrame, ablation: bool):
    root_path = Path('results', 'UM', model_type)
    layers = [2, 7, 12]
    max_nums = [] if ablation else [10, 50, 150]
    lans = {p.name for p in root_path.glob('*') if p.is_dir()}
    for lan in progressbar(lans):
        lan_path = Path(root_path, lan)
        atts = [p.name for p in lan_path.glob('*') if p.is_dir()]
        for att in atts:
            for layer in layers:
                if ablation:
                    curr_res = load_res(model_type, lan, att, layer, ablation)
                    metrics = curr_res.keys()
                    for metric in metrics:
                        rankings = curr_res[metric].keys()
                        for ranking in rankings:
                            for num_ablated, res in curr_res[metric][ranking]:
                                df[(lan, att, layer, ranking)][(metric, num_ablated)] = res
                else:
                    for max_num in max_nums:
                        curr_res = load_res(model_type, lan, att, layer, ablation, max_num)
                        rankings = curr_res.keys()
                        for ranking in rankings:
                            df[(lan, att, layer, ranking)][max_num] = curr_res[ranking][max_num - 1]

def ablation_analysis(df:pd.DataFrame):
    idx = pd.IndexSlice
    rankings = df.columns.get_level_values(3).unique()
    metrics = df.index.get_level_values(0).unique()
    results = {}
    for metric in metrics:
        results[metric] = {}
        for ranking in rankings:
            results[metric][ranking] = df.loc[idx[metric, :], idx[:, :, :, [ranking]]].mean(axis=1)
            results[metric][ranking].plot(xlabel='num ablated', ylabel=metric, title=ranking)
            plt.show()
    print(results)

def get_setting_results(df: pd.DataFrame, languages, attributes, layers):
    results = {}
    idx = pd.IndexSlice
    settings = df.columns.get_level_values(3).unique()
    rankings = set([s[s.index('by'):] for s in settings])
    probes = set([s[:s.index(' by')] for s in settings])
    # metrics = df.index.get_level_values(0).unique()
    max_nums = [10, 50, 150]
    for max_num in max_nums:
        results[max_num] = {}
        for first, second in itertools.product(settings, repeat=2):
            first_res = df.loc[idx[max_num], idx[languages, attributes, layers, first]].dropna()
            second_res = df.loc[idx[max_num], idx[languages, attributes, layers, second]].dropna()
            if first_res.size != second_res.size:
                raise Exception
            if np.count_nonzero(first_res.values - second_res.values) == 0:
                continue
            res = wilcoxon(first_res, second_res, alternative='less')
            if first not in results[max_num].keys():
                results[max_num][first] = {}
            results[max_num][first][second] = round(res[1], 2)
        # for setting in settings:
        #     relevant_data = df.loc[idx[max_num], idx[languages, attributes, layers, [setting]]]
        #     results[max_num][setting] = (round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
        # for ranking in rankings:
        #     relevant_settings = [s for s in settings if s.endswith(ranking)]
        #     relevant_data = df.loc[idx[max_num], idx[languages, attributes, layers, relevant_settings]]
        #     results[max_num][ranking] = (round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
        # for probe in probes:
        #     relevant_settings = [s for s in settings if s.startswith(probe)]
        #     relevant_data = df.loc[idx[max_num], idx[languages, attributes, layers, relevant_settings]]
        #     results[max_num][probe] = (round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
    return results

def probing_analysis(df: pd.DataFrame):
    layers = df.columns.get_level_values(2).unique()
    languages = df.columns.get_level_values(0).unique()
    attributes = df.columns.get_level_values(1).unique()
    lang_results = {}
    layer_results = {}
    att_results = {}
    idx = pd.IndexSlice
    for lang in languages:
        lang_results[lang] = get_setting_results(df, [lang], attributes, layers)
        print(f'{lang}:')
        for num_neurons in [10, 50, 150]:
            curr_res = pd.DataFrame.from_dict(lang_results[lang][num_neurons]).reindex(
                order).reindex(columns=order)
            # curr_res = pd.DataFrame.from_dict(lang_results[lang][num_neurons]).sort_index().sort_index(axis=1)
            samples = df.loc[idx[num_neurons], idx[lang, attributes, layers, 'linear by top avg']].count().sum()
            plot_heatmap(curr_res, f'{lang}_{num_neurons}_neurons', f'samples per ranking: {samples}', Path('UM', lang))
    for layer in layers:
        layer_results[layer] = get_setting_results(df, languages, attributes, [layer])
        print(f'{layer}:')
        for num_neurons in [10, 50, 150]:
            curr_res = pd.DataFrame.from_dict(layer_results[layer][num_neurons]).reindex(
                order).reindex(columns=order)
            # curr_res = pd.DataFrame.from_dict(layer_results[layer][num_neurons]).sort_index().sort_index(axis=1)
            samples = df.loc[idx[num_neurons], idx[languages, attributes, layer, 'linear by top avg']].count().sum()
            plot_heatmap(curr_res, f'layer_{layer}_{num_neurons}_neurons', f'samples per ranking: {samples}',
                         Path('wilcoxon', f'layer {layer}'))
    for att in attributes:
        att_results[att] = get_setting_results(df, languages, att, layers)
        print(f'{att}:')
        for num_neurons in [10, 50, 150]:
            curr_res = pd.DataFrame.from_dict(att_results[att][num_neurons]).reindex(
                order).reindex(columns=order)
            # curr_res = pd.DataFrame.from_dict(att_results[att][num_neurons]).sort_index().sort_index(axis=1)
            samples = df.loc[idx[num_neurons], idx[languages, att, layers, 'linear by top avg']].count().sum()
            plot_heatmap(curr_res, f'{att}_{num_neurons}_neurons', f'samples per ranking: {samples}',
                         Path('wilcoxon', att))
    global_results = get_setting_results(df, languages, attributes, layers)
    print('all:')
    for num_neurons in [10, 50, 150]:
        curr_res = pd.DataFrame.from_dict(global_results[num_neurons]).reindex(order).reindex(
            columns=order)
        # curr_res = pd.DataFrame.from_dict(global_results[num_neurons]).sort_index().sort_index(axis=1)
        samples = df.loc[idx[num_neurons], idx[languages, attributes, layers, 'linear by top avg']].count().sum()
        plot_heatmap(curr_res, f'global_{num_neurons}_neurons', f'samples per ranking: {samples}',
                     Path('wilcoxon'))


def plot_heatmap(data: pd.DataFrame, title: str, subtitle:str, save_path: Path):
    h = sns.heatmap(data, annot=True, xticklabels=True, yticklabels=True, cmap='Blues')
    h.set_facecolor('grey')
    h.set_xticklabels(h.get_xmajorticklabels(), fontsize=8)
    h.set_yticklabels(h.get_ymajorticklabels(), fontsize=8)
    plt.suptitle(title)
    plt.title(subtitle)
    h.figure.tight_layout()
    res_root_path = Path('results')
    if not Path(res_root_path, save_path).exists():
        Path(res_root_path, save_path).mkdir()
    plt.savefig(Path(res_root_path, save_path, title+'_wilcoxon_matrix.png'))
    plt.close()
    # plt.show()

def cluster_results(model_type):
    all_res = {}
    root_path = Path('results', 'UM', model_type)
    for lan in [f.name for f in root_path.glob('*') if f.is_dir()]:
        lan_path = Path(root_path, lan)
        for att in [f.name for f in lan_path.glob('*') if f.is_dir()]:
            att_path = Path(lan_path, att)
            for layer in [f.name for f in att_path.glob('*') if f.is_dir() and f.name.startswith('layer')]:
                layer_path = Path(att_path, layer)
                with open(Path(layer_path, 'test_acc_results.pkl'),'rb') as f:
                    curr_res = pickle.load(f)
                settings = list(curr_res.keys())
                settings.sort()
                all_res[f'{lan} {att} {layer[len("layer "):]}'] = [curr_res[s][:150] for s in settings if 'bottom' not in s and 'worst' not in s]
    data_matrix = np.array(list(all_res.values())).reshape([len(all_res), len(all_res['ara Aspect 2']) * len(all_res['ara Aspect 2'][0])])
    data_matrix = normalize(data_matrix)
    # data_matrix = np.array(list(all_res.values()))
    cluster_centers = data_matrix[[list(all_res.keys()).index('bul Definiteness 7'),
                                  list(all_res.keys()).index('hin Part of Speech 12'),
                                  list(all_res.keys()).index('rus Animacy 2')]]
    labels = KMeans(n_clusters=3, init=cluster_centers).fit_predict(data_matrix)
    transformed_data = PCA(n_components=50).fit_transform(data_matrix)
    transformed_data = TSNE(n_components=2, init='pca').fit_transform(transformed_data)
    bins = np.bincount(labels)
    label_names = ['Standard', 'G>L', 'L>G']
    plt.figure(figsize=[6.4, 4.0])
    for i in range(3):
        plt.scatter(transformed_data[labels == i, 0], transformed_data[labels == i, 1], label=label_names[i])
    names = ['' for n in list(all_res.keys())]
    for i, n in enumerate(list(all_res.keys())):
        for j, w in enumerate(n.split()):
            if j == 0 or w.startswith('1'):
                chars = w[:2]
            elif w == 'Polarity':
                chars = 'Pl'
            elif w == 'Possesion':
                chars = 'Ps'
            elif w == 'Person':
                chars = 'Pe'
            elif w == 'Animacy':
                chars = 'An'
            elif w == 'Aspect':
                chars = 'As'
            elif w == 'and' or w == 'Noun' or w == 'Class':
                chars = ''
            else:
                chars = w[0]
            names[i] += chars
        plt.annotate(names[i], transformed_data[i], size=6)
    plt.legend(fontsize=14)
    plt.yticks([], [])
    plt.xticks([], [])
    plt.tight_layout()
    plt.savefig(Path('results', 'UM', model_type, 'clusters'))
    plt.close()
    # initial_centers = kmeans_plusplus_initializer(transformed_data, 2).initialize()
    # xmeans_instance = xmeans(transformed_data, initial_centers)
    # xmeans_instance.process()
    # clusters = xmeans_instance.get_clusters()
    # centers = xmeans_instance.get_centers()
    # visualizer = cluster_visualizer()
    # visualizer.append_clusters(clusters, transformed_data)
    # visualizer.append_cluster(centers, None, marker='*', markersize=10)
    # visualizer.show()

if __name__ == '__main__':
    abl = False
    model_type = 'xlm'
    # data = create_empty_df(model_type, abl)
    # fill_df(model_type, data, abl)
    # if abl:
    #     ablation_analysis(data)
    # else:
    #     probing_analysis(data)
    cluster_results(model_type)
