import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm as progressbar
import matplotlib.pyplot as plt

def load_res(lan, att, layer, ablation: bool, max_num=None):
    if ablation:
        dump_file = Path('results', 'UM', lan, att, 'layer ' + str(layer), 'ablation by attr', 'results.pkl')
    else:
        dump_file = Path('results', 'UM', lan, att, 'layer ' + str(layer), 'figs', str(max_num), 'results.pkl')
    if not dump_file.exists():
        raise Exception(f'{lan} {att} {layer} does not exist')
    with open(dump_file, 'rb') as f:
        res = pickle.load(f)
    return res

def create_empty_df(ablation: bool):
    root_path = Path('results','UM')
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
                    curr_res = load_res(lan, att, layer, ablation)
                    metrics = curr_res.keys()
                    all_mets.update(metrics)
                    for metric in metrics:
                        rankings = curr_res[metric].keys()
                        all_rankings.update(rankings)
                        for ranking in rankings:
                            all_abl.update({abl for abl, r in curr_res[metric][ranking]})
                else:
                    for max_num in max_nums:
                        curr_res = load_res(lan, att, layer, ablation, max_num)
                        metrics = curr_res.keys()
                        all_mets.update(metrics)
                        for metric in metrics:
                            rankings = curr_res[metric].keys()
                            all_rankings.update(rankings)
    col_mi = pd.MultiIndex.from_product([lans, all_atts, layers, all_rankings],
                                    names=['language', 'attribute', 'layer', 'ranking'])
    rows = [all_mets, all_abl] if ablation else [all_mets, max_nums]
    rows_names = ['metric', 'ablated'] if ablation else ['metric', 'max num']
    row_mi = pd.MultiIndex.from_product(rows, names=rows_names)
    df = pd.DataFrame(index=row_mi, columns=col_mi).sort_index().sort_index(axis=1)
    return df

def fill_df(df: pd.DataFrame, ablation: bool):
    root_path = Path('results', 'UM')
    layers = [2, 7, 12]
    max_nums = [] if ablation else [10, 50, 150]
    lans = {p.name for p in root_path.glob('*') if p.is_dir()}
    for lan in progressbar(lans):
        lan_path = Path(root_path, lan)
        atts = [p.name for p in lan_path.glob('*') if p.is_dir()]
        for att in atts:
            for layer in layers:
                if ablation:
                    curr_res = load_res(lan, att, layer, ablation)
                    metrics = curr_res.keys()
                    for metric in metrics:
                        rankings = curr_res[metric].keys()
                        for ranking in rankings:
                            for num_ablated, res in curr_res[metric][ranking]:
                                df[(lan, att, layer, ranking)][(metric, num_ablated)] = res
                else:
                    for max_num in max_nums:
                        curr_res = load_res(lan, att, layer, ablation, max_num)
                        metrics = curr_res.keys()
                        for metric in metrics:
                            rankings = curr_res[metric].keys()
                            for ranking in rankings:
                                df[(lan, att, layer, ranking)][(metric, max_num)] = curr_res[metric][ranking]

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
    metrics = df.index.get_level_values(0).unique()
    max_nums = [10, 50, 150]
    for metric in metrics:
        results[metric] = {}
        for max_num in max_nums:
            results[metric][max_num] = {}
            for setting in settings:
                relevant_data = df.loc[idx[metric, max_num], idx[languages, attributes, layers, [setting]]]
                results[metric][max_num][setting] = (round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
            for ranking in rankings:
                relevant_settings = [s for s in settings if s.endswith(ranking)]
                relevant_data = df.loc[idx[metric, max_num], idx[languages, attributes, layers, relevant_settings]]
                results[metric][max_num][ranking] = (round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
            for probe in probes:
                relevant_settings = [s for s in settings if s.startswith(probe)]
                relevant_data = df.loc[idx[metric, max_num], idx[languages, attributes, layers, relevant_settings]]
                results[metric][max_num][probe] = (round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
    return results

def probing_analysis(df: pd.DataFrame):
    layers = df.columns.get_level_values(2).unique()
    languages = df.columns.get_level_values(0).unique()
    attributes = df.columns.get_level_values(1).unique()
    lang_results = {}
    layer_results = {}
    att_results = {}
    for lang in languages:
        lang_results[lang] = get_setting_results(df, [lang], attributes, layers)
        print(f'{lang}:')
        print(pd.DataFrame.from_dict(lang_results[lang]['acc_auc']))
    for layer in layers:
        layer_results[layer] = get_setting_results(df, languages, attributes, [layer])
        print(f'{layer}:')
        print(pd.DataFrame.from_dict(layer_results[layer]['acc_auc']))
    for att in attributes:
        att_results[att] = get_setting_results(df, languages, att, layers)
        print(f'{att}:')
        print(pd.DataFrame.from_dict(att_results[att]['acc_auc']))
    global_results = get_setting_results(df, languages, attributes, layers)
    print('all:')
    print('acc:')
    print(pd.DataFrame.from_dict(global_results['acc_auc']))
    print('sel:')
    print(pd.DataFrame.from_dict(global_results["sel_auc"]))


if __name__ == '__main__':
    abl = False
    data = create_empty_df(abl)
    fill_df(data, abl)
    if abl:
        ablation_analysis(data)
    else:
        probing_analysis(data)
