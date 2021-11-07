from argparse import ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import itertools
import sys


def get_values_avg(root_path, att, layer):
    with open(Path(root_path, 'train_parsed.pkl'), 'rb') as g:
        parsed_train = pickle.load(g)
    values_to_ignore = set()
    ignore_path = Path(root_path, att, 'values_to_ignore.pkl')
    if ignore_path.exists():
        with open(ignore_path, 'rb') as g:
            values_to_ignore = pickle.load(g)
    relevant_vals = {word['attributes'][att] for word in parsed_train if att in word['attributes']} - values_to_ignore
    embeddings_by_val = {val: [] for val in relevant_vals}
    for word in parsed_train:
        if att in word['attributes'] and word['attributes'][att] in relevant_vals:
            embeddings_by_val[word['attributes'][att]].append(word['embedding'][layer])
    avg_embeds_with_labels = {label: np.mean(embeds, axis=0) for label, embeds in embeddings_by_val.items()}
    avg_embeds = list(avg_embeds_with_labels.values())
    return avg_embeds, avg_embeds_with_labels


def get_diff_sum(arr):
    diff = np.zeros_like(arr[0])
    for couple in itertools.combinations(arr, 2):
        diff += np.abs(couple[0] - couple[1])
    return diff


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-model', type=str)
    argparser.add_argument('-language', type=str)
    argparser.add_argument('-attribute', type=str)
    argparser.add_argument('-layer', type=int)
    args = argparser.parse_args()
    model_type = args.model
    language = args.language
    attribute = args.attribute
    layer = args.layer
    print(f'model: {model_type}')
    print(f'language: {language}')
    print(f'attribute: {attribute}')
    print(f'layer: {layer}')
    root_path = Path('pickles', 'UM', model_type, language)
    if not Path(root_path, attribute).exists():
        sys.exit('WRONG SETTING')
    values_avg, values_avg_with_labels = get_values_avg(root_path, attribute, layer)
    diff = get_diff_sum(values_avg)
    ranking = np.argsort(diff)[::-1].tolist()
    dump_dir = Path(root_path, attribute, str(layer))
    if not dump_dir.exists():
        dump_dir.mkdir()
    with open(Path(dump_dir, 'cluster_ranking.pkl'), 'wb+') as f:
        pickle.dump(ranking, f)
    with open(Path(dump_dir, 'avg_embeddings_by_label.pkl'), 'wb+') as f:
        pickle.dump(values_avg_with_labels, f)
    print('dumped')
