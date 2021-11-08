from train_and_test import train, test
from dataHandler import UMDataHandler
import consts
import torch
from torch.utils.data.dataloader import DataLoader
import logging
from pathlib import Path
from argparse import ArgumentParser
import sys
import pickle

logging.basicConfig(level=logging.INFO)


def print_statistics(data_loader: DataLoader):
    labels_list = [example[1] for example in data_loader.dataset]
    labels_num = len(set(labels_list))
    print('num of labels: {}'.format(labels_num))
    hist = torch.histc(torch.tensor(labels_list, dtype=float), labels_num)
    print('majority: {}'.format(max(hist).item() / len(labels_list)))


if __name__ == "__main__":
    torch.manual_seed(consts.SEED)
    parser = ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-language', type=str)
    parser.add_argument('-attribute', type=str)
    parser.add_argument('-layer', type=int)
    parser.add_argument('--control', default=False, action='store_true')
    args = parser.parse_args()
    min_count = 100
    model_type = args.model
    language = args.language
    attribute = args.attribute
    layer = args.layer
    data_name = 'UM'
    control = args.control
    control_str = '_control' if control else ''
    save_path = Path('pickles', data_name, model_type, language, args.attribute)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    file_name = f'best_model_whole_vector_layer_{str(layer)}{control_str}'
    save_path = Path(save_path, file_name)
    train_path = Path('pickles', data_name, model_type, language, 'train_parsed.pkl')
    dev_path = Path('pickles', data_name, model_type, language, 'dev_parsed.pkl')
    test_path = Path('pickles', data_name, model_type, language, 'test_parsed.pkl')
    res_file_dir = Path('results', data_name, model_type, language, args.attribute, 'layer ' + str(layer))
    if not res_file_dir.exists():
        res_file_dir.mkdir(parents=True, exist_ok=True)
    res_file_name = 'whole vector' + control_str
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
        sys.stdout = f
        print('data: ', data_name)
        print('model:', model_type)
        print('language:', language)
        print('attribute: ', attribute)
        print('layer: ', layer)
        print('control: ', control)
        data_model = UMDataHandler
        print('creating dataset')
        train_data_handler = data_model(train_path, data_name, model_type=model_type, layer=layer, control=control,
                                        language=language, attribute=attribute)
        dev_data_handler = data_model(dev_path, data_name, model_type=model_type, layer=layer, control=control,
                                      language=language, attribute=attribute)
        test_data_handler = data_model(test_path, data_name, model_type=model_type, layer=layer, control=control,
                                       language=language, attribute=attribute)
        values_to_ignore = set()
        for data_set in [train_data_handler, dev_data_handler, test_data_handler]:
            data_set.create_dicts()
            # data_set.get_features()
            histogram = data_set.count_values_for_att()
            for value, words in histogram.items():
                if len(words) < min_count:
                    values_to_ignore.add(value)
        with open(Path('pickles', data_name, model_type, language, attribute, 'values_to_ignore.pkl'), 'wb+') as g:
            pickle.dump(values_to_ignore, g)
        print('ignoring labels: {}'.format(values_to_ignore))
        print('creating train and test datasets')
        train_data_loader = DataLoader(train_data_handler.create_dataset(), batch_size=consts.BATCH_SIZE)
        dev_data_loader = DataLoader(dev_data_handler.create_dataset(), batch_size=consts.BATCH_SIZE)
        test_data_loader = DataLoader(test_data_handler.create_dataset(), batch_size=consts.BATCH_SIZE)
        print('training')
        print_statistics(test_data_loader)
        classifier = train(train_data_loader, lambda1=0.001, lambda2=0.01, model_name='wholeVector',
                           save_path=save_path)
        test(classifier, test_data_loader)
