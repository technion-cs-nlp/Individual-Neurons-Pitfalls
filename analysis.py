import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path
import numpy as np
from scipy.interpolate import interpolate
import torch
import consts
import utils
import copy
from argparse import ArgumentParser
import ast
import pickle
import numpy as np

class plots():
    def __init__(self, dir_path:Path, max_num:int, ablation: bool = False):
        # self.colors = ['black', 'gray', 'brown', 'red', 'orange', 'purple', 'blue', 'olive', 'palegreen', 'green', 'teal',
        #                'slateblue', 'orchid']
        self.dir_path = dir_path
        self.max_num = max_num
        self.ablation = ablation
        self.save_path = Path(dir_path, 'figs')
        if not self.save_path.exists():
            self.save_path.mkdir()

    def draw_plot(self, ax, sorted_results, auc=False):
        raise NotImplementedError

    def prep_plot(self, title, results, save_file_name, xlabel, ax, to_save=False):
        results = {name:res for name,res in results.items() if res}
        if not results:
            return None, None
        # fig = plt.figure(figsize=[7.2, 4.8])
        auc = True if save_file_name == 'test accuracy' else False
        if to_save:
            if auc:
                fig = plt.figure(figsize=[9.8,5.8])
                ax = plt.subplot(111)
                fig.suptitle(title)
            else:
                plt.figure(figsize=[7.2, 4.8])
                ax = plt.subplot(111, title=title)
        sorted_results = list(results.items())
        curr_max = min(self.max_num, len(sorted_results[0][1]))
        def sort_plots_by_last_val(l):
            if curr_max <= len(l[1]):
                return l[1][curr_max-1]
            else:
                return l[1][-1]
        def sort_plots_by_name(l):
            return l[0]
        def sort_plots_by_auc(l):
            return self.test_auc[l[0]]
        # sorted_results.sort(key=sort_plots_by_name)
        if auc:
            sorted_results.sort(key=sort_plots_by_auc,reverse=True)
        else:
            sorted_results.sort(key=sort_plots_by_last_val,reverse=True)
        # find a better way for auc condition

        ax, legend = self.draw_plot(ax, sorted_results, auc=auc)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(xlabel)
        if to_save:
            box = ax.get_position()
            if auc:
                ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
                ax.legend(legend, ncol=2, loc='upper center', prop={'size': 9}, bbox_to_anchor=(0.5,1.29))
            else:
                ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
                ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
                ax.set_title(title)
            # fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.savefig(Path(self.save_path, save_file_name))
            plt.close()
        return ax, legend

class probing(plots):
    def __init__(self, dir_path:Path, names, layer:int, small:bool, max_num:int, model_type:str='all'):
        super(probing, self).__init__(dir_path=dir_path, max_num=max_num)
        if model_type != 'all':
            names = [name for name in names if name.startswith(model_type)]
        self.colors = {'bayes by bayes mi': 'black', 'linear by bayes mi': 'gray',
                       'bayes by worst mi': 'brown', 'linear by top avg': 'red',
                       'bayes by bottom avg': 'orange', 'linear by worst mi': 'purple',
                        'bayes by top avg': 'blue', 'linear by random': 'olive',
                       'linear by bottom avg': 'palegreen', 'bayes by random': 'green',
                       'bayes': 'black', 'linear': 'red', 'by bayes mi': 'black', 'by top avg': 'red',
                       'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green'}
        self.names = names
        # self.lca = 'with lca' if 'lca' in ''.join(names) else 'no lca'
        self.layer = layer
        self.small = small
        self.model_type = model_type
        self.save_path = Path(self.save_path, str(self.max_num), model_type)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
        self.language = dir_path.parts[2]
        self.attribute = dir_path.parts[3]
        self.layer_str = 'layer ' + str(self.layer)
        self.load_results()

    def load_results(self):
        self.train_acc_results = {name: [] for name in self.names if 'original' not in str(name)}
        self.train_mi_results = {name: [] for name in self.names if 'original' not in str(name)}
        self.train_nmi_results = {name: [] for name in self.names if 'original' not in str(name)}
        self.test_acc_results = {name: [] for name in self.names if 'original' not in str(name)}
        self.test_mi_results = {name: [] for name in self.names if 'original' not in str(name)}
        self.test_nmi_results = {name: [] for name in self.names if 'original' not in str(name)}
        self.test_auc = {}
        for name in self.names:
            if 'original' in name:
                continue
            with open(Path(self.dir_path, name), 'r') as f:
                for line in f.readlines():
                    if 'accuracy on train set' in line:
                        self.train_acc_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'final accuracy on test' in line:
                        self.test_acc_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'mi on train' in line:
                        self.train_mi_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'mi on test' in line:
                        self.test_mi_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'nmi on train' in line:
                        self.train_nmi_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'nmi on test' in line:
                        self.test_nmi_results[name].append(round(float(line.split()[-1]), ndigits=4))
            self.test_auc[name] = np.trapz(self.test_acc_results[name][:self.max_num], dx=1)
        control_names = [name + '_control' for name in self.names
                         if Path(self.dir_path, name + '_control').is_file()]
        self.train_controls = {name: [] for name in control_names}
        self.train_acc_for_control = copy.deepcopy(self.train_acc_results)
        self.test_controls = {name: [] for name in control_names}
        self.test_acc_for_control = copy.deepcopy(self.test_acc_results)
        for control_name in control_names:
            if 'original' in control_name:
                self.train_acc_for_control[control_name[:-8]] = \
                    self.train_acc_for_control[control_name.replace('original ','')[:-8]]
                self.test_acc_for_control[control_name[:-8]] = \
                    self.test_acc_for_control[control_name.replace('original ', '')[:-8]]
            with open(Path(self.dir_path, control_name), 'r') as f:
                for line in f.readlines():
                    if 'accuracy on train set' in line:
                        self.train_controls[control_name].append(float(line.split()[-1]))
                    if 'final accuracy on test' in line:
                        self.test_controls[control_name].append(float(line.split()[-1]))
        self.train_selectivities = {control_name[:-8]: [] for control_name in control_names}
        self.test_selectivities = {control_name[:-8]: [] for control_name in control_names}

    def draw_plot(self, ax, sorted_results, auc=False):
        legend = []
        for name, res in sorted_results:
            # TODO change 'layer==2' condition to something that makes sense
            # it's there in order to place labels only once
            if self.layer == 2:
                ax.plot(res[:self.max_num], color=self.colors[name], label=name)
            else:
                ax.plot(res[:self.max_num], color=self.colors[name])
            name_for_legend = ' '.join([name, '(AUC:{:.2f})'.format(self.test_auc[name])]) if auc else name
            legend.append(name_for_legend)
        return ax, legend

    def plot_acc_and_nmi(self, ax, to_save, metric):
        if not self.train_acc_results:
            return
        graph_types = {'accuracy': self.test_acc_results, 'nmi': self.test_nmi_results}
        results = graph_types[metric]
        # Uncomment if train results are wanted
        # for train_results, metric in zip([self.train_acc_results, self.train_nmi_results], ['accuracy', 'nmi']):
        #     title = ' '.join([self.language, self.attribute, self.layer_str]) +\
        #             ' - train ' + metric + (' (small dataset)' if self.small else '')
        #     self.prep_plot(title, train_results, 'train ' + metric, 'neurons', ax, to_save)
        # for test_results, metric in zip([self.test_acc_results, self.test_nmi_results], ['accuracy', 'nmi']):
        title = ' '.join([self.language, self.attribute, self.layer_str]) +\
                ' - test ' + metric + (' (small dataset)' if self.small else '')
        ax, legend = self.prep_plot(title, results, 'test ' + metric, 'neurons', ax, to_save)
        return ax,legend

    def plot_selectivity(self, ax, to_save):
        if not self.train_selectivities:
            return
        # Uncomment if train results are wanted
        # title = ' '.join([self.language, self.attribute, self.layer_str]) \
        #             + ' - train selectivity' + (' (small dataset)' if self.small else '')
        # for (name, res), color in zip(self.train_selectivities.items(), self.colors):
        #     for acc, cont in zip(self.train_acc_for_control[name], self.train_controls[name + '_control']):
        #         res.append(acc - cont)
        # self.prep_plot(title, self.train_selectivities, 'train selectivity', 'neurons')
        title = ' '.join([self.language, self.attribute, self.layer_str]) \
                    +' - test selectivity' + (' (small dataset)' if self.small else '')
        for name, res in self.test_selectivities.items():
            for acc, cont in zip(self.test_acc_for_control[name], self.test_controls[name + '_control']):
                res.append(acc - cont)
        ax, legend = self.prep_plot(title, self.test_selectivities, 'test selectivity', 'neurons', ax, to_save)
        return ax,legend

    def plot_avgs(self, ax, to_save, avg_type):
        if self.model_type != 'all':
            return
        avg_names = ['linear', 'bayes', 'by top avg', 'by bottom avg', 'by random', 'by bayes mi',
                      'by worst mi']
        train_acc_avgs = {name: [] for name in avg_names}
        test_acc_avgs = {name: [] for name in avg_names}
        train_sel_avgs = {name: [] for name in avg_names}
        test_sel_avgs = {name: [] for name in avg_names}
        for avg_name in avg_names:
            train_acc_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in self.train_acc_results.items()
                                if name.startswith(avg_name) or name.endswith(avg_name)]
            test_acc_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in self.test_acc_results.items()
                                if name.startswith(avg_name) or name.endswith(avg_name)]
            train_sel_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in self.train_selectivities.items()
                                     if name.startswith(avg_name) or name.endswith(avg_name)]
            test_sel_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in self.test_selectivities.items()
                                     if name.startswith(avg_name) or name.endswith(avg_name)]
            if not train_acc_relevant_results:
                continue
            min_train_acc_len = min([t.shape[0] for t in train_acc_relevant_results])
            if min_train_acc_len < self.max_num:
                train_acc_relevant_results = [t[:min_train_acc_len] for t in train_acc_relevant_results]
            train_acc_avgs[avg_name]=torch.stack(train_acc_relevant_results).mean(dim=0).tolist()
            min_test_acc_len = min([t.shape[0] for t in test_acc_relevant_results])
            if min_test_acc_len < self.max_num:
                test_acc_relevant_results = [t[:min_test_acc_len] for t in test_acc_relevant_results]
            test_acc_avgs[avg_name]=torch.stack(test_acc_relevant_results).mean(dim=0).tolist()
            min_train_sel_len = min([t.shape[0] for t in train_sel_relevant_results])
            if min_train_sel_len < self.max_num:
                train_sel_relevant_results = [t[:min_train_sel_len] for t in train_sel_relevant_results]
            train_sel_avgs[avg_name] = torch.stack(train_sel_relevant_results).mean(dim=0).tolist()
            min_test_sel_len = min([t.shape[0] for t in test_sel_relevant_results])
            if min_test_sel_len < self.max_num:
                test_sel_relevant_results = [t[:min_test_sel_len] for t in test_sel_relevant_results]
            test_sel_avgs[avg_name] = torch.stack(test_sel_relevant_results).mean(dim=0).tolist()
        class_names = ['linear', 'bayes']
        train_acc_class_avgs = {name: res for name,res in train_acc_avgs.items() if name in class_names}
        test_acc_class_avgs = {name: res for name, res in test_acc_avgs.items() if name in class_names}
        train_sel_class_avgs = {name: res for name, res in train_sel_avgs.items() if name in class_names}
        test_sel_class_avgs = {name: res for name, res in test_sel_avgs.items() if name in class_names}
        rank_names = ['by top avg', 'by bottom avg', 'by random', 'by bayes mi', 'by worst mi',
                  'by mixed k=20', 'by mixed k=40', 'by zigzag']
        train_acc_rank_avgs = {name: res for name, res in train_acc_avgs.items() if name in rank_names}
        test_acc_rank_avgs = {name: res for name, res in test_acc_avgs.items() if name in rank_names}
        train_sel_rank_avgs = {name: res for name, res in train_sel_avgs.items() if name in rank_names}
        test_sel_rank_avgs = {name: res for name, res in test_sel_avgs.items() if name in rank_names}
        train_avgs = [train_acc_class_avgs, train_acc_rank_avgs, train_sel_class_avgs, train_sel_rank_avgs]
        test_avgs = [test_acc_class_avgs, test_acc_rank_avgs, test_sel_class_avgs, test_sel_rank_avgs]
        for i, (train_res, test_res) in enumerate(zip(train_avgs, test_avgs)):
            if avg_type == 'ranking' and i==0:
                continue
            # avg_type = 'classifiers' if i==0 or i==2 else 'ranking'
            metric = 'acc' if i<2 else 'sel'
            small_str = ' (small dataset)' if self.small else ''
            title = ' '.join([self.language, self.attribute, self.layer_str, 'train',avg_type, metric,'avgs',small_str])
            file_name = ' '.join(['train',avg_type,metric,'avgs'])
            # Uncomment if train results are wanted
            # self.prep_plot(title, train_res, file_name, 'neurons')
            title = ' '.join(
                [self.language, self.attribute, self.layer_str, 'test', avg_type, metric, 'avgs', small_str])
            file_name = ' '.join(['test', avg_type, metric, 'avgs'])
            ax, legend = self.prep_plot(title, test_res, file_name, 'neurons', ax, to_save)
            return ax, legend

class ablation(plots):
    def __init__(self, dir_path, names, layer, max_num=consts.BERT_OUTPUT_DIM):
        super(ablation, self).__init__(dir_path=dir_path, max_num=max_num, ablation=True)
        self.colors = {'by bayes mi': 'black', 'by top avg': 'red',
                       'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green'}
        self.names = names
        self.layer = layer
        self.language = dir_path.parts[2]
        self.attribute = dir_path.parts[3]
        self.load_results()
        self.save_path = Path(self.save_path, str(self.max_num))
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
        self.layer_str = 'layer ' + str(self.layer)

    def load_results(self):
        self.total_accs = {name: [] for name in self.names}
        self.loss_results = {name: [] for name in self.names}
        self.relevant_accs = {name: [] for name in self.names}
        self.irrelevant_accs = {name: [] for name in self.names}
        self.lemma_preds = {name: [] for name in self.names}
        self.lemma_ranks = {name: [] for name in self.names}
        self.lemma_log_ranks = {name: [] for name in self.names}
        self.lemma_top_10 = {name: [] for name in self.names}
        self.lemma_top_100 = {name: [] for name in self.names}
        num_neurons = 0
        for name in self.names:
            with open(Path(self.dir_path, name), 'r') as f:
                for line in f.readlines():
                    if line.startswith('using') and 'neurons' in line:
                        num_neurons = int(line.split()[1])
                    if line.startswith('loss'):
                        # self.loss_results[name].insert(0, round(float(line.split()[-1]), ndigits=4))
                        self.loss_results[name].append((consts.BERT_OUTPUT_DIM - num_neurons,
                                                        round(float(line.split()[-1]), ndigits=4)))
                    if line.startswith('accuracy'):
                        # self.total_accs[name].insert(0, round(float(line.split()[-1]), ndigits=4))
                        self.total_accs[name].append((consts.BERT_OUTPUT_DIM - num_neurons,
                                                      round(float(line.split()[-1]), ndigits=4)))
                    if line.startswith('relevant words accuracy'):
                        # self.relevant_accs[name].insert(0, round(float(line.split()[-1]), ndigits=4))
                        self.relevant_accs[name].append((consts.BERT_OUTPUT_DIM - num_neurons,
                                                         round(float(line.split()[-1]), ndigits=4)))
                    if line.startswith('irrelevant words accuracy'):
                        # self.irrelevant_accs[name].insert(0, round(float(line.split()[-1]), ndigits=4))
                        self.irrelevant_accs[name].append((consts.BERT_OUTPUT_DIM - num_neurons,
                                                           round(float(line.split()[-1]), ndigits=4)))
                    if line.startswith('lemma predictions:'):
                        self.lemma_preds[name].append((consts.BERT_OUTPUT_DIM - num_neurons,
                                                      int(line.split()[-1])))
        lemma_ranks_path = Path('pickles','UM',self.language, self.attribute, str(self.layer))
        for name in self.names:
            cur_path = Path(lemma_ranks_path, 'ablation_lemmas_ranks_by_' + name[3:] + '.pkl')
            if not cur_path.exists():
                continue
            with open(cur_path,'rb') as f:
                res = pickle.load(f)
                self.lemma_ranks[name] = [(num_ablated, np.mean(r)) for num_ablated, r in res.items()]
                self.lemma_log_ranks[name] = [(num_ablated, np.mean(np.ma.filled(np.ma.log(r),0))) for num_ablated, r in res.items()]
                self.lemma_top_10[name] = [(num_ablated, (np.where(np.array(r) < 10)[0].size) / len(r)) for num_ablated, r in res.items()]
                self.lemma_top_100[name] = [(num_ablated, (np.where(np.array(r) < 100)[0].size) / len(r)) for num_ablated, r in
                                           res.items()]

    def draw_plot(self, ax, sorted_results, **kwargs):
        legend = []
        for name, res in sorted_results:
            x_axis = [r[0] for r in res]
            y_axis = [r[1] for r in res]
            try:
                max_num_idx = x_axis.index(self.max_num)
            except ValueError:
                max_num_idx = 0
            if self.layer == 2:
                ax.plot(x_axis[max_num_idx:], y_axis[max_num_idx:], color=self.colors[name], label=name)
            else:
                ax.plot(x_axis[max_num_idx:], y_axis[max_num_idx:], color=self.colors[name])
            legend.append(name)
        return ax, legend

    def plot_metric(self, ax, to_save, metric):
        graph_types = {'total accuracy': self.total_accs, 'loss': self.loss_results,
                       'ablated words accuracy': self.relevant_accs,
                       'non-ablated words accuracy': self.irrelevant_accs,
                        'lemma predictions': self.lemma_preds,
                       'avg lemma rank': self.lemma_ranks, 'avg lemma log rank': self.lemma_log_ranks,
                       'lemmas in top 10': self.lemma_top_10, 'lemmas in top 100': self.lemma_top_100}
        results = graph_types[metric]
        title = ' '.join([self.language, self.attribute, self.layer_str,'ablation ']) + metric
        ax, legend = self.prep_plot(title, results, metric, xlabel='ablated neurons', ax=ax, to_save=to_save)
        return ax, legend

class morphologyAblation(plots):
    def __init__(self, dir_path, names, layer, max_num=760):
        super(morphologyAblation, self).__init__(dir_path, max_num)
        self.colors = {'by bayes mi': 'black', 'by top avg': 'red',
                       'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green',
                        'wrong words': 'black', 'correct lemmas': 'red',
                       'kept attribute': 'orange', 'correct values': 'purple',
                       'split words': 'green'}
        self.names = names
        # bayes doesn't start from 0 ablated and we need the num of errors for 0 ablated
        assert self.names[0] == 'by bayes mi' and self.names[3] == 'by top avg'
        self.names[0], self.names[3] = 'by top avg', 'by bayes mi'
        self.language = dir_path.parts[2]
        self.attribute = dir_path.parts[3]
        self.layer = layer
        self.layer_str = 'layer ' + str(self.layer)
        self.load_results()
        self.dump_results()

    def load_results(self):
        self.wrong_word = {name: [] for name in self.names}
        self.correct_lemma = {name: [] for name in self.names}
        # self.wrong_lemma = {name: [] for name in self.names}
        self.kept_attribute = {name: [] for name in self.names}
        # self.no_attribute = {name: [] for name in self.names}
        self.correct_val = {name: [] for name in self.names}
        # self.wrong_val = {name: [] for name in self.names}
        self.split_words = {name: [] for name in self.names}

        num_ablated = 0
        for name in self.names:
            with open(Path(self.dir_path, name), 'r') as f:
                for line in f.readlines():
                    if line.startswith('num ablated'):
                        num_ablated = int(line.split()[-1])
                    if line.startswith('{'):
                        curr_stats = ast.literal_eval(line)
                        if name == 'by top avg' and num_ablated == 0:
                            self.initial_errors = curr_stats['wrong word']
                            self.initial_correct_lemma = curr_stats['correct lemma']
                            # self.initial_wrong_lemma = curr_stats['wrong lemma']
                            self.initial_no_attribute = curr_stats['no attribute']
                        self.wrong_word[name].append((num_ablated, curr_stats['wrong word'] /
                                                      (curr_stats['relevant'] - curr_stats['pred split'])))
                        # new_errors = curr_stats['wrong word'] - self.initial_errors
                        # new_correct_lemma = curr_stats['correct lemma'] - self.initial_correct_lemma
                        # new_wrong_lemma = curr_stats['wrong lemma'] - self.initial_wrong_lemma
                        # new_no_attribute = curr_stats['no attribute'] - self.initial_no_attribute
                        # if new_errors < 0 or new_no_attribute < 0 or new_correct_lemma < 0:
                        #     print('here')
                        self.correct_lemma[name].append((num_ablated,
                                                         utils.divide_zero(curr_stats['correct lemma'],
                                                                           curr_stats['wrong word'])))
                        # if utils.divide_zero(new_correct_lemma, new_errors) > 1:
                        #     print('here')
                        self.kept_attribute[name].append((num_ablated,
                                                        utils.divide_zero(curr_stats['kept attribute'],
                                                                          curr_stats['wrong word'])))
                        # self.no_attribute[name].append((num_ablated,
                        #                                 utils.divide_zero(curr_stats['no attribute'],
                        #                                                   curr_stats['wrong word'])))
                        if curr_stats['kept attribute'] != 0:
                            self.correct_val[name].append((num_ablated, curr_stats['correct val'] /
                                                           curr_stats['kept attribute']))
                        self.split_words[name].append((num_ablated, curr_stats['pred split'] /
                                                       curr_stats['relevant']))

    def dump_results(self):
        wrong_words_path = Path(self.dir_path, 'wrong words')
        correct_lemmas_path = Path(self.dir_path, 'correct lemmas')
        kept_att_path = Path(self.dir_path, 'kept attribute')
        correct_val_path = Path(self.dir_path, 'correct val')
        split_words_path = Path(self.dir_path, 'split words')
        paths = [wrong_words_path, correct_lemmas_path, kept_att_path, correct_val_path, split_words_path]
        for p in paths:
            if not p.exists():
                p.mkdir()
        for p, rankings_results in zip(paths,[self.wrong_word, self.correct_lemma, self.kept_attribute,
                                 self.correct_val, self.split_words]):
            for name, res in rankings_results.items():
                with open(Path(p,name),'wb+') as f:
                    pickle.dump(res,f)

    def plot_metric(self, ax, to_save, metric):
        graph_types = {'wrong words': self.wrong_word, 'correct lemmas': self.correct_lemma,
                       'kept attribute': self.kept_attribute,'correct values': self.correct_val,
                       'split words': self.split_words}
        results = graph_types[metric]
        title = ' '.join([self.language, self.attribute, self.layer_str, 'ablation ']) + metric
        ax, legend = self.prep_plot(title, results, metric, xlabel='ablated neurons', ax=ax, to_save=to_save)
        return ax, legend

    def plot_ranking(self, ax, to_save, ranking):
        all_results = {'wrong words': self.wrong_word, 'correct lemmas': self.correct_lemma,
                       'kept attribute': self.kept_attribute, 'correct values': self.correct_val,
                       'split words': self.split_words}
        results = {metric: r[ranking] for metric, r in all_results.items()}
        title = ' '.join([self.language, self.attribute, self.layer_str, 'ablation ']) + ranking
        ax, legend = self.prep_plot(title, results, ranking, xlabel='ablated neurons', ax=ax, to_save=to_save)
        return ax, legend

    def draw_plot(self, ax, sorted_results, **kwargs):
        legend = []
        for name, res in sorted_results:
            x_axis = [r[0] for r in res]
            y_axis = [r[1] for r in res]
            try:
                max_num_idx = x_axis.index(self.max_num)
            except ValueError:
                max_num_idx = 0
            # TODO this if-else is only for drawing together, otherwise should not give labels?
            if self.layer == 2:
                ax.plot(x_axis[max_num_idx:], y_axis[max_num_idx:], color=self.colors[name], label=name)
            else:
                ax.plot(x_axis[max_num_idx:], y_axis[max_num_idx:], color=self.colors[name])
            legend.append(name)
        return ax, legend


def run_all_probing(dir_path, plot_separate):
    axs = [0] * 3
    small_dataset = False
    max_nums = [50, 150] if plot_separate else [150]
    # model_types = ['all','linear','bayes'] if plot_separate else ['all']
    model_types = ['all']
    for metric in ['acc','nmi','selectivity','ranking avg', 'classifiers avg']:
        if not plot_separate:
            fig, axs = plt.subplots(3, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['probing',dir_path.parts[-2], dir_path.parts[-1], metric, 'per layer']))
            legend=None
        for i, layer in enumerate([2, 7, 12]):
            for max_num in max_nums:
                for model_type in model_types:
                    layer_dir = Path(dir_path, 'layer '+str(layer))
                    res_files_names = [f.parts[-1] for f in layer_dir.glob('*') if
                                 f.is_file() and not f.parts[-1].startswith('whole')
                                       and not f.parts[-1].endswith('control')]
                    if not res_files_names:
                        continue
                    def plot_metric(plotting: probing, metric):
                        if metric == 'acc':
                            return plotting.plot_acc_and_nmi(axs[i], plot_separate, 'accuracy')
                        if metric == 'nmi':
                            return plotting.plot_acc_and_nmi(axs[i], plot_separate, 'nmi')
                        if metric == 'selectivity':
                            return plotting.plot_selectivity(axs[i], plot_separate)
                        if metric == 'ranking avg':
                            return plotting.plot_avgs(axs[i], plot_separate, 'ranking')
                        if metric == 'classifiers avg':
                            return plotting.plot_avgs(axs[i], plot_separate, 'classifiers')
                    plotting = probing(layer_dir,res_files_names,layer,small_dataset,max_num,model_type)
                    res = plot_metric(plotting, metric)
                    if not plot_separate:
                        axs[i], legend = res
                        axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.savefig(Path(dir_path, ' '.join(['probing', metric, 'by layers'])))

def run_ablation(dir_path, plot_separate):
    axs = [0]*3
    # for metric in ['total accuracy', 'loss', 'ablated words accuracy', 'non-ablated words accuracy',
    #                'lemma predictions']:
    for metric in ['avg lemma rank', 'avg lemma log rank', 'lemmas in top 10', 'lemmas in top 100']:
        if not plot_separate:
            fig, axs = plt.subplots(3, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['ablation',dir_path.parts[-2], dir_path.parts[-1], metric, 'per layer']))
            legend=None
        for i,layer in enumerate([2, 7, 12]):
            max_nums = [0,400,600] if plot_separate else [0]
            for max_num in max_nums:
                ablation_root_path = Path(dir_path, 'layer '+str(layer), 'ablation by attr')
                if not ablation_root_path.exists():
                    continue
                res_files_names = [f.parts[-1] for f in ablation_root_path.glob('*') if
                                   f.is_file()]
                ab = ablation(dir_path=ablation_root_path, names=res_files_names, layer=layer, max_num=max_num)
                axs[i], legend=ab.plot_metric(axs[i], plot_separate,metric)
                if not plot_separate:
                    axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            plt.savefig(Path(dir_path, ' '.join(['ablation', metric, 'by layers'])))

def run_morph(dir_path, plot_separate, all_rankings):
    num_subplots = 3
    axs = [0] * num_subplots
    iter_list = ['wrong words', 'correct lemmas', 'kept attribute', 'correct values', 'split words'] if all_rankings \
        else ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi', 'by random']
    for name in iter_list:
        if not plot_separate:
            fig, axs = plt.subplots(num_subplots, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['ablation', dir_path.parts[-2], dir_path.parts[-1], name, 'per layer']))
            legend = None
        for i, layer in enumerate([2, 7, 12]):
            max_nums = [0, 400, 600] if plot_separate else [0]
            for max_num in max_nums:
                # for max_num in [0]:
                spacy_root_path = Path(dir_path, 'layer ' + str(layer), 'spacy')
                if not spacy_root_path.exists():
                    continue
                res_files_names = [f.parts[-1] for f in spacy_root_path.glob('*') if
                                   f.is_file()]
                ma = morphologyAblation(dir_path=spacy_root_path, names=res_files_names, layer=layer, max_num=max_num)
                axs[i], legend = ma.plot_metric(axs[i], plot_separate, name) if all_rankings \
                    else ma.plot_ranking(axs[i], plot_separate, name)
                if not plot_separate:
                    axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            if not Path(dir_path,'spacy figs').exists():
                Path(dir_path,'spacy figs').mkdir()
            plt.savefig(Path(dir_path, 'spacy figs', ' '.join(['ablation', name, 'by layers'])))


if __name__ == "__main__":
    data_name = 'UM'
    languages = ['rus']
    for lan in languages:
        root_path = Path('results',data_name,lan)
        atts_path = [p for p in root_path.glob('*') if not p.is_file()]
        for att_path in atts_path:
            # run_all_probing(att_path, plot_separate=True)
            # run_ablation(att_path, plot_separate=False)
            run_morph(att_path, plot_separate=False, all_rankings=False)
