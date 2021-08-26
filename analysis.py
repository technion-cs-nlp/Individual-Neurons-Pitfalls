import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import timedelta
from pathlib import Path
import torch
import consts
from utils import divide_zero
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

    def prep_plot(self, title, results, save_file_name, xlabel, ylabel, ax, to_save=False):
        results = {name:res for name,res in results.items() if res}
        if not results:
            return None, None
        # fig = plt.figure(figsize=[7.2, 4.8])
        # auc = True if save_file_name == 'test accuracy' else False
        auc = False
        if to_save:
            if auc:
                fig = plt.figure(figsize=[9.8,5.8])
                ax = plt.subplot(111)
                fig.suptitle(title)
            else:
                plt.figure(figsize=[6.4, 4.8])
                # plt.figure()
                # ax = plt.subplot(111, title=title)
                ax = plt.subplot(111)
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
            return self.test_acc_auc[l[0]]
        # sorted_results.sort(key=sort_plots_by_name)
        if auc:
            sorted_results.sort(key=sort_plots_by_auc,reverse=True)
        else:
            sorted_results.sort(key=sort_plots_by_last_val,reverse=True)
        # find a better way for auc condition

        ax, legend_labels, legend_lines = self.draw_plot(ax, sorted_results, auc=auc)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        # ax.yticks(fontsize=10)
        if to_save:
            box = ax.get_position()
            if auc:
                ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
                ax.legend(legend_labels, ncol=3, loc='upper center', prop={'size': 9}, bbox_to_anchor=(0.5,1.29))
            else:
                ax.set_position([box.x0, box.y0, box.width, box.height])
                # ax.legend(legend_lines, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
                # figlegend = plt.figure(figsize=[12., 4.8])
                # figlegend.legend(legend_lines, legend_labels, ncol=4, loc='center', fontsize=12)
                # plt.tight_layout()
                # figlegend.show()
                # plt.savefig(Path(self.save_path, 'only_legend.png', bbox_inches='tight'))
                # ax.set_title(title)
            # fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.tight_layout()
            plt.savefig(Path(self.save_path, save_file_name))
            # plt.show()
            plt.close()
        return ax, legend_labels

class probing(plots):
    def __init__(self, dir_path:Path, names, layer:int, small:bool, max_num:int, model_type:str='all'):
        super(probing, self).__init__(dir_path=dir_path, max_num=max_num)
        if model_type != 'all':
            names = [name for name in names if name.startswith(model_type)]
        # self.colors = {'bayes by bayes mi': 'black', 'linear by bayes mi': 'gray',
        #                'bayes by worst mi': 'brown', 'linear by top avg': 'red',
        #                'bayes by bottom avg': 'orange', 'linear by worst mi': 'purple',
        #                 'bayes by top avg': 'blue', 'linear by random': 'olive',
        #                'linear by bottom avg': 'palegreen', 'bayes by random': 'green',
        #                'linear by top cluster': 'aquamarine', 'linear by bottom cluster': 'skyblue',
        #                'bayes by top cluster': 'khaki', 'bayes by bottom cluster': 'thistle',
        #                'bayes': 'black', 'linear': 'red', 'by bayes mi': 'black', 'by top avg': 'red',
        #                'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green',
        #                'by top cluster': 'aquamarine', 'by bottom cluster': 'lightslategray'}
        self.colors = {'bayes by bayes mi': 'black', 'linear by bayes mi': 'gray',
                       'bayes by worst mi': 'black', 'linear by top avg': 'red',
                       'bayes by bottom avg': 'blue', 'linear by worst mi': 'gray',
                       'bayes by top avg': 'blue', 'linear by random': 'olive',
                       'linear by bottom avg': 'red', 'bayes by random': 'green',
                       'linear by top cluster': 'aquamarine', 'linear by bottom cluster': 'aquamarine',
                       'bayes by top cluster': 'khaki', 'bayes by bottom cluster': 'khaki',
                       'bayes': 'black', 'linear': 'red', 'by bayes mi': 'black', 'by top avg': 'red',
                       'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green',
                       'by top cluster': 'aquamarine', 'by bottom cluster': 'lightslategray'}
        colors_cmap = plt.get_cmap('Paired').colors
        colors_cmap = colors_cmap[1::2] + ('black', 'gray')
        settings = ['bayes by bayes mi', 'linear by bayes mi', 'bayes by top avg', 'linear by top avg',
                    'bayes by top cluster', 'linear by top cluster', 'bayes by random', 'linear by random',
                    'bayes by worst mi', 'linear by worst mi', 'bayes by bottom avg', 'linear by bottom avg',
                    'bayes by bottom cluster', 'linear by bottom cluster']
        self.colors = {k: v for k, v in zip(settings, colors_cmap+colors_cmap)}
        self.linestyles = {k: 'dotted' if 'bottom' in k or 'worst' in k else 'dashed' if 'random' in k else 'solid'
                           for k in self.colors.keys()}
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
        self.test_acc_auc = {}
        self.test_sel_auc = {}
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
            self.test_acc_auc[name] = np.trapz(self.test_acc_results[name][:self.max_num], dx=1)
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

    def dump_results(self):
        dump_path = Path(self.dir_path, 'test_acc_results.pkl')
        res_to_dump = self.test_acc_results
        with open(dump_path,'wb+') as f:
            pickle.dump(res_to_dump,f)

    def draw_plot(self, ax, sorted_results, auc=False):
        legend_labels, legend_lines = [], []
        for name, res in sorted_results:
            # TODO change 'layer==2' condition to something that makes sense
            # it's there in order to place labels only once
            if self.layer == 2:
                # line = Line2D(list(range(self.max_num)), res[:self.max_num], color=self.colors[name], label=name, linestyle=self.linestyles[name])
                line, = ax.plot(res[:self.max_num], color=self.colors[name], label=name, linestyle=self.linestyles[name])
            else:
                # line = Line2D(list(range(self.max_num)), res[:self.max_num], color=self.colors[name], linestyle=self.linestyles[name])
                line, = ax.plot(res[:self.max_num], color=self.colors[name], linestyle=self.linestyles[name])
            # name_for_legend = ' '.join([name, '(AUC:{:.2f})'.format(self.test_acc_auc[name])]) if auc else name
            name_for_legend = ' '.join(name.split()[:2]+name.split()[-1:])
            if not ('bottom' in name or 'worst' in name):
                legend_labels.append(name_for_legend)
                legend_lines.append(line)
        return ax, legend_labels, legend_lines

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
        paper_str = '_paper'
        title = ' '.join([self.language, self.attribute, self.layer_str]) +\
                ' - test ' + metric + paper_str
        ax, legend = self.prep_plot(title, results, 'test ' + metric + paper_str, 'neurons', metric, ax, to_save)
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
            self.test_sel_auc[name] = np.trapz(self.test_selectivities[name][:self.max_num], dx=1)
        paper_str = '_paper'
        ax, legend = self.prep_plot(title, self.test_selectivities, 'test selectivity'+paper_str, 'neurons', 'selectivity', ax, to_save)
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
            ax, legend = self.prep_plot(title, test_res, file_name, 'neurons', metric, ax, to_save)
            return ax, legend

class ablation(plots):
    def __init__(self, dir_path, names, layer, max_num=consts.BERT_OUTPUT_DIM):
        super(ablation, self).__init__(dir_path=dir_path, max_num=max_num, ablation=True)
        self.colors = {'by bayes mi': 'black', 'by top avg': 'red',
                       'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green',
                       'by top cluster': 'aquamarine', 'by bottom cluster': 'lightslategray'}
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
            cur_path = Path(lemma_ranks_path, 'ablation_lemmas_ranks_by_' + name[len('sparsed by '):] + '.pkl')
            if not cur_path.exists():
                continue
            with open(cur_path,'rb') as f:
                res = pickle.load(f)
                self.lemma_ranks[name] = [(num_ablated, np.mean(r)) for num_ablated, r in res.items()]
                self.lemma_log_ranks[name] = [(num_ablated, np.mean(np.ma.filled(np.ma.log(r),0))) for num_ablated, r in res.items()]
                self.lemma_top_10[name] = [(num_ablated, (np.where(np.array(r) < 10)[0].size) / len(r)) for num_ablated, r in res.items()]
                self.lemma_top_100[name] = [(num_ablated, (np.where(np.array(r) < 100)[0].size) / len(r)) for num_ablated, r in
                                           res.items()]

    def dump_results(self):
        dump_path = Path(self.dir_path, 'results.pkl')
        res_to_dump = {'lemma_log_rank': self.lemma_log_ranks,
                       'lemma_top_10': self.lemma_top_10,
                       'relevant acc': self.relevant_accs}
        with open(dump_path,'wb+') as f:
            pickle.dump(res_to_dump,f)


    def draw_plot(self, ax, sorted_results, **kwargs):
        legend = []
        for name, res in sorted_results:
            x_axis = [r[0] for r in res]
            y_axis = [r[1] for r in res]
            try:
                max_num_idx = x_axis.index(self.max_num)
            except ValueError:
                max_num_idx = 0
            prefix = len('sparsed ')
            if self.layer == 2:
                ax.plot(x_axis[max_num_idx:], y_axis[max_num_idx:], color=self.colors[name[prefix:]],
                        label=name[prefix:] if name.startswith('sparsed') else name)
            else:
                ax.plot(x_axis[max_num_idx:], y_axis[max_num_idx:], color=self.colors[name[prefix:]])
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
        ax, legend = self.prep_plot(title, results, metric, xlabel='ablated neurons', ylabel=metric, ax=ax, to_save=to_save)
        return ax, legend

class morphologyAblation(plots):
    def __init__(self, dir_path, names, layer, max_num=760, only_dump=False):
        super(morphologyAblation, self).__init__(dir_path, max_num)
        self.colors = {'by bayes mi': 'black', 'by top avg': 'red',
                       'by bottom avg': 'orange', 'by worst mi': 'purple', 'by random': 'green',
                        'wrong words': 'black', 'correct lemmas': 'red',
                       'kept attribute': 'orange', 'correct values': 'purple',
                       'split words': 'green'}
        self.names = names
        # bayes doesn't start from 0 ablated and we need the num of errors for 0 ablated
        if not only_dump:
            assert self.names[0] == 'by bayes mi' and self.names[4] == 'by top avg'
            self.names[0], self.names[4] = 'by top avg', 'by bayes mi'
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
        self.correct_lemma_correct_val = {name: [] for name in self.names}
        self.correct_lemma_wrong_val = {name: [] for name in self.names}
        self.wrong_lemma_correct_val = {name: [] for name in self.names}
        self.wrong_lemma_wrong_val = {name: [] for name in self.names}
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
                        # self.wrong_word[name].append((num_ablated, curr_stats['wrong word'] /
                        #                               (curr_stats['relevant'] - curr_stats['pred split'])))
                        self.wrong_word[name].append((num_ablated, curr_stats['wrong word'] /
                                                      curr_stats['relevant']))
                        # new_errors = curr_stats['wrong word'] - self.initial_errors
                        # new_correct_lemma = curr_stats['correct lemma'] - self.initial_correct_lemma
                        # new_wrong_lemma = curr_stats['wrong lemma'] - self.initial_wrong_lemma
                        # new_no_attribute = curr_stats['no attribute'] - self.initial_no_attribute
                        self.correct_lemma[name].append((num_ablated,
                                                         divide_zero(curr_stats['correct lemma'],
                                                                           curr_stats['wrong word'])))
                        self.kept_attribute[name].append((num_ablated,
                                                        divide_zero(curr_stats['kept attribute'],
                                                                          curr_stats['wrong word'])))
                        # self.no_attribute[name].append((num_ablated,
                        #                                 divide_zero(curr_stats['no attribute'],
                        #                                                   curr_stats['wrong word'])))
                        if curr_stats['kept attribute'] != 0:
                            self.correct_val[name].append((num_ablated, curr_stats['correct val'] /
                                                           curr_stats['kept attribute']))
                        self.split_words[name].append((num_ablated, curr_stats['pred split'] /
                                                       curr_stats['relevant']))
                        self.correct_lemma_correct_val[name].append(
                            (num_ablated, curr_stats['correct lemma, correct value'] /
                             curr_stats['wrong word']))
                        self.correct_lemma_wrong_val[name].append(
                            (num_ablated, curr_stats['correct lemma, wrong value'] /
                             curr_stats['wrong word']))
                        self.wrong_lemma_correct_val[name].append(
                            (num_ablated, curr_stats['wrong lemma, correct value'] /
                             curr_stats['wrong word']))
                        self.wrong_lemma_wrong_val[name].append(
                            (num_ablated, curr_stats['wrong lemma, wrong value'] /
                             curr_stats['wrong word']))

    def dump_results(self):
        wrong_words_path = Path(self.dir_path, 'wrong words')
        correct_lemmas_path = Path(self.dir_path, 'correct lemmas')
        kept_att_path = Path(self.dir_path, 'kept attribute')
        correct_val_path = Path(self.dir_path, 'correct val')
        split_words_path = Path(self.dir_path, 'split words')
        c_lemma_c_val_path = Path(self.dir_path, 'c lemmas c val')
        c_lemma_w_val_path = Path(self.dir_path, 'c lemmas w val')
        w_lemma_c_val_path = Path(self.dir_path, 'w lemmas c val')
        w_lemma_w_val_path = Path(self.dir_path, 'w lemmas w val')
        paths = [wrong_words_path, correct_lemmas_path, kept_att_path, correct_val_path, split_words_path,
                 c_lemma_c_val_path, c_lemma_w_val_path, w_lemma_c_val_path, w_lemma_w_val_path]
        for p in paths:
            if not p.exists():
                p.mkdir()
        stats = [self.wrong_word, self.correct_lemma, self.kept_attribute,
                            self.correct_val, self.split_words,
                            self.correct_lemma_correct_val, self.correct_lemma_wrong_val,
                            self.wrong_lemma_correct_val, self.wrong_lemma_wrong_val]
        for p, rankings_results in zip(paths,stats):
            for name, res in rankings_results.items():
                with open(Path(p,name),'wb+') as f:
                    pickle.dump(res,f)

    def plot_metric(self, ax, to_save, metric):
        graph_types = {'wrong words': self.wrong_word, 'correct lemmas': self.correct_lemma,
                       'kept attribute': self.kept_attribute,'correct values': self.correct_val,
                       'split words': self.split_words}
        results = graph_types[metric]
        title = ' '.join([self.language, self.attribute, self.layer_str, 'ablation ']) + metric
        ax, legend = self.prep_plot(title, results, metric, xlabel='ablated neurons', ylabel=metric, ax=ax, to_save=to_save)
        return ax, legend

    def plot_ranking(self, ax, to_save, ranking):
        all_results = {'wrong words': self.wrong_word, 'correct lemmas': self.correct_lemma,
                       'kept attribute': self.kept_attribute, 'correct values': self.correct_val,
                       'split words': self.split_words}
        results = {metric: r[ranking] for metric, r in all_results.items()}
        title = ' '.join([self.language, self.attribute, self.layer_str, 'ablation ']) + ranking
        ax, legend = self.prep_plot(title, results, ranking, xlabel='ablated neurons', ylabel='', ax=ax, to_save=to_save)
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


def run_all_probing(dir_path, plot_separate, only_dump=False):
    axs = [0] * 3
    small_dataset = False
    max_nums = [10, 50, 150] if plot_separate else [150]
    max_nums = [150]
    # model_types = ['all','linear','bayes'] if plot_separate else ['all']
    model_types = ['all']
    # metrics = ['acc','nmi','selectivity','ranking avg', 'classifiers avg']
    metrics = ['acc']
    for metric in metrics:
        if not plot_separate:
            fig, axs = plt.subplots(3, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['probing',dir_path.parts[-2], dir_path.parts[-1], metric, 'per layer']))
            legend=None
        # for i, layer in enumerate([2, 7, 12]):
        for i, layer in enumerate([7]):
            for max_num in max_nums:
                for model_type in model_types:
                    layer_dir = Path(dir_path, 'layer '+str(layer))
                    res_files_names = [f.name for f in layer_dir.glob('*') if
                                 f.is_file() and not f.name.startswith('whole')
                                       and not f.name.endswith('control') and not f.name.endswith('.pkl')]
                    if not res_files_names:
                        continue
                    res_files_names = ['bayes by bayes mi', 'bayes by worst mi', 'linear by bayes mi',
                                       'linear by top avg', 'bayes by random', 'linear by bottom avg']
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
                    # if metric == 'acc':
                    #     plotting.dump_results()
                    if only_dump:
                        continue
                    res = plot_metric(plotting, metric)
                    if not plot_separate:
                        axs[i], legend = res
                        axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
                    # if metric == 'selectivity':
                    #     plotting.dump_results()

        if not plot_separate and not only_dump:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.savefig(Path(dir_path, ' '.join(['probing', metric, 'by layers'])))

def run_ablation(dir_path, plot_separate):
    if dir_path.name == 'Part of Speech':
        return
    axs = [0]*3
    # metrics = ['total accuracy', 'loss', 'ablated words accuracy', 'non-ablated words accuracy',
    #            'avg lemma rank', 'avg lemma log rank', 'lemmas in top 10',
    #            'lemmas in top 100']
    metrics = ['total accuracy', 'loss', 'ablated words accuracy', 'non-ablated words accuracy']
    # for metric in ['avg lemma rank', 'avg lemma log rank', 'lemmas in top 10', 'lemmas in top 100']:
    for metric in metrics:
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
                # res_files_names = [f.parts[-1] for f in ablation_root_path.glob('*') if
                #                    f.is_file()]
                res_files_names = [f.name for f in ablation_root_path.glob('*') if
                                   f.is_file() and f.name.startswith('sparsed') and 'intervention' not in f.name]
                ab = ablation(dir_path=ablation_root_path, names=res_files_names, layer=layer, max_num=max_num)
                # ab.dump_results()
                axs[i], legend=ab.plot_metric(axs[i], plot_separate,metric)
                if not plot_separate:
                    axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            plt.savefig(Path(dir_path, ' '.join(['ablation', metric, 'by layers'])))

def run_morph(dir_path, plot_separate, all_rankings, only_dump=False):
    num_subplots = 3
    axs = [0] * num_subplots
    iter_list = ['wrong words', 'correct lemmas', 'kept attribute', 'correct values', 'split words'] if all_rankings \
        else ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi', 'by random', 'by top cluster', 'by bottom cluster']
    if only_dump:
        iter_list = ['by top avg']
    for name in iter_list:
        if not plot_separate and not only_dump:
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
                res_files_names = [f.name for f in spacy_root_path.glob('*') if
                                   f.is_file() and f.name.endswith('lnspace')]
                ma = morphologyAblation(dir_path=spacy_root_path, names=res_files_names, layer=layer,
                                        max_num=max_num, only_dump=only_dump)
                if only_dump:
                    continue
                axs[i], legend = ma.plot_metric(axs[i], plot_separate, name) if all_rankings \
                    else ma.plot_ranking(axs[i], plot_separate, name)
                if not plot_separate:
                    axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate and not only_dump:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            if not Path(dir_path,'spacy figs').exists():
                Path(dir_path,'spacy figs').mkdir()
            plt.savefig(Path(dir_path, 'spacy figs', ' '.join(['ablation', name, 'by layers'])))


if __name__ == "__main__":
    data_name = 'UM'
    # languages = ['eng', 'ara', 'hin', 'rus', 'fin', 'bul', 'tur', 'spa', 'fra']
    languages = ['bul']
    for lan in languages:
        print(lan)
        root_path = Path('results',data_name,lan)
        atts_path = [p for p in root_path.glob('*') if not p.is_file()]
        for att_path in atts_path:
            if 'Definiteness' != att_path.name:
                continue
            run_all_probing(att_path, plot_separate=True, only_dump=False)
            # run_ablation(att_path, plot_separate=False)
            # run_morph(att_path, plot_separate=False, all_rankings=False, only_dump=True)
