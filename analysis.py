import matplotlib.pyplot as plt
from pathlib import Path
import torch
import consts
from utils import divide_zero
import copy
from argparse import ArgumentParser
import ast
import pickle

class plots():
    def __init__(self, dir_path: Path, max_num: int, ablation: bool = False):
        self.dir_path = dir_path
        self.max_num = max_num
        self.ablation = ablation
        self.save_path = Path(dir_path, 'figs')
        if not self.save_path.exists():
            self.save_path.mkdir()

    def draw_plot(self, ax, sorted_results):
        raise NotImplementedError

    def prep_plot(self, title, results, save_file_name, xlabel, ylabel, ax, to_save=False):
        results = {name: res for name, res in results.items() if res}
        if not results:
            return None, None
        if to_save:
            plt.figure(figsize=[6.4, 4.0])
            # plt.figure()
            # ax = plt.subplot(111, title=title)
            ax = plt.subplot(111)
        sorted_results = list(results.items())
        curr_max = min(self.max_num, len(sorted_results[0][1]))

        def sort_plots_by_last_val(l):
            if curr_max <= len(l[1]):
                return l[1][curr_max - 1]
            else:
                return l[1][-1]

        def sort_plots_by_name(l):
            return l[0]

        # sorted_results.sort(key=sort_plots_by_name)
        sorted_results.sort(key=sort_plots_by_last_val, reverse=True)

        ax, legend_labels, legend_lines = self.draw_plot(ax, sorted_results)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        # ax.yticks(fontsize=10)
        if to_save:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height])
            # ax.legend(legend_lines, legend_labels, ncol=3, loc='upper center', bbox_to_anchor=(0.45, 1.15), fontsize=8)
            # figlegend = plt.figure(figsize=[12., 0.6])
            # legend_labels = ['Linear by Linear', 'Gaussian by Gaussian', 'Linear by Gaussian',
            #                  'Gaussian by random', 'Linear by random']
            # figlegend.legend(legend_lines, legend_labels, ncol=3, loc='center', fontsize=12)
            # plt.tight_layout()
            # figlegend.show()
            # plt.savefig(Path(self.save_path, 'only_legend.png', bbox_inches='tight'))
            # ax.set_title(title)
            # ax.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.tight_layout()
            plt.savefig(Path(self.save_path, save_file_name))
            # plt.show()
            plt.close()
        return ax, legend_labels


class probing(plots):
    def __init__(self, dir_path: Path, names, layer: int, max_num: int, model_type: str = 'all'):
        super(probing, self).__init__(dir_path=dir_path, max_num=max_num)
        if model_type != 'all':
            names = [name for name in names if name.startswith(model_type)]
        colors_cmap = plt.get_cmap('Paired').colors
        colors_cmap = colors_cmap[1::2] + ('black', 'gray')
        settings = ['bayes by bayes mi', 'linear by bayes mi', 'bayes by top avg', 'linear by top avg',
                    'bayes by top cluster', 'linear by top cluster', 'bayes by random', 'linear by random',
                    'bayes by worst mi', 'linear by worst mi', 'bayes by bottom avg', 'linear by bottom avg',
                    'bayes by bottom cluster', 'linear by bottom cluster']
        self.colors = {k: v for k, v in zip(settings, colors_cmap + colors_cmap)}
        self.linestyles = {k: 'dotted' if 'bottom' in k or 'worst' in k else 'dashed' if 'random' in k else 'solid'
                           for k in self.colors.keys()}
        self.names = names
        self.layer = layer
        self.model_type = model_type
        self.save_path = Path(self.save_path, str(self.max_num), model_type)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
        self.language = dir_path.parts[2]
        self.attribute = dir_path.parts[3]
        self.layer_str = 'layer ' + str(self.layer)
        self.load_results()

    def load_results(self):
        self.train_acc_results = {name: [] for name in self.names}
        self.train_mi_results = {name: [] for name in self.names}
        self.train_nmi_results = {name: [] for name in self.names}
        self.test_acc_results = {name: [] for name in self.names}
        self.test_mi_results = {name: [] for name in self.names}
        self.test_nmi_results = {name: [] for name in self.names}
        for name in self.names:
            with open(Path(self.dir_path, name), 'r') as f:
                for line in f.readlines():
                    if 'accuracy on train set' in line:
                        self.train_acc_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'final accuracy on test' in line:
                        if line.split()[-1].replace('.', '', 1).isdigit():
                            self.test_acc_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'mi on train' in line:
                        self.train_mi_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'mi on test' in line:
                        self.test_mi_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'nmi on train' in line:
                        self.train_nmi_results[name].append(round(float(line.split()[-1]), ndigits=4))
                    if 'nmi on test' in line:
                        self.test_nmi_results[name].append(round(float(line.split()[-1]), ndigits=4))
        control_names = [name + '_control' for name in self.names
                         if Path(self.dir_path, name + '_control').is_file()]
        self.train_controls = {name: [] for name in control_names}
        self.train_acc_for_control = copy.deepcopy(self.train_acc_results)
        self.test_controls = {name: [] for name in control_names}
        self.test_acc_for_control = copy.deepcopy(self.test_acc_results)
        for control_name in control_names:
            with open(Path(self.dir_path, control_name), 'r') as f:
                for line in f.readlines():
                    if 'accuracy on train set' in line:
                        self.train_controls[control_name].append(float(line.split()[-1]))
                    if 'final accuracy on test' in line:
                        if line.split()[-1].replace('.', '', 1).isdigit():
                            self.test_controls[control_name].append(float(line.split()[-1]))
        self.train_selectivities = {control_name[:-8]: [] for control_name in control_names}
        self.test_selectivities = {control_name[:-8]: [] for control_name in control_names}

    def dump_results(self):
        dump_path = Path(self.dir_path, 'test_acc_results.pkl')
        res_to_dump = self.test_acc_results
        with open(dump_path, 'wb+') as f:
            pickle.dump(res_to_dump, f)

    def draw_plot(self, ax, sorted_results):
        legend_labels, legend_lines = [], []
        for name, res in sorted_results:
            name_for_legend = ' '.join(name.split()[:2] + name.split()[-1:])
            # TODO change 'layer==2' condition to something that makes sense
            # it's there in order to place labels only once when plotting all layers together
            if self.layer == 2:
                if 'bottom' not in name and 'worst' not in name:
                    line, = ax.plot(res[:self.max_num], color=self.colors[name], label=name_for_legend,
                                    linestyle=self.linestyles[name])
                else:
                    line, = ax.plot(res[:self.max_num], color=self.colors[name], linestyle=self.linestyles[name])
            else:
                line, = ax.plot(res[:self.max_num], color=self.colors[name], linestyle=self.linestyles[name])
            if not ('bottom' in name or 'worst' in name):
                legend_labels.append(name_for_legend)
                legend_lines.append(line)
        return ax, legend_labels, legend_lines

    def plot_acc_and_nmi(self, ax, to_save, metric):
        if not self.train_acc_results:
            return
        graph_types = {'accuracy': self.test_acc_results, 'nmi': self.test_nmi_results}
        results = graph_types[metric]
        # paper_str = '_paper'
        paper_str = ''
        title = ' '.join([self.language, self.attribute, self.layer_str]) + \
                ' - test ' + metric + paper_str
        ax, legend = self.prep_plot(title, results, 'test ' + metric + paper_str, 'neurons', metric, ax, to_save)
        return ax, legend

    def plot_selectivity(self, ax, to_save):
        if not self.train_selectivities:
            return
        title = ' '.join([self.language, self.attribute, self.layer_str]) \
                + ' - test selectivity'
        for name, res in self.test_selectivities.items():
            for acc, cont in zip(self.test_acc_for_control[name], self.test_controls[name + '_control']):
                res.append(acc - cont)
        # paper_str = '_paper'
        paper_str = ''
        ax, legend = self.prep_plot(title, self.test_selectivities, 'test selectivity' + paper_str, 'neurons',
                                    'selectivity', ax, to_save)
        return ax, legend

    '''
    used for plotting mean accuracy and selectivity across probe / rankings
    not up to date - probably doesn't work
    '''

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
            train_acc_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in
                                          self.train_acc_results.items()
                                          if name.startswith(avg_name) or name.endswith(avg_name)]
            test_acc_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in self.test_acc_results.items()
                                         if name.startswith(avg_name) or name.endswith(avg_name)]
            train_sel_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in
                                          self.train_selectivities.items()
                                          if name.startswith(avg_name) or name.endswith(avg_name)]
            test_sel_relevant_results = [torch.tensor(res[:self.max_num]) for name, res in
                                         self.test_selectivities.items()
                                         if name.startswith(avg_name) or name.endswith(avg_name)]
            if not train_acc_relevant_results:
                continue
            min_train_acc_len = min([t.shape[0] for t in train_acc_relevant_results])
            if min_train_acc_len < self.max_num:
                train_acc_relevant_results = [t[:min_train_acc_len] for t in train_acc_relevant_results]
            train_acc_avgs[avg_name] = torch.stack(train_acc_relevant_results).mean(dim=0).tolist()
            min_test_acc_len = min([t.shape[0] for t in test_acc_relevant_results])
            if min_test_acc_len < self.max_num:
                test_acc_relevant_results = [t[:min_test_acc_len] for t in test_acc_relevant_results]
            test_acc_avgs[avg_name] = torch.stack(test_acc_relevant_results).mean(dim=0).tolist()
            min_train_sel_len = min([t.shape[0] for t in train_sel_relevant_results])
            if min_train_sel_len < self.max_num:
                train_sel_relevant_results = [t[:min_train_sel_len] for t in train_sel_relevant_results]
            train_sel_avgs[avg_name] = torch.stack(train_sel_relevant_results).mean(dim=0).tolist()
            min_test_sel_len = min([t.shape[0] for t in test_sel_relevant_results])
            if min_test_sel_len < self.max_num:
                test_sel_relevant_results = [t[:min_test_sel_len] for t in test_sel_relevant_results]
            test_sel_avgs[avg_name] = torch.stack(test_sel_relevant_results).mean(dim=0).tolist()
        class_names = ['linear', 'bayes']
        train_acc_class_avgs = {name: res for name, res in train_acc_avgs.items() if name in class_names}
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
            if avg_type == 'ranking' and i == 0:
                continue
            metric = 'acc' if i < 2 else 'sel'
            title = ' '.join(
                [self.language, self.attribute, self.layer_str, 'test', avg_type, metric, 'avgs'])
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

    def dump_results(self):
        dump_path = Path(self.dir_path, 'results.pkl')
        res_to_dump = {'relevant acc': self.relevant_accs}
        with open(dump_path, 'wb+') as f:
            pickle.dump(res_to_dump, f)

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
                       'non-ablated words accuracy': self.irrelevant_accs}
        results = graph_types[metric]
        title = ' '.join([self.language, self.attribute, self.layer_str, 'ablation ']) + metric
        ax, legend = self.prep_plot(title, results, metric, xlabel='ablated neurons', ylabel=metric, ax=ax,
                                    to_save=to_save)
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
                        self.wrong_word[name].append((num_ablated, curr_stats['wrong word'] /
                                                      curr_stats['relevant']))
                        self.correct_lemma[name].append((num_ablated,
                                                         divide_zero(curr_stats['correct lemma'],
                                                                     curr_stats['wrong word'])))
                        self.kept_attribute[name].append((num_ablated,
                                                          divide_zero(curr_stats['kept attribute'],
                                                                      curr_stats['wrong word'])))
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
        for p, rankings_results in zip(paths, stats):
            for name, res in rankings_results.items():
                with open(Path(p, name), 'wb+') as f:
                    pickle.dump(res, f)


def run_all_probing(dir_path, plot_separate, only_dump=False):
    axs = [0] * 3
    max_nums = [150]
    # model_types = ['all','linear','bayes'] if plot_separate else ['all']
    probe_types = ['all']
    # metrics = ['acc','selectivity','ranking avg', 'classifiers avg']
    metrics = ['selectivity']
    for metric in metrics:
        if not plot_separate:
            fig, axs = plt.subplots(3, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['probing', dir_path.parts[-2], dir_path.parts[-1], metric, 'per layer']))
            legend = None
        # for i, layer in enumerate([2, 7, 12]):
        for i, layer in enumerate([12]):
            for max_num in max_nums:
                for probe_type in probe_types:
                    layer_dir = Path(dir_path, 'layer ' + str(layer))
                    res_files_names = [f.name for f in layer_dir.glob('*') if
                                       f.is_file() and not f.name.startswith('whole')
                                       and not f.name.endswith('control') and not f.name.endswith('.pkl')]
                    if not res_files_names:
                        continue
                    res_files_names = ['bayes by bayes mi', 'bayes by worst mi', 'linear by bayes mi',
                                       'linear by top avg', 'linear by random',
                                       'bayes by random', 'linear by bottom avg']

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

                    plotting = probing(layer_dir, res_files_names, layer, max_num, probe_type)
                    if only_dump:
                        continue
                    res = plot_metric(plotting, metric)
                    if not plot_separate:
                        axs[i], legend = res
                        axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate and not only_dump:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            fig.legend(ncol=4, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.savefig(Path(dir_path, ' '.join(['probing', metric, 'by layers'])))


def run_ablation(dir_path, plot_separate):
    if dir_path.name == 'Part of Speech':
        return
    axs = [0] * 3
    metrics = ['total accuracy', 'loss', 'ablated words accuracy', 'non-ablated words accuracy']
    for metric in metrics:
        if not plot_separate:
            fig, axs = plt.subplots(3, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['ablation', dir_path.parts[-2], dir_path.parts[-1], metric, 'per layer']))
            legend = None
        for i, layer in enumerate([2, 7, 12]):
            max_nums = [0, 400, 600] if plot_separate else [0]
            for max_num in max_nums:
                ablation_root_path = Path(dir_path, 'layer ' + str(layer), 'ablation by attr')
                if not ablation_root_path.exists():
                    continue
                res_files_names = [f.name for f in ablation_root_path.glob('*') if
                                   f.is_file() and f.name.startswith('sparsed') and 'intervention' not in f.name]
                ab = ablation(dir_path=ablation_root_path, names=res_files_names, layer=layer, max_num=max_num)
                # ab.dump_results()
                axs[i], legend = ab.plot_metric(axs[i], plot_separate, metric)
                if not plot_separate:
                    axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size':8}, bbox_to_anchor=(0.5,0.95))
            fig.legend(ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.savefig(Path(dir_path, ' '.join(['ablation', metric, 'by layers'])))


def run_morph(dir_path, set_type, all_rankings):
    num_subplots = 3
    axs = [0] * num_subplots
    iter_list = ['wrong words', 'correct lemmas', 'kept attribute', 'correct values',
                 'split words'] if all_rankings else ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi',
                                                      'by random', 'by top cluster', 'by bottom cluster']
    for i, layer in enumerate([2, 7, 12]):
        # for i, layer in enumerate([1]):
        spacy_root_path = Path(dir_path, 'layer ' + str(layer), 'spacy', set_type)
        if not spacy_root_path.exists():
            continue
        res_files_names = [f.name for f in spacy_root_path.glob('*') if
                           f.is_file() and f.name.endswith('lnspace')]
        morphologyAblation(dir_path=spacy_root_path, names=res_files_names, layer=layer)


if __name__ == "__main__":
    data_name = 'UM'
    model_type = 'bert'
    set_type = 'test'
    # languages = ['eng', 'ara', 'hin', 'rus', 'fin', 'bul', 'tur', 'spa', 'fra']
    languages = ['eng', 'spa', 'fra']
    for lan in languages:
        print(lan)
        root_path = Path('results', data_name, model_type, lan)
        atts_path = [p for p in root_path.glob('*') if not p.is_file()]
        for att_path in atts_path:
            # if 'Part of Speech' != att_path.name:
            #     continue
            # run_all_probing(att_path, plot_separate=True, only_dump=False)
            # run_ablation(att_path, plot_separate=False)
            run_morph(att_path, set_type, all_rankings=False)
