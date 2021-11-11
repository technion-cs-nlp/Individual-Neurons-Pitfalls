import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pathlib import Path
import torch
import consts
from utils import divide_zero
import copy
from argparse import ArgumentParser
import ast
import pickle
import numpy as np


class plots:
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
        settings = ['gaussian by ttb gaussian', 'linear by ttb gaussian', 'gaussian by ttb linear',
                    'linear by ttb linear',
                    'gaussian by ttb probeless', 'linear by ttb probeless', 'gaussian by random', 'linear by random',
                    'gaussian by btt gaussian', 'linear by btt gaussian', 'gaussian by btt linear',
                    'linear by btt linear',
                    'gaussian by btt probeless', 'linear by btt probeless']
        self.colors = {k: v for k, v in zip(settings, colors_cmap + colors_cmap)}
        self.linestyles = {k: 'dotted' if 'btt' in k else 'dashed' if 'random' in k else 'solid'
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
        avg_names = ['linear', 'gaussian', 'by ttb linear', 'by btt linear', 'by random', 'by ttb gaussian',
                     'by btt gaussian']
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
        class_names = ['linear', 'gaussian']
        train_acc_class_avgs = {name: res for name, res in train_acc_avgs.items() if name in class_names}
        test_acc_class_avgs = {name: res for name, res in test_acc_avgs.items() if name in class_names}
        train_sel_class_avgs = {name: res for name, res in train_sel_avgs.items() if name in class_names}
        test_sel_class_avgs = {name: res for name, res in test_sel_avgs.items() if name in class_names}
        rank_names = ['by ttb linear', 'by btt linear', 'by random', 'by ttb gaussian', 'by btt gaussian']
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


class InterDump(plots):
    def __init__(self, dir_path, names, layer, max_num=760):
        super(InterDump, self).__init__(dir_path, max_num)
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


class InterPlot:
    def __init__(self, model_type, set_type, language, attr, layer):
        self.model_type = model_type
        self.set_type = set_type
        self.language = language
        self.attribute = attr
        self.layer = layer
        cmap = plt.get_cmap('Paired').colors
        self.line_colors_all_rankings = dict(zip(['by ttb linear', 'by ttb probeless', 'by ttb gaussian', 'by random',
                                                  'by btt linear', 'by btt probeless', 'by btt gaussian'],
                                                 cmap))
        self.linestyles = {'error rate': 'solid', 'CLWV': 'dashed'}
        self.root_path = Path('results', 'UM', self.model_type, self.language, self.attribute,
                              'layer ' + str(self.layer), 'spacy', self.set_type)

    def load_data(self):
        wrong_words_path = Path(self.root_path, 'wrong words')
        correct_lemmas_path = Path(self.root_path, 'correct lemmas')
        kept_att_path = Path(self.root_path, 'kept attribute')
        correct_val_path = Path(self.root_path, 'correct val')
        c_lemma_c_val_path = Path(self.root_path, 'c lemmas c val')
        c_lemma_w_val_path = Path(self.root_path, 'c lemmas w val')
        w_lemma_c_val_path = Path(self.root_path, 'w lemmas c val')
        w_lemma_w_val_path = Path(self.root_path, 'w lemmas w val')
        rankings = ['by ttb linear', 'by btt linear', 'by ttb gaussian', 'by btt gaussian',
                    'by ttb probeless', 'by btt probeless', 'by random']
        self.rankings = rankings + \
                        [f'{r}_translation_{step}_{alpha}' for r in rankings for step in [10] for alpha in
                         [1, 2, 4, 6, 8]] + \
                        [f'{r}_translation_{step}_{alpha}_scaled' for r in rankings for step in [10] for alpha in
                         [2.0, 6.0, 8.0, 10.0, 12.0]]

        stat_names = ['wrong words', 'correct lemmas', 'kept att', 'correct val', 'c lemma c val',
                      'c lemma w val', 'w lemma c val', 'w lemma w val']
        self.res = dict(zip(self.rankings, [dict.fromkeys(stat_names) for _ in self.rankings]))
        for ranking in self.rankings:
            if not Path(wrong_words_path, ranking).exists():
                continue
            with open(Path(wrong_words_path, ranking), 'rb') as f:
                self.res[ranking]['wrong words'] = pickle.load(f)
            with open(Path(correct_lemmas_path, ranking), 'rb') as f:
                self.res[ranking]['correct lemmas'] = pickle.load(f)
            with open(Path(kept_att_path, ranking), 'rb') as f:
                self.res[ranking]['kept att'] = pickle.load(f)
            with open(Path(correct_val_path, ranking), 'rb') as f:
                self.res[ranking]['correct val'] = pickle.load(f)
            with open(Path(c_lemma_c_val_path, ranking), 'rb') as f:
                self.res[ranking]['c lemma c val'] = pickle.load(f)
            with open(Path(c_lemma_w_val_path, ranking), 'rb') as f:
                self.res[ranking]['c lemma w val'] = pickle.load(f)
            with open(Path(w_lemma_c_val_path, ranking), 'rb') as f:
                self.res[ranking]['w lemma c val'] = pickle.load(f)
            with open(Path(w_lemma_w_val_path, ranking), 'rb') as f:
                self.res[ranking]['w lemma w val'] = pickle.load(f)

    def plot_line(self, alpha, scaled):
        all_rankings_res = {}
        # for ranking in self.rankings[:7]:
        for ranking in ['by ttb linear', 'by ttb probeless', 'by ttb gaussian', 'by random', 'by btt linear',
                        'by btt probeless', 'by btt gaussian']:
            # for ranking in ['by ttb linear', 'by ttb probeless']:
            # num_ablated = [str(r[0]) for r in self.res[ranking]['wrong words']]
            to_plot = {}
            inter_types = ['ablation', r'$\alpha='+str(alpha)+'$', r'scaled $\alpha='+str(alpha)+'$']

            for inter_type in inter_types:
                method = ranking if inter_type == 'ablation' else f'{ranking}_translation_10_{inter_type[-2]}' \
                    if 'scale' not in inter_type else \
                        f'{ranking}_translation_10_{float(inter_type[inter_type.index("=") + 1:-1])}_scaled'
                if self.res[method]['wrong words']:
                    wrong_preds = np.array([r[1] for r in self.res[method]['wrong words']])
                    clwv = np.array([r[1] for r in self.res[method]['c lemma w val']]) * wrong_preds
                    to_plot[inter_type] = (wrong_preds, clwv)
            if alpha == 0:
                all_rankings_res[ranking] = to_plot['ablation']
            elif not scaled:
                all_rankings_res[ranking] = to_plot[r'$\alpha=' + str(alpha) + '$']
            else:
                all_rankings_res[ranking] = to_plot[r'scaled $\alpha=' + str(alpha) + '$']
        ticks = [i * 10 for i in range(len(list(all_rankings_res.values())[0][0]))]
        # num_ablated = [str(r[0]) for r in self.res['by ttb linear_intervention_10_8.0_lnspace']['wrong words']]
        title = 'ablation' if alpha == 0 else f'beta_{alpha}_scaled' if scaled else f'beta_{alpha}'
        self.plot_by_plt(ticks, all_rankings_res, title, False)
        max_points = {ranking: self.find_saturation_point(stats[1], 1.05, 0.1, ticks) for ranking, stats in
                      all_rankings_res.items()}
        return max_points

    def find_saturation_point(self, stats: np.array, k1, k2, x_ticks):
        for i in range(len(stats) - 2):
            improvement1, improvement2 = stats[i + 1] / stats[i], stats[i + 2] / stats[i]
            if improvement1 < k1 and improvement2 < k1:
                if stats[-1] - stats[i] > k2:
                    continue
                return {'max': stats[i], 'argmax': x_ticks[i]}
        return {'max': stats[-1], 'argmax': x_ticks[len(stats) - 1]}

    def plot_by_plt(self, x_ticks, stats, title, within_ranking):
        plt.figure(figsize=[6.8, 5.4])
        ax = plt.subplot(111)
        legend_labels, legend_lines = [], []
        # colors = self.line_colors_one_ranking if within_ranking else self.line_colors_all_rankings
        colors = self.line_colors_all_rankings
        for name, (wrong_words, clwv) in stats.items():
            end_ticks = min(31, len(wrong_words)) if title != 'ablation' else min(76, len(wrong_words))
            line, = ax.plot(x_ticks[:end_ticks], wrong_words[:end_ticks], color=colors[name], label=name,
                            linestyle=self.linestyles['error rate'])
            legend_labels.append(name)
            legend_lines.append(line)
            line, = ax.plot(x_ticks[:end_ticks], clwv[:end_ticks], color=colors[name], label=name,
                            linestyle=self.linestyles['CLWV'])
        ax.set_xlabel('neurons', fontsize=16)
        ax.set_ylabel('fraction of all predictions', fontsize=16)
        # ax.annotate('saturation point', xy=(5.0, 0.37), xytext=(12, 0.55), size=16,
        #             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8),
        #             )
        loc = plticker.MultipleLocator(base=50)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis='both', which='major', labelsize=16)
        legend_labels = ['by ttb Linear', 'by ttb Probeless', 'by ttb Gaussian', 'by random',
                         'by btt Linear', 'by btt Probeless', 'by btt Gaussian']
        ax.legend(legend_lines, legend_labels, ncol=3, loc='center left', bbox_to_anchor=(-0.145, 1.12), fontsize=12.5)
        plt.tight_layout()
        plt.savefig(Path(self.root_path, 'figs', title + '_combined.png'))
        plt.close()


def run_all_probing(dir_path, layer, metric, plot_separate, only_dump=False):
    axs = [0] * 3
    max_nums = [150]
    # model_types = ['all','linear','gaussian'] if plot_separate else ['all']
    probe_types = ['all']
    # metrics = ['acc','selectivity','ranking avg', 'classifiers avg']
    metrics = [metric]
    for met in metrics:
        if not plot_separate:
            fig, axs = plt.subplots(3, figsize=[8.4, 6.8])
            fig.suptitle(' '.join(['probing', dir_path.parts[-2], dir_path.parts[-1], met, 'per layer']))
            legend = None
        # for i, layer in enumerate([2, 7, 12]):
        for i, l in enumerate([layer]):
            for max_num in max_nums:
                for probe_type in probe_types:
                    layer_dir = Path(dir_path, 'layer ' + str(l))
                    res_files_names = [f.name for f in layer_dir.glob('*') if
                                       f.is_file() and not f.name.startswith('whole')
                                       and not f.name.endswith('control') and not f.name.endswith('.pkl')]
                    if not res_files_names:
                        continue
                    # res_files_names = ['gaussian by ttb gaussian', 'gaussian by btt gaussian', 'linear by ttb gaussian',
                    #                    'linear by ttb linear', 'linear by random',
                    #                    'gaussian by random', 'linear by btt linear']

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

                    plotting = probing(layer_dir, res_files_names, l, max_num, probe_type)
                    plotting.dump_results()
                    if only_dump:
                        continue
                    res = plot_metric(plotting, met)
                    if not plot_separate:
                        axs[i], legend = res
                        axs[i].text(1.01, 0.5, 'layer ' + str(layer), transform=axs[i].transAxes)
        if not plot_separate and not only_dump:
            for ax in axs:
                ax.label_outer()
            # fig.legend(legend, ncol=5, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            fig.legend(ncol=4, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 0.95))
            plt.savefig(Path(dir_path, ' '.join(['probing', met, 'by layers'])))


def run_interventions(model_type, set_type, language, attribute, layer, alpha=8, scaled=True):
    spacy_root_path = Path('results', 'UM', model_type, language, attribute, 'layer ' + str(layer), 'spacy', set_type)
    if not spacy_root_path.exists():
        sys.exit('WRONG SETTING')

    def relevant_file(f):
        if alpha == 0:
            return 'translation' not in f.name
        elif not scaled:
            return 'translation' in f.name and f.name.endswith(str(alpha))
        else:
            return f.name.endswith('scaled')

    res_files_names = [f.name for f in spacy_root_path.glob('*') if
                       f.is_file() and relevant_file(f)]
    InterDump(dir_path=spacy_root_path, names=res_files_names, layer=layer)
    inter_plt = InterPlot(model_type, set_type, language, attribute, layer)
    inter_plt.load_data()
    inter_plt.plot_line(alpha, scaled)


if __name__ == "__main__":
    data_name = 'UM'
    parser = ArgumentParser()
    parser.add_argument('-experiments', type=str, help='must be either \'probing\' or \'interventions\'')
    parser.add_argument('-set', type=str, help='data set to analyze results on, can be dev or test, default is test. '
                                               'Only relevant if experiments==\'interventions\'')
    parser.add_argument('-model', type=str, help='language model: \'bert\' or \'xlm\'')
    parser.add_argument('-language', type=str, help='3-chars language code')
    parser.add_argument('-attribute', type=str, help='name of attribute as defined by UM schema')
    parser.add_argument('-layer', type=int, help='model\'s layer to analyze (0-12)')
    parser.add_argument('-beta', type=int, default=8, help='value of beta, default is 8. '
                                                           'Only relevant if experiments==\'interventions\'')
    parser.add_argument('--scaled', default=False, action='store_true',
                        help='whether to use a scaled coefficients vector (alpha) or a constant coefficient '
                             'for all neurons, default is False (constant). '
                             'Only relevant if experiments==\'interventions\'')
    parser.add_argument('-metric', type=str, default='acc', help='acc or selectivity, default is acc. '
                                                                 'Only relevant if experiments==\'probing\'')
    args = parser.parse_args()
    experiments = args.experiments
    if experiments != 'probing' and experiments != 'interventions':
        sys.exit('WRONG SETTING')
    set_type = args.set
    if set_type is None:
        set_type = 'test'
    model_type = args.model
    language = args.language
    attribute = args.attribute
    layer = args.layer
    alpha = args.beta
    scaled = args.scaled
    metric = args.metric
    # languages = ['eng', 'ara', 'hin', 'rus', 'fin', 'bul', 'tur', 'spa', 'fra']
    att_path = Path('results', data_name, model_type, language, attribute)
    if not Path(att_path, f'layer {str(layer)}').exists():
        sys.exit('WRONG SETTING')
    if experiments == 'probing':
        run_all_probing(att_path, layer, metric, plot_separate=True, only_dump=False)
    else:
        run_interventions(model_type, set_type, language, attribute, layer, alpha, scaled)
