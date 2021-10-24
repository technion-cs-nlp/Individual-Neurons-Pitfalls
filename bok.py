from bokeh.io import output_file, show, save, export_png
from bokeh.plotting import figure
from bokeh.models import CDSView, BooleanFilter, ColumnDataSource, Label, Legend
from bokeh.palettes import Category10
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


class bokehPlots():
    def __init__(self, model_type, set_type, language, attr, layer):
        self.model_type = model_type
        self.set_type = set_type
        self.language = language
        self.attribute = attr
        self.layer = layer
        self.bar_colors = ["lightslategrey", "cornflowerblue", 'lightgreen', 'khaki']
        # self.line_colors = ['blue', 'orange']
        # self.line_colors = Category10[4]
        # self.line_colors = plt.get_cmap('Set1').colors[:2] * 2
        cmap = plt.get_cmap('Paired').colors
        # self.line_colors = {'ablation error rate': cmap[0], 'mirroring error rate': cmap[1],
        #                    'ablation CLWV': cmap[0], 'mirroring CLWV': cmap[1]}
        # self.line_colors = dict(zip(['ablation', 'bugged scaling', r'$\alpha=1$', r'$\alpha=2$', r'$\alpha=4$',
        #                              r'$\alpha=6$', r'$\alpha=8$', r'scaled $\alpha=2$', r'scaled $\alpha=6$',
        #                              r'scaled $\alpha=8$', r'scaled $\alpha=10$', r'scaled $\alpha=12$'],
        #                         cmap))
        self.line_colors_one_ranking = dict(zip(['ablation', r'$\alpha=2$', r'$\alpha=8$', r'ranking scale $\alpha=6$',
                                     r'ranking scale $\alpha=8$', r'ranking scale $\alpha=12$', r'ln scale $\alpha=6$',
                                     r'ln scale $\alpha=8$', r'ln scale $\alpha=12$'],
                                    cmap))
        self.line_colors_all_rankings = dict(zip(['by top avg', 'by top cluster', 'by bayes mi', 'by random',
                                                 'by bottom avg', 'by bottom cluster', 'by worst mi'],
                                                cmap))
        # self.linestyles = {'ablation error rate': 'solid', 'mirroring error rate': 'solid',
        #                    'ablation CLWV': 'dashed', 'mirroring CLWV': 'dashed'}
        self.linestyles = {'error rate': 'solid', 'CLWV': 'dashed'}
        # self.markers = {'ablation total mistakes': 'solid', 'mirroring total mistakes': 'solid',
        #                    'ablation CLWV': 'dashed', 'mirroring CLWV': 'dashed'}
        self.metrics = ['correct lemma, correct value', 'correct lemma, wrong value',
                        'wrong lemma, correct value', 'wrong lemma, wrong value']
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
        rankings = ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi',
                         'by top cluster', 'by bottom cluster', 'by random']
        self.rankings = rankings + [r + '_intervention' for r in rankings] + \
                        [f'{r}_intervention_{step}_{alpha}' for r in rankings for step in [10] for alpha in [1, 2, 4, 6, 8]] + \
                        [f'{r}_intervention_{step}_{alpha}__scaled' for r in rankings for step in [10] for alpha in
                         [2.0, 6.0, 8.0, 10.0, 12.0]] + \
                        [f'{r}_intervention_{step}_{alpha}_lnspace' for r in rankings for step in [10] for alpha in
                         [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]]

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

    def plot_bar(self, ranking, ablation, absolute):
        ranking += '' if ablation else '_intervention'
        num_ablated = [str(r[0]) for r in self.res[ranking]['wrong words']]
        wrong_preds = np.array([r[1] for r in self.res[ranking]['wrong words']])
        if max(wrong_preds) < 0.05:
            return
        booleans = [True if wp >= 0.05 else False for wp in wrong_preds]
        start_point = booleans.index(True)
        if not absolute:
            wrong_preds = np.ones_like(wrong_preds)
        data = {'ablated': num_ablated,
                'correct lemma, correct value': [r[1] for r in self.res[ranking]['c lemma c val']] * wrong_preds,
                'correct lemma, wrong value': [r[1] for r in self.res[ranking]['c lemma w val']] * wrong_preds,
                'wrong lemma, correct value': [r[1] for r in self.res[ranking]['w lemma c val']] * wrong_preds,
                'wrong lemma, wrong value': [r[1] for r in self.res[ranking]['w lemma w val']] * wrong_preds}
        clwv = np.array(data['correct lemma, wrong value'])
        max_clwv, argmax_clwv = clwv[booleans].max(), clwv[booleans].argmax()
        argmax_clwv = data['ablated'][start_point + argmax_clwv]
        label = Label(text=f'max correct lemma wrong value: {max_clwv:.2f}, argmax: {argmax_clwv}', x=590, y=370,
                      x_units='screen', y_units='screen',
                      render_mode='css', border_line_color='black',
                      border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=1.0)
        data = ColumnDataSource(data)
        view = CDSView(source=data, filters=[BooleanFilter(booleans)])
        title = ' '.join([self.language, self.attribute, 'layer', str(self.layer), ranking])
        # title = ''
        output_file(title)
        x_range = [num for i, num in enumerate(num_ablated) if booleans[i]]
        tooltips = "$name @ablated: @$name{(0.0000)}"
        # tooltips = ''
        p = figure(x_range=x_range, plot_height=500, plot_width=1000, title=title,
                   toolbar_location=None, tools="hover", tooltips=tooltips)
        # p = figure(x_range=x_range, plot_height=500, plot_width=1000, toolbar_location=None)
        r_list = [p.square(fill_color=c, line_color=c) for c in self.bar_colors]
        p.xaxis.major_label_orientation = 1.2
        p.vbar_stack(self.metrics, x='ablated', width=0.9, source=data, color=self.bar_colors,
                     legend_label=self.metrics, view=view)
        # p.vbar_stack(self.metrics, x='ablated', width=0.9, source=data, color=self.colors, view=view)
        legend = Legend(items=[(m, [r]) for m, r in zip(self.metrics, r_list)], location='center')
        legend.orientation = 'horizontal'
        legend.label_text_font_size = '16px'
        p.add_layout(legend, 'above')

        # p.vbar(x=x, top=y, width=0.9, color="red")
        p.xgrid.grid_line_color = None
        # p.y_range.start = 0
        # p.yaxis.visible = False
        p.outline_line_color = None
        # p.legend.orientation = "horizontal"
        # show(p)
        abs_str = ' absolute' if absolute else ' normalized'
        save(p, filename=Path(self.root_path, 'figs', ranking + abs_str + '.html'))
        # export_png(p, filename=Path(root_path, 'figs', ranking + abs_str + '.png').__str__())
        res = {'max': max_clwv, 'argmax': argmax_clwv}
        return res

    def plot_line(self, alpha):
        # ranking += '' if ablation else '_intervention'
        res = {}
        all_rankings_res = {}
        # for ranking in self.rankings[:7]:
        # for ranking in ['by top avg', 'by top cluster', 'by bayes mi', 'by random', 'by bottom avg',
        #                 'by bottom cluster', 'by worst mi']:
        for ranking in ['by top avg', 'by top cluster']:
            # num_ablated = [str(r[0]) for r in self.res[ranking]['wrong words']]
            to_plot = {}
            inter_types = ['ablation', 'bugged scaling', r'$\alpha=1$', r'$\alpha=2$', r'$\alpha=4$', r'$\alpha=6$',
                               r'$\alpha=8$', r'scaled $\alpha=2$', r'scaled $\alpha=6$', r'scaled $\alpha=8$',
                               r'scaled $\alpha=10$', r'scaled $\alpha=12$']
            inter_types = ['ablation', 'bugged scaling', r'$\alpha=2$',
                               r'$\alpha=8$', r'scaled $\alpha=6$', r'scaled $\alpha=8$',
                               r'scaled $\alpha=12$']
            inter_types = ['ablation', r'$\alpha=2$',
                           r'$\alpha=8$', r'ranking scale $\alpha=6$', r'ranking scale $\alpha=8$',
                           r'ranking scale $\alpha=12$', r'ln scale $\alpha=2$',
                            r'ln scale $\alpha=4$', r'ln scale $\alpha=6$', r'ln scale $\alpha=8$',
                           r'ln scale $\alpha=10$', r'ln scale $\alpha=12$']

            for inter_type in inter_types:
                method = ranking if inter_type == 'ablation' else f'{ranking}_intervention_10_{inter_type[-2]}'\
                    if 'scale' not in inter_type else\
                    f'{ranking}_intervention_10_{float(inter_type[inter_type.index("=")+1:-1])}__scaled' \
                    if 'ln' not in inter_type else\
                    f'{ranking}_intervention_10_{float(inter_type[inter_type.index("=")+1:-1])}_lnspace'
                if self.res[method]['wrong words']:
                    wrong_preds = np.array([r[1] for r in self.res[method]['wrong words']])
                    clwv = np.array([r[1] for r in self.res[method]['c lemma w val']]) * wrong_preds
                    # to_plot[inter_type] = (wrong_preds[:31], clwv[:31])
                    to_plot[inter_type] = (wrong_preds, clwv)
            all_rankings_res[ranking] = to_plot[r'ln scale $\alpha=' + str(alpha) +'$']
            # all_rankings_res[ranking] = to_plot['ablation']
            # wrong_preds_ablation = np.array([r[1] for r in self.res[ranking]['wrong words']])
            # clwv_ablation = np.array([r[1] for r in self.res[ranking]['c lemma w val']]) * wrong_preds_ablation
            # wrong_preds_intervention = np.array([r[1] for r in self.res[ranking+'_intervention']['wrong words']])
            # clwv_intervention = np.array([r[1] for r in self.res[ranking+'_intervention']['c lemma w val']]) * wrong_preds_intervention
            # max_clwv_ablation, argmax_clwv_ablation = clwv_ablation.max(), clwv_ablation.argmax()
            # max_clwv_intervention, argmax_clwv_intervention = clwv_intervention.max(), clwv_intervention.argmax()
            # to_plot = [wrong_preds_intervention, clwv_intervention, wrong_preds_ablation, clwv_ablation]
            # names = ['mirroring error rate', 'mirroring CLWV', 'ablation error rate', 'ablation CLWV']
            # self.plot_by_plt(num_ablated, to_plot, ranking, True)
            # continue
            # title = ' '.join([self.language, self.attribute, 'layer', str(self.layer), ranking])
            # # title = ''
            # output_file(title)
            # p = figure(x_range=num_ablated, plot_height=500, plot_width=1000, toolbar_location=None)
            # p.xaxis.major_label_orientation = 1.2
            # legend_items = []
            # p.line(num_ablated, wrong_preds_ablation, color=self.line_colors[0])
            # # p.square(num_ablated, wrong_preds_ablation, fill_color=self.line_colors[0], line_color=self.line_colors[0])
            # legend_items.append(('ablation error rate',
            #                      [p.line(color=self.line_colors[0])]))
            # p.line(num_ablated, clwv_ablation, color=self.line_colors[1])
            # # p.square(num_ablated, clwv_ablation, self.line_colors[1], line_color=self.line_colors[1])
            # legend_items.append(('ablation CLWV',
            #                      [p.line(color=self.line_colors[1])]))
            # p.line(num_ablated, wrong_preds_intervention, color=self.line_colors[2])
            # # p.circle(num_ablated, wrong_preds_intervention, fill_color=self.line_colors[0], line_color=self.line_colors[0])
            # legend_items.append(('mirroring error rate',
            #                      [p.line(color=self.line_colors[2])]))
            # p.line(num_ablated, clwv_intervention, color=self.line_colors[3])
            # # p.circle(num_ablated, clwv_intervention, fill_color=self.line_colors[1], line_color=self.line_colors[1])
            # legend_items.append(('mirroring CLWV',
            #                      [p.line(color=self.line_colors[3])]))
            # legend = Legend(items=legend_items, location='center')
            # legend.orientation = 'horizontal'
            # legend.label_text_font_size = '16px'
            # p.add_layout(legend, 'above')
            # p.xgrid.grid_line_color = None
            # p.outline_line_color = None
            # # show(p)
            # save(p, filename=Path(self.root_path, 'figs', ranking + '_combined.html'))
            # # export_png(p, filename=Path(self.root_path, 'figs', ranking + 'combined' + '.png').__str__())
        num_ablated = [str(r[0]) for r in self.res['by top avg_intervention_10_8.0_lnspace']['wrong words']]
        self.plot_by_plt(num_ablated, all_rankings_res, f'alpha_{alpha}', False)
        max_points = {ranking: self.find_knee_point(stats[1], 1.05, 0.1, num_ablated) for ranking, stats in all_rankings_res.items()}
        return max_points

    def find_knee_point(self, stats:np.array, k1, k2, x_ticks):
        for i in range(len(stats)-2):
            improvement1, improvement2 = stats[i+1] / stats[i], stats[i+2] / stats[i]
            if improvement1 < k1 and improvement2 < k1:
                if stats[-1] - stats[i] > k2:
                    continue
                return {'max': stats[i], 'argmax': x_ticks[i]}
        return {'max': stats[-1], 'argmax': x_ticks[len(stats)-1]}

    def plot_by_plt(self, x_ticks, stats, ranking, within_ranking):
        # if ranking != 'by top cluster':
        #     return
        # plt.figure(figsize=[12., 4.8])
        plt.figure(figsize=[6.8, 5.4])
        ax = plt.subplot(111)
        legend_labels, legend_lines = [], []
        # colors = {name: color for name, color in zip(names, self.line_colors)}
        colors = self.line_colors_one_ranking if within_ranking else self.line_colors_all_rankings
        for name, (wrong_words, clwv) in stats.items():
            # line, = ax.plot(x_ticks[:len(wrong_words)], wrong_words, color=colors[name], label=name,
            #                 linestyle=self.linestyles['error rate'])
            end_ticks = min(31, len(wrong_words))
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
        loc = plticker.MultipleLocator(base=5.0)
        ax.xaxis.set_major_locator(loc)
        # ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.tick_params(axis='both', which='major', labelsize=16)
        legend_labels = ['by ttb Linear', 'by ttb Probeless', 'by ttb Gaussian', 'by random',
                         'by btt Linear', 'by btt Probeless', 'by btt Gaussian']
        ax.legend(legend_lines, legend_labels, ncol=3, loc='center left', bbox_to_anchor=(-0.145, 1.12), fontsize=12.5)
        plt.tight_layout()
        plt.savefig(Path(self.root_path, 'figs', ranking + '_combined.png'))
        plt.close()

def plt_smt(lan, att, layer, absolute: bool):
    root_path = Path('results','UM', lan, att, 'layer ' + str(layer), 'spacy')
    wrong_words_path = Path(root_path,'wrong words')
    correct_lemmas_path = Path(root_path,'correct lemmas')
    kept_att_path = Path(root_path, 'kept attribute')
    correct_val_path = Path(root_path, 'correct val')
    c_lemma_c_val_path = Path(root_path, 'c lemmas c val')
    c_lemma_w_val_path = Path(root_path, 'c lemmas w val')
    w_lemma_c_val_path = Path(root_path, 'w lemmas c val')
    w_lemma_w_val_path = Path(root_path, 'w lemmas w val')
    res = {}
    rankings = ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi',
                    'by top cluster', 'by bottom cluster', 'by random']
    rankings += [r+'_intervention' for r in rankings]
    for ranking in rankings:
        colors = ["lightslategrey", "cornflowerblue", 'lightgreen', 'khaki']
        if not Path(wrong_words_path, ranking).exists():
            continue
        with open(Path(wrong_words_path, ranking),'rb') as f:
            wrong_words_res = pickle.load(f)
        with open(Path(correct_lemmas_path, ranking),'rb') as f:
            correct_lemmas_res = pickle.load(f)
        with open(Path(kept_att_path, ranking),'rb') as f:
            kept_att_res = pickle.load(f)
        with open(Path(correct_val_path, ranking), 'rb') as f:
            correct_val_res = pickle.load(f)
        with open(Path(c_lemma_c_val_path, ranking), 'rb') as f:
            c_lemma_c_val_res = pickle.load(f)
        with open(Path(c_lemma_w_val_path, ranking), 'rb') as f:
            c_lemma_w_val_res = pickle.load(f)
        with open(Path(w_lemma_c_val_path, ranking), 'rb') as f:
            w_lemma_c_val_res = pickle.load(f)
        with open(Path(w_lemma_w_val_path, ranking), 'rb') as f:
            w_lemma_w_val_res = pickle.load(f)
        # metrics = ['wrong preds', 'correct lemmas', 'kept attribute', 'correct values']
        metrics = ['correct lemma, correct value', 'correct lemma, wrong value',
                   'wrong lemma, correct value', 'wrong lemma, wrong value']
        num_ablated = [str(r[0]) for r in wrong_words_res]
        wrong_preds = np.array([r[1] for r in wrong_words_res])
        if max(wrong_preds) < 0.05:
            continue
        # booleans = [True if wp >= 0.05 else False for wp in wrong_preds]
        booleans = [True for wp in wrong_preds]
        start_point = booleans.index(True)
        if not absolute:
            wrong_preds = np.ones_like(wrong_preds)
        # data = {'ablated': num_ablated,
        #         'wrong preds': [r[1] for r in wrong_words_res],
        #         'correct lemmas': [r[1] for r in correct_lemmas_res],
        #         'kept attribute': [r[1] for r in kept_att_res],
        #         'correct values': [r[1] for r in correct_val_res]}
        data = {'ablated': num_ablated,
                'correct lemma, correct value': [r[1] for r in c_lemma_c_val_res] * wrong_preds,
                'correct lemma, wrong value': [r[1] for r in c_lemma_w_val_res] * wrong_preds,
                'wrong lemma, correct value': [r[1] for r in w_lemma_c_val_res] * wrong_preds,
                'wrong lemma, wrong value': [r[1] for r in w_lemma_w_val_res] * wrong_preds}
        # correct_lemmas_sum = np.array([clcv + clwv for i, (clcv, clwv) in enumerate(zip(
        #     data['correct lemma, correct value'], data['correct lemma, wrong value'])) if booleans[i]])
        clwv = np.array(data['correct lemma, wrong value'])
        max_clwv, argmax_clwv = clwv[booleans].max(), clwv[booleans].argmax()
        argmax_clwv = data['ablated'][start_point + argmax_clwv]
        label = Label(text=f'max correct lemma wrong value: {max_clwv:.2f}, argmax: {argmax_clwv}', x=590, y=370, x_units='screen', y_units='screen',
                      render_mode='css', border_line_color='black',
                      border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=1.0)
        data = ColumnDataSource(data)

        view = CDSView(source=data, filters=[BooleanFilter(booleans)])
        title = ' '.join([lan, att, 'layer', str(layer), ranking])
        title = ''
        output_file(title)
        x_range = [num for i, num in enumerate(num_ablated) if booleans[i]]
        tooltips = "$name @ablated: @$name{(0.0000)}"
        tooltips = ''
        # p = figure(x_range=x_range, plot_height=500, plot_width=1000, title=title,
        #            toolbar_location=None, tools="hover", tooltips=tooltips)
        p = figure(x_range=x_range, plot_height=500, plot_width=1000, toolbar_location=None)
        r_list = [p.square(fill_color=c, line_color=c) for c in colors]
        p.xaxis.major_label_orientation = 1.2
        # p.vbar_stack(metrics, x='ablated', width=0.9, source=data, color=colors, legend_label=metrics, view=view)
        p.vbar_stack(metrics, x='ablated', width=0.9, source=data, color=colors, view=view)
        legend = Legend(items=[(m,[r]) for m,r in zip(metrics,r_list)], location='center')
        legend.orientation = 'horizontal'
        legend.label_text_font_size = '16px'
        p.add_layout(legend, 'above')

        # p.vbar(x=x, top=y, width=0.9, color="red")
        p.xgrid.grid_line_color = None
        # p.y_range.start = 0
        # p.yaxis.visible = False
        p.outline_line_color = None
        # p.legend.orientation = "horizontal"
        # show(p)
        abs_str = ' absolute' if absolute else ' normalized'
        save(p, filename=Path(root_path, 'figs', ranking + abs_str + '.html'))
        # export_png(p, filename=Path(root_path, 'figs', ranking + abs_str + '.png').__str__())
        res[ranking] = {'max': max_clwv, 'argmax': argmax_clwv}
    return res


def run_all(lan, absolute):
    lan_root_path = Path('results', 'UM', lan)
    atts_path = [p for p in lan_root_path.glob('*') if not p.is_file()]
    res = {}
    for att_path in atts_path:
        if att_path.name == 'Part of Speech':
            continue
        res[att_path.name] = {}
        for layer in [2, 7, 12]:
            res[att_path.name][layer] = plt_smt(lan, att_path.name, layer, absolute)
    return res

def create_dataset(model_type, languages, alpha):
    other_model = 'bert' if model_type == 'xlm' else 'xlm'
    attributes = ['Number', 'Tense', 'Gender and Noun Class']
    layers = [2, 7, 12]
    rankings = ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi',
                'by top cluster', 'by bottom cluster', 'by random']
    # rankings += [r + '_intervention' for r in rankings]
    # ratios = ['absolute', 'normalized']
    # ratios = ['absolute']
    metrics = [f'max_{model_type}', f'argmax_{model_type}']
    metrics_other = [f'max_{other_model}', f'argmax_{other_model}']
    cols = pd.MultiIndex.from_product([languages, attributes, layers, rankings])
    rows = pd.MultiIndex.from_product([metrics])
    rows_other = pd.MultiIndex.from_product([metrics_other])
    df = pd.DataFrame(index=rows, columns=cols).sort_index().sort_index(axis=1)
    with open(Path('results', 'UM', model_type, f'max_clwv_lnscale_alpha_{alpha}.pkl'), 'rb') as f:
        res = pickle.load(f)

    with open(Path('results', 'UM', other_model, f'max_clwv_lnscale_alpha_{alpha}.pkl'), 'rb') as f:
        res_other = pickle.load(f)
    df_other = pd.DataFrame(index=rows_other, columns=cols).sort_index().sort_index(axis=1)
    for lan, l_res in res.items():
        # for ratio, ratio_res in l_res.items():
        for att, a_res in l_res.items():
            if att == 'Part of Speech':
                continue
            for layer, layer_res in a_res.items():
                for ranking, r_res in layer_res.items():
                    df[(lan, att, layer, ranking)][f'max_{model_type}'] = round(r_res['max'],2)
                    df[(lan, att, layer, ranking)][f'argmax_{model_type}'] = r_res['argmax']
                    # df[(lan, att, layer, ranking)][(ratio, 'argmax')] = r_res['argmax']
    for lan, l_res in res_other.items():
        # for ratio, ratio_res in l_res.items():
        for att, a_res in l_res.items():
            if att == 'Part of Speech':
                continue
            for layer, layer_res in a_res.items():
                for ranking, r_res in layer_res.items():
                    df_other[(lan, att, layer, ranking)][f'max_{other_model}'] = round(r_res['max'],2)
                    df_other[(lan, att, layer, ranking)][f'argmax_{other_model}'] = r_res['argmax']
    idx = pd.IndexSlice
    # for lang in languages:
    #     print(f'{lang}:')
    #     for ranking in rankings:
    #         relevant_data = df.loc[idx['absolute', 'max, argmax'], idx[[lang], attributes, layers, [ranking]]]
    #         # print(f'{ranking}:')
    #         # print(round(relevant_data.mean(), 2), round(relevant_data.std(), 2))

    # for ranking in rankings:
    #     relevant_data = df.loc[idx['absolute', 'max, argmax'], idx[languages, attributes, layers, [ranking]]]
    #     print(f'{ranking}:')
    #     print(relevant_data)
    #     # print('avg:')
    #     # print(round(relevant_data.mean(), 2), round(relevant_data.std(), 2))


    for lan in languages:
        for att in attributes:
            relevant_data = df.loc[idx[[f'max_{model_type}', f'argmax_{model_type}']], idx[[lan], [att], layers, rankings]]
            relevant_data_other = df_other.loc[idx[[f'max_{other_model}', f'argmax_{other_model}']], idx[[lan], [att], layers, rankings]]
            conc = pd.concat([relevant_data, relevant_data_other])
            with open(Path('results', 'UM', 'interventions_comp', str(alpha), f'{lan}_{att}.csv'),'wb+') as f:
                conc.to_csv(f)
            print(f'{att}:')
            print('absolute:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(relevant_data)
            # relevant_data = df.loc[idx['normalized', 'max, argmax'], idx[[lan], [att], layers, rankings]]
            # print('normalized:')
            # print(relevant_data)
            # print('avg:')
            # print(round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
    # print('all:')
    # print(df)

def plot_and_dump(langs):
    res = {}
    for lan in langs:
        res[lan] = {}
        for absolute in [True, False]:
            res[lan]['absolute' if absolute else 'normalized'] = run_all(lan, absolute)
    with open(Path('results', 'UM', 'max_clwv.pkl'),'wb+') as f:
        pickle.dump(res, f)
    return res

def new_plot_and_dump(model_type, set_type, langs, alpha, dump=False):
    res = {}
    for lan in langs:
        res[lan] = {}
        lan_root_path = Path('results', 'UM', model_type, lan)
        atts_path = [p for p in lan_root_path.glob('*') if not p.is_file()]
        for att_path in atts_path:
            if att_path.name == 'Part of Speech':
                continue
            res[lan][att_path.name] = {}
            # if att_path.name != 'Tense':
            #     continue
            # for layer in [2, 7, 12]:
            for layer in [1]:
                bok = bokehPlots(model_type, set_type, lan, att_path.name, layer)
                bok.load_data()
                res[lan][att_path.name][layer] = bok.plot_line(alpha)
    if dump:
        with open(Path('results', 'UM', model_type, f'max_clwv_lnscale_alpha_{alpha}.pkl'),'wb+') as f:
            pickle.dump(res, f)

if __name__ == "__main__":
    # languages = ['eng', 'spa', 'fra']
    model_type = 'bert'
    set_type = 'test'
    languages = ['eng', 'fra', 'spa']
    # plot_and_dump(languages)
    # create_dataset(languages)
    # for alpha in [2, 4, 6, 8, 10, 12]:
    for alpha in [8]:
        new_plot_and_dump(model_type, set_type, languages, alpha, dump=False)
    # for alpha in [2, 4, 8]:
    for alpha in [8]:
        print(f'alpha: {alpha}')
        create_dataset(model_type, languages, alpha)



