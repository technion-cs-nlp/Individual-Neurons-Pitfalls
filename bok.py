from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import CDSView, BooleanFilter, ColumnDataSource, Label
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

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
    for ranking in ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi', 'by random',
                    'by top cluster', 'by bottom cluster', 'by top cluster_intervention', 'by bottom cluster_intervention']:
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
        booleans = [True if wp >= 0.05 else False for wp in wrong_preds]
        start_point = booleans.index(True)
        if absolute:
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
        output_file(title)
        x_range = [num for i, num in enumerate(num_ablated) if booleans[i]]
        p = figure(x_range=x_range, plot_height=500, plot_width=1000, title=title,
                   toolbar_location=None, tools="hover", tooltips="$name @ablated: @$name{(0.0000)}")
        p.xaxis.major_label_orientation = 1.2
        p.vbar_stack(metrics, x='ablated', width=0.9, source=data, color=colors, legend_label=metrics, view=view)
        p.add_layout(label)

        # p.vbar(x=x, top=y, width=0.9, color="red")
        p.xgrid.grid_line_color = None
        # p.y_range.start = 0
        # p.yaxis.visible = False
        p.outline_line_color = None
        p.legend.orientation = "horizontal"
        # show(p)
        abs_str = ' absolute' if absolute else ' normalized'
        save(p, filename=Path(root_path, 'figs', ranking + abs_str + '.html'))
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

def create_dataset(languages):
    attributes = ['Number', 'Tense', 'Gender and Noun Class']
    layers = [2, 7, 12]
    rankings = ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi', 'by random',
                'by top cluster', 'by bottom cluster', 'by top cluster_intervention', 'by bottom cluster_intervention']
    ratios = ['absolute', 'normalized']
    metrics = ['max, argmax']
    cols = pd.MultiIndex.from_product([languages, attributes, layers, rankings])
    rows = pd.MultiIndex.from_product([ratios, metrics])
    df = pd.DataFrame(index=rows, columns=cols).sort_index().sort_index(axis=1)
    with open(Path('results', 'UM', 'max_clwv.pkl'), 'rb') as f:
        res = pickle.load(f)
    for lan, l_res in res.items():
        for ratio, ratio_res in l_res.items():
            for att, a_res in ratio_res.items():
                if att == 'Part of Speech':
                    continue
                for layer, layer_res in a_res.items():
                    for ranking, r_res in layer_res.items():
                        df[(lan, att, layer, ranking)][(ratio, 'max, argmax')] = round(r_res['max'],2), r_res['argmax']
                        # df[(lan, att, layer, ranking)][(ratio, 'argmax')] = r_res['argmax']
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
    for att in attributes:
        relevant_data = df.loc[idx['absolute', 'max, argmax'], idx[languages, [att], layers, rankings]]
        print(f'{att}:')
        print('absolute:')
        print(relevant_data)
        relevant_data = df.loc[idx['normalized', 'max, argmax'], idx[languages, [att], layers, rankings]]
        print('normalized:')
        print(relevant_data)
        # print('avg:')
        # print(round(relevant_data.mean(), 2), round(relevant_data.std(), 2))
    print('all:')
    print(df)

def plot_and_dump(langs):
    res = {}
    for lan in langs:
        res[lan] = {}
        for absolute in [True, False]:
            res[lan]['absolute' if absolute else 'normalized'] = run_all(lan, absolute)
    with open(Path('results', 'UM', 'max_clwv.pkl'),'wb+') as f:
        pickle.dump(res, f)
    return res

if __name__ == "__main__":
    languages = ['eng', 'spa']
    plot_and_dump(languages)
    create_dataset(languages)


