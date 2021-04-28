from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import CDSView, BooleanFilter, ColumnDataSource
import pickle
from pathlib import Path
import numpy as np

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
    for ranking in ['by top avg', 'by bottom avg', 'by bayes mi', 'by worst mi', 'by random']:
        colors = ["lightslategrey", "cornflowerblue", 'lightgreen', 'khaki']
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
        booleans = [True if wp >= 0.1 else False for wp in wrong_preds]
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
        data = ColumnDataSource(data)

        view = CDSView(source=data, filters=[BooleanFilter(booleans)])
        title = ' '.join([lan, att, 'layer', str(layer), ranking])
        output_file(title)
        x_range = [num for i, num in enumerate(num_ablated) if booleans[i]]
        p = figure(x_range=x_range, plot_height=500, plot_width=1000, title=title,
                   toolbar_location=None, tools="hover", tooltips="$name @ablated: @$name{(0.0000)}")
        p.xaxis.major_label_orientation = 1.2
        p.vbar_stack(metrics, x='ablated', width=0.9, source=data, color=colors, legend_label=metrics, view=view)


        # p.vbar(x=x, top=y, width=0.9, color="red")
        p.xgrid.grid_line_color = None
        # p.y_range.start = 0
        # p.yaxis.visible = False
        p.outline_line_color = None
        p.legend.orientation = "horizontal"
        # show(p)
        abs_str = ' absolute' if absolute else ' normalized'
        save(p, filename=Path(root_path, 'figs', ranking + abs_str + '.html'))

def run_all(lan, absolute):
    lan_root_path = Path('results', 'UM', lan)
    atts_path = [p for p in lan_root_path.glob('*') if not p.is_file()]
    for att_path in atts_path:
        for layer in [2, 7, 12]:
            plt_smt(lan, att_path.parts[-1], layer, absolute)

if __name__ == "__main__":
    languages = ['eng','rus']
    for lan in languages:
        for absolute in [True, False]:
            run_all(lan, absolute)
