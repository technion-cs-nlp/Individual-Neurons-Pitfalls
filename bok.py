from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import CDSView, BooleanFilter, ColumnDataSource
import pickle
from pathlib import Path

def plt_smt(lan, att, layer):
    root_path = Path('results','UM', lan, att, 'layer ' + str(layer), 'spacy')
    wrong_words_path = Path(root_path,'wrong words')
    correct_lemmas_path = Path(root_path,'correct lemmas')
    kept_att_path = Path(root_path, 'kept attribute')
    correct_val_path = Path(root_path, 'correct val')
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
        metrics = ['wrong preds', 'correct lemmas', 'kept attribute', 'correct values']
        num_ablated = [str(r[0]) for r in wrong_words_res]
        data = {'ablated': num_ablated,
                'wrong preds': [r[1] for r in wrong_words_res],
                'correct lemmas': [r[1] for r in correct_lemmas_res],
                'kept attribute': [r[1] for r in kept_att_res],
                'correct values': [r[1] for r in correct_val_res]}
        data = ColumnDataSource(data)
        booleans = [True if wp >= 0.1 else False for wp in data.data['wrong preds']]
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
        p.yaxis.visible = False
        p.outline_line_color = None
        p.legend.orientation = "horizontal"
        # show(p)
        save(p, filename=Path(root_path, 'figs', ranking + '.html'))

def run_all(lan):
    lan_root_path = Path('results', 'UM', lan)
    atts_path = [p for p in lan_root_path.glob('*') if not p.is_file()]
    for att_path in atts_path:
        for layer in [2, 7, 12]:
            plt_smt(lan, att_path.parts[-1], layer)

if __name__ == "__main__":
    lan = 'eng'
    run_all(lan)
