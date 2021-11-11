# On the Pitfalls of Analyzing Individual Neurons in Language Models

Code for reproducing results from the paper [On the Pitfalls of Analyzing Individual Neurons](https://arxiv.org/abs/2110.07483), 
named "Are All Neurons Created Equal? Interpreting and Controlling BERT through Individual Neurons" in a previous version.

## Setup

1. Install the required libraries by running `pip install -r requirements.txt`.
2. Download the datasets of the languages you would like to experiment with from the [Universal Dependencies site](https://universaldependencies.org/) and place them in `data/UD`.
*The supported languages are the ones we worked with in the paper. If you would like to experiment with a different one, you should explicilty add the relevant paths in consts.py.*
3. Clone the modified [UD converter](https://github.com/ltorroba/ud-compatibility) to this repository's folder.
4. Create an empty directory `data/UM`.
5. Run `ud_to_um.sh`.
This will place converted UM data files in the directory.
6. Run `python parsing.py -model MODEL -language LANGUAGE` where MODEL can be 'bert' or 'xlm' and LANGUAGE is a three-letter language code.
This will dump some parsed data files to be used by the experiment files.

## Probing Experiments

1. Run `python LinearWholeVector.py -model MODEL -language LANGUAGE -attribute ATT -layer LAYER`. 
This will fit a linear classifier on the desired config (language, attribute, layer) to obtain the Linear ranking, as well as dump some more necessary files.
2. Obtain the Probeless ranking by running `python Probeless -model MODEL -language LANGUAGE -attribute ATT -layer LAYER`. 
3. To test the Gaussian classifier, run `python -model MODEL -language LANGUAGE -attribute ATT -layer LAYER -ranking RANKING`, where RANKING can be one of:
`'ttb gaussian', 'btt gaussian', 'ttb linear', 'btt linear', 'ttb probeless', 'btt probeless', 'random'`. 
Choosing 'ttb gaussian' here would obtain the gaussian ranking, but note the run time will be significantly longer.
4. To test the Linear classifier, run `python LinearSubset.py -model MODEL -language -LANGUAGE -attribute ATT -layer LAYER -ranking RANKING`, 
where Ranking can be any of the rankings from the previous step. 
5. To view a graph of your results, run `python analysis.py -experiments probing -model MODEL -language LANGUAGE -attribute ATT -layer LAYER`.
The matching graph will be placed at `results/UM/MODEL/LANGUAGE/ATTRIBUTE/LAYER/figs/`.

## Intervention Experiments

1. Produce any of the three ranking of your choosing by running the appropriate commands from 1-3 in the probing experiments.
*Note that you must create the Linear ranking (step 1) first in order to obtain any of the other rankings*.
2. Run `python interventions.py -model MODEL -language LANGUAGE -attribute ATT -layer LAYER -ranking RANKING --translation -beta BETA --scaled`.
This will execute interventions by the translation method with the specifier beta (in the paper we show results for beta=8).
Not setting the `--scaled` flag would apply the same coefficient, beta, to all neurons.
Not setting the `--translation` flag would apply ablation rather translation, and `beta` and `scaled` params will be ignored.
3. Run `python spacyParsing.py -model MODEL -language LANGUAGE -attribute ATT -layer LAYER -ranking RANKING --translation -beta BETA --scaled`.
This will parse the output from the experiment from the previous step using spaCy.
5. To view a graph of your results, run `python analysis.py -experiments interventions -model MODEL -language LANGUAGE -attribute ATT -layer LAYER -beta BETA --scaled`.
Here, if beta=0 it will be considered an ablation experiment, otherwise it is translation. 
The matching graph will be placed at `results/UM/MODEL/LANGUAGE/ATTRIBUTE/LAYER/spacy/test/figs`.

