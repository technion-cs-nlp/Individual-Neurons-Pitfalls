import torch
import consts
import numpy as np
import torch.distributions.multivariate_normal as mn
import utils
from tqdm import tqdm as progressbar
from linearCorrelationAnalysis import LCA
import pickle
from pathlib import Path
import time
from argparse import ArgumentParser
import sys

def get_ranking(args):
    func, path = args
    if path != None:
        rank = func(path)
    else:
        rank = func()
    return rank

class Bayesian():
    def __init__(self, layer, data_name, control=False, small_dataset = False, language='', attribute='POS'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.control = control
        self.language = language
        self.attribute = attribute
        small_dataset_str = '_small' if small_dataset else ''
        if data_name == 'UM':
            self.UM_data_prep()
        else:
            self.train_features = utils.load_obj('features_tensor_layer_'+str(self.layer)+small_dataset_str,
                                                 self.device,'train_',data_name).to(self.device)
            if self.control:
                self.train_labels = torch.tensor(utils.load_obj(
                    'control_tags'+small_dataset_str,self.device,'train_',data_name)).to(self.device)
            else:
                self.train_labels = utils.load_obj(
                    'labels_tensor'+small_dataset_str,self.device,'train_',data_name).to(self.device)
            self.dev_features = utils.load_obj('features_tensor_layer_'+str(self.layer)+small_dataset_str,
                                                 self.device,'dev_',data_name).to(self.device)
            if self.control:
                self.dev_labels = torch.tensor(utils.load_obj(
                    'control_tags'+small_dataset_str,self.device,'dev_',data_name)).to(self.device)
            else:
                self.dev_labels = utils.load_obj(
                    'labels_tensor'+small_dataset_str,self.device,'dev_',data_name).to(self.device)
            self.test_features = utils.load_obj(
                'features_tensor_layer_'+str(self.layer)+small_dataset_str,self.device,'test_',data_name).to(self.device)
            if self.control:
                self.test_labels = torch.tensor(utils.load_obj(
                    'control_tags'+small_dataset_str,self.device,'test_',data_name)).to(self.device)
            else:
                self.test_labels = utils.load_obj(
                    'labels_tensor'+small_dataset_str,self.device,'test_',data_name).to(self.device)
        self.labels_dim = len(set(self.train_labels.tolist()))
        self.get_categorical()
        self.get_mean_and_cov()
        self.features_sets = {'train': self.train_features, 'dev': self.dev_features, 'test': self.test_features}
        self.labels_sets = {'train': self.train_labels, 'dev': self.dev_labels, 'test': self.test_labels}
        self.categorical_sets = {'train': self.train_categorical, 'dev': self.dev_categorical, 'test': self.test_categorical}

    def UM_data_prep(self):
        self.train_features, self.dev_features, self.test_features = [], [], []
        self.train_labels, self.dev_labels, self.test_labels = [], [], []
        att_path = Path('pickles', 'UM', self.language, self.attribute)
        label_to_idx_file = Path(att_path, 'label_to_idx.pkl')
        with open(label_to_idx_file, 'rb') as f:
            label_to_idx = pickle.load(f)
        for set_name, set_features, set_labels in zip(['train_','dev_','test_'],
                                                      [self.train_features,self.dev_features,self.test_features],
                                                      [self.train_labels,self.dev_labels,self.test_labels]):
            parsed_data_path = Path('pickles', 'UM', self.language, set_name+'parsed.pkl')
            with open(parsed_data_path, 'rb') as f:
                parsed_data = pickle.load(f)
            with open(Path(att_path, 'values_to_ignore.pkl'),'rb') as f:
                values_to_ignore = pickle.load(f)
            control_labels_path = Path(att_path, set_name+'control_labels')
            with open(control_labels_path,'rb') as f:
                control_labels = pickle.load(f)
            for word in parsed_data:
                if not word['attributes'].get(self.attribute):
                    continue
                if word['attributes'][self.attribute] in values_to_ignore:
                    continue
                if self.control:
                    label = control_labels[word['word']]
                else:
                    label = label_to_idx[word['attributes'][self.attribute]]
                emb = torch.tensor(word['embedding'][self.layer])
                set_features.append(emb)
                set_labels.append(label)

        self.train_features = torch.stack(self.train_features).to(self.device)
        self.dev_features = torch.stack(self.dev_features).to(self.device)
        self.test_features = torch.stack(self.test_features).to(self.device)
        self.train_labels = torch.tensor(self.train_labels).to(self.device)
        self.dev_labels = torch.tensor(self.dev_labels).to(self.device)
        self.test_labels = torch.tensor(self.test_labels).to(self.device)

    def get_categorical(self):
        print('computing categorical distribution')
        train_counts = torch.histc(self.train_labels.float(), bins=self.labels_dim)
        self.train_categorical = (train_counts / self.train_labels.size()[0]).to(self.device)
        dev_counts = torch.histc(self.dev_labels.float(), bins=self.labels_dim)
        self.dev_categorical = (dev_counts / self.dev_labels.size()[0]).to(self.device)
        test_counts = torch.histc(self.test_labels.float(), bins=self.labels_dim)
        self.test_categorical = (test_counts / self.test_labels.size()[0]).to(self.device)

    def get_mean_and_cov(self):
        print('computing mean and covariance matrix for multivariate normal distribution')
        # compute mean and cov matrix for each label
        self.features_by_label = [self.train_features[(self.train_labels == label).nonzero(as_tuple=False)].squeeze(1)
                              for label in range(self.labels_dim)]
        empirical_means = torch.stack([features.mean(dim=0) for features in self.features_by_label])
        empirical_covs = [torch.tensor(np.cov(features.cpu(), rowvar=False)) for features in self.features_by_label]
        ###diagonalize covs, use hyperparams from paper
        mu_0 = empirical_means  # [44,768]
        # lambda_0 is eigenvalues of sigma
        # lambda_0 = torch.stack([torch.diag(torch.eig(cov).eigenvalues[:,0]) for cov in empirical_covs]) #[44,768,768]
        # lambda_0 is diagonal of sigma
        lambda_0 = torch.stack([torch.diag(torch.diagonal(cov)) for cov in empirical_covs]).to(self.device)
        v_0 = torch.tensor(self.train_features.shape[1] + 2).to(self.device)  # int
        k_0 = torch.tensor(0.01).to(self.device)
        N_v = torch.tensor([features.shape[0] for features in self.features_by_label]).to(self.device)  # [44]
        k_n = k_0 + N_v  # [44]
        v_n = v_0 + N_v  # [44]
        mu_n = (k_0 * mu_0 + N_v.unsqueeze(1) * empirical_means) / k_n.unsqueeze(1)  # [44,768]
        S=[]
        for label in range(self.labels_dim):
            features_minus_mean = self.features_by_label[label] - empirical_means[label]
            S.append(features_minus_mean.T @ features_minus_mean)
        S = torch.stack(S).to(self.device)
        lambda_n = lambda_0 + S
        self.mu_star = mu_n
        sigma_star = lambda_n / (v_n + self.train_features.shape[1] + 2).view(self.labels_dim,1,1)
        #sigma star may be not PSD because of floating point errors so we add small values to the diagonal
        min_eig = []
        for sigma in sigma_star:
            eigs = torch.eig(sigma).eigenvalues[:,0]
            min_eig.append(eigs.min())
        min_eig = torch.tensor(min_eig).to(self.device)
        sigma_star[min_eig < 0] -= min_eig.view(sigma_star.shape[0], 1, 1)[min_eig < 0] * \
                                   torch.eye(sigma_star.shape[1]).to(self.device) * torch.tensor(10).to(self.device)
        self.sigma_star = sigma_star

    def get_distributions(self, selected_features):

        ############################
        # changed for conversion of penn to ud
        # self.distributions = [mn.MultivariateNormal(
        #     self.mu_star[label,selected_features].double(),
        #     self.sigma_star[label,selected_features][:,selected_features])
        #     for label in range(constants.LABEL_DIM)]

        self.distributions = []
        for label in range(self.labels_dim):
            if self.train_categorical[label].item() == 0:
                self.distributions.append(torch.distributions.normal.Normal(0.,0.))
            else:
                self.distributions.append(mn.MultivariateNormal(
                    self.mu_star[label, selected_features].double(),
                    self.sigma_star[label, selected_features][:, selected_features]
                ))


    def compute_probs(self, selected_features, set_name: str):
        features = self.features_sets[set_name]
        with torch.no_grad():
            ###################
            # changed for conversion of penn to ud
            log_probs = []
            for i in range(self.labels_dim):
                if self.train_categorical[i] == 0:
                    log_probs.append(torch.zeros(features.shape[0],dtype=torch.double))
                else:
                    log_probs.append(self.distributions[i].log_prob(features[:, selected_features]))
            log_probs = torch.stack(log_probs, dim=1)
            ##########################
            # log_probs = torch.stack([dist.log_prob(features[:, selected_features])
            #                          for dist in self.distributions],dim=1).to(self.device).double()
            log_prob_times_cat = log_probs + self.train_categorical.log()
            self.not_normalized_probs = log_prob_times_cat
            self.normalizer = log_prob_times_cat.logsumexp(dim=1)

        self.probs = log_prob_times_cat - self.normalizer.unsqueeze(1)

    def predict(self, set_name: str):
        preds = self.probs.argmax(dim=1)
        # preds = torch.tensor([self.original_tag_to_new_tag[pred.item()] for pred in preds])
        labels = self.labels_sets[set_name]
        # labels = self.train_labels if train else self.test_labels
        accuracy = ((preds == labels)).nonzero(as_tuple=False).shape[0] / labels.shape[0]
        categorical = self.categorical_sets[set_name]
        # categorical = self.train_categorical if train else self.test_categorical
        entropy = torch.distributions.Categorical(categorical).entropy()
        # labels_in_new_tags = torch.tensor([self.new_tag_to_original_tag[label.item()] for label in labels])
        with torch.no_grad():
            conditional_entropy = -self.probs[list(range(self.probs.shape[0])),labels].mean()
        mutual_inf = (entropy - conditional_entropy) / torch.tensor(2.0).log()
        return accuracy, mutual_inf.item(), (mutual_inf / entropy).item()

def greedy_selection(bayes:Bayesian, by_mi:bool, by_best=True):
    selected_neurons = []
    for num_of_neurons in progressbar(range(len(selected_neurons), 500)):
    # for num_of_neurons in range(len(selected_neurons), constants.SUBSET_SIZE):
        best_neuron = -1
        best_acc = 0.
        if by_best:
            best_mi, best_nmi = float('-inf'), float('-inf')
        else:
            best_mi, best_nmi = float('inf'), float('inf')
        acc_on_best_mi = 0.
        mi_on_best_acc, nmi_on_best_acc = 0., 0.
        print('using ', num_of_neurons + 1, 'neurons')
        start = time.time()
        for neuron in range(consts.BERT_OUTPUT_DIM):
            if neuron in selected_neurons:
                continue
            bayes.get_distributions(selected_neurons + [neuron])
            bayes.compute_probs(selected_neurons + [neuron], 'dev')
            acc, mi, nmi = bayes.predict('dev')
            if by_mi:
                if by_best:
                    if mi > best_mi:
                        best_mi = mi
                        best_nmi = nmi
                        best_neuron = neuron
                        acc_on_best_mi = acc
                else:
                    if mi < best_mi:
                        best_mi = mi
                        best_nmi = nmi
                        best_neuron = neuron
                        acc_on_best_mi = acc
            else:
                if acc > best_acc:
                    best_acc = acc
                    mi_on_best_acc = mi
                    nmi_on_best_acc = nmi
                    best_neuron = neuron
        selected_neurons.append(best_neuron)
        print('added neuron ', best_neuron)
        bayes.get_distributions(selected_neurons)
        bayes.compute_probs(selected_neurons, 'train')
        train_acc, train_mi, train_nmi = bayes.predict('train')
        print('accuracy on train set: ',train_acc)
        print('mi on train set: ',train_mi)
        print('nmi on train set: ', train_nmi)
        print('accuracy on dev set: ', acc_on_best_mi)
        print('mi on dev set: ', best_mi)
        print('nmi on dev set: ', best_nmi)
        bayes.compute_probs(selected_neurons, 'test')
        test_acc, test_mi, test_nmi = bayes.predict('test')
        print('final accuracy on test: ',test_acc)
        print('mi on test: ',test_mi)
        print('nmi on test: ',test_nmi)
        print('selected_neurons: ', selected_neurons)
        print('time for iteration: {} seconds'.format(time.time()-start))
    print('selected neurons: ', selected_neurons)

def run_bayes_on_subset(bayes:Bayesian, neurons):
    for i in range(1, consts.SUBSET_SIZE + 1):
        print('using ',i, ' neurons')
        bayes.get_distributions(neurons[:i])
        bayes.compute_probs(neurons[:i], 'train')
        train_acc, train_mi, train_nmi = bayes.predict('train')
        bayes.compute_probs(neurons[:i],'dev')
        dev_acc, dev_mi, dev_nmi = bayes.predict('dev')
        bayes.compute_probs(neurons[:i], 'test')
        test_acc, test_mi, test_nmi = bayes.predict('test')
        print('mi on train set: ', train_mi)
        print('nmi on train set: ', train_nmi)
        print('accuracy on train set: ', train_acc)
        print('mi on dev set: ', dev_mi)
        print('nmi on dev set: ', dev_nmi)
        print('accuracy on dev set: ', dev_acc)
        print('mi on test: ', test_mi)
        print('nmi on test: ', test_nmi)
        print('final accuracy on test:', test_acc)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-language', type=str)
    parser.add_argument('-attribute', type=str)
    parser.add_argument('-layer', type=int)
    parser.add_argument('-ranking', type=str)
    parser.add_argument('--control', default=False, action='store_true')
    args = parser.parse_args()
    language = args.language
    layer = args.layer
    attribute = args.attribute
    ranking = args.ranking
    control = args.control
    small_dataset = False
    control_str = '_control' if control else ''
    small_dataset_str = '_small' if small_dataset else ''
    data_name = 'UM'
    greedy = True if ranking.startswith('greedy') else False
    res_file_dir = Path('results', data_name, language, attribute, 'layer ' + str(layer))
    if not res_file_dir.exists():
        res_file_dir.mkdir(parents=True, exist_ok=True)
    linear_model_path = Path('pickles', data_name, language, attribute,
                             'best_model_whole_vector_layer_' + str(layer) + control_str + small_dataset_str)
    bayes_res_path = Path(res_file_dir,'bayes by bayes mi'+control_str)
    worst_bayes_res_path = Path(res_file_dir, 'bayes by worst mi'+control_str)
    bayes = Bayesian(layer=layer, control=control, small_dataset=small_dataset, data_name=data_name,
                     language=language, attribute=attribute)

    if ranking == 'greedy best':
        res_suffix = 'bayes mi'
    elif ranking == 'greedy worst':
        res_suffix = 'worst mi'
    else:
        res_suffix = ranking
    res_file_name = 'bayes by ' + res_suffix + control_str
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
        sys.stdout = f
        print('layer: ', layer)
        print('control: ', control)
        print('small: ', small_dataset)
        print('language: ', language)
        print('attribute: ', attribute)
        if greedy:
            by_best = True if ranking.endswith('best') else False
            print('by best: ', by_best)
            greedy_selection(bayes, by_mi=True, by_best=by_best)
        else:
            ranking_params = {'top avg': (utils.sort_neurons_by_avg_weights, linear_model_path),
                              'bottom avg': (utils.sort_neurons_by_avg_weights, linear_model_path),
                              'bayes mi': (utils.sort_neurons_by_bayes_mi, bayes_res_path),
                              'worst mi': (utils.sort_neurons_by_bayes_mi, worst_bayes_res_path),
                              'random': (utils.sort_neurons_by_random, None)}
            neurons_list = get_ranking(ranking_params[ranking])
            if ranking == 'bottom avg':
                neurons_list = list(reversed(neurons_list))
            run_bayes_on_subset(bayes, neurons_list)

