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

def greedy_selection(bayes:Bayesian, by_mi:bool, by_best=True, from_log=False):
    selected_neurons = []
    if from_log:
        logs = {2: {True: [654, 512, 92, 360, 321, 144, 712, 497, 430, 265, 98, 556, 147, 487, 213, 209, 685, 104, 320, 363, 703, 19, 495, 291, 327, 405, 377, 501, 623, 157, 510, 44, 462, 208, 159, 375, 563, 103, 515, 273, 698, 443, 96, 463, 402, 502, 646, 294, 605, 399, 545, 410, 529, 30, 207, 63, 132, 651, 178, 356, 176, 446, 124, 456, 429, 575, 332, 393, 165, 197, 283, 286, 1, 9, 573, 521, 439, 34, 656, 668, 470, 746, 518, 226, 672, 740, 509, 282, 205, 93, 384, 274, 534, 546, 51, 436, 408, 60, 733, 225, 263, 578, 522, 693, 749, 170, 709, 628, 284, 65, 73, 189, 695, 755, 691, 121, 452, 288, 448, 572, 171, 336, 116, 617, 391, 117, 652, 293, 682, 420, 650, 599, 174, 15, 514, 396, 642, 689, 423, 417, 218, 358, 700, 664],
                    False: [190, 188, 241, 325, 577, 315, 57, 709, 387, 724, 259, 374, 382, 469, 661, 310, 422, 47, 192, 252, 548, 136, 565, 288, 223, 762, 76, 28, 120, 677, 753, 377, 154, 285, 409, 49, 499, 68, 388, 690, 36, 240, 257, 452, 542, 180, 24, 567, 403, 413, 172, 84, 675, 705, 309, 271, 115, 559, 461, 369, 576, 362, 129, 638, 614, 528, 345, 251, 702, 228, 322, 719, 237, 135, 500, 624, 56, 734, 300, 7, 431, 699, 348, 511, 555, 74, 629, 125, 581, 175, 701, 204, 238, 652, 571, 367, 264, 186, 589, 301, 134, 508, 330, 681, 177, 200, 678, 662, 160, 725, 674, 128, 739, 471, 626, 554, 562, 606, 368, 353, 89, 411, 569, 334, 654, 53, 298, 99, 130, 506, 328, 80, 193, 31, 490, 520, 639, 71, 630, 102, 550, 660, 50, 758, 272]},
                7:{True: [145, 757, 305, 12, 685, 321, 41, 522, 234, 404, 298, 393, 267, 242, 101, 292, 307, 365, 662, 220, 317, 508, 689, 565, 408, 752, 52, 415, 342, 759, 640, 459, 359, 631, 734, 722, 245, 495, 262, 740, 202, 582, 225, 601, 224, 688, 165, 314, 633, 384, 186, 497, 504, 475, 30, 155, 250, 257, 518, 311, 320, 741, 196, 536, 74, 523, 627, 217, 478, 546, 679, 767, 665, 92, 98, 22, 43, 719, 549, 44, 556, 514, 103, 179, 620, 728, 680, 356, 231, 201, 414, 594, 18, 266, 547, 388, 737, 711, 531, 158, 709, 223, 750, 272, 391, 507, 88, 328, 51, 456, 193, 34, 608, 470, 400, 287, 485, 488, 647, 170, 425, 554, 732, 684, 230, 625, 286, 335, 443, 96, 297, 394, 584, 730, 592, 13, 48, 332, 708, 441, 67, 432, 446, 207],
                   False: [190, 576, 283, 705, 260, 494, 251, 411, 296, 427, 696, 575, 702, 691, 710, 319, 119, 8, 351, 529, 254, 527, 329, 469, 403, 492, 24, 367, 438, 386, 370, 744, 645, 236, 675, 347, 607, 611, 338, 81, 554, 247, 144, 457, 295, 128, 520, 33, 690, 323, 634, 293, 356, 154, 212, 751, 743, 600, 5, 676, 374, 309, 681, 671, 169, 256, 200, 235, 56, 651, 614, 294, 268, 75, 649, 493, 546, 604, 376, 765, 70, 87, 264, 588, 26, 159, 273, 198, 587, 129, 289, 120, 324, 342, 279, 158, 401, 454, 729, 670, 344, 541, 148, 172, 359, 688, 149, 85, 179, 544, 280, 486, 468, 701, 232, 725, 659, 297, 660, 177, 313, 348, 22, 175, 167, 434, 712, 153, 341, 23, 181, 657, 300, 271, 204, 543, 134, 609, 222, 446, 552, 224, 564, 723]},
                12:{True: [305, 321, 145, 101, 252, 267, 735, 762, 626, 722, 661, 18, 114, 685, 249, 214, 31, 479, 42, 684, 601, 397, 322, 580, 596, 719, 720, 430, 433, 563, 149, 646, 188, 627, 130, 759, 518, 550, 519, 423, 728, 410, 254, 659, 253, 66, 607, 578, 227, 439, 307, 320, 242, 110, 206, 292, 480, 456, 314, 454, 329, 675, 3, 408, 536, 379, 421, 734, 632, 582, 218, 11, 706, 514, 67, 339, 333, 38, 319, 612, 16, 633, 683, 286, 128, 517, 691, 462, 516, 718, 170, 119, 76, 207, 391, 505, 169, 275, 20, 651, 117, 682, 597, 613, 615, 724, 162, 561, 392, 443, 25, 396, 502, 389, 723, 236, 248, 556, 303, 19, 368, 638, 623, 449, 504, 246, 201, 140, 432, 703, 664, 196, 727, 239, 508, 259, 541, 736, 232, 277, 372, 606, 138, 171],
                    False: [754, 348, 198, 82, 77, 734, 24, 119, 610, 677, 284, 243, 148, 529, 190, 753, 761, 564, 230, 109, 68, 251, 469, 105, 334, 660, 364, 755, 588, 158, 552, 277, 53, 8, 78, 672, 414, 766, 534, 102, 293, 425, 163, 743, 270, 260, 545, 422, 528, 169, 338, 589, 399, 312, 600, 132, 706, 193, 538, 690, 322, 701, 225, 120, 670, 705, 209, 47, 278, 52, 501, 141, 63, 98, 704, 289, 168, 208, 288, 220, 467, 17, 217, 241, 212, 535, 139, 122, 49, 302, 5, 637, 394, 328, 235, 653, 269, 330, 143, 187, 301, 702, 563, 29, 51, 242, 6, 555, 266, 616, 595, 273, 290, 320, 224, 484, 557, 285, 424, 59, 465, 173, 419, 154, 498, 644, 25, 304, 315, 540, 381, 717, 256, 459, 618, 149, 395, 162, 457, 33, 335, 509, 279, 297, 490]}}
        selected_neurons = logs[bayes.layer][by_best]
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
    parser.add_argument('--from_log', default=False, action='store_true')
    args = parser.parse_args()
    language = args.language
    layer = args.layer
    attribute = args.attribute
    ranking = args.ranking
    control = args.control
    from_log = args.from_log
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
    cluster_ranking_path = Path('pickles', 'UM', language, attribute, str(layer), 'cluster_ranking.pkl')
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
    from_log_str = 'from_log' if from_log else ''
    res_file_name = 'bayes by ' + res_suffix + control_str + from_log_str
    # res_file_name += '_tmp' #TODO
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
            greedy_selection(bayes, by_mi=True, by_best=by_best, from_log=from_log)
        else:
            ranking_params = {'top avg': (utils.sort_neurons_by_avg_weights, linear_model_path),
                              'bottom avg': (utils.sort_neurons_by_avg_weights, linear_model_path),
                              'bayes mi': (utils.sort_neurons_by_bayes_mi, bayes_res_path),
                              'worst mi': (utils.sort_neurons_by_bayes_mi, worst_bayes_res_path),
                              'random': (utils.sort_neurons_by_random, None),
                              'top cluster': (utils.sort_neurons_by_clusters, cluster_ranking_path),
                              'bottom cluster': (utils.sort_neurons_by_clusters, cluster_ranking_path)}
            try:
                neurons_list = get_ranking(ranking_params[ranking])
            except FileNotFoundError:
                sys.exit('WRONG SETTING')
            if ranking == 'bottom avg' or ranking == 'bottom cluster':
                neurons_list = list(reversed(neurons_list))
            run_bayes_on_subset(bayes, neurons_list)

