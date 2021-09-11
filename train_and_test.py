import torch
import torch.nn as nn
import torch.optim as optim
from model import PosTaggerWholeVector, PosTaggerSingleNeuron, PosTaggerSingleWithHidden, SinglePosPredictor,\
    PosTaggerSubset
import consts
import copy
import pickle

models = {'wholeVector': PosTaggerWholeVector,
          'singleNoHidden': PosTaggerSingleNeuron,
          'singleWithHidden': PosTaggerSingleWithHidden,
          'singlePos': SinglePosPredictor,
          'subset':PosTaggerSubset}

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())

def train(dataloader, model_name: str, lambda1=0.001, lambda2=0.01,
          save_path:str='', verbose:bool=False, reg:bool=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(set([sample[1] for sample in dataloader.dataset]))
    classifier = models[model_name](first_layer_size=dataloader.dataset[0][0].shape[0], num_labels=num_labels).to(device)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=consts.LEARNING_RATE)
    best_loss=consts.INF
    acc_on_best_loss, mi_on_best_loss = 0.0, 0.0
    best_model=None
    label_ratios = (torch.histc(torch.tensor([sample[1] for sample in dataloader.dataset]).float(),
                                bins=consts.LABEL_DIM)) / len(dataloader.dataset)
    entropy = torch.distributions.Categorical(label_ratios).entropy()
    for epoch in range(consts.EPOCHS):
        if verbose:
            print('epoch ',epoch)
        running_loss = 0.0
        accuracy = 0.0
        cross_entropy_sum = torch.tensor(0.0).to(device)
        for i, features_and_labels in enumerate(dataloader):
            optimizer.zero_grad()
            word_features, word_label = features_and_labels
            if model_name != 'wholeVector' and model_name != 'subset':
                word_features = word_features.unsqueeze(1)
            outputs = classifier(word_features)
            word_label = word_label.to(device)
            preds = outputs.argmax(dim=1).to(device)
            correct_preds = preds == word_label
            accuracy += torch.sum(correct_preds).item()
            cross_entropy_loss = cross_entropy(outputs,word_label)
            cross_entropy_sum += cross_entropy_loss * dataloader.batch_size
            weights = list(classifier.parameters())[0]
            if reg:
                l1_loss = lambda1 * l1_penalty(weights)
                l2_loss = torch.tensor(lambda2) * l2_penalty(weights)
                loss = cross_entropy_loss + l1_loss + l2_loss
            else:
                loss = cross_entropy_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if verbose:
                if i % 5000 == 4999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i * consts.BATCH_SIZE, running_loss / i * consts.BATCH_SIZE))
                    print('accuracy after %d: %.3f' % (i * consts.BATCH_SIZE, accuracy / (i * consts.BATCH_SIZE)))
        if running_loss < best_loss:
            acc_on_best_loss = accuracy
            best_loss = running_loss
            best_model = copy.deepcopy(classifier)
            cross_entropy_avg = cross_entropy_sum / len(dataloader.dataset)
            mi_on_best_loss = (entropy - cross_entropy_avg) / torch.tensor(2.0).log()
    if save_path != '':
        best_model.to('cpu')
        torch.save(best_model.state_dict(), save_path)
    print('accuracy on train set: %.5f' % (acc_on_best_loss/len(dataloader.dataset)))
    print('mi on train set: %.5f' % mi_on_best_loss)
    print('nmi on train set: %.5f' % (mi_on_best_loss / entropy).item())
    return classifier

def test(classifier, dataloader, save_path=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0.0
    running_loss = 0.0
    cross_entropy_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_true_labels = torch.tensor([]), torch.tensor([])
    for i, features_and_labels in enumerate(dataloader):
        word_features, word_label = features_and_labels
        if type(classifier) != PosTaggerWholeVector and type(classifier) != PosTaggerSubset:
            word_features = word_features.unsqueeze(1)
        outputs = classifier(word_features)
        word_label = word_label.to(device)
        preds = outputs.argmax(dim=1).to(device)
        if save_path != '':
            all_preds = torch.cat([all_preds,preds.float()])
            all_true_labels = torch.cat([all_true_labels,word_label.float()])
        correct_preds = preds == word_label
        accuracy += torch.sum(correct_preds).item()
        loss = criterion(outputs, word_label)
        cross_entropy_sum += loss * dataloader.batch_size
        running_loss += loss.item()
        # if i % 1000 == 999:
        #     print('[%5d] loss: %.3f' %
        #           (i * BATCH_SIZE, running_loss / i*BATCH_SIZE))
        #     print('accuracy after %d: %.3f' % (i * BATCH_SIZE, accuracy / (i * BATCH_SIZE)))
    final_acc = accuracy/len(dataloader.dataset)
    print('final accuracy on test: %.5f' % (accuracy/len(dataloader.dataset)))
    label_ratios = (torch.histc(torch.tensor([sample[1] for sample in dataloader.dataset]).float(),
                                bins=consts.LABEL_DIM)) / len(dataloader.dataset)
    entropy = torch.distributions.Categorical(label_ratios).entropy()
    cross_entropy_avg = cross_entropy_sum / len(dataloader.dataset)
    mi = (entropy - cross_entropy_avg) / torch.tensor(2.0).log()
    print('mi on test: %.5f' % mi)
    print('nmi on test set: %.5f' % (mi / entropy).item())
    if save_path != '':
        torch.save(all_preds,save_path)
        torch.save(all_true_labels,'pickles/true_labels')
    return final_acc
