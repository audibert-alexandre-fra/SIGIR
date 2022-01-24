from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import os
import numpy as np

    
#Def constant
loss_classification = nn.CrossEntropyLoss()
loss_regression = nn.L1Loss()


def new_score_classification(y_true:list, y_pred:list, path:str, average:str ='macro'):
    """[Print and save results, f1 score]

    Args:
        y_true (list): [true label]
        y_pred (list): [pred label]
        path (str): [path to save the model]
        average (str, optional): [ type of F1 score]]. Defaults to 'macro'.
    """
    f1 = f1_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    accuracy = sum(y_pred == y_true) / len(y_pred)
    results = 'Accuracy: {0} \n F1: {1} \n precision : {2} \n recall : {3} '.format(accuracy, f1, precision, recall)
    print(results)
    with open( path + "results.txt", "a+") as a_file:
        a_file.write("\n")
        a_file.write("Classification")
        a_file.write(results)
        
        
def new_score_regression(y_true: list, y_pred: list, path:str):
    """[Print and save results, mae]

    Args:
        y_true (list): [true label]
        y_pred (list): [pred label]
        path (str): [path to save the model]
    """
    results_mae = abs(y_pred-y_true).mean()
    print(results_mae)
    with open(path + "results.txt", "a+") as a_file:
        a_file.write("\n")
        a_file.write("Regression")
        a_file.write(str(results_mae))
        
        
@torch.no_grad()
def get_all_preds(model, loader, regression=False):
    """[Return all prediction]

    Args:
        model (nn.Module): [trained Model]
        loader ([Dataloader]): [dataloader]

    Returns:
        [tensor]: [return the prediction corresponding to the loader]
    """
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds[int(regression)].cpu()), dim=0)
    return all_preds


def multitaskclassificationregression(preds: list, classification: list, regression: list, lamb: float):
    """[Loss function]

    Args:
        preds (list): [prediction of our pretrain model]
        classification (list): [true label for classification]
        regression (list): [true score for regression]
        lamb (float): [ponderation for joint-learning]

    Returns:
        [float]: [loss]
    """
    mae, cross_entropy = nn.L1Loss(), nn.CrossEntropyLoss()
    loss0 = cross_entropy(preds[0], classification)
    loss1 = mae(preds[1].squeeze(1), regression)
    return (lamb*loss0 + (1-lamb)*loss1)


class CallBack():
    """[simple class to save the best model]
    """
    def __init__(self, lr, factor=0.1, patient=2, path='valider.pth'):
        self.memory_val = []
        self.path = path
        
    def add_val(self, mean_loss):
        self.memory_val.append(mean_loss)

    def save_best(self, model):
        if self.memory_val[-1] == min(self.memory_val):
            print('====== Save New model ===== ')
            print(self.memory_val)
            torch.save(model.state_dict(), self.path)


def train_callback(model: nn.Module, trainer: list, valider: list, tester: list, Y_test: list,
                   weight_decay: float=1e-4, epochs: int=10, lr: float=1e-4, cross_stich: bool=False ,lamb: float=0.85):
    """[trainer for multi-task]

    Args:
        model (nn.Module): [Model]
        trainer (list): [train srt]
        valider (list): [validation set]
        tester (list): [test set]
        Y_test (list): [true answeres]
        weight_decay (float, optional): [parameter for Adam]. Defaults to 1e-4.
        epochs (int, optional): [parameter for Adam]. Defaults to 10.
        lr (float, optional): [parameter for Adam]. Defaults to 1e-4.
        lamb (float, optional): [hyperparemeter for the weighted loss]. Defaults to 0.85.

    Returns:
        [str]: [path to the best model]
    """
    
    name_path = 'model/' + model.name + '/' 
    complete_name = name_path + str(len(os.listdir(name_path))) + '.pth'
    callback = CallBack(lr, path=complete_name)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if cross_stich:
        general_param , cross_stitch_param = model.parameter_different_learning()
        optimizer = optim.Adam([
                {'params': general_param},
                {'params': cross_stitch_param, 'lr': lr*10}
            ], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience = 3)
    for epoch in range(epochs):  
        running_loss = 0.0
        mean = 0.0
        nb = 0
        for i, data in enumerate(trainer):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = multitaskclassificationregression(
                    outputs, labels[:, 0].long(), labels[:, 1].float(), lamb)        
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Affichage training part results
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                nb += 1
                mean += running_loss/50
                running_loss = 0.0
        print('---- Mean loss = %.3f ----' % (mean / nb), end='')
        
        # Work on validation set
        model.eval()
        with torch.no_grad():
            loss_val = 0
            for i, data in enumerate(valider):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss_val = multitaskclassificationregression(
                        outputs, labels[:, 0].long(), labels[:, 1].float(), lamb)                  
            print('---- Loss Val %.3f----' % loss_val)
            callback.add_val(loss_val)
            callback.save_best(model)
        model.train()
        curr_lr = optimizer.param_groups[0]['lr']
        print('=== Current lr: %.3f===' % curr_lr)
        scheduler.step(loss_val)
    print('Finished Training')
    print('Start Evaluation Model')
    model.load_state_dict(torch.load(complete_name))
    model.eval()
    y_pred = get_all_preds(model, tester, regression=False)
    proba, predicted = torch.max(y_pred, 1)
    new_score_classification(Y_test[:, 0].numpy(), predicted.numpy(), path=name_path)

    y_pred = get_all_preds(model, tester, regression=True)
    new_score_regression(Y_test[:, 1].numpy(), y_pred.numpy(), path=name_path)
    return complete_name