from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import os
import numpy as np

#General LOSS
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
def get_all_preds(model: nn.Module, loader: list):
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
            (all_preds, preds.cpu()), dim=0)
    return all_preds
    

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



def train_callback(model: nn.Module, trainer: list, valider: list, tester: list, Y_test:torch.tensor,
                   loss_function, classification: bool=True,
                   weight_decay: float=1e-4, epochs: int=10, lr: float=1e-4):
    """[Train function. train on trainer, vald on valider, and test
        on the tester]

    Args:
        model (nn.Module): [Pretrained Model]
        trainer (list): [trainset]
        valider (list): [valid set]]
        tester (list): [test set]
        Y_test (torch.tensor): [answer]
        loss_function ([type]): [function of error]
        classification (bool, optional): [if true classification trainer, else regression trainer]. Defaults to True.
        weight_decay (float, optional): [parameter of optimizer Adam]. Defaults to 1e-4.
        epochs (int, optional): [parameter of optimizer Adam]. Defaults to 10.
        lr (float, optional): [parameter of optimizer Adam]. Defaults to 1e-4.

    Returns:
        [str]: [path of the best model]
    """
    
    # Initialisation of criterion, correct path, optimizer, and scheduler
    criterion = loss_function
    name_path = 'model/' + model.name + '/' 
    complete_name = name_path + str(len(os.listdir(name_path))) + '.pth'
    callback = CallBack(lr, path=complete_name)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience = 3)
    
    #Begining of training
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
            if classification:
                loss = criterion(outputs, labels.long())
            else:
                loss = criterion(outputs.reshape(-1), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #Display results
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
                if classification:
                    loss_val += criterion(outputs, labels.long())
                else:
                    loss_val += criterion(outputs.reshape(-1), labels.float())
            print('---- Loss Val %.3f----' % loss_val)
            callback.add_val(loss_val)
            callback.save_best(model)
        model.train()
        curr_lr = optimizer.param_groups[0]['lr']
        print('=== Current lr: %.3f===' % curr_lr)
        scheduler.step(loss_val)
    print('Finished Training')
    print('Start Evaluation Model')
    
    #Save Evaluate the best model
    model.load_state_dict(torch.load(complete_name))
    model.eval()
    y_pred = get_all_preds(model, tester)
    if classification:
        proba, predicted = torch.max(y_pred, 1)
        new_score_classification(Y_test.numpy(), predicted.numpy(), path=name_path)
    else:
        new_score_regression(Y_test.numpy(), y_pred.numpy(), path=name_path)
    return complete_name