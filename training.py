from preprocessing import data_preparation
import os
import torch
import dataloader.dataloader as dataloader
from torch.utils.data import DataLoader
from model import CNN_OD, CNN_LSTM_ATTENTION, HAHNN
from trainer import loss, loss_len

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def kim_training(epochs_rep=5):
    print(device)
    path = os.path.abspath('')
    dataset, pretrained_matrix = data_preparation.data_preparation(path=path, min_freq=5, remove_stop_words=True)  
    X_train, Y_train = dataset[0]
    X_val, Y_val = dataset[1]
    X_test, Y_test = dataset[2]
    batch_size = 32
    epochs = 25
    print("Classification")
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 0]),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val[:, 0]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test[:, 0]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)
        
        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim = kim.to(device)
        load_best = loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs)

        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 0]),
                           batch_size=batch_size//2,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs//2, lr=1e-5)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16, dim=1)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs//2, lr=1e-5)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16, orthogonal=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs//2, lr=1e-5)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition_fusion(16)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs//2, lr=1e-4)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition_fusion(16, dim=1)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs//2, lr=1e-4)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=True)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition_fusion(16, orthogonal=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs//2, lr=1e-4)

    print("Regression")
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 1]),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val[:, 1]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test[:, 1]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)
        
        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim = kim.to(device)
        load_best = loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs)
        
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 1]),
                           batch_size=batch_size//2,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs//2, lr=1e-5)
        
        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16, dim=1)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs//2, lr=1e-5)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16, orthogonal=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs//2, lr=1e-5)


        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition_fusion(16)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs//2, lr=1e-4)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition_fusion(16, orthogonal=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs//2, lr=1e-4)

        kim = CNN_OD.CnnKim(pretrained_matrix=pretrained_matrix, classification=False)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition_fusion(16, dim=1)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs//2, lr=1e-4)     


def main():
    print("The type of device is:{0}".format(device))
    kim_training(10)
    #CNN_LSTM(5)
    #Hahnn(5, len_sentence=15)

if __name__ == "__main__":
    main()




"""
def CNN_LSTM(epochs_rep=5):
    print(device)
    path = os.path.abspath('')
    dataset, pretrained_matrix = data_preparation.data_preparation(path=path, min_freq=5, remove_stop_words=True)  
    X_train, Y_train = dataset[0]
    X_val, Y_val = dataset[1]
    X_test, Y_test = dataset[2]
    batch_size = 32
    epochs = 22
    print("Classification")
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 0]),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val[:, 0]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test[:, 0]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)
        
        kim = CNN_LSTM_ATTENTION.CnnLstmAttention(pretrained_matrix=pretrained_matrix, classification=True)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs)
    
    print("Regression")
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 1]),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val[:, 1]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test[:, 1]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)
        
        kim = CNN_LSTM_ATTENTION.CnnLstmAttention(pretrained_matrix=pretrained_matrix, classification=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs)


def CNN_LSTM(epochs_rep=5):
    print(device)
    path = os.path.abspath('')
    dataset, pretrained_matrix = data_preparation.data_preparation(path=path, min_freq=5, remove_stop_words=True)  
    X_train, Y_train = dataset[0]
    X_val, Y_val = dataset[1]
    X_test, Y_test = dataset[2]
    batch_size = 32
    epochs = 22
    print("Classification")
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 0]),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val[:, 0]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test[:, 0]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)
        
        kim = CNN_LSTM_ATTENTION.CnnLstmAttention(pretrained_matrix=pretrained_matrix, classification=True)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 0],
                    loss_function=loss.loss_classification, classification=True, epochs=epochs)
    
    print("Regression")
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train[:, 1]),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val[:, 1]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test[:, 1]),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)
        
        kim = CNN_LSTM_ATTENTION.CnnLstmAttention(pretrained_matrix=pretrained_matrix, classification=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=iter(tester),Y_test=Y_test[:, 1],
                    loss_function=loss.loss_regression, classification=False, epochs=epochs)
"""