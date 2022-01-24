from importlib.resources import path
from preprocessing import data_preparation
import os
import torch
import dataloader.dataloader as dataloader
from torch.utils.data import DataLoader
from model import hard_parameter_sharing as hard_sharing
from trainer import loss_join_learning as loss
from model import soft_cross_stitch, hard_parameter_sharing, decomp_kim, decomp_my_proposition
import gc

def clean_cuda():
    """
    Remove Useless element on GPU
    """
    gc.collect()
    torch.cuda.empty_cache()
    print("Clean!")

clean_cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def kim_proposition(epochs_rep=5):
    print(device)
    path = os.path.abspath('')
 
    dataset, pretrained_matrix = data_preparation.data_preparation(path=path, min_freq=5, remove_stop_words=True)  
    X_train, Y_train = dataset[0]
    X_val, Y_val = dataset[1]
    X_test, Y_test = dataset[2]

    batch_size = 16
    epochs = 25
    for _ in range(epochs_rep):
        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train),
                           batch_size=batch_size,
                           collate_fn=dataloader.collate_batch, shuffle=True)

        data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        tester = DataLoader(dataloader.DataSet_1(X_test, Y_test),
                           batch_size=1,
                           collate_fn=dataloader.collate_batch, shuffle=False)

        kim = hard_parameter_sharing.CnnKimHardShared(pretrained_matrix=pretrained_matrix)
        kim = kim.to(device)
        load_best = loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs, lr=1e-4)

        kim = soft_cross_stitch.CnnKimCrossStitch(pretrained_matrix=pretrained_matrix)
        kim = kim.to(device)
        load_best_cross = loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs, lr=1e-4)

        kim = soft_cross_stitch.CnnKimCrossStitch(pretrained_matrix=pretrained_matrix)
        kim.load_state_dict(torch.load(load_best_cross))
        kim.start_multi_task(0.95)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs, lr=1e-4, cross_stich=True)

        kim = decomp_kim.DecompKim(pretrained_matrix=pretrained_matrix, nb_channel=32)
        kim = kim.to(device)
        load_best_decomp = loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs, lr=1e-4)

        kim = decomp_kim.DecompKim(pretrained_matrix=pretrained_matrix,  nb_channel=32)
        kim.load_state_dict(torch.load(load_best_decomp))
        kim.decomp_conv()
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs, lr=1e-4)


        clean_cuda()
        kim = decomp_my_proposition.KimMultiDec(pretrained_matrix=pretrained_matrix)
        kim = kim.to(device)
        load_best_decomp_2 = loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs, lr=1e-4)        

        data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train),
                           batch_size=batch_size//2,
                           collate_fn=dataloader.collate_batch, shuffle=True)
        clean_cuda()
        kim = hard_parameter_sharing.CnnKimHardShared(pretrained_matrix=pretrained_matrix)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16, orthogonal=False)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs//2, lr=1e-5)

        clean_cuda()
        kim = hard_parameter_sharing.CnnKimHardShared(pretrained_matrix=pretrained_matrix)
        kim.load_state_dict(torch.load(load_best))
        kim.apply_cp_decomposition(16, orthogonal=True)
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs//2, lr=1e-5)

        clean_cuda()
        kim = decomp_my_proposition.KimMultiDec(pretrained_matrix=pretrained_matrix)
        kim.load_state_dict(torch.load(load_best_decomp_2))
        kim.apply_cp_decomposition_fusion(16)
        kim = kim.to(device)
        load_best_decomp_2 = loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs//2, lr=1e-4)

        clean_cuda()
        kim = decomp_my_proposition.KimMultiDec(pretrained_matrix=pretrained_matrix)
        kim.apply_cp_decomposition_fusion()
        kim.load_state_dict(torch.load(load_best_decomp_2))
        kim.apply_cp_decomposition_tucker()        
        kim = kim.to(device)
        loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
        epochs=epochs//2, lr=1e-4)

def kim_proposition_path(path_2):
    path = os.path.abspath('')
    dataset, pretrained_matrix = data_preparation.data_preparation(path=path, min_freq=5, remove_stop_words=True)  
    X_train, Y_train = dataset[0]
    X_val, Y_val = dataset[1]
    X_test, Y_test = dataset[2]

    batch_size = 16
    epochs = 25

    data_val = DataLoader(dataloader.DataSet_1(X_val, Y_val),
                        batch_size=1,
                        collate_fn=dataloader.collate_batch, shuffle=False)

    tester = DataLoader(dataloader.DataSet_1(X_test, Y_test),
                        batch_size=1,
                        collate_fn=dataloader.collate_batch, shuffle=False) 

    data_training = DataLoader(dataloader.DataSet_1(X_train, Y_train),
                        batch_size=batch_size//2,
                        collate_fn=dataloader.collate_batch, shuffle=True)
    clean_cuda()
    kim = hard_parameter_sharing.CnnKimHardShared(pretrained_matrix=pretrained_matrix)
    kim.load_state_dict(torch.load(path_2))
    kim.apply_cp_decomposition(16, orthogonal=True, dim=0)
    kim = kim.to(device)
    loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
    epochs=epochs//2, lr=1e-5)

    kim = hard_parameter_sharing.CnnKimHardShared(pretrained_matrix=pretrained_matrix)
    kim.load_state_dict(torch.load(path_2))
    kim.apply_cp_decomposition(16, orthogonal=True, dim=1)
    kim = kim.to(device)
    loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
    epochs=epochs//2, lr=1e-5)

    kim = hard_parameter_sharing.CnnKimHardShared(pretrained_matrix=pretrained_matrix)
    kim.load_state_dict(torch.load(path_2))
    kim.apply_cp_decomposition(16, orthogonal=False)
    kim = kim.to(device)
    loss.train_callback(kim ,list(data_training), list(data_val), tester=list(tester), Y_test=Y_test,
    epochs=epochs//2, lr=1e-5)            


def main():
    print("The type of device is:{0}".format(device))
#    for i in range(10):
#        kim_proposition(1)
    for path_2 in os.listdir('model/kim_1d_hard_parameter/'):
        if path_2[-3:] == 'pth':
            kim_proposition_path('model/kim_1d_hard_parameter/' + path_2)


if __name__ == "__main__":
    main()
