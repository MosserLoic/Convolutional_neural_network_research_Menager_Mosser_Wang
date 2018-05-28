# -*- coding: utf-8 -*-

# %% partie importation de données

import numpy as np
import arff
from keras.utils import np_utils
import random

# %% importation des données

def generation_liste_aleatoire_entiers_uniques(size):
    liste = [i for i in range(0,size)]
    random.shuffle(liste)
    return liste
    
def split_list_three(Sub_dataset_X,Sub_dataset_Y,Percent_train_test,Percent_Validation_set):
    length = len(Sub_dataset_X)
    length_train_set = int(Percent_train_test*length)
    length_valid_set = int(Percent_Validation_set*length_train_set)
    length_train_set = length_train_set-length_valid_set
    X_train = []
    X_valid = []
    X_test = []
    Y_train = []
    Y_valid = []
    Y_test = []
    for i in range(0,length):
        if i<length_valid_set : 
            X_valid.append(Sub_dataset_X[i])
            Y_valid.append(Sub_dataset_Y[i])
        elif i <(length_valid_set+length_train_set):
            X_train.append(Sub_dataset_X[i])
            Y_train.append(Sub_dataset_Y[i])
        else :
            X_test.append(Sub_dataset_X[i])
            Y_test.append(Sub_dataset_Y[i])
    return (X_train,Y_train,X_test,Y_test,X_valid,Y_valid)
    
def melange_croise(X,Y):
    numero = generation_liste_aleatoire_entiers_uniques(len(X))
    return ([X[i] for i in numero],[Y[i] for i in numero])

def Load_Arff_File(File_Name,Percent_train_test=0.7,Percent_Validation_set = 0.1):
    data = arff.load(open(File_Name, 'r'))
    Dataset = data["data"]
    
    attributes = data["attributes"][-1][1]
    nb_class = len(attributes)
    
    datas = {}
    Classes = {}
    
    X_train_gene = []
    X_valid_gene = []
    X_test_gene = []
    Y_train_gene = []
    Y_valid_gene = []
    Y_test_gene = []
    
    for i in range(0,nb_class):
        Classes[i] = []
        datas[i] = []
        
    Length_dataset = len(Dataset)
    for i in range(0,Length_dataset):
        Classes[int(Dataset[i][-1])-1].append(int(Dataset[i][-1])-1)
        datas[int(Dataset[i][-1])-1].append(Dataset[i][0:-1])
        
    for i in range(0,nb_class):
        random.shuffle(datas[i])
        (X_train,Y_train,X_test,Y_test,X_valid,Y_valid) = split_list_three(datas[i],Classes[i],Percent_train_test,Percent_Validation_set)
       
        X_train_gene = X_train_gene+X_train
        X_valid_gene = X_valid_gene+X_valid
        X_test_gene = X_test_gene+X_test
        Y_train_gene = Y_train_gene+Y_train
        Y_valid_gene = Y_valid_gene+Y_valid
        Y_test_gene = Y_test_gene+Y_test
    
    
    length_time_series = len(X_train_gene[0])
    
    (X_train,Y_train) = melange_croise(X_train_gene,Y_train_gene)
    (X_valid,Y_valid) = melange_croise(X_valid_gene,Y_valid_gene)
    (X_test,Y_test) = melange_croise(X_test_gene,Y_test_gene)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_validation = np.array(X_valid)
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_validation = np.array(Y_valid)
    
    Nb_examples_train = X_train.shape[0]
    Nb_examples_test = X_test.shape[0]
    Nb_examples_validation = X_validation.shape[0]
    
    return ( X_train.reshape(Nb_examples_train,length_time_series,1) , np_utils.to_categorical(Y_train,nb_class) ) , ( X_test.reshape(Nb_examples_test,length_time_series,1) , np_utils.to_categorical(Y_test,nb_class) ), ( X_validation.reshape(Nb_examples_validation,length_time_series,1) , np_utils.to_categorical(Y_validation,nb_class) ) , nb_class
