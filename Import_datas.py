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
    


def Load_Arff_File(File_Name,Percent_train_test=0.7,Percent_Validation_set = 0.1):
    data = arff.load(open(File_Name, 'r'))
    Dataset = data["data"]
    
    Length_dataset = len(Dataset)

    Length_train = int(Percent_train_test*Length_dataset)
    
    Length_validation = int(Percent_Validation_set*Length_train)
    
    X_data_train = []
    Y_data_train = []
    
    X_data_validation = []
    Y_data_validation = []
    
    X_data_test = []
    Y_data_test = []
    numero = generation_liste_aleatoire_entiers_uniques(Length_dataset)
    
    for i in range(0,Length_dataset):
        if i<Length_train:
            if i<Length_validation : 
                X_data_validation.append(Dataset[numero[i]][0:-1])
                Y_data_validation.append(int(Dataset[numero[i]][-1])-1)
            else:
                X_data_train.append(Dataset[numero[i]][0:-1])
                Y_data_train.append(int(Dataset[numero[i]][-1])-1)
        else:
            X_data_test.append(Dataset[numero[i]][0:-1])
            Y_data_test.append(int(Dataset[numero[i]][-1])-1)
    attributes = data["attributes"][-1][1]
    Nb_class = len(attributes)
    
    X_train = np.array(X_data_train)
    X_test = np.array(X_data_test)
    X_validation = np.array(X_data_validation)
    
    Y_train = np.array(Y_data_train)
    Y_test = np.array(Y_data_test)
    Y_validation = np.array(Y_data_validation)
    
    Nb_examples_train = X_train.shape[0]
    Nb_examples_test = X_test.shape[0]
    Nb_examples_validation = X_validation.shape[0]
    
    return ( X_train.reshape(Nb_examples_train,96,1) , np_utils.to_categorical(Y_train,7) ) , ( X_test.reshape(Nb_examples_test,96,1) , np_utils.to_categorical(Y_test,7) ), ( X_validation.reshape(Nb_examples_validation,96,1) , np_utils.to_categorical(Y_validation,7) ) , Nb_class
