# -*- coding: utf-8 -*-

from Import_datas import Load_Arff_File
from Model_definition import define_model_conv_mult,train_model
from Shapelet_list_creation import Grabocka_Shapelets
from keras.callbacks import EarlyStopping
from Special_Callbacks import Plot_acc_Losses_end_train,Plot_acc_Losses_during_train,Shapelet_Analysis
from Regularisation import l2_reg

# %% import des données du dataset en trois parties distinctes piochées aléatoirement dans celui ci avec 72 % du dataset dans le train set,
# 28 % dans le test set et 15 % du train set dans le validation set

(X_train,Y_train),(X_test,Y_test),(X_val,Y_val),Nb_class = Load_Arff_File("ElectricDevices.arff",Percent_train_test=0.72,Percent_Validation_set = 0.15)

# %% choix de l'heuristique : 

Shapelets = Grabocka_Shapelets()    # Heuristique de Grabocka
# Shapelets = Creation_Shapelets_list([30,35,40,45,50])     # Heuristique Maison

# %% definition du modèle et des Callbacks

Model = define_model_conv_mult(Shapelets,l2_reg)  # definition du modèle sans régulation

Plot_acc_Losses_end_train = Plot_acc_Losses_end_train(filename = "Train_results.png")
Plot_acc_Losses_during_train = Plot_acc_Losses_during_train(donnee_suivi="Acc") # donnee_suivi="Loss"
Shapelet_Analysis = Shapelet_Analysis(Model = Model,X_train = X_train,num_shapelet=0,num_couche_conv = 1)
Early_stoping = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto',)   # Early_stopping with patience on the loss metric variable
Callbacks = [Early_stoping,Shapelet_Analysis] # definition des Callbacks

# %% definition de la compilation et de l'apprentissage

Model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
(model,accuracy,History) = train_model(Model ,X_train ,Y_train ,X_test ,Y_test ,X_val ,Y_val ,nb_epoch = 400 ,Callbacks = Callbacks,ver = 1)
