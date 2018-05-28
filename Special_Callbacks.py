# -*- coding: utf-8 -*-

import keras.callbacks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Shapelets_analysis import Shapelet_Analysis_Function,Shapelet_recup
from numpy import transpose

# %% Callbacks : affichage de la progression de l'apprentissage en fin de celui_ci

class Plot_acc_Losses_end_train(keras.callbacks.Callback):
    def __init__(self, filename = "training_datas.png"):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.abscisse = []
        self.iterateur = 0
        self.filename = filename

    def on_epoch_end(self, epoch, logs={}):
        
        
        self.iterateur += 1
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.abscisse.append(self.iterateur)
        
    def on_train_end(self, logs={}):
        fig = plt.figure()
        plot_Acc = fig.add_subplot(211)
        plot_Losses = fig.add_subplot(212)
        
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.2, hspace=0.8)
        
        red_acc = mpatches.Patch(color='red', label='Train Accuracy')
        blue_acc = mpatches.Patch(color='blue', label='Validation Accuracy')
        red_losses = mpatches.Patch(color='red', label='Train Losses')
        blue_losses = mpatches.Patch(color='blue', label='Validation Losses')
        
        
        plot_Acc.plot(self.abscisse,self.acc,"r-")
        plot_Acc.plot(self.abscisse,self.val_acc,"b-")
        plot_Acc.set(xlabel='Epochs', ylabel='Accuracy',title='Accuracy of the model during train')
        plot_Acc.legend(handles=[red_acc,blue_acc])
        
        plot_Losses.plot(self.abscisse,self.losses,"r-")
        plot_Losses.plot(self.abscisse,self.val_losses,"b-")
        plot_Losses.set(xlabel='Epochs', ylabel='Losses',title='Losses of the model during train')
        plot_Losses.legend(handles=[red_losses,blue_losses])
        fig.savefig(self.filename)
        fig.show()
        
# %% Callbacks : affichage de la progression de l'apprentissage en fin de celui_ci et à chaque epoch

class Plot_acc_Losses_during_train(keras.callbacks.Callback):
    def __init__(self, filename = "training_datas.png",donnee_suivi = "Acc"):
        self.losses = [0]
        self.acc = [0]
        self.val_losses = [0]
        self.val_acc = [0]
        self.abscisse = [0]
        self.iterateur = 0
        self.filename = filename
        
        plt.show()
        self.axes = plt.gca()
        self.donnee_suivi = donnee_suivi
        
        if self.donnee_suivi == "Acc":
            self.plot_Acc, = self.axes.plot(self.abscisse, self.acc, 'r-')
            self.plot_val_Acc, = self.axes.plot(self.abscisse, self.val_acc, 'b-')
        else: 
            self.plot_Loss, = self.axes.plot(self.abscisse, self.losses, 'r-')
            self.plot_val_Loss, = self.axes.plot(self.abscisse, self.val_losses, 'b-')
        
    def on_epoch_end(self, epoch, logs={}):
        
        if self.abscisse[0]==0: # supprime l'initialisation
            self.abscisse = self.abscisse[1:]
            self.losses = self.losses[1:]
            self.val_losses = self.val_losses[1:]
            self.acc = self.acc[1:]
            self.val_acc = self.val_acc[1:]
        
        self.iterateur += 1
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.abscisse.append(self.iterateur)
        
        if self.donnee_suivi == "Acc":

            self.plot_Acc.set_xdata(self.abscisse)
            self.plot_Acc.set_ydata(self.acc)

            self.plot_val_Acc.set_xdata(self.abscisse)
            self.plot_val_Acc.set_ydata(self.val_acc)

            self.axes.set_xlim(0,max(self.abscisse)+1)
            self.axes.set_ylim(min(min(self.acc),min(self.val_acc))-0.05,max(max(self.acc),max(self.val_acc))+0.05)
            plt.ylabel("précision (%)")
            plt.title("suivi de la précision au cours de l'apprentissage")
            
        else :
            self.plot_Loss.set_xdata(self.abscisse)
            self.plot_Loss.set_ydata(self.losses)
            
            self.plot_val_Loss.set_xdata(self.abscisse)
            self.plot_val_Loss.set_ydata(self.val_losses)
            
            self.axes.set_xlim(0,max(self.abscisse)+1)
            self.axes.set_ylim(min(min(self.losses),min(self.val_losses))-0.05,max(max(self.losses),max(self.val_losses))+0.05)
            plt.ylabel("erreur (%)")
            plt.title("suivi de l'erreur au cours de l'apprentissage")
            
        plt.xlabel("epochs")
        plt.draw()
        plt.pause(1e-12)
        
    def on_train_end(self, logs={}):
        fig = plt.figure()
        plot_Acc = fig.add_subplot(211)
        plot_Losses = fig.add_subplot(212)
        
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.2, hspace=0.8)
        
        red_acc = mpatches.Patch(color='red', label='Train Accuracy')
        blue_acc = mpatches.Patch(color='blue', label='Validation Accuracy')
        red_losses = mpatches.Patch(color='red', label='Train Losses')
        blue_losses = mpatches.Patch(color='blue', label='Validation Losses')
        
        
        plot_Acc.plot(self.abscisse,self.acc,"r-")
        plot_Acc.plot(self.abscisse,self.val_acc,"b-")
        plot_Acc.set(xlabel='Epochs', ylabel='Accuracy',title='Accuracy of the model during train')
        plot_Acc.legend(handles=[red_acc,blue_acc])
        
        plot_Losses.plot(self.abscisse,self.losses,"r-")
        plot_Losses.plot(self.abscisse,self.val_losses,"b-")
        plot_Losses.set(xlabel='Epochs', ylabel='Losses',title='Losses of the model during train')
        plot_Losses.legend(handles=[red_losses,blue_losses])
        fig.savefig(self.filename)
        fig.show()
        
# %% Callbacks : Analyse de shapelet en live

class Shapelet_Analysis(keras.callbacks.Callback):
    
    def __init__(self,Model,num_couche_conv,X_train,num_shapelet=0,nb_shapelets = 3):

        self.nb_shapelets = nb_shapelets
        self.modele = Model
        self.iterateur = 0
        self.num_couche_conv = num_couche_conv
        self.num_shapelet = num_shapelet
        self.X_train = X_train
        self.sauvegarde_shapelets = {}
        
        for i in range(0,self.nb_shapelets):
            self.sauvegarde_shapelets[i] = []

    def on_epoch_end(self,epoch,logs = {}):
        self.iterateur+=1
        for i in range(0,self.nb_shapelets):
            self.sauvegarde_shapelets[i].append(Shapelet_recup(self.modele,self.num_couche_conv+i,num_shapelet=0))

    def on_train_end( self, epoch, logs={} ):
        for j in range(0,self.nb_shapelets):
            ((X_curve,Y_curve),(X_shapelet,Y_Shapelet))=Shapelet_Analysis_Function(self.modele,self.num_couche_conv+j,self.X_train,self.num_shapelet)
            plt.plot(X_curve,Y_curve,"r-")
            plt.plot(X_shapelet,Y_Shapelet,"b-")
            plt.title("visualisation d'un shapelet à l'epoch n° " + str(self.iterateur))
            plt.xlabel("Echantillons temporels")
            plt.ylabel("Grandeur physique associée au Dataset")
            plt.savefig("./Figures/fig_"+str(len(X_shapelet))+"_"+str(self.iterateur)+".png")
            plt.clf()
            print("taille x = ",len(X_shapelet)," taille y = ",len(Y_Shapelet))
    
            print(self.iterateur)
            for i in range(0,len(self.sauvegarde_shapelets[j])-1):
                plt.plot(X_curve,Y_curve,"r-")
                Y_Shapelet =self.sauvegarde_shapelets[j][i]
                print("taille x = ",len(X_shapelet)," taille y = ",len(Y_Shapelet))
                plt.plot(X_shapelet,Y_Shapelet,"b-")
                plt.title("visualisation d'un shapelet à l'epoch n° " + str(i+1))
                plt.xlabel("Echantillons temporels")
                plt.ylabel("Grandeur physique associée au Dataset")
                plt.savefig("./Figures/fig_"+str(len(X_shapelet))+"_"+str(i+1)+".png")
                plt.clf()
