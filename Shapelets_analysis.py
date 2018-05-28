# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from numpy import sqrt
from numpy import transpose

def calcul_distance(Courbe,Shapelet):
    somme = 0
    for i in range(0,len(Courbe)):
        somme = somme+((Courbe[i]-Shapelet[i])**2)
    return sqrt(somme)

def Rang_distance_la_plus_courte(Courbe,Shapelet):
    L_shapelet = len(Shapelet)
    L_courbe = len(Courbe)
    Erreurs = [calcul_distance(Courbe[i:i+len(Shapelet)],Shapelet) for i in range(0,L_courbe-L_shapelet)]
    return Erreurs.index(min(Erreurs))

def Erreur_distance_la_plus_courte(Courbe,Shapelet):
    L_shapelet = len(Shapelet)
    L_courbe = len(Courbe)
    Erreurs = [calcul_distance(Courbe[i:i+len(Shapelet)],Shapelet) for i in range(0,L_courbe-L_shapelet)]
    return min(Erreurs)

def Rang_Erreur_distance_la_plus_courte(Courbe,Shapelet):
    L_shapelet = len(Shapelet)
    L_courbe = len(Courbe)
    Erreurs = [calcul_distance(Courbe[i:i+len(Shapelet)],Shapelet) for i in range(0,L_courbe-L_shapelet)]
    return (Erreurs.index(min(Erreurs)),min(Erreurs))


def Find_optimal_position(Courbe,Shapelet,name_fig=None):
    L_shapelet = len(Shapelet)
    L_courbe = len(Courbe)
    i = Rang_distance_la_plus_courte(Courbe,Shapelet)
    abscisse_Courbe = [j for j in range(0,L_courbe)]
    abscisse_Shapelet = [j+i for j in range(0,L_shapelet)]
    Ordonnee_Courbe = [Courbe[i] for i in range(0,L_courbe)]
    Ordonnee_Shapelet = [Shapelet[i] for i in range(0,L_shapelet)]
    plt.plot(abscisse_Courbe,Ordonnee_Courbe,"r-")
    plt.plot(abscisse_Shapelet,Ordonnee_Shapelet,"-")
    plt.xlabel("Echantillons temporels")
    plt.ylabel("Grandeur physique associée au Dataset")
    plt.title("Visualisation d'une courbe du dataset pour un shapelet\n à la position ou celui ci mimise la distance euclidienne à la courbe ")
    if name_fig!=None:
        plt.savefig(name_fig)
        plt.clf()
    else:
        plt.show()

def Find_optimal_position_for_optimal_plot(Courbes,Shapelet,name_fig=None):
    L_coub = len(Courbes)
    erreur = []
    for i in range(L_coub):
        erreur.append(Erreur_distance_la_plus_courte(Courbes[i],Shapelet))
    mini = min(erreur)
    Rang = erreur.index(mini)
    Courbe_optimale = Courbes[Rang]
    Find_optimal_position(Courbe_optimale,Shapelet,name_fig)

def return_curves(Courbes,Shapelet):
    L_coub = len(Courbes)
    erreur = []
    for i in range(L_coub):
        erreur.append(Erreur_distance_la_plus_courte(Courbes[i],Shapelet))
    mini = min(erreur)
    Rang = erreur.index(mini)
    Courbe_optimale = Courbes[Rang]
    return Courbe_optimale

def Shapelet_Analysis_Function(model,num_couche_conv,X_train,num_shapelet=0):
    
    poids_biais_couche_convolution = model.layers[num_couche_conv].get_weights()
    poids_couche_convolution = transpose(poids_biais_couche_convolution[0])
    Courbes = transpose(transpose(X_train)[0])
    Shapelet = poids_couche_convolution[num_shapelet][0]
    Curve = return_curves(Courbes,Shapelet)
    
    L_shapelet = len(Shapelet)
    L_courbe = len(Curve)
    i = Rang_distance_la_plus_courte(Curve,Shapelet)
    abscisse_Courbe = [j for j in range(0,L_courbe)]
    abscisse_Shapelet = [j+i for j in range(0,L_shapelet)]
    Ordonnee_Courbe = [Curve[i] for i in range(0,L_courbe)]
    Ordonnee_Shapelet = [Shapelet[i] for i in range(0,L_shapelet)]
    
    return ((abscisse_Courbe,Ordonnee_Courbe),(abscisse_Shapelet,Ordonnee_Shapelet))

def Shapelet_recup(model,num_couche_conv,num_shapelet=0):
    poids_biais_couche_convolution = model.layers[num_couche_conv].get_weights()
    poids_couche_convolution = transpose(poids_biais_couche_convolution[0])
    Shapelet = poids_couche_convolution[num_shapelet][0]
    return Shapelet
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
