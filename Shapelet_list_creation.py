# -*- coding: utf-8 -*-

from math import log

# %% Heuristique de Grabocka

def Grabocka_Shapelets(dataset_length=16637,Series_length=96,Class_number=7,L=0.1,R=3):  # déjà parametrée pour notre dataset
    List_shapelet_length = [int(i*L*Series_length) for i in range(1,R+1)]
    Liste_end = []
    for i in range(0,len(List_shapelet_length)):
        Liste_end.append((int(log((Series_length+1-List_shapelet_length[i])*dataset_length)*(Class_number-1)),List_shapelet_length[i]))
    return Liste_end

# %% Heuristique plus simple
    
def Creation_Shapelets_list(Length_list):
    L = len(Length_list)
    List_returned = []
    for i in range(0,L):
        List_returned.append((96-Length_list[i],Length_list[i]))
    return List_returned
