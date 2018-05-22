# -*- coding: utf-8 -*-
#
from keras.models import Model #model type
from keras.layers import Input,Dense #core layers
from keras.layers import Conv1D, GlobalMaxPooling1D,concatenate #convolution
from math import log

# %% Fonction de grabocka pour le calcul des tailles et du nombre de shapelets avec le formalisme deja utilisé

def Grabocka_Shapelets(dataset_length=16637,Series_length=96,Class_number=7,L=0.1,R=3):  # déjà parametré pour notre dataset
    List_shapelet_length = [int(i*L*Series_length) for i in range(1,R+1)]
    Liste_end = []
    for i in range(0,len(List_shapelet_length)):
        Liste_end.append((int(log((Series_length+1-List_shapelet_length[i])*dataset_length)*(Class_number-1)),List_shapelet_length[i]))
    return Liste_end


# %% definition des modèles : 
    
def definition_model_conv_simple(nb_shapelets,length_Shapelets):

    # Version manuelle : 

    main_input = Input(shape=(96,1), dtype='float32', name='main_input')

    Conv =  Conv1D(nb_shapelets, length_Shapelets, activation='relu')(main_input)
    Pool = GlobalMaxPooling1D()(Conv)
    main_output = Dense(7, activation='softmax', name='main_output')(Pool)

    model = Model(inputs=[main_input], outputs=[main_output])

    # Version librairie
    """
    model = Sequential()
    model.add(Conv1D(20, 9, activation='relu',input_shape=(96,1)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(7, activation='sigmoid'))
    """
    return model


def define_model_conv_mult(Liste_Shapelets,regulation):
    main_input = []
    Conv = []
    Pool = []
    main_input=Input(shape=(96,1), dtype='float32', name=('main_input'))
    for Shapelets in Liste_Shapelets:
        (nb_shapelets,length_Shapelets) = Shapelets
        Conv.append(Conv1D(nb_shapelets, length_Shapelets, activation='relu' ,kernel_regularizer=regulation)(main_input))
        Pool.append(GlobalMaxPooling1D()(Conv[-1]))
    Axis_select = -1
    Pool = concatenate(Pool,axis = Axis_select)
    main_output = Dense(7, activation='softmax', name='main_output')(Pool)
    model = Model(inputs=[main_input], outputs=[main_output])
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# %% train fonction
    
def train_model(model ,X_train ,Y_train ,X_test ,Y_test ,X_validation ,Y_validation ,nb_epoch):
    History = model.fit(X_train,Y_train,nb_epoch=nb_epoch,validation_data=(X_validation,Y_validation))
    result = model.evaluate(X_test, Y_test,verbose=0)
    return (model ,result[1], History)
