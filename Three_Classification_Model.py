import numpy as np
import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten, Conv1D, MaxPooling1D, BatchNormalization,InputLayer
from keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


# This function builds the desired network #
def create_model (input_len, num_classes):
    
    model = Sequential()
    model.add(InputLayer(input_shape=(input_len, 1))) 
    
    # Normalization
    model.add(BatchNormalization())
    
    # Conv + Maxpooling
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=5 ))
    
    # Dropout
    model.add(Dropout(0.2))
    
    # Converting 3D feature to 1D feature Vector
    model.add(Flatten())
    
    # Fully Connected Layer
    model.add(Dense(16, activation='tanh'))
    
    # Normalization
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    return model 

################# load Data ###################
# get the path of data save 
diraction = "/pssm/"

# load training set
trainData = np.load(diraction+"Train/pssm_nr2011/monobi_prob_pssm_nr2011.npy")
trainLbls = np.load(diraction+"Train/class.npy")

# load independentTest set
testData  = np.load(diraction+"Test/pssm_nr2011/monobi_tprob_pssm_nr2011.npy")
testLbls = np.load(diraction+"Test/class.npy")


# Enter the path where you want the network to be stored
save_path= "/CNN"

batch_size = 150
epochs = 40
seed =0
np.random.seed(seed)
num_classes = 2
input_len = len(trainData[0])
n_fold = 10
kfold = KFold(n_splits=n_fold,shuffle=True, random_state=seed)
T = 0

Accuracy = np.zeros((n_fold,1))
fmeasure = np.zeros((n_fold,1))
precision = np.zeros((n_fold,1))
recall = np.zeros((n_fold,1))
AUC = np.zeros((n_fold,1))
MCC =np.zeros((n_fold,1)) 
Specificity = np.zeros((n_fold,1)) 

Accuracy_val = np.zeros((n_fold,1))
fmeasure_val = np.zeros((n_fold,1))
precision_val = np.zeros((n_fold,1))
recall_val = np.zeros((n_fold,1))
AUC_val = np.zeros((n_fold,1))
MCC_val =np.zeros((n_fold,1)) 
Specificity_val = np.zeros((n_fold,1)) 

x_test = testData.reshape(testData.shape[0], input_len,1).astype('float32')
y_test = keras.utils.to_categorical(testLbls, num_classes)

for  (train, val) in (kfold.split(trainData)):
    print('T= ',T)
    x_train, x_val= trainData[train], trainData[val]
    y_train, y_val = trainLbls[train], trainLbls[val]
    
    x_train = x_train.reshape(x_train.shape[0], input_len,1).astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    x_val = x_val.reshape(x_val.shape[0], input_len,1).astype('float32')
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = create_model(input_len, num_classes)
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    histories = []
    filepath=save_path +"/model_"+np.str(T)+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True)#, mode='auto' val_loss val_acc rmsprop

    history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[
              checkpoint
          ]
         )
    histories.append(history.history)
    model.load_weights(filepath)
    
    ################## evaluating network on evalution set###########
    estmLbls_val = model.predict_classes(x_val)
    f_val = f1_score(trainLbls[val], estmLbls_val, average='macro') 
    acc_val = accuracy_score(trainLbls[val], estmLbls_val)
    rec_val = recall_score(trainLbls[val], estmLbls_val, average='macro')
    pr_val = precision_score(trainLbls[val], estmLbls_val, average='macro')
    roc_auc_val = roc_auc_score(trainLbls[val],estmLbls_val)
    mcc_val  = matthews_corrcoef(trainLbls[val],estmLbls_val)
    cm_val = confusion_matrix(trainLbls[val],estmLbls_val)
    sp_val = cm_val[0,0]/(cm_val[0,1]+cm_val[0,0]) #tn/(tn+fp)
        
    Accuracy_val[T] = acc_val
    fmeasure_val[T] = f_val
    precision_val[T] = pr_val
    recall_val[T] = rec_val
    AUC_val[T] = roc_auc_val
    MCC_val[T] = mcc_val
    Specificity_val[T] = sp_val
    
    ################## evaluating network on independentTest set###########
    estmLbls = model.predict_classes(x_test)
    f = f1_score(testLbls, estmLbls, average='macro') 
    acc = accuracy_score(testLbls, estmLbls)
    rec = recall_score(testLbls, estmLbls, average='macro')
    pr = precision_score(testLbls, estmLbls, average='macro')
    roc_auc = roc_auc_score(testLbls,estmLbls)
    mcc = matthews_corrcoef(testLbls, estmLbls)
    cm = confusion_matrix(testLbls, estmLbls)
    sp= cm[0,0]/(cm[0,1]+cm[0,0]) #tn/(tn+fp)
   
    Accuracy[T] = acc
    fmeasure[T] = f
    precision[T] = pr
    recall[T] = rec
    AUC[T] = roc_auc
    MCC[T] = mcc
    Specificity[T] = sp
       
    T = T+1 
    
print('**************************************************************')
indx = np.argmax(Accuracy)
print('The results on independent test set:')
print ('Accuracy= ', Accuracy[indx])
print ('F measure= ', fmeasure[indx])
print ('precision= ',precision[indx])
print ('recall= ', recall[indx])
print ('AUC= ', AUC[indx])
print ('MCC= ', MCC[indx])
print ('Specificity= ', Specificity[indx])
print ('                                    ')

print('The 10-fold cross-validation performance measures on training set:')
print ('Accuracy= ', np.mean(Accuracy_val))
print ('F measure= ', np.mean(fmeasure_val))
print ('precision= ', np.mean(precision_val))
print ('Recall= ', np.mean(recall_val))
print ('AUC= ', np.mean(AUC_val))
print ('MCC= ', np.mean(MCC_val))
print ('Specificity= ', np.mean(Specificity_val))

