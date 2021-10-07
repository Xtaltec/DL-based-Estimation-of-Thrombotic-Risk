# -*- coding: utf-8 -*-
"""
This is the main python script containing the training loop.
Input hyperparameters can be set up through argparse.

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

import os, pyvista as pv, numpy as np,scipy.io as sio,time, random, matlab.engine, argparse
from hyperdash import Experiment
from pathlib import Path
from glob import glob
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers

#%% Define functions

def thresholding(pred,label):
    
    conf =[]
    for i in thresholds:
        
        pr_th,lab_th = (pred>i),(label>i)
        conf += confusion(pr_th,lab_th)
    
    return np.array(conf).reshape(-1,4)

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = np.sum(confusion_vector == 1)
    false_positives = np.sum(confusion_vector == float('inf'))
    true_negatives = np.sum(np.isnan(confusion_vector))
    false_negatives = np.sum(confusion_vector == 0)

    return [true_positives, false_positives, true_negatives, false_negatives]

#%% Parse input arguments

def parseArguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p",  type=str, default='D:\\PhD\\Frontiers\\GitHub\\fcn',
                        help="Base path with the code and data folders")
    parser.add_argument("--data", "-d",  type=str, default='D:\\PhD\\Frontiers\\geo\\data\\LAA_Smoothed',
                        help="Choose dataset to be employed when running the code.")
    parser.add_argument("--experiment", "-exp",  type=str, default='Prueba',
                        help="Choose the name of the experiment to be logged into hyperdash.")
    parser.add_argument("--folds", "-f", type=int, default=8,
                        help="Number of folds if cross-validation == True (Not list).")
    parser.add_argument("--num_epoch", "-ep", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--learn_rate", "-lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--batch_size", "-bs",type=int, default=16,
                        help="Number of folds if cross-validation == True")
    parser.add_argument("--drop_rate", "-dr", type=float, default=0.2,
                        help="Drop rate")
    parser.add_argument("--principal_component", "-pc", type=float, default=32,
                        help="Number of principal components preserved when performing truncated PCA")
    parser.add_argument("--split", "-s", type=float, default=0.8,
                        help="Training-testing split")
    parser.add_argument("--seed", "-se", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--loss_func", "-loss", type=str, default='L1',
                        help="Loss function. Options 'L1','SmoothL1','MSE")
    parser.add_argument("--cross", "-cr", type=bool, default=False,
                        help="Cross-validation or hyperparameter tuning (Not list)")
    parser.add_argument("--results_by", "-rb",  nargs='+', type=str, default=[],
                        help="Activation function. 'elu' or 'relu'")
    parser.add_argument("--threshold", "-th",  nargs='+', type=float, default=[1,2,3,4,5,6],
                        help="Thresholds for binary classification")
    
    args = parser.parse_args()

    return args

args = parseArguments()

os.chdir(args.path+'\\code\\')
data_path = args.path+'\\data\\'
result_path =args.path+'\\results\\'

#%% Load the data from the simulation results (.vtk,.vtu,.vtp ... meshes)

geo_list = glob(args.data+'\\*.vt*') # Find all files with vt file extension

# Read the final ECAP data from the resulting meshes
point,ECAP = [],[]

for i in geo_list:
    
    mesh = pv.read(i)
    point += [mesh.points]
    ECAP += [mesh.point_arrays['ECAP_Both']]
    
point, ECAP = np.array(point),np.array(ECAP)

ShapeData = np.moveaxis(np.reshape(point,[point.shape[0],point.shape[1]*3]),1,0)
StressData = np.moveaxis(ECAP,1,0)

# Save the files for the unsupervised learning in Matlab
ShapeDataFile = data_path+'Shape_Final.mat'
StressDataFile = data_path+'ECAP_Final.mat'

sio.savemat(ShapeDataFile,{'ShapeData':ShapeData})
sio.savemat(StressDataFile,{'StressData':StressData})

geo_ind = np.asarray(np.squeeze(np.load(data_path+'test_index.npy')),dtype=int)-1
original_ind = np.asarray(np.load(data_path+'case_list.npy'),dtype=int)-1

TempDataFile = data_path+'TempData.mat' # File where all the MATLAB unsupervised learning data is going to be stored
ResultFile = result_path+'DL_ECAP_result.mat' # Name of the file with the results of the DL analysis

#%% Hyperparameters

experiment_name = args.experiment # Name of the experiment
nNodes=StressData.shape[0] #Number of nodes in geometry
SV_Shape=args.principal_component #Retained Single Values of Shape
split = args.split # Train-test split
batchS=args.batch_size # Batch size
Drop= args.drop_rate # Drop rate
nEpoch= args.num_epoch # Number of epochs, tested best value to avoid overfitting 
Act='relu' # Activation unit type
lr=args.learn_rate # Learning rate
thresholds = args.threshold # Threshold for binary classification

var,var_test =[],[]
conf_all = np.zeros([len(thresholds),4])

#%% Unsupervised Learning is done in Matlab
def UnsupervisedLearning(DataFile, ShapeDataFile, StressDataFile, IdxList_train, IdxList_test,SV_Shape,nNodes):
    train_ind_mat=matlab.double(list(IdxList_train+1)) #+1 to matlab index
    test_ind_mat=matlab.double(list(IdxList_test+1)) #+1 to matlab index
    DataFlag = eng.UnsupervisedLearning(DataFile, ShapeDataFile, StressDataFile, train_ind_mat,test_ind_mat,SV_Shape,nNodes)
    MatData=sio.loadmat(DataFile)
    
    X=MatData['ShapeCode_train']
    X=np.asmatrix(X)
    X=X.transpose()

    X_t=MatData['ShapeCode_test']
    X_t=np.asmatrix(X_t)
    X_t=X_t.transpose()
  
    S=MatData['StressData_train']
    S=np.asmatrix(S)

    S_t=MatData['StressData_test']
    S_t=np.asmatrix(S_t)

    Proj=MatData['Proj']

    MeanShape=MatData['MeanShape']
    Var=MatData['V123']
    
    return X, X_t, S, S_t,Proj, MeanShape,Var

#%% Define the fully connected mlp to perform the non linear mapping

def CreateModel_NonlinearMapping2(Xshape, Yshape,lr):
    
    opt = optimizers.Adam(learning_rate=lr)
    
    model = Sequential()
    model.add(Dense(128, input_dim=Xshape[1], kernel_initializer='normal', activation=Act))
    model.add(Dense(256, kernel_initializer='normal', activation=Act))    
    model.add(Dense(512, kernel_initializer='normal', activation=Act))
    model.add(Dropout(Drop))        
    model.add(Dense(1024, kernel_initializer='normal', activation=Act))
    model.add(Dropout(Drop))  
    model.add(Dense(2048, kernel_initializer='normal', activation=Act))
    model.add(Dropout(Drop))
    model.add(Dense(Yshape[1], kernel_initializer='normal',activation='linear')) 

    model.compile(loss='mse', optimizer=opt , metrics=['accuracy','mse','mae','mape','cosine'])
    
    return model

#%% Initialize all the data

os.chdir(args.path+'\\code\\')
eng = matlab.engine.start_matlab() # Start the MATLAB engine

num_cases=ShapeData.shape[1]; # Total number of simulations
IndexList= np.arange(0, num_cases, 1) #Lista de numero de simulaciones

indices = list(np.arange(num_cases))
random.seed(args.seed)
random.shuffle(indices)

if args.cross == True:
    folds = np.array_split(indices,args.folds)
    it = args.folds # Training iterations
else:
    it = 1 # Training iterations
           
    train_all = random.sample(indices,int(round(num_cases*split)))
    val_ind = random.sample(train_all,int(np.ceil(len(train_all)*0.1)))
    train_ind = list(np.setdiff1d(train_all,val_ind))
    test_ind = list(np.setdiff1d(indices,train_all))
    
#%% Loop over training mode 
mae_val, mae_test, mape_test = [],[],[]
    
#%%     
for i in range(it):
    
    exp = Experiment(experiment_name) # Log experiment in hyperdash
    
    IndexList_train,IndexList_val, IndexList_test = [],[],[]; # List of testing dataset

    if args.cross == True:    
        test_ind = folds[i] # Select testing cases for the iteration
        train_ind = list(np.setdiff1d(indices, test_ind)) # Remaining cases for training
        val_ind = random.sample(list(train_ind),round(len(train_ind)*0.1)) # Validation case %10 of training data
        train_ind = list(np.setdiff1d(train_ind, val_ind)) # Remove validation data from training
    
    # Print training, validation and testing dataset legth
    print('\nTrain : '+str(len(train_ind)))
    print('\nVal : '+str(len(val_ind)))
    print('Test : '+str(len(test_ind))+'\n')  
    print(str(test_ind)+'\n')
    
    IndexList_train.append(train_ind),IndexList_val.append(val_ind),IndexList_test.append(test_ind) 
    ShapeData_train,ShapeData_val,ShapeData_test = np.transpose(ShapeData[:,train_ind]),np.transpose(ShapeData[:,val_ind]),np.transpose(ShapeData[:,test_ind])
    StressData_train,StressData_val,StressData_test =np.transpose(StressData[:,train_ind]),np.transpose(StressData[:,val_ind]),np.transpose(StressData[:,test_ind]) 
    #%% Truncated - PCA for dimensionality reduction
    
    t1=time.perf_counter()
    
    [X, X_val, S, S_val,Proj, MeanShape,Var]=UnsupervisedLearning(TempDataFile, ShapeDataFile, StressDataFile
                                                                        ,np.array(train_ind), np.array(val_ind),SV_Shape,nNodes)
    
    var +=[Var]
    [X, X_t, S, S_t,Proj, MeanShape,Var]=UnsupervisedLearning(TempDataFile, ShapeDataFile, StressDataFile
                                                                        ,np.array(train_ind), np.array(test_ind),SV_Shape,nNodes)
    var_test+=[Var]
    
    #%% Fit model
    
    # Create the neural network model and perform the non linear mapping
    NMapper=CreateModel_NonlinearMapping2(X.shape, StressData_train.shape,lr)
    
    # Training of the network
    history = NMapper.fit(X,StressData_train , epochs=nEpoch, batch_size=batchS, 
                          validation_data=(X_val, StressData_val),verbose=0)
    
    # Predict the low dimensional ECAP representations
    Yp=NMapper.predict(X_t, batch_size=len(test_ind), verbose=0)
    results = NMapper.evaluate(X_t,StressData_test, verbose=0)

    conf_all += thresholding(Yp, StressData_test)/(Yp.shape[0]*Yp.shape[1])
    
    print('\nResults: ')
    print('MAE validation: ', np.min(history.history['val_mae']))
    print('MAE testing: ',results[3])
        
    if args.cross==True:
        
        mae_val += [np.min(history.history['val_mae'])]
        mae_test += [results[3]]
         
    exp.end()

#%% Print final results   

conf_all = conf_all/(i+1)

exp = Experiment(experiment_name)     

if args.cross==True:
     print('\nFinal results from cross-validation training: \n')
     print('MAE validation: {:.5f} + {:.5f}'.format(np.mean(mae_val),np.std(mae_val))) 
     print('MAE test: {:.5f} + {:.5f}'.format(np.mean(mae_test),np.std(mae_test)))

print('\nThresholding results:\n')
for n,i in enumerate(conf_all):
    print('Th={:01d} - TP: {:.2f}, FP: {:.2f}, TN: {:.2f}, FN: {:.2f}'.
          format(thresholds[n],i[0],i[1],i[2],i[3]))
    
def safe(x,y):
    if y == 0:
        return 0
    return x / y   
        
    
print('\nRates:')
for n,i in enumerate(conf_all):

    print('Th={:01d} - TPR: {:.2f}, FPR: {:.2f}'.
          format(thresholds[n],safe(i[0],i[0]+i[1]),safe(i[2],i[2]+i[3])))     
   
exp.end()

exp_path = os.path.join(result_path,experiment_name)
Path(exp_path).mkdir(parents=True, exist_ok=True)
np.save(os.path.join(exp_path,'mae_val.npy'),mae_val)
np.save(os.path.join(exp_path,'mae_test.npy'),mae_test)
np.save(os.path.join(exp_path,'prediction.npy'),Yp)
    
