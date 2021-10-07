# -*- coding: utf-8 -*-
"""
This is the main python script containing the training loop.
Input hyperparameters can be set up through argparse.

Based on the work:
    Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

import os, random, argparse, numpy as np, pandas as pd, scipy.io as sio  
from pathlib import Path
from os.path import normpath
from itertools import product
from skimage.draw import disk

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from hyperdash import Experiment

#%% Parse arguments

def parseArguments():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p",  type=str, default='C:\\Users\\Xabier\\PhD\\Frontiers\\GitHub\\unet', #'D:\\PhD\\Frontiers\\GitHub\\unet',  
                        help="Base path with the code and data folders")
    parser.add_argument("--data", "-d",  type=str, default='ECAP',
                        help="Choose dataset to be employed when running the code.")
    parser.add_argument("--experiment", "-exp",  type=str, default='Prueba',
                        help="Choose the name of the experiment to be logged into hyperdash.")
    parser.add_argument("--data_type", "-dt",  nargs='+', type=str, default=['c'],
                        help="Choose data representation. c: Cartesian grid, b: Bullseye plot")
    parser.add_argument("--folds", "-f", type=int, default=8,
                        help="Number of folds if cross-validation == True (Not list).")
    parser.add_argument("--num_epoch", "-ep",  nargs='+', type=int, default=[300],
                        help="Number of epochs")
    parser.add_argument("--learn_rate", "-lr",  nargs='+', type=float, default=[0.001],
                        help="Learning rate")
    parser.add_argument("--batch_size", "-bs", nargs='+',type=int, default=[16],
                        help="Number of folds if cross-validation == True")
    parser.add_argument("--drop_rate", "-dr",  nargs='+', type=float, default=[0.1],
                        help="Drop rate")
    parser.add_argument("--exponential", "-expo", nargs='+', type=int, default=[6],
                        help="Channel exponent to control network size")
    parser.add_argument("--decaylr", "-dlr",  type=bool, default=False,
                        help="If True learning rate decay is employed")
    parser.add_argument("--split", "-s",  nargs='+', type=float, default=[0.8],
                        help="Training-testing split")
    parser.add_argument("--seed", "-se",  nargs='+', type=int, default=[None],
                        help="Random seed")
    parser.add_argument("--loss_func", "-loss",  nargs='+', type=str, default=['L1'],
                        help="Loss function. Options 'L1','SmoothL1','MSE")
    parser.add_argument("--cross", "-cr", type=bool, default=False,
                        help="Cross-validation or hyperparameter tuning (Not list)")
    parser.add_argument("--results_by", "-rb",  nargs='+', type=str, default=[],
                        help="Activation function. 'elu' or 'relu'")
    parser.add_argument("--threshold", "-th",  nargs='+', type=float, default=[1,2,3,4,5,6,20,30,50],
                        help="Thresholds for binary classification")
    
    args = parser.parse_args()

    return args

args = parseArguments()

#%% Change dir and import models

os.chdir(args.path+'\\code\\')
data_path = args.path +'\\data\\'

from DfpNet import TurbNetG, weights_init
import utils

#%% Function definitions

def train(loader):
    
    # Set the model in training mode: Just affects to dropout and batchnorm
    model.train()
    
    #Epoch loss
    loss_all = 0
    
    for data in loader: # Get Batch
    
        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, num_epoch, lr*0.1, lr)
            if currLr < lr:
                for g in optimizer.param_groups:
                    g['lr'] = currLr
        
        shape, ecap = data
        shape, ecap = shape.to(device),ecap.to(device) # Data to GPU
        batch_len = shape.shape[0]
        optimizer.zero_grad() # Zero out the gradients
        output = model(shape) # Forward pass batch
        
        if data_type == 'b':
              
            filt_temp = torch.tensor(filt_l.repeat(batch_len,0)).float().to(device)
            ecap,output = torch.mul(ecap,filt_temp), torch.mul(output,filt_temp)
            filt_rat= filt_ratio.to(device)
            
        # Compute the batch loss
        loss = crit(output, ecap)*filt_rat if data_type == 'b' else crit(output, ecap)
        
        #loss = crit(output, ecap)*filt_rat if data_type == 'b' else crit(output, ecap) # Compute the batch loss
        loss.backward() # Calculate gradients (backpropagation)
        loss_all += batch_len * loss.item() 
        optimizer.step() # Update weights
        
    return loss_all / train_len

def evaluate(loader,len_data):
    
    #Set the model for evaluation
    model.eval()
    loss_all = 0
    conf = np.zeros([len(thresholds),4]) 

    with torch.no_grad():
        for data in loader:
            
            shape, ecap = data
            shape, ecap = shape.to(device),ecap.to(device) # Data to GPU
            batch_len = shape.shape[0]
            pred = model(shape)
            conf += thresholding(pred, ecap)/(ecap.shape[3]**2*ecap.shape[0])
            
            if not data_type == 'c':
                
                filt_temp = torch.tensor(filt_l.repeat(batch_len,0)).float().to(device)
                ecap,pred =  torch.mul(ecap,filt_temp), torch.mul(pred,filt_temp)
                filt_rat= filt_ratio.to(device)
            
            loss = crit(pred, ecap) if data_type == 'c' else crit(pred, ecap)*filt_rat
            
            loss_all += batch_len * loss.item()
    
    return loss_all/len_data, conf/len_data
            
def predict(loader):
    
    #Set the model for evaluation
    model.eval()

    predictions = []
    labels = []
    shape_out = []

    with torch.no_grad():
        for data in loader:

            shape, ecap = data
            shape = shape.to(device)
            shape_new = shape.detach().squeeze().cpu().numpy()* max_ori + min_ori
            
            if data_type == 'b': 
                shape_out.append(np.multiply(shape_new,filt)) 
                    
            else:
                shape_out.append(shape_new)

            # Ground truth            
            lab = ecap.detach().squeeze().cpu().numpy()
            labels.append(lab)
            
            # Prediction
            pred = model(shape).squeeze().detach().cpu().numpy()
            predictions.append(pred)
    
    return np.array(labels),np.array(predictions),np.array(shape_out)

def thresholding(pred,label):
    
    conf =[]
    if data_type == 'b':
        pred = pred[label!=0]
        label = label[label!=0]
        
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

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return [true_positives, false_positives, true_negatives, false_negatives]

#%% Set hyperparameters:

results_by = args.results_by # Results can be grouped by keys 

thresholds = args.threshold
   
parameters = dict(
    
    data_type = list(args.data_type), # c: image, b: bullseye
    folds = [args.folds], # Cross-validation folds
    num_epoch = list(args.num_epoch), # Number of epochs for each fold
    learn_rate = list(args.learn_rate), # Learning rate
    batch_size = list(args.batch_size), # Batch size
    drop_rate = list(args.drop_rate), # Drop rate
    expo = list(args.exponential), # Channel exponent to control network size
    split =list(args.split), # Training-testing split
    seed=list(args.seed), # Random seed
    loss_func = list(args.loss_func), # Options 'L1','SmoothL1','MSE'
    decayLr = [args.decaylr], # Learning rate decay
    cross = [args.cross] # If True perform cross-validation

)

#%% Datafram to store all the data according to the employed keys

if parameters['cross'][0] == True:
    
    if sum([len(i) for i in [v for v in parameters.values()]])> len(parameters):
        raise ValueError('In cross-validation use a single value per hyperparameter')
        
    parameters['num_epoch'] = parameters['num_epoch']*parameters['folds'][0] # Repeat training as many times as folds

df = pd.DataFrame(columns = list(['Val_Mae','Test_Mae'])+list(parameters.keys()))
df['Val_Mae'],df['Test_Mae']= df['Val_Mae'].astype(float),df['Test_Mae'].astype(float)
conf_all = np.zeros([len(thresholds),4])

# Set experiment name
experiment = args.experiment 
param_values = [v for v in parameters.values()]

#%% Grid search loop
i = 0

for hyper in product(*param_values): 
    
    i+=1 # Run ID   
    
    #%% Create log in hyperdash for monitoring
    
    exp = Experiment(args.experiment) # Log hyperparameters in hyperdash
    
    data_type = exp.param("Data representation", hyper[0])
    fold_num = exp.param("Folds", hyper[1])  
    num_epoch = exp.param("Number of epochs", hyper[2])
    lr = exp.param("Learning rate", hyper[3])
    batch_size = exp.param("Batch size", hyper[4]) 
    dropout = exp.param("Dropout", hyper[5])
    expo = exp.param("Network size",  hyper[6])
    split = exp.param("Dataset split", hyper[7])
    seed = exp.param("Seed",  hyper[8])
    loss_func = exp.param('Loss function', hyper[9])
    decayLr = exp.param("Lr decay", hyper[10])
    cross = exp.param('Cross-validation', hyper[11])
    
    print('\n')
    
    #%% Save results depending on experiment
  
    result_path = normpath(os.path.join(args.path,'results/',data_type+'_'+experiment))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    

    #%% Load data

    if cross == False or cross == True and i == 1:
        
        # Obtain min-max values to reconstruct shape
        Shape_ori = sio.loadmat(data_path+'Image/Shape.mat')['mapShape']
        min_ori = np.min(Shape_ori)
        max_ori = np.max(Shape_ori-min_ori)
        
        if data_type == 'b':
            
            ECAP = np.expand_dims(sio.loadmat(data_path+'Bullseye/ECAP.mat')['ECAP'],axis=1)
            ECAP = ECAP / np.max(ECAP) * np.max(sio.loadmat(data_path+'Image/ECAP.mat')['mapECAP'])
            
            X = sio.loadmat(data_path+'Bullseye/X.mat')['X'].astype(int)
            Y = sio.loadmat(data_path+'Bullseye/Y.mat')['Y'].astype(int)
            Z = sio.loadmat(data_path+'Bullseye/Z.mat')['Z'].astype(int)
            
            Shape = np.stack((X,Y,Z),axis=1).astype(int)
            
            Shape = Shape-np.min(Shape[Shape != 0])
            Shape = Shape / np.max(Shape)
            
            # Create circular filter to ignore outside the bullseye
            _,shape_r,shape_c,_= Shape_ori.shape
            filt = np.zeros([shape_r,shape_c])
            r,c = disk([shape_r/2,shape_c/2],shape_r/2-3)
            filt[r,c]=1
            filt = filt.astype(int)
            filt_l = np.expand_dims(filt,axis=(0,1))
            filt_ratio = torch.tensor(filt.size /np.sum(filt==1)).float()
            
        else:
        
            ECAP = np.expand_dims(sio.loadmat(data_path+'Image/ECAP.mat')['mapECAP'].transpose(0,2,1),axis=1)
            Shape = np.moveaxis(sio.loadmat(data_path+'Image/Shape.mat')['mapShape'],-1,1).transpose(0,1,3,2)
            
            Shape = Shape-np.min(Shape)
            Shape = Shape / np.max(Shape)
        
    #%% Training-testing data divide
    
    # Get dataset sizes and indices
    num_cases,_,image_size,_ = ECAP.shape
    train_len,test_len = int(np.ceil(num_cases*split)),int(np.floor(num_cases*(1-split)))
    
    if i ==1 : 
        
        iteration_best =1000 # Placeholder for best value
        indices = list(np.arange(num_cases))
        
        # Cross Validation
        if cross == True:
        
            # Shuffle dataset indices
            random.shuffle(indices)
            
            # Divide the dataset in folds
            folds = np.array_split(indices, fold_num)
            
        else:
            
            torch.manual_seed(seed) if seed != None else False # Set random seed if not None
            
            train_all = random.sample(indices,int(round(num_cases*split)))
            val_ind = random.sample(train_all,int(np.ceil(len(train_all)*0.1)))
            train_ind = list(np.setdiff1d(train_all,val_ind))
            test_ind = list(np.setdiff1d(indices,train_all))
           
    #%% Choose each fold for each iteration
    
    if cross == True:    
        test_ind = folds[i-1] # Select testing cases for the iteration
        train_ind = list(np.setdiff1d(indices, test_ind)) # Remaining cases for training
        val_ind = random.sample(list(train_ind),round(len(train_ind)*0.1)) # Validation case %10 of training data
        train_ind = list(np.setdiff1d(train_ind, val_ind)) # Remove validation data from training
    
    print('\n'+str(len(train_ind)));print(str(len(val_ind)));print(str(len(test_ind)))

    #%% Generate datasets
    x_train, y_train = torch.from_numpy(Shape[train_ind,:,:,:]), torch.from_numpy(ECAP[train_ind,:,:,:])                     
    x_val, y_val = torch.from_numpy(Shape[val_ind,:,:,:]), torch.from_numpy(ECAP[val_ind,:,:,:])   
    x_test, y_test = torch.from_numpy(Shape[test_ind,:,:,:]), torch.from_numpy(ECAP[test_ind,:,:,:])                   
    
    train_dataset, val_dataset = TensorDataset(x_train.float(),y_train.float()),TensorDataset(x_val.float(),y_val.float())
    test_dataset = TensorDataset(x_test.float(),y_test.float())
    
    #%% Instantiate loaders   
    train_loader,val_loader,test_loader = DataLoader(train_dataset, batch_size=batch_size),DataLoader(val_dataset, batch_size=batch_size),DataLoader(test_dataset, batch_size=1) 
    
    #%% Check device availability and instantiate model
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(TurbNetG(channelExponent=expo, dropout=dropout)) if torch.cuda.device_count() > 1 else TurbNetG(channelExponent=expo, dropout=dropout)
    model.to(device)
    
    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized TurbNet with {} trainable params ".format(params))
    
    #%% Initialize weights
    
    model.apply(weights_init)
    
    #%% Loss criterion
    if loss_func == 'MSE':
        crit = torch.nn.MSELoss()
    elif loss_func == 'SmoothL1':
        crit = torch.nn.SmoothL1Loss()
    else:
        crit = torch.nn.L1Loss()
                
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
    
    #%% Training loop
    results = np.zeros([num_epoch,3]) # Array to save results
    thres = np.zeros([num_epoch,len(thresholds),4])
    
    for epoch in range(num_epoch):
        
        loss = train(train_loader)
        
        loss_val,conf_val= evaluate(val_loader,len(val_dataset))
        loss_test,conf_test= evaluate(test_loader,len(test_dataset))

        print('\nEpoch: {:03d}'. format(epoch))
        exp.metric("Training loss", loss)
        exp.metric("Validation loss", loss_val)
        exp.metric("Test loss", loss_test)

        results[epoch,0],results[epoch,1],results[epoch,2]= loss,loss_val,loss_test
        thres[epoch,:,:]=conf_test
            
        if iteration_best >loss_val:
            
            iteration_best = loss_val
            labels,predictions,shape_out = predict(test_loader)
            np.save(os.path.join(result_path,'GT'+str(i)+'.npy'),labels)
            np.save(os.path.join(result_path,'pred'+str(i)+'.npy'),predictions)
            np.save(os.path.join(result_path,'shape'+str(i)+'.npy'),shape_out)
            np.save(os.path.join(result_path,'test_index'+str(i)+'.npy'),test_ind)
            
    #%% Print best results in iteration and save in the dataframe
    
    min_val = np.min(results[:,1])
    mae_test = float(results[np.where(np.isclose(results[:,1],min_val)),2].squeeze())
    conf_final = thres[np.where(np.isclose(results[:,1],min_val))[0][0],:,:]
    
    df.loc[i] = list([min_val,mae_test])+list(hyper) 
    
    print('\nFinal - Train Loss: {:.5f}, Test Loss: {:.5f}\n'.
          format(np.min(results[:,0]),np.min(results[:,2])))
    
    print('\nFinal Confusion matrix Th=4 - TP: {:.2f}, FP: {:.2f}, TN: {:.2f}, FN: {:.2f}\n'.
          format(conf_final[3,0],conf_final[3,1],conf_final[3,2],conf_final[3,3]))
    
    conf_all += conf_final
    
    #%% Cleanup and mark that the experiment successfully completed  
    exp.end()
    
#%% Print experiment summary

exp = Experiment(args.experiment) # Log hyperparameters in hyperdash    

df.to_pickle(os.path.join(result_path,'..','dataframe_'+str(args.experiment)+'.npy'))
conf_all = conf_all/i

# Plot mean accuracy for each hyperparameter with multiple values
keys = list( parameters.keys())

print('\n\nGeneral results:')          

#%% Plot cross-validation result

if cross==True:
    
    mean = df.Val_Mae.mean()
    std = df.Val_Mae.std()
    
    print('\nValidation MAE accuracy: {:.5f} + {:.5f}\n'.format(mean,std)) 
    
    for j in range(len(df)):
        print(str(j)+': {:.5f}'.format(df.iloc[j][0]))
        
    mean = df.Test_Mae.mean()
    std = df.Test_Mae.std()
        
    print('\nTesting MAE accuracy: {:.5f} + {:.5f}\n'.format(mean,std)) 
    
    for j in range(len(df)):
        print(str(j)+': {:.5f}'.format(df.iloc[j][1]))    

#%% Plot results grouped by hyperparameters

for i in keys:
    if len(df[i].unique())>1:
        
        print('\n Val MAE value for '+i)            
        mean = df.groupby(df[i])['Val_Mae'].mean()
        std = df.groupby(df[i])['Val_Mae'].std()
        
        for j,k,m in zip(mean.index,mean.values,std.values):
            print(str(j)+': {:.5f} + {:.5f}'.format(k,m))
            
        print('\n Test MAE value for '+i)            
        mean = df.groupby(df[i])['Test_Mae'].mean()
        std = df.groupby(df[i])['Test_Mae'].std()
        
        for j,k,m in zip(mean.index,mean.values,std.values):
            print(str(j)+': {:.5f} + {:.5f}'.format(k,m))
 
if cross == False:
    group_by = [keys.index(i) for i in results_by] 
    
    for g in group_by:
        
        print('\n\nBy '+keys[g]+':')        
        
        for i in list(keys.copy()[:g]+keys.copy()[g+1:]):
            if len(df[i].unique())>1:
                
                print('\nMean value for '+i)            
                mean = df.groupby([df[keys[g]],df[i]])['Test_Mae'].mean()
                std = df.groupby([df[keys[g]],df[i]])['Test_Mae'].std()
                
                for j,k,m in zip(mean.index,mean.values,std.values):
                    print(str(j)+': {:.5f} + {:.5f}'.format(k,m)) 
    print('\n')


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

#%% Cleanup and mark that the experiment successfully completed
exp.end()