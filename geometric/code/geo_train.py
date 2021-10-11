# -*- coding: utf-8 -*-
"""
This is the main python script containing the training loop.
Input hyperparameters can be set up through argparse.

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

import os, random, torch , argparse, numpy as np, pandas as pd
from pathlib import Path
from os.path import normpath,join
from itertools import product
import wandb # Hyperdash is not supported anymore. Replaced by Weights & Bias
from torch.nn import DataParallel
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

#%% Parse arguments

def parseArguments():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-pro",  type=str, default='Frontiers',
                        help="Choose the name of the project to be logged into W&B.")
    parser.add_argument("--group", "-gr",  type=str, default='Cross',
                        help="Choose the name of the group of the runs.")
    parser.add_argument("--name", "-na",  type=str, default='Run',
                        help="Choose the name of each of the runs.")
    parser.add_argument("--data", "-d",  nargs='+', type=str, default=['ECAP','A'],
                        help="Choose dataset to be employed when running the code.")
    parser.add_argument("--path", "-p",  type=str, default='D:\\PhD\\DL\\Frontiers\\GitHub\\geometric',
                        help="Base path with the code and data folders")
    parser.add_argument("--folds", "-f", type=int, default=8,
                        help="Number of folds if cross-validation == True (Not list).")
    parser.add_argument("--num_epoch", "-ep",  nargs='+', type=int, default=[100],
                        help="Number of epochs")
    parser.add_argument("--learn_rate", "-lr",  nargs='+', type=float, default=[0.001],
                        help="Learning rate")
    parser.add_argument("--batch_size", "-bs", nargs='+',type=int, default=[16],
                        help="Number of folds if cross-validation == True")
    parser.add_argument("--drop_rate", "-dr",  nargs='+', type=float, default=[0.1],
                        help="Drop rate")
    parser.add_argument("--depth", "-dp",  nargs='+', type=int, default=[12],
                        help="Number of hidden layers")
    parser.add_argument("--hidden", "-hid",  nargs='+', type=int, default=[32],
                        help="Depth of hidden layers")
    parser.add_argument("--activation", "-act",  nargs='+', type=str, default=['elu'],
                        help="Activation function. 'elu' or 'relu'")
    parser.add_argument("--spline", "-sp",  nargs='+', type=bool, default=[True],
                        help="SplineConv (True) or SAGEConv (False)'")
    parser.add_argument("--kernel_size", "-k",  nargs='+', type=int, default=[5],
                        help="Kernel size")
    parser.add_argument("--weight_decay", "-wd",  nargs='+', type=float, default=[0],
                        help="Weight decay")
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

os.chdir(join(args.path,'code'))
data_path = join(args.path,'data')
    
from modelGeo import ECAPdataset # If I do not import the dataset class it raises and error
from modelGeo import GeometricPointNet

#%% Define all functions   

def train(loader):
    
    # Set the model in training mode: Just affects to dropout and batchnorm
    model.train()
    
    #Epoch loss
    loss_all = 0

    for data in loader: # Get Batch
        
        data = data.to(device) # Data to GPU
        optimizer.zero_grad() # Zero out the gradients
        output = model(data) # Forward pass batch
        label = data.y.to(device) # Extract labels
        
        loss = crit(output, label) # Compute loss
            
        loss.backward() # Calculate gradients (backpropagation)
        optimizer.step() # Update weights
        
        loss_all += data.num_graphs * loss.item() # Multiply loss by batch size (necessary due to graph dataset)
        
    return loss_all / len(train_dataset)

def evaluate(loader,len_data):
    
    #Set the model for evaluation
    model.eval()

    loss_all = 0
    conf = np.zeros([len(thresholds),4]) 

    with torch.no_grad():
        for data in loader:

            data = data.to(device) # Data to GPU
            pred = model(data) # Predict
            label = data.y.to(device) # Label to GPU
            loss = crit(pred, label) # Compute loss
            conf += thresholding(pred, label)/data.y.shape[0] # Compute correct higher percentiles

            loss_all += data.num_graphs * loss.item()  # Multiply loss by batch size (necessary due to graph dataset)
            
            
    return loss_all / len_data, conf/len_data

def predict(loader):
    
    #Set the model for evaluation
    model.eval()

    predictions = []
    labels = []
    indices = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device) # Batch to GPU
            pred = model(data) # Predict
             
            label = data.y.detach().cpu().numpy() # Ground truth
            pred = pred.detach().cpu().numpy() # Prediction
        
            labels.append(label) # Save batch Ground truth
            predictions.append(pred) # Save batch Prediction
        
            indices.append(data.case) # Test_indices
    
    return labels,predictions,indices

def thresholding(pred,label):
    
    # Given the threshold return boolean matrix with 1 if > thres 0 if <= 1
    
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

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return [true_positives, false_positives, true_negatives, false_negatives]

#%% Set hyperparameters:

results_by = args.results_by # Results can be grouped by keys 

thresholds = args.threshold

parameters = dict(
     
    dat = list(args.data)
    ,folds = [args.folds] # Cross-validation folds
    ,num_epoch = list(args.num_epoch) # Number of epochs for each fold
    ,learn_rate = list(args.learn_rate) # Learning rate
    ,batch_size = list(args.batch_size) # Batch size
    ,drop_rate = list(args.drop_rate) # Drop rate
    ,depth =list(args.depth) # Depth of hidden layers in local feature extractor
    ,hidden =list(args.hidden) # Number of channels in hidden layers
    ,activation = list(args.activation) # Activation function
    ,spline = list(args.spline) # SplineConv or SAGEConv
    ,kernel_size = list(args.kernel_size) # Kernel size
    ,weight_decay =list(args.weight_decay) # Weight decay
    ,split =list(args.split) # Training-testing split
    ,seed=list(args.seed) # Random seed
    ,loss_func = list(args.loss_func) # Options 'L1','SmoothL1','MSE'
    ,cross = [args.cross] # If True perform cross-validation
    
)

hyp_name  = ["Dataset","Folds","Epochs", "Learning rate", "Batch size", "Drop rate","Layer depth", "Hidden features","Activation"
         ,"Spline or Sage", "Kernel size", "Weight decay","Split", "Seed", 'Loss','Cross-validation']

#%% Dataframe to store all the data according to the employed keys

if parameters['cross'][0] == True:
    
    if sum([len(i) for i in [v for v in parameters.values()]])> len(parameters):
        raise ValueError('In cross-validation use a single value per hyperparameter')
        
    parameters['num_epoch'] = parameters['num_epoch']*parameters['folds'][0] # Repeat training as many times as folds

df = pd.DataFrame(columns = list(['Val_Mae','Test_Mae'])+list(parameters.keys()))
df['Val_Mae'],df['Test_Mae']= df['Val_Mae'].astype(float),df['Test_Mae'].astype(float)
conf_all = np.zeros([len(thresholds),4])

# Set experiment name
param_values = [v for v in parameters.values()]

#%% Grid search loop
   
i = 0

for hyper in product(*param_values): 
     
    i+=1    

    #%% Create log in hyperdash for monitoring
    config = dict(zip(hyp_name,hyper))
    run = wandb.init(project=args.project, group=args.group, name=config['Dataset'], job_type='Run',config=config)

    wandb.define_metric("Train loss", summary="min")
    wandb.define_metric("Validation loss", summary="min")
    wandb.define_metric("Test loss", summary="min")
    
    print('\n')
    
    #%% Save results depending on experiment
  
    result_path = normpath(join(args.path,'results/','_'.join([args.project,args.group,args.name])))
    Path(result_path).mkdir(parents=True, exist_ok=True)

    #%% Load dataset and divide the data in training and testing
    
    if config['Cross-validation'] == False:
        
        dataset = torch.load(join(data_path,config['Dataset']+'.dataset'))
    
        # Normalize vertex-wise coordinates
        norm = T.NormalizeScale()
        norm(dataset.data)
        
        num_cases = dataset.len() # Number of cases
        indices = list(np.arange(num_cases)) # Indices of all cases
        
        torch.manual_seed(config['Seed']) if config['Seed'] != None else False # Set random seed if not None
        
        train_all = random.sample(indices,int(round(num_cases*config['Split'])))
        val_ind = random.sample(train_all,int(np.ceil(len(train_all)*0.1)))
        train_ind = list(np.setdiff1d(train_all,val_ind))
        test_ind = list(np.setdiff1d(indices,train_all))
        
    elif i == 1 and config['Cross-validation'] == True:
        
        dataset = torch.load(join(data_path,config['Dataset']+'.dataset'))
        
        # Shuffle the dataset
        dataset = dataset.shuffle()

        # Divide the dataset in folds
        folds = np.array_split(indices, config['Folds'])
            
    #%% Choose each fold for each iteration and loader instantiation
    
    if config['Cross-validation'] == True:    
        test_ind = folds[i-1] # Select testing cases for the iteration
        train_ind = list(np.setdiff1d(indices, test_ind)) # Remaining cases for training
        val_ind = random.sample(list(train_ind),round(len(train_ind)*0.1)) # Validation case %10 of training data
        train_ind = list(np.setdiff1d(train_ind, val_ind)) # Remove validation data from training
        
    train_dataset,val_dataset,test_dataset = dataset[list(train_ind)],dataset[list(val_ind)],dataset[list(test_ind)]

    print(train_dataset);print(val_dataset);print(test_dataset)
    
    # Pass the data to the loader
    train_loader,val_loader,test_loader = DataLoader(train_dataset, batch_size=config['Batch size']),DataLoader(val_dataset, batch_size=config['Batch size']),DataLoader(test_dataset, batch_size=1) 
   
    #%% Check GPU device availability
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
     
    #%% Create model instance
    
    model = GeometricPointNet(config['Spline or Sage'],config['Layer depth'],config['Hidden features'],config['Drop rate'],config['Kernel size'],config['Activation'])
    
    # If more than one GPU available paralelize
    if torch.cuda.device_count() > 1: 
        model = DataParallel(model) 

    model.to(device)
    
    #num_epoch = int(iterations/train_len + 0.5)
    print(model) # print full net
    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized model with {} trainable params ".format(params))
    
    #%% Model loss function and optimizer
    
    if config['Loss'] == 'MSE':
        crit = torch.nn.MSELoss()
    elif config['Loss'] == 'SmoothL1':
        crit = torch.nn.SmoothL1Loss()
    else:
        crit = torch.nn.L1Loss()
        
    optimizer = torch.optim.Adam(model.parameters(),lr=config['Learning rate'], weight_decay=config['Weight decay'])
    
    #%% Training loop    
    
    results = np.zeros([config['Epochs'],3])
    thres = np.zeros([config['Epochs'],len(thresholds),4])
    
    for epoch in range(config['Epochs']):
        
        loss = train(train_loader)
        
        loss_val,conf_val= evaluate(val_loader,len(val_dataset)) 
        loss_test,conf_test= evaluate(test_loader,len(test_dataset))
                        
        print('\nEpoch: {:03d}'. format(epoch))
        
        wandb.log({"Train loss": loss, "Validation loss": loss_val, "Test loss": loss_test})
    
        test_metric = loss_val 
       
        results[epoch,0],results[epoch,1],results[epoch,2]= loss, test_metric,loss_test
        thres[epoch,:,:]=conf_test
            
    #%% Print best results in iteration and save
    
    min_val = np.min(results[:,1])
    mae_test = float(results[np.where(np.isclose(results[:,1],min_val)),2].squeeze())
    conf_final = thres[np.where(np.isclose(results[:,1],min_val))[0][0],:,:]
    
    df.loc[i] = list([min_val,mae_test])+list(hyper) 
        
    print('\nFinal - Loss: {:.5f}, Accuracy: {:.5f}\n'.
          format(np.min(results[:,0]),np.min(results[:,2])))
    
    print('\nFinal Confusion matrix Th=4 - TP: {:.2f}, FP: {:.2f}, TN: {:.2f}, FN: {:.2f}\n'.
          format(conf_final[3,0],conf_final[3,1],conf_final[3,2],conf_final[3,3]))
    
    conf_all += conf_final
    
    #%% Cleanup and mark successfull completion
    
    run.finish()
    
#%% Print experiment summary

run = wandb.init(project=args.project, group=args.group, name=args.name+'_Summary', job_type= 'Summary',config=config) # Log hyperparameters WandB

df.to_pickle(os.path.join(result_path,'..','dataframe_'+'_'.join([args.project,args.group,args.name])+'.npy'))
conf_all = conf_all/i
    
keys = list(parameters.keys())

print('\n\nGeneral results:')          

#%% Plot cross-validation result

if config['Cross-validation'] == True:
    
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

if config['Cross-validation'] == False:
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
        
    
print('\nRates:\n')
for n,i in enumerate(conf_all):

    print('Th={:01d} - TPR: {:.2f}, FPR: {:.2f}'.
          format(thresholds[n],safe(i[0],i[0]+i[1]),safe(i[2],i[2]+i[3])))

#%% Cleanup and mark successfull completion

run.finish()


