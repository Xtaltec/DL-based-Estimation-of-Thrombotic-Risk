# -*- coding: utf-8 -*-
"""
This is the main python script containing the training loop.
Input hyperparameters can be set up through argparse.

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

import os, random, torch, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from os.path import join
from itertools import product
import wandb # Hyperdash is not supported anymore. Replaced by Weights & Bias
from torch.nn import DataParallel
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from utils import parseArguments,train,evaluate,predict,result_plotting,ECAPdataset # If I do not import the dataset Class it raises an error
from modelGeo import GeometricPointNet

#%% Main code
if __name__ == "__main__":
    
    args = parseArguments()

    #%% Change dir and import models
    
    os.chdir(join(args.path,'code'))
    data_path = join(args.path,'data')

    #%% Set hyperparameters:
    
    results_by = args.results_by # Results can be grouped by keys 
    thresholds = args.threshold # Values for the thresholding
    
    # Dictionary with all input parameters
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
    
    # Names of all parameters to better display in W&B
    hyp_name  = ["Dataset","Folds","Epochs", "Learning rate", "Batch size", "Drop rate","Layer depth", "Hidden features","Activation"
             ,"Spline or Sage", "Kernel size", "Weight decay","Split", "Seed", 'Loss','Cross-validation']

    hyp_short  = np.array(["","f","ep", "lr", "bs", "dr","depth", "hidden","act"
             ,"spline", "ks", "wd","sp", "seed", 'loss','cross'])
    
    #%% Dataframe to store all the data according to the employed keys
    
    if parameters['cross'][0] == False: # Ensure that folds = 1 if cross = False
        parameters['folds'][0]=1
        
    name_list = np.array([1 if len(i)>1 else 0 for i in parameters.values()]).astype(bool) # Array with the differential hyperparameters
    name_list[0]=1 # Always use at least the dataset as run name
    
    df = pd.DataFrame(columns = list(['Val_Mae','Test_Mae'])+list(parameters.keys())) # Add the columns for the validation and test values
    df['Val_Mae'],df['Test_Mae']= df['Val_Mae'].astype(float),df['Test_Mae'].astype(float)
    
    # Set experiment name
    param_values = [v for v in parameters.values()]
    
    #%% Grid search loop
       
    it,setup_old = 0,None
    
    # Repeat run with given hyperparameters as many times as folds
    for hyper in product(*param_values):
        
        conf_all = np.zeros([len(thresholds),4]) # Array to safe the confusion matrix data for cross-validation
        
        for _ in range(parameters['folds'][0]):
             
            it+=1    
            
            # Run name based on the parameters used in the grid search
            run_name = '_'.join(['_'.join([x,y]) for x,y in zip(hyp_short[name_list],np.array(hyper).astype(str)[name_list])])[1:]
            
            #%% Create log in W&B for monitoring
            
            setup = dict(zip(hyp_name,hyper))
            run = wandb.init(project=args.project, group=args.group, name=run_name, job_type='Run',config=setup)
        
            wandb.define_metric("Train loss", summary="min")
            wandb.define_metric("Validation loss", summary="min")
            wandb.define_metric("Test loss", summary="min")
            
            print('\n')
            
            #%% Save results depending on project-group-run name
          
            result_path = join(args.path,'results',args.project,args.group,run_name)
            Path(result_path).mkdir(parents=True, exist_ok=True)
        
            #%% Load dataset and divide the data in training and testing
            
            if setup['Cross-validation'] == False:
                
                random.seed(setup['Seed']) if setup['Seed'] != None else False # Set random seed if not None
                
                dataset = torch.load(join(data_path,setup['Dataset']+'.dataset'))
            
                # Normalize vertex-wise coordinates
                norm = T.NormalizeScale()
                norm(dataset.data)
                
                num_cases = dataset.len() # Number of cases
                aux_ind = list(np.arange(num_cases)) # Indices of all cases
                
                train_all = random.sample(aux_ind,int(round(num_cases*setup['Split'])))
                val_ind = random.sample(train_all,int(np.ceil(len(train_all)*0.1)))
                train_ind = list(np.setdiff1d(train_all,val_ind))
                test_ind = list(np.setdiff1d(aux_ind,train_all))
                
                f = 0 # Counter for folds
                
            elif setup_old != setup and setup['Cross-validation'] == True:
                
                torch.manual_seed(setup['Seed']) if setup['Seed'] != None else False # Set random seed if not None

                dataset = torch.load(join(data_path,setup['Dataset']+'.dataset'))
                
                # Normalize vertex-wise coordinates
                norm = T.NormalizeScale()
                norm(dataset.data)
                
                num_cases = dataset.len() # Number of cases
                aux_ind = list(np.arange(num_cases)) # Indices of all cases
                
                # Shuffle the dataset
                dataset = dataset.shuffle()
        
                # Divide the dataset in folds
                folds = np.array_split(aux_ind, setup['Folds'])
                
                setup_old = setup.copy() # Placeholder to detect when setup changes
                f = -1 # Counter for folds
                    
            #%% Choose each fold for each iteration and loader instantiation
            
            if setup['Cross-validation'] == True:    
                
                test_ind = folds[f] # Select testing cases for the iteration
                train_ind = list(np.setdiff1d(aux_ind, test_ind)) # Remaining cases for training
                val_ind = random.sample(list(train_ind),round(len(train_ind)*0.1)) # Validation case %10 of training data
                train_ind = list(np.setdiff1d(train_ind, val_ind)) # Remove validation data from training
                f += 1 # Counter for cross-validations folds
                
            train_dataset,val_dataset,test_dataset = dataset[list(train_ind)],dataset[list(val_ind)],dataset[list(test_ind)]
        
            print(train_dataset);print(val_dataset);print(test_dataset);print('\n')
            print(test_ind);print('\n')
            
            # Pass the data to the loader
            train_loader,val_loader,test_loader = DataLoader(train_dataset, batch_size=setup['Batch size']),DataLoader(val_dataset, batch_size=setup['Batch size']),DataLoader(test_dataset, batch_size=1) 
           
            #%% Check GPU device availability
            
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            print('Let\'s use', torch.cuda.device_count(), 'GPUs! \n')
             
            #%% Create model instance
            
            model = GeometricPointNet(setup['Spline or Sage'],setup['Layer depth'],setup['Hidden features'],setup['Drop rate'],setup['Kernel size'],setup['Activation'])
            
            # If more than one GPU available paralelize
            if torch.cuda.device_count() > 1: 
                model = DataParallel(model) 
        
            model.to(device)
            
            #num_epoch = int(iterations/train_len + 0.5)
            print(model) # print full net
            model_parameters = filter(lambda p: p.requires_grad,model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("\nInitialized model with {} trainable params \n".format(params))
            
            #%% Model loss function and optimizer
            
            if setup['Loss'] == 'MSE':
                crit = torch.nn.MSELoss()
            elif setup['Loss'] == 'SmoothL1':
                crit = torch.nn.SmoothL1Loss()
            else:
                crit = torch.nn.L1Loss()
                
            optimizer = torch.optim.Adam(model.parameters(),lr=setup['Learning rate'], weight_decay=setup['Weight decay'])
            
            #%% Training loop    
            
            results = 1000*np.ones([setup['Epochs'],3])
            thres = np.zeros([setup['Epochs'],len(thresholds),4])
            
            for epoch in range(setup['Epochs']):
                
                loss = train(model,optimizer,device,crit,train_loader,len(train_dataset))
                
                loss_val,conf_val= evaluate(model,optimizer,device,crit,val_loader,setup,thresholds,len(val_dataset))
                loss_test,conf_test= evaluate(model,optimizer,device,crit,test_loader,setup,thresholds,len(test_dataset))
                                
                print('\nEpoch: {:03d}'. format(epoch))
                
                wandb.log({"Train loss": loss, "Validation loss": loss_val, "Test loss": loss_test})
                
                # Plot prediction images in wandb
                if loss_test < np.min(results[:,2]):
                
                    labels,predictions,indices = predict(model,device,test_loader,setup) # Get predictions
                    result_plotting(args.n_images,epoch,dataset,result_path,labels,predictions,indices) # Plot predictions in W&B
                
                results[epoch,0],results[epoch,1],results[epoch,2]= loss,loss_val,loss_test
                thres[epoch,:,:]=conf_test
                    
            #%% Print best results in iteration and save
            
            min_val = np.min(results[:,1]) # Minimum validation mae
            mae_test = float(results[np.where(np.isclose(results[:,1],min_val)),2].squeeze()) # Testing value for minimum validation
            conf_final = thres[np.where(np.isclose(results[:,1],min_val))[0][0],:,:] # Iteration confusion matrix
            
            df.loc[it] = list([min_val,mae_test])+list(hyper) # Save the best results in dataframe
                
            print('\nFinal - Loss: {:.5f}, Accuracy: {:.5f}\n'.
                  format(np.min(results[:,0]),np.min(results[:,2])))
            
            print('\nFinal Confusion matrix Th=4 - TP: {:.2f}, FP: {:.2f}, TN: {:.2f}, FN: {:.2f}\n'.
                  format(conf_final[3,0],conf_final[3,1],conf_final[3,2],conf_final[3,3]))
            
            conf_all += conf_final # Sum the confusion matrix results for all folds 
            
            run.finish() # Cleanup and mark successfull completion
        
        #%% Print experiment summary and save dataframe
        
        print('\n\nGeneral results:')
        
        run = wandb.init(project=args.project, group=args.group, name=run_name+'_Summary', job_type= 'Summary',config=setup) # Log hyperparameters WandB
        
        keys = list(parameters.keys())
        conf_all = conf_all/(f+1) # Divide by number of folds(+1 because it starts from 0)
        
        # Save dataframe with data from the cross-validation run 
        pd.set_option("display.max_columns", None) # To avoid weird pandas issue
        df_cross = df.loc[it-f:it]
        df_cross.to_pickle(os.path.join(result_path,'dataframe_'+'_'.join([args.project,args.group,run_name])+'.npy'))

        #%% Plot cross-validation result

        if setup['Cross-validation'] == True:
            
            mean = df_cross.Val_Mae.mean()
            std = df_cross.Val_Mae.std()
            
            print('\nValidation MAE accuracy: {:.5f} + {:.5f}\n'.format(mean,std)) 
            
            for j in range(len(df_cross)):
                print(str(j)+': {:.5f}'.format(df_cross.iloc[j][0]))
                
            mean = df_cross.Test_Mae.mean()
            std = df_cross.Test_Mae.std()
                
            print('\nTesting MAE accuracy: {:.5f} + {:.5f}\n'.format(mean,std)) 
            
            for j in range(len(df_cross)):
                print(str(j)+': {:.5f}'.format(df_cross.iloc[j][1]))
                
            run.finish() # Cleanup and mark successfull completion
            
    #%% Final group summary grouped by hyperparameters
      
    run = wandb.init(project=args.project, group=args.group, name='_'.join([args.group,'Summary',datetime.now().strftime('%y_%m_%d_%H%M')]), job_type= 'Summary') # Log hyperparameters WandB

    # By hyperparameters with more than one value
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
    
    group_by = [keys.index(i) for i in results_by] 
    
    # By keys defined by the user
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
        
    # Save dataframe with all runs
    df.to_pickle(os.path.join(result_path,'..','dataframe_'+'_'.join([args.project,args.group,datetime.now().strftime('%y_%m_%d_%H%M')])+'.npy'))

    run.finish() # Cleanup and mark successfull completion
    
   