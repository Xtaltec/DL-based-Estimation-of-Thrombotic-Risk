# -*- coding: utf-8 -*-
"""
Utility functions

@author: Xabier Morales Ferez - xabier.morales@upf.edu
"""

import torch, wandb, subprocess, argparse, numpy as np
from os.path import join
from torch_geometric.data import InMemoryDataset

#%% Parse arguments

def parseArguments():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-pro",  type=str, default='Frontiers',
                        help="Choose the name of the project to be logged into W&B.")
    parser.add_argument("--group", "-gr",  type=str, default='Prueba',
                        help="Choose the name of the group of the runs.")
    parser.add_argument("--data", "-d",  nargs='+', type=str, default=['ECAP','ECAP_Log','ECAP_Rotated','ECAP_Rotated_Log'],
                        help="Choose dataset to be employed when running the code.")
    parser.add_argument("--path", "-p",  type=str, default='C:\\Users\\Xabier\\PhD\\Frontiers\\GitHub\\geometric',#'D:\\PhD\\DL\\Frontiers\\GitHub\\geometric',
                        help="Base path with the code and data folders")
    parser.add_argument("--folds", "-f", type=int, default=2,#1,
                        help="Number of folds if cross-validation == True (Not list).")
    parser.add_argument("--num_epoch", "-ep",  nargs='+', type=int, default=[2],#100],
                        help="Number of epochs")
    parser.add_argument("--learn_rate", "-lr",  nargs='+', type=float, default=[0.001],
                        help="Learning rate")
    parser.add_argument("--batch_size", "-bs", nargs='+',type=int, default=[2],#16],
                        help="Number of folds if cross-validation == True")
    parser.add_argument("--drop_rate", "-dr",  nargs='+', type=float, default=[0.1],
                        help="Drop rate")
    parser.add_argument("--depth", "-dp",  nargs='+', type=int, default=[2],#12],
                        help="Number of hidden layers")
    parser.add_argument("--hidden", "-hid",  nargs='+', type=int, default=[2],#32],
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
    parser.add_argument("--cross", "-cr", type=bool, default=True,#False,
                        help="Cross-validation or hyperparameter tuning (Not list)")
    parser.add_argument("--results_by", "-rb",  nargs='+', type=str, default=[],
                        help="Activation function. 'elu' or 'relu'")
    parser.add_argument("--threshold", "-th",  nargs='+', type=float, default=[1,2,3,4,5,6,20,30,50],
                        help="Thresholds for binary classification")
    parser.add_argument("--n_images", "-ni",  type=int, default=5,
                        help="Number of images to plot")
    
    args = parser.parse_args()

    return args

def train(m,opt,dev,crit,loader,len_data):
    
    # Set the model in training mode: Just affects to dropout and batchnorm
    m.train()
    
    #Epoch loss
    loss_all = 0

    for data in loader: # Get Batch
        
        data = data.to(dev) # Data to GPU
        opt.zero_grad() # Zero out the gradients
        output = m(data) # Forward pass batch
        label = data.y.to(dev) # Extract labels
        
        loss = crit(output, label) # Compute loss
            
        loss.backward() # Calculate gradients (backpropagation)
        opt.step() # Update weights
        
        loss_all += data.num_graphs * loss.item() # Multiply loss by batch size (necessary due to graph dataset)
        
    return loss_all / len_data

def evaluate(m,opt,dev,crit,loader,setup,thres,len_data):
    
    #Set the model for evaluation
    m.eval()

    loss_all = 0
    conf = np.zeros([len(thres),4]) 

    with torch.no_grad():
        for data in loader:

            data = data.to(dev) # Data to GPU
            pred = m(data) if 'Log' not in setup["Dataset"] else torch.exp(m(data)) # Predict
            label = data.y.to(dev) if 'Log' not in setup["Dataset"] else torch.exp(data.y.to(dev)) # Label to GPU
            
            loss = crit(pred, label) # Compute loss
            conf += thresholding(pred, label,thres)/data.y.shape[0] # Compute correct higher percentiles

            loss_all += data.num_graphs * loss.item()  # Multiply loss by batch size (necessary due to graph dataset)
            
            
    return loss_all / len_data, conf/len_data

def predict(m,dev,loader,setup):
    
    #Set the model for evaluation
    m.eval()

    predictions = []
    labels = []
    indices = []

    with torch.no_grad():
        for data in loader:

            data = data.to(dev) # Batch to GPU
            pred = m(data) # Predict
            
            label = data.y.detach().cpu().numpy() if 'Log' not in setup["Dataset"] else np.exp(data.y.detach().cpu().numpy()) # Ground truth
            pred = pred.detach().cpu().numpy() if 'Log' not in setup["Dataset"] else np.exp(pred.detach().cpu().numpy()) # Prediction
        
            labels.append(label) # Save batch Ground truth
            predictions.append(pred) # Save batch Prediction
        
            indices.append(data.case) # Test_indices
    
    return labels,predictions,indices

def thresholding(pred,label,thres):
    
    # Given the threshold return boolean matrix with 1 if > thres 0 if <= 1
    
    conf =[]
    
    for i in thres:
        
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
    
def result_plotting(n_images,epoch,dataset,result_path,labels,predictions,indices):
    """ Takes predictions and plots them alongside the GT in W&B """
    
    coord,connectivity = [],[] # List of prediction and ground truth images
    for k in range(n_images):
        
        or_case = int(np.where(np.array(dataset.data.case)==indices[k])[0][0]) # Get index of case
        
        # Get the coordinate and connectivity data
        coord += [dataset[or_case].coord.numpy()] 
        connectivity += [dataset[or_case].connectivity.numpy()]

    # Save npz array with the results of this iteration
    np.savez(join(result_path,'Temp.npz'),labels=labels,predictions=predictions,indices=indices,coord=coord,connectivity=connectivity)

    # Call mesh_plotter to create the figures
    subprocess.call(['python', 'mesh_plotter.py','-p',result_path,'-e',str(epoch),'-ni',str(n_images)])
        
    # Send image for visualization in W&B
    images = wandb.Image(join(result_path,'Result_'+str(epoch).zfill(3)+'.png'),caption = "Epoch "+str(epoch))
    wandb.log({"Prediction": images})


# Defining the InMemoryDataset class
class ECAPdataset(InMemoryDataset):
    def __init__(self, input_data, output_path, transform=None, pre_transform=None):
        super(ECAPdataset, self).__init__(None, transform, pre_transform)
        self.data, self.slices = self.collate(input_data)
        


def flatten(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])
