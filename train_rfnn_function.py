import torch
import numpy as np
from sklearn.metrics import roc_auc_score

'''
f_train_rfnn_model trains a rfnn_model.

Input parameters:
    
    - model: The rfnn_a model to be trained
    
    - model_device: The device of the model
    
    - X_train, X_test, y_train, y_test: The torch.tensors containing the training and test data.
                                        The number and order of samples have to be consitent between X_* and y_*
    
    - lr_rate: The learning rate to use for the training of each individual tnn-model.
    
    - num_epochs: The number of epochs for which to train each individual tnn-model.
    
    - nums_tnn: Total number of trees to construct within the training process.
'''

def f_train_rfnn_model (model, model_device, X_train, X_test, y_train, y_test, lr_rate, num_epochs, nums_tnn):
    
    test_roc_aucs = np.zeros((nums_tnn))
    
    # For each new tree DO:
    
    for no_t in range(nums_tnn):
        
        # Create the new tree
        
        if no_t !=0:
            model.add_tree(model_device)
        
        # Calculate the current output not including the new tree
        
        pre_model_odds = np.zeros((X_train.shape[0], max(1, no_t)))
        pre_model_odds_val = np.zeros((X_test.shape[0], max(1, no_t)))
        
        model.eval()
        with torch.no_grad():
            
            for n in range(no_t):
                pre_model_odds[:,n] = model.single_tree_forward(n,torch.tensor(X_train.values, dtype = torch.float32).to(model_device), return_log_odds = True).detach().clone().cpu().numpy()
                pre_model_odds_val[:,n] = model.single_tree_forward(n,torch.tensor(X_test.values, dtype = torch.float32).to(model_device), return_log_odds = True).detach().clone().cpu().numpy()
        model.train()
        
        # Create the optimizer for the new tree
        
        optim_dict = model.get_tree_optim(no_t,lr_rate)
        curr_tree = optim_dict['tnn']
        curr_optim = optim_dict['optim']
        
        # Define the weights for the sum of log-odds of all trees.
        
        if no_t > 0:
            weights = torch.ones(no_t + 1).to(model_device) * model.grow_shrinkage
            weights[0] = 1
        else:
            weights = 1
        
        # For each training epoch DO:
        
        for n in range(num_epochs):
            
            # Create output of the current tree
            
            curr_output = model.single_tree_forward(no_t, torch.tensor(X_train.values, dtype = torch.float32).to(model_device), return_log_odds = True).to(model_device)
            
            # Add output to the former trees' log-odds
            
            if no_t > 0:
                full_output = (torch.tensor(pre_model_odds, dtype = torch.float32).to(model_device) * weights[:-1]).sum(1) + curr_output * weights[-1]
            else:
                full_output = curr_output
            
            # Backward pass on the summed output of all trees (calculating gradients only for the new tree)    
            
            loss = torch.nn.BCELoss()(torch.sigmoid(full_output), y_train)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()
            
            
            # During the last epoch calculate the model's performance after training the new tree
            
            if n == (num_epochs - 1):
                curr_tree.eval()
                curr_output_val = model.single_tree_forward(no_t, torch.tensor(X_test.values, dtype = torch.float32).to(model_device), return_log_odds = True).to(model_device)
                
                if no_t > 0:
                    full_output_val = (torch.tensor(pre_model_odds_val, dtype = torch.float32).to(model_device) * weights[:-1]).sum(1) + curr_output_val * weights[-1]
                else:
                    full_output_val = curr_output_val
                
                curr_tree.train()
                test_roc_aucs[no_t] = roc_auc_score(y_test.detach().clone().cpu().numpy(), torch.sigmoid(full_output_val).detach().clone().cpu().numpy())
    
        print("Finished tree number ", no_t + 1)
        
    return test_roc_aucs