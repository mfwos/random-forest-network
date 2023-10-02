import torch
from tnn_model import tnn

'''
RFNN_a: RFNN model that uses grow_shrinkage and randomly initialized new
trees to form a boosted forest.

Input parameters:
    
    - depth: Depth of the trees
    
    - n_features: Number of input features for the tree models
    
    - grow_shrinkage: The growing learning-rate for the added log-odds of additional trees
    
    - add_tree_leaf_odds: array of size 2**depth indicating the initialization log-odds for the leaf nodes
                          of a new tree. If add_tree_leaf_odds = None, the leaves are initialized randomly
                          
    - sigmoid_steepness:  float-parameter; indicates the steepness of all sigmoid activation functions
                          used within a new added tree. If sigmoid_steepness == None, the steepness parameters are
                          initialized randomly for each new tree, but with a minimum steepness of 1.
'''

class rfnn_a (torch.nn.Module):
    
    
    # Initially a single tree is constructed 
    
    def __init__(self, depth, n_features, grow_shrinkage, add_tree_leaf_odds = None, sigmoid_steepness = None):
        
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.grow_shrinkage = grow_shrinkage
        self.sigmoid_steepness = sigmoid_steepness
        self.add_tree_leaf_odds = add_tree_leaf_odds
        
        self.trees = torch.nn.ModuleList([tnn(depth, n_features, sigmoid_steepness)])
        self.n_trees = 1
        self.eval()
    
    # add_tree adds a new tree to the forest model    
    
    def add_tree(self, device):
        new_tree = tnn(self.depth, self.n_features, self.add_tree_leaf_odds, self.sigmoid_steepness).to(device)
        new_tree.eval()
        self.trees.append(new_tree)
        self.n_trees += 1
    
    # get_tree_optim creates an AdamW optimizer for tree of tree index tree_idx
    # using the given learning rate.
    
    def get_tree_optim(self, tree_idx, l_rate):
        self.eval()
        self.trees[tree_idx].train()
        optim = torch.optim.AdamW(self.trees[tree_idx].parameters(), lr = l_rate)
        return {'tnn': self.trees[tree_idx], 'optim': optim}
    
    # Create a new prediction on x_input using the sum of log-odds of each individual 
    # tree model, taking the grow_shrinkage factor into account.
    
    def forward(self, x_input):
        
        tree_outputs = torch.zeros((x_input.shape[0], self.n_trees))
        for i in range(self.n_trees):
            curr_output = self.trees[i](x_input)
            tree_outputs[:,i] = torch.log(curr_output / (1 - curr_output))
        
        if self.n_trees > 1:
            weights = torch.ones(self.n_trees) * self.grow_shrinkage
            weights[0] = 1
        else:
            weights = 1
        
        return torch.sigmoid((weights * tree_outputs).sum(1))
    
    # Create a single prediction using the tree at index no_t
    
    def single_tree_forward(self, no_t, x_input, return_log_odds = False):
        
        tree_output = self.trees[no_t](x_input)
        
        if return_log_odds:
            return torch.log(tree_output/(1 - tree_output))
        else:
            return tree_output
        
        
        
        