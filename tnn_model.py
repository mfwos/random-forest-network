import torch

'''
TNN replicating Random Tree as a neural net;
input features are:
    
    - depth: The depth of the tree
    
    - n_features: The number of input features to every node
    
    - leaf_odds: array of size 2**depth indicating the initialization log-odds for the leaf nodes
                 If leaf_odds = None, the leaves are initialized randomly
                 
    - sig_steepness:  float-parameter; indicates the steepness of all sigmoid activation functions
                      used within the tree. If sig_steepness == None, the steepness parameters are
                      initialized randomly, but with a minimum steepness of 1.
'''

class tnn (torch.nn.Module):
    
    def __init__(self, depth, n_features, leaf_odds = None, sigmoid_steepness = None):
        
        super().__init__()
        self.depth = depth
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_features,2**d) for d in range(self.depth)])
        self.sig_steepness = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.maximum(torch.randn(2**d),torch.tensor(1)) if sigmoid_steepness is None else torch.ones(2**d) * sigmoid_steepness ) for d in range(self.depth)])
        self.leaves = torch.nn.parameter.Parameter(torch.randn(2**(self.depth))) if leaf_odds == None else torch.nn.parameter.Parameter(torch.ones(2**(self.depth)) * leaf_odds ) 
        
        
    def forward(self, x_input, verbose = False):
        
        # Calculate the probabilities for every node for each layer (that is every level of depth)
        layer_probs = {}
        for k, layer in enumerate(self.layers):
            layer_probs[k] = torch.sigmoid(self.layers[k](x_input) * self.sig_steepness[k])
        
        output_probs = {}
        
        # calculate the probability along each path of the tree by building
        # the probabilities successively for each layer of depth
        
        for d in range(self.depth):
            no_layers = 2**d
            output_probs[d] = {}
            
            for no_l in range(no_layers):
                
                no_father_layer = str(int(no_l/2))
                father_side = '_left' if no_l % 2 == 0 else '_right'
                
                output_probs[d][str(no_l)+'_left'] = output_probs[d-1][no_father_layer + father_side] * layer_probs[d][:,no_l] if d != 0 else layer_probs[d][:,no_l]
                output_probs[d][str(no_l)+'_right'] = output_probs[d-1][no_father_layer + father_side] * (1-layer_probs[d][:,no_l]) if d!=0 else (1-layer_probs[d][:,no_l])
                
        if verbose:
            print(output_probs)
        
        # calculate the final output probability of the model
        
        final_probs = torch.stack(list(output_probs[self.depth-1].values()), dim = 1)
        output = (torch.sigmoid(self.leaves) * final_probs).sum(1)
        
        return output
    
    
    # copy_tree copies a tnn-model tree and initializes the model's parameters
    # with the parameters of that model
    
    def copy_tree(self, other_tnn):
        
        assert isinstance(other_tnn, tnn), "The object to copy must be of type TNN as well."
        assert other_tnn.depth <= self.depth, "Depth of the TNN to copy must be less or equal to the TNN's depth."
        
        for layer in range(other_tnn.depth):
            
            self.layers[layer].weight = torch.nn.Parameter(other_tnn.layers[layer].weight.detach().clone())
            self.layers[layer].bias = torch.nn.Parameter(other_tnn.layers[layer].bias.detach().clone())
            self.sig_steepness[layer] = torch.nn.Parameter(other_tnn.sig_steepness[layer].detach().clone())
    
    
    # copy the leaf values of an existing tnn-model
    
    def copy_leaves(self, other_tnn):
        
        assert isinstance(other_tnn, tnn), "The object to copy must be of type TNN as well."
        assert other_tnn.depth <= self.depth, "Depth of the TNN to copy must be less or equal to the TNN's depth."
        
        depth_factor = int(2**(self.depth - other_tnn.depth))
        
        self.leaves = torch.nn.parameter.Parameter(torch.cat([other_tnn.leaves[i].repeat(depth_factor) for i in range(2**other_tnn.depth)]))