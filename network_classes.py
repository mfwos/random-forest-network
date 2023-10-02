# Simple fully connected network

import torch

class fc_nn (torch.nn.Module):
    
    def __init__(self, no_input_nodes, use_input_batchnorm, no_layers, l_nodes,
                 l_activation_funcs, l_use_batch_norm, output_nodes, output_activation):
        
        super().__init__()
        
        # Should an input batchnorm be learned?
        self.use_input_batchnorm = use_input_batchnorm
        
        if use_input_batchnorm:
            self.input_batchnorm = torch.nn.BatchNorm1d(no_input_nodes)
        
        # Define the number of layers, nodes per layer, the activation fucnction
        # and wether a batchnorm should be used with every layer
        self.no_layers = no_layers
        self.nodes = [no_input_nodes] + l_nodes if type(l_nodes) == list else [no_input_nodes] + [l_nodes for m in range(no_layers)]
        self.activation_func = l_activation_funcs if type(l_activation_funcs) == list else [l_activation_funcs for m in range(no_layers)]
        self.use_batch_norm = l_use_batch_norm if type(l_use_batch_norm) == list else [l_use_batch_norm for m in range(no_layers)]
        
        # Initialize the layers with the parameters defied above as well as the output layer
        self.fcn_layers = torch.nn.ModuleList([fcn_layer(self.nodes[i], self.nodes[i+1], self.use_batch_norm[i], self.activation_func[i]) for i in range(no_layers)])
        self.output_layer = torch.nn.Linear(self.nodes[-1], output_nodes)
        self.output_activation = output_activation
        
    
    # Model call for forward pass
    def forward(self, x):
        
        if self.use_input_batchnorm:
            x = self.input_batchnorm(x)
        
        for m in range(self.no_layers):
            x = self.fcn_layers[m](x)
        
        return self.output_activation(self.output_layer(x)).squeeze()
    
    def get_features(self, x, layer_idx = None):
        if layer_idx == None:
            layer_idx = self.no_layers
        
        if self.use_input_batchnorm:
            x = self.input_batchnorm(x)
        
        for m in range(layer_idx):
            x = self.fcn_layers[m](x)
        
        return x

# Class for a single layer in a fully connected network
class fcn_layer (torch.nn.Module):
    
    def __init__(self, input_nodes, output_nodes, use_batch_norm, activation_func):
        
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        self.layer = torch.nn.Linear(input_nodes, output_nodes)
        
        if use_batch_norm:
            self.norm = torch.nn.BatchNorm1d(output_nodes)
        
        self.act_func = activation_func
        
    def forward(self, x):
        
        x = self.layer(x)
        if self.use_batch_norm:
            x = self.norm(x)
        
        x = self.act_func(x)
        
        return x      