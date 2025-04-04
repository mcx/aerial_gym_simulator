from tqdm import tqdm as tqdm
from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np
from sample_factory.utils.typing import ActionSpace
from sample_factory.algo.utils.action_distributions import ContinuousActionDistribution
import torch
import torch.nn as nn
import os

class ModelDeploy(nn.Module):
    def __init__(self, layer_sizes):
        super(ModelDeploy, self).__init__()
        self.control_stack = nn.ModuleList([])
        self.allocation_stack = nn.ModuleList([])
        

        self.control_stack.append(nn.Linear(layer_sizes[0], layer_sizes[0])) # normalize observations
        
        # control layers + last layer
        self.control_stack.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
        for layer_size_in, layer_size_out in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.control_stack.append(nn.Tanh())
            self.control_stack.append(nn.Linear(layer_size_in, layer_size_out).to(torch.float)).cpu()



    def forward(self, x):
        network_input = x[:]
        for i, layer in enumerate(self.control_stack):
            network_input = layer(network_input)
        return network_input

def convert_model_to_script_model(nn_model_full):

    nn_dims = [nn_model_full.running_mean.shape[0], 
               dict(nn_model_full.actor_mlp.named_parameters())["0.bias"].shape[0],
               dict(nn_model_full.actor_mlp.named_parameters())["2.bias"].shape[0],
               dict(nn_model_full.actor_mlp.named_parameters())["4.bias"].shape[0],
               dict(nn_model_full.mu_layer.named_parameters())["bias"].shape[0],
    ]

    nn_model_deploy = ModelDeploy(nn_dims)

    # # Observation normalization
    nn_model_deploy.control_stack[0].weight.data.zero_()
    nn_model_deploy.control_stack[0].weight.data.diagonal().copy_(1.0 / torch.sqrt(nn_model_full.running_var.data + 1e-8))
    nn_model_deploy.control_stack[0].bias.data[:] = -nn_model_full.running_mean.data / torch.sqrt(nn_model_full.running_var.data + 1e-8)

    # Control layers
    nn_model_deploy.control_stack[1].weight.data[:] = dict(nn_model_full.actor_mlp.named_parameters())["0.weight"].data
    nn_model_deploy.control_stack[1].bias.data[:]   = dict(nn_model_full.actor_mlp.named_parameters())["0.bias"].data
    nn_model_deploy.control_stack[3].weight.data[:] = dict(nn_model_full.actor_mlp.named_parameters())["2.weight"].data
    nn_model_deploy.control_stack[3].bias.data[:]   = dict(nn_model_full.actor_mlp.named_parameters())["2.bias"].data
    nn_model_deploy.control_stack[5].weight.data[:] = dict(nn_model_full.actor_mlp.named_parameters())["4.weight"].data
    nn_model_deploy.control_stack[5].bias.data[:]   = dict(nn_model_full.actor_mlp.named_parameters())["4.bias"].data

    # # last layer
    nn_model_deploy.control_stack[7].weight.data[:] = dict(nn_model_full.mu_layer.named_parameters())["weight"].data
    nn_model_deploy.control_stack[7].bias.data[:]   = dict(nn_model_full.mu_layer.named_parameters())["bias"].data

    random_input = torch.randn(nn_model_full.running_mean.shape[0])
    print("Expected output:", nn_model_full(random_input))
    print("Observed output:", nn_model_deploy(random_input))

    sm = torch.jit.script(nn_model_deploy)

    if not os.path.exists("./deployment/deployed_models"):
        os.makedirs("./deployment/deployed_models")

    torch.jit.save(sm, "./deployment/deployed_models/etor_task_b.pt")

    print('Size normal (B):', os.path.getsize("./deployment/deployed_models/etor_task_b.pt"))
    
    return sm