from torch import nn
import ray

@ray.remote
def distributed_get_parameters(model, predicate):
    """Distributed parameter collection"""
    params = []
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                params.append(param)
    return params

@ray.remote
def distributed_get_parameters_conv(model, name):
    return distributed_get_parameters(model, 
        lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)

@ray.remote
def distributed_get_parameters_conv_depthwise(model, name):
    return distributed_get_parameters(model, 
        lambda m, p: isinstance(m, nn.Conv2d) and m.groups == m.in_channels 
                    and m.in_channels == m.out_channels and p == name)

@ray.remote
def distributed_get_parameters_bn(model, name):
    return distributed_get_parameters(model, 
        lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)

# Keep original functions for compatibility
get_parameters = distributed_get_parameters
get_parameters_conv = distributed_get_parameters_conv
get_parameters_conv_depthwise = distributed_get_parameters_conv_depthwise
get_parameters_bn = distributed_get_parameters_bn
