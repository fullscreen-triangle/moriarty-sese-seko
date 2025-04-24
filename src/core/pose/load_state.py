import collections
import ray

@ray.remote
def distributed_load_state(net, checkpoint):
    """Distributed state loading"""
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    
    # Process state dict in parallel
    futures = []
    for target_key, target_value in target_state.items():
        futures.append(process_state_item.remote(target_key, target_value, source_state))
    
    results = ray.get(futures)
    for key, value in results:
        new_target_state[key] = value

    return new_target_state

@ray.remote
def distributed_load_from_mobilenet(net, checkpoint):
    """Distributed MobileNet state loading"""
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    
    futures = []
    for target_key, target_value in target_state.items():
        futures.append(process_mobilenet_state_item.remote(target_key, target_value, source_state))
    
    results = ray.get(futures)
    for key, value in results:
        new_target_state[key] = value

    return new_target_state

# Keep original functions for compatibility
load_state = distributed_load_state
load_from_mobilenet = distributed_load_from_mobilenet
