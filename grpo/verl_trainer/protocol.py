import numpy as np
import torch
from verl import DataProto


def packet(name, objects, **metadata):
    array = np.empty(len(objects), dtype=object)
    array[:] = objects
    return DataProto.from_dict(tensors={'item_id': torch.arange(len(objects))},
        non_tensors={name: array}, meta_info=metadata)


def groups_to_episodes(data):
    return [ep for group in data.non_tensor_batch['groups'] for ep in group]
