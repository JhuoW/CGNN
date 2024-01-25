
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, add_self_loops

from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
)

def get_dataset(name, root_dir, self_loops, undirected):
    path = f"{root_dir}/"
    if name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root = path, name = name, transform = T.NormalizeFeatures())
        data = dataset[0]
    

    if undirected:
        data.edge_index = to_undirected(data.edge_index)
    if self_loops:
        data.edge_index = add_self_loops(data.edge_index)

    return dataset, data


def get_split(name, data, mask_id):
    if name in ["chameleon", "squirrel"]:
        return (data.train_mask[:, mask_id],
                data.val_mask[:, mask_id],
                data.test_mask[:, mask_id])