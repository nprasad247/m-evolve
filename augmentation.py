import numpy as np
import networkx as nx
import torch
from torch.utils.data import WeightedRandomSampler
from torchmetrics import ConfusionMatrix
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger as TBL



"""
Implementation of M-Evolve: M-Evolve: Structural-Mapping-Based Data Augmentation for Graph Classification.
(Jiajun Zhou, Jie Shen, Shanqing Yu, Guanrong Chen, Qi Xuan)
Paper can be found at https://arxiv.org/abs/2007.05700
"""

def length_2_paths(graph, node1, node2):
    """ Get the open triads described in the paper"""

    paths = []
    for path in nx.all_simple_paths(graph, node1, node2, 2):      
        if len(path) == 3:
            paths.append([[path[0], path[1]], [path[1], path[2]]])
    return paths

def get_candidate_edges(graph):
    """See equation (5) in the paper."""
    edges = []
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            
            if node1 < node2:
                
                if not graph.has_edge(node1, node2) and len(length_2_paths(graph, node1, node2)) > 0:
                    edges.append([node1, node2])
    
    return edges

"""For the next 4 functions, see equation (6) in the paper."""
def ra_score(graph, node1, node2):
    
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    
    common_neighbors = list(neighbors1 & neighbors2)
    
    return sum(1 / graph.degree[node] for node in common_neighbors)
    


def all_ra_scores(graph, candidate_edges):
  
    return torch.tensor([ra_score(graph, *edge) for edge in candidate_edges])


def addition_weight(graph, node1, node2, ra_scores):
    
    return ra_score(graph, node1, node2) / ra_scores.sum()
    


def all_addition_weights(graph, candidate_edges, ra_scores):
    
    weights = [addition_weight(graph, *edge, ra_scores) for edge in candidate_edges]
    return torch.tensor(weights)



def deletion_weight(graph, triad_edge, ra_scores):
    return 1 - addition_weight(graph, *triad_edge, ra_scores)


def all_deletion_weights(graph, triad, ra_scores):
    weight_0 = deletion_weight(graph, triad[0], ra_scores)
    return torch.tensor([weight_0, 1 - weight_0])


def augmented_graph(data, budget=0.15):
    """The overall graph augmentation described in the paper.


    Parameters:
        data: torch_geometric Data object representing the graph.
        budget: The fraction of edges (m) we want to modify. We add budget * m new edges
                and delete budget * m existing edges.
    """
    graph = to_networkx(data, to_undirected=True)
    candidate_edges = get_candidate_edges(graph)
    ra_scores = all_ra_scores(graph, candidate_edges)
    add_weights = all_addition_weights(graph, candidate_edges, ra_scores)
    
    num_samples = int(np.ceil(budget * len(candidate_edges)))
    E_add_indices = list(WeightedRandomSampler(add_weights, num_samples, replacement=False))
    E_add = torch.tensor(candidate_edges)[E_add_indices].tolist()
    E_del = []

    
    for edge in E_add:

        for triad in length_2_paths(graph, *edge):
      
            del_ra_scores = all_ra_scores(graph, triad)
            del_weights = all_deletion_weights(graph, triad, del_ra_scores)
            del_index = list(WeightedRandomSampler(del_weights, 1, replacement=False))
            E_del.extend(torch.tensor(triad)[del_index].tolist())
   
    graph.add_edges_from(E_add)
    graph.remove_edges_from(E_del)
    new_data = from_networkx(graph)
    new_data.x = data.x
    new_data.y = data.y
    new_data.csmiles = data.csmiles
    new_data.graph_feats = data.graph_feats

    return new_data
    

def confusion_matrix(y_probs):
    """ Equation (8) in the paper """
    confusion_matrix =  torch.t(ConfusionMatrix(num_classes=2, normalize='true')(torch.t(y_probs[:2,:].cpu()), y_probs[2,:].int().cpu()))
    return confusion_matrix[:, y_probs[2,:].long().cpu()]


def label_reliability(y_probs, confusion_matrix):
    """ Equation (9) in the paper """
    
    return torch.diagonal(torch.t(y_probs[:2,:].cpu()) @ confusion_matrix)


def g(y_probs):
    """ Returns 1 for correct classifications and -1 otherwise """
    bool_remap = torch.tensor([-1, 1])
    preds = y_probs[:2,:].argmax(dim=0)
    
    correct = (preds == y_probs[2,:].long()).long()
    
    return bool_remap[correct]

def theta(y_probs, r):
    """ Equation (10) of the paper """
    g_val = g(y_probs)
    values = torch.linspace(0, 1)
    candidates = torch.tensor([torch.relu((val - r) * g_val).sum() for val in values])
    return values[candidates.argmin()]


def m_evolve(model, train_graphs, val_graphs, train_batch_size, val_batch_size, num_iter=5):
    """ The final M-Evolve algorithm from the paper. Iteratively adds data to the training set
        by modifying edge connections in the previous dataset. Only adds new points with sufficient
        label reliability.
    """

    logger = TBL('tb_logs', name='Augmentation Experiment')
    trainer = pl.Trainer(gpus=1, precision=32, limit_train_batches=0.5, auto_lr_find=True, logger=logger)
    train_loader = DataLoader(train_graphs, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=val_batch_size,shuffle=False)
    trainer.fit(model, train_loader, val_loader)

    for t in range(num_iter):
        print(f"Beginning augmentation step {t}...")
        train_pool = [augmented_graph(graph, budget=0.15) for graph in train_graphs]
        print(f"Finished augmentation step {t}!")
        pool_loader = DataLoader(train_pool, batch_size=32, shuffle=False)
        y_probs = model.y_probs  #line 5
        Q = confusion_matrix(y_probs) #line 6
        r = label_reliability(y_probs, Q) #line 7 
        theta_val = theta(y_probs, r) #line 8 
        trainer.test(model,pool_loader)
        pool_probs = model.val_probs
        r_pool = label_reliability(pool_probs, Q)  #line 9 part 1

        
        for x, y in zip(train_pool, (r_pool > theta_val).tolist()):
            if y:
                train_graphs.append(x)
        logger = TBL('tb_logs', name='Augmentation Experiment')
        trainer = pl.Trainer(gpus=1, precision=32, limit_train_batches=0.5, auto_lr_find=True, logger=logger)
        train_loader = DataLoader(train_graphs, batch_size=train_batch_size, shuffle=True)
        trainer.fit(model, train_loader, val_loader)

    return model
