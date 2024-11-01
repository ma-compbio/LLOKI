import torch
import torch.nn.functional as F
import numpy as np
import squidpy as sq
from anndata import AnnData 

def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).to(raw_graph.device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def post_processing(cur_raw_adj, add_self_loop=True, sym=True, gcn_norm=False):

    if add_self_loop:
        num_nodes = cur_raw_adj.size(0)
        cur_raw_adj = cur_raw_adj + torch.diag(torch.ones(num_nodes)).to(cur_raw_adj.device)
    
    if sym:
        cur_raw_adj = cur_raw_adj + cur_raw_adj.t()
        cur_raw_adj /= 2

    deg = cur_raw_adj.sum(1)

    if gcn_norm:

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)

        cur_adj = torch.mm(deg_inv_sqrt, cur_raw_adj)
        cur_adj = torch.mm(cur_adj, deg_inv_sqrt)

    else:

        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)

        cur_adj = torch.mm(deg_inv_sqrt, cur_raw_adj)
    
    return cur_adj

def knn_graph(embeddings, k, gcn_norm=False, sym=True):
    #print(embeddings.shape)
    device = embeddings.device
    embeddings = F.normalize(embeddings, dim=1, p=2)
    similarity_graph = torch.mm(embeddings, embeddings.t())
    
    X = top_k(similarity_graph.to(device), k + 1)
    similarity_graph = F.relu(X)

    cur_adj = post_processing(similarity_graph, gcn_norm=gcn_norm, sym=sym)

    sparse_adj = cur_adj.to_sparse()
    edge_index = sparse_adj.indices().detach()
    edge_weight = sparse_adj.values()
    #print(sparse_adj.shape)
    return edge_index, edge_weight



def create_combined_graph(embeddings, adata, k, device, gcn_norm=False, sym=True):
    knn_edge_index, knn_edge_weight = knn_graph(embeddings, 100, gcn_norm=gcn_norm, sym=sym)
    
    # Normalize embeddings graph
    num_nodes = embeddings.size(0)
    knn_adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    knn_adj_matrix[knn_edge_index[0], knn_edge_index[1]] = knn_edge_weight
    knn_max_value = knn_adj_matrix.max()

    # Create Delaunay graph from spatial data
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=k)
    spatial_adj_matrix = torch.tensor(adata.obsp['spatial_connectivities'].toarray(), dtype=torch.float32, device=device)


    assert knn_adj_matrix.shape == spatial_adj_matrix.shape, "Matrix dimensions do not match."


    # Calculate Euclidean distances for spatial data
    spatial_coords = torch.tensor(adata.obsm['spatial'], dtype=torch.float32, device=device)
    distance_matrix = torch.tensor(adata.obsp['spatial_distances'].toarray(), dtype=torch.float32, device=device)

    min_distance_threshold = 1e-5

    # Refine KNN neighbors based on spatial distances
    refined_adj_matrix = torch.zeros_like(knn_adj_matrix)
    for i in range(num_nodes):
        neighbors = knn_edge_index[1][knn_edge_index[0] == i]
        #print(len(neighbors))
        if len(neighbors) > 0:
            distances = distance_matrix[i][neighbors]

            # Filter out zero distances and non-neighbors (values below threshold)
            valid_mask = distances > min_distance_threshold
            valid_neighbors = neighbors[valid_mask]
            valid_distances = distances[valid_mask]

            #sort based on distance and selct top-k/2 neighbors
            _, indices = torch.sort(valid_distances)
            top_indices = indices[:int(k/2)]  
            refined_adj_matrix[i, valid_neighbors[top_indices]] = knn_adj_matrix[i, valid_neighbors[top_indices]]

    #print(f"Non-zero elements in Refined adjacency matrix: {torch.nonzero(refined_adj_matrix, as_tuple=False).size(0)}")


    # Use element-wise multiplication to combine KNN and spatial graphs
    #combined_adj_matrix = refined_adj_matrix + spatial_adj_matrix
    combined_adj_matrix = refined_adj_matrix
    #print(f"Non-zero elements in Combined adjacency matrix: {torch.nonzero(combined_adj_matrix, as_tuple=False).size(0)}")



    # Symmetrization and GCN normalization
    if sym:
        combined_adj_matrix = (combined_adj_matrix + combined_adj_matrix.t()) / 2

    if gcn_norm:
        combined_adj_matrix = apply_gcn_normalization(combined_adj_matrix)


    # Convert combined adjacency matrix back to sparse format and extract indices and weights
    combined_sparse_matrix = combined_adj_matrix.to_sparse().coalesce()
    combined_edge_index = combined_sparse_matrix.indices().detach()
    combined_edge_weight = combined_sparse_matrix.values().detach()

    
    return combined_edge_index, combined_edge_weight



def apply_gcn_normalization(adj_matrix):
    # Compute the degree matrix
    deg = adj_matrix.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt = torch.diag(deg_inv_sqrt)
    # Apply normalization
    normalized_adj_matrix = deg_inv_sqrt @ adj_matrix @ deg_inv_sqrt
    return normalized_adj_matrix


def apply_gcn_normalization(adj_matrix):
    # Compute degree of each node
    deg = adj_matrix.sum(1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # Normalize adjacency matrix with GCN normalization
    normalized_adj_matrix = deg_inv_sqrt.mm(adj_matrix).mm(deg_inv_sqrt)
    return normalized_adj_matrix



def create_spatially_weighted_knn_graph(embeddings, adata, k, device, gcn_norm=False, sym=True):
    knn_edge_index, knn_edge_weight = knn_graph(embeddings, k, gcn_norm=gcn_norm, sym=sym)
    
    # Normalize embeddings graph
    num_nodes = embeddings.size(0)
    knn_adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    knn_adj_matrix[knn_edge_index[0], knn_edge_index[1]] = knn_edge_weight

    

    
    refined_adj_matrix = torch.zeros_like(knn_adj_matrix)
    spatial_coords=torch.tensor(adata.obsm["spatial"]).to(embeddings.device)


    # Refine KNN neighbors based on spatial distances
    p1=list(np.random.choice(list(range(num_nodes)),100))
    p2=list(np.random.choice(list(range(num_nodes)),100))
    avg_dist=((spatial_coords[p2]-spatial_coords[p1])**2).sum(-1).sqrt().mean()

    avg_dist=avg_dist


    for i in range(num_nodes):
        neighbors = knn_edge_index[1][knn_edge_index[0] == i]
        if len(neighbors) > 0:
            distances=spatial_coords[neighbors]-spatial_coords[i]
            distances=(distances**2).sum(-1).sqrt()
            distance_weighting=torch.exp(-1*(distances**2)/(2*avg_dist*avg_dist))
            refined_adj_matrix[i, neighbors] = distance_weighting.float()
            refined_adj_matrix[i,i]=1.0
    
    combined_adj_matrix = refined_adj_matrix

    # Symmetrization and GCN normalization
    if sym:
        combined_adj_matrix = (combined_adj_matrix + combined_adj_matrix.t()) / 2

    if gcn_norm:
        combined_adj_matrix = apply_gcn_normalization(combined_adj_matrix)


    # Convert combined adjacency matrix back to sparse format and extract indices and weights
    combined_sparse_matrix = combined_adj_matrix.to_sparse().coalesce()
    combined_edge_index = combined_sparse_matrix.indices().detach()
    combined_edge_weight = combined_sparse_matrix.values().detach()

    
    return combined_edge_index, combined_edge_weight
