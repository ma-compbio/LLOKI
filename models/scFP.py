import torch
import copy
from embedder import embedder
from misc.graph_construction import create_delaunay_graph, knn_graph, create_combined_graph
import numpy as np
from sklearn.decomposition import PCA
import os
import scanpy as sc
from sklearn.metrics import mean_squared_error
import scipy
import pandas as pd
import scipy.stats as stats
from pyemd import emd_samples

class scFP_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.args.n_nodes = self.adata.X.shape[0]
        self.args.n_feat = self.adata.X.shape[1]

    

    def save_adata(self):
        # Construct the file path where the AnnData object will be saved.
    	save_path = f'output/sec37_500gene/{self.args.name}_processed.h5ad'
    	os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists.
        # Save the AnnData object to the specified path.
        #self.adata.X = self.adata.obsm['denoised']
    	self.adata.write_h5ad(save_path)
    	print(f"Saved imputed and processed AnnData to {save_path}")


    
    def train(self):
        cell_data = self.adata.X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata.X

        cell_data = torch.Tensor(cell_data).to(self.args.device)
        cell_data = cell_data / torch.max(cell_data)  # Normalize the input data
        #  Hard FP
        #print((self.adata.X.toarray().sum()))
        print('Start Hard Feature Propagation ...!')
        #edge_index, edge_weight = create_combined_graph(cell_data, self.adata, self.args.k, self.args.device, gcn_norm=False, sym=True)
        edge_index, edge_weight = knn_graph(cell_data, self.args.k)
        self.model = FeaturePropagation(num_iterations=self.args.iter, adata=self.adata, mask=True, alpha=0.0, trainer=self)
        self.model = self.model.to(self.device)

        denoised_matrix = self.model(cell_data, edge_index, edge_weight)
        #self.adata.X = denoised_matrix.detach().cpu().numpy()  
        denoised_matrix_np = denoised_matrix.detach().cpu().numpy()
        
        print((denoised_matrix_np == 0).sum())
        # Soft FP
        print('Start Softs Feature Propagation ...!')
        #edge_index_new, edge_weight_new = create_combined_graph(denoised_matrix, self.adata, self.args.k, self.args.device, gcn_norm=False, sym=True)
        edge_index_new, edge_weight_new = knn_graph(denoised_matrix, self.args.device, gcn_norm=True, sym=True)
        print(self.args.alpha)
        self.model = FeaturePropagation(num_iterations=self.args.iter, adata=self.adata, mask=False, alpha=self.args.alpha, trainer=self)        
        self.model = self.model.to(self.device)

        denoised_matrix = self.model(denoised_matrix, edge_index_new, edge_weight_new)
        print('post prop')
        #print((self.adata.X.toarray().sum()))
  
        # reduced
        pca = PCA(n_components = 32)
        denoised_matrix = denoised_matrix.detach().cpu().numpy()
        reduced = pca.fit_transform(denoised_matrix)
            
        self.adata.obsm['denoised'] = denoised_matrix
        self.adata.obsm['reduced'] = reduced
        #self.save_adata()    
        return self.evaluate()
        
class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations, adata, mask, alpha=0.0, early_stopping=True, patience=15, trainer=None):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.patience = patience
        self.adata = adata
        self.trainer = trainer 

    def calc_avg_expression(self, adata):
        if scipy.sparse.issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X
    
        gene_means = data.mean(axis=0)
        return pd.Series(gene_means, index=adata.var_names)

    def calc_pearson_corr(self, merf, adata):
        scrna = sc.read_h5ad('/work/magroup/ehaber/UCE/data/sc_ref/scref_500gene.h5ad')
        scrna = scrna[~scrna.obs['class'].isna()].copy()
        labels = list(set(scrna.obs['class']) & set(adata.obs['class']))

        for label in labels:
            scrna_subset = scrna[scrna.obs['class'] == label]
            merf_subset = adata[adata.obs['class'] == label]
    
            # Calculate average expression values
            scrna_avg = self.calc_avg_expression(scrna_subset)  # Assume returns Series with gene names as index
            merf_avg = self.calc_avg_expression(merf_subset)   # Same assumption
            pearson_corr = emd_samples(merf_avg, scrna_avg)
            print(f'Pearson Correlation Coefficient for {label}: {pearson_corr:.4f}')
        return pearson_corr

    def forward(self, x, edge_index, edge_weight):
        original_x = copy.copy(x)
        nonzero_idx = torch.nonzero(x)
        nonzero_i, nonzero_j = nonzero_idx.t()
        scale = x.max()
        out = x
        n_nodes = x.shape[0]
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
        adj = adj.float()

        res = (1 - self.alpha) * out

    
        for i in range(self.num_iterations):
            previous_out = out.clone()
            out = torch.sparse.mm(adj, out)
            change = torch.norm(out - previous_out)
            
            if self.mask:
                out[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
            else:
                out.mul_(self.alpha).add_(res)

            #print(out.max())
            
            # if (i + 1) % 5 == 0:
            #     self.adata.X = out.detach().cpu().numpy()
            #     self.trainer.save_adata(iteration=i + 1)  # Call save_adata using the trainer instance
        # out *= scale
        return out

# def forward(self, x, edge_index, edge_weight):
#         original_x = copy.copy(x)
#         nonzero_idx = torch.nonzero(x)
#         nonzero_i, nonzero_j = nonzero_idx.t()

#         out = x
#         n_nodes = x.shape[0]
#         adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
#         adj = adj.float()
        
    
#         res = (1-self.alpha) * out
#         for _ in range(self.num_iterations):
#             out = torch.sparse.mm(adj, out)
#             if self.mask:
#                 out[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
#             else:
#                 out.mul_(self.alpha).add_(res) 
                
#         return out