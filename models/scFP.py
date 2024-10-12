import torch
import copy
from embedder import embedder
from misc.graph_construction import  knn_graph, create_combined_graph
import numpy as np
from sklearn.decomposition import PCA
import os
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import mean_squared_error
import scipy
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d
from pyemd import emd_samples

class scFP_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.args.n_nodes = self.adata.X.shape[0]
        self.args.n_feat = self.adata.X.shape[1]



    def save_adata(self):
        # Construct the file path where the AnnData object will be saved.
    	save_path = os.path.join(self.args.output_dir, f'{self.args.name}_processed.h5ad')
    	os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists.
        
        # Save the AnnData object to the specified path.
    	self.adata.write_h5ad(save_path)
    	print(f"Saved imputed and processed AnnData to {save_path}")

    def calculate_sparsity(self, adata):
        """
        Calculate the sparsity per cell for an AnnData object.
    
        Parameters:
            adata (AnnData): The AnnData object containing gene expression data.
    
        Returns:
            np.ndarray: A 1D array of sparsity values per cell.
        """
        if issparse(adata.X):
            # For sparse matrices
            sparsity_per_cell = 1 - (adata.X.getnnz(axis=1) / adata.X.shape[1])
        else:
            # For dense matrices
            sparsity_per_cell = np.mean(adata.X == 0, axis=1)
        return sparsity_per_cell.flatten()


    def compute_empirical_cdf(self, sparsity_values):
        """
        Compute the empirical CDF for a sorted array of sparsity values.
    
        Parameters:
            sparsity_values (np.ndarray): A 1D array of sparsity values.
    
        Returns:
            tuple: A tuple containing the sorted sparsity values and their corresponding CDF values.
        """
        sorted_sparsity = np.sort(sparsity_values)
        cdf = np.linspace(0, 1, len(sorted_sparsity))
        return sorted_sparsity, cdf


    def map_sparsity_values(self, cdf_source, sorted_source, cdf_target, sorted_target):
        """
        Map sparsity values from the source distribution to the target distribution using interpolation.
    
        Parameters:
            cdf_source : CDF values for the source distribution.
            sorted_source : Sorted sparsity values for the source distribution.
            cdf_target : CDF values for the target distribution.
            sorted_target : Sorted sparsity values for the target distribution.
    
        Returns:
            np.ndarray: Mapped sparsity values for the source data.
        """

        print("Source CDF:", cdf_source)
        print("Target CDF:", cdf_target)
        print("Target Sparsity (sorted):", sorted_target)


        # Create interpolation function for the target distribution
        interp_func = interp1d(cdf_target, sorted_target, bounds_error=False, fill_value=(sorted_target[0], sorted_target[-1]))
        # Map the source CDF to the target sparsity values
        mapped_sparsity = interp_func(cdf_source)

        # Print the mapped sparsity for debugging
        print("Mapped Sparsity Values:", mapped_sparsity)

        return mapped_sparsity


    def get_target_sparsity_per_cell(self, original_sparsity, sorted_sparsity, mapped_sparsity):
        """
        Rearrange mapped sparsity values to match the original cell order.
    
        Parameters:
            original_sparsity (np.ndarray): Original sparsity values per cell.
            sorted_sparsity (np.ndarray): Sorted original sparsity values.
            mapped_sparsity (np.ndarray): Mapped sparsity values after OT.
    
        Returns:
            np.ndarray: Target sparsity per cell, aligned with the original cell order.
        """
        # Get the indices that would sort the original sparsity array
        sorted_indices = np.argsort(original_sparsity)
        # Create an array to hold the target sparsity per cell
        target_sparsity = np.zeros_like(original_sparsity)
        # Place the mapped sparsity values back into their original positions
        target_sparsity[sorted_indices] = mapped_sparsity
        return target_sparsity

    
    def train(self):
        # Convert adata.X to dense format if necessary
        cell_data = self.adata.X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata.X.copy()
    
        ### Start of OT implementation ###
    
        # 1. Calculate sparsity per cell for scRNA-seq data (reference)
        adata_scrna = sc.read_h5ad('/work/magroup/ehaber/UCE/data/MERFISH_BICCN/scref_full.h5ad')  # Update with your path
        sparsity_scrna = self.calculate_sparsity(adata_scrna)
        print('scRNA-seq sparsity:', sparsity_scrna)
    
        # 2. Calculate sparsity per cell for MERFISH data (to be adjusted)
        sparsity_merfish_original = self.calculate_sparsity(self.adata)
        print('MERFISH sparsity before adjustment:', sparsity_merfish_original)
    
        # 3. Compute empirical CDFs
        sorted_sparsity_scrna, cdf_scrna = self.compute_empirical_cdf(sparsity_scrna)
        sorted_sparsity_merfish, cdf_merfish = self.compute_empirical_cdf(sparsity_merfish_original)
    
        # 4. Perform OT mapping
        mapped_sparsity_merfish = self.map_sparsity_values(
            cdf_merfish, sorted_sparsity_merfish, cdf_scrna, sorted_sparsity_scrna
        )
    
        # 5. Get target sparsity per cell
        target_sparsity_per_cell = self.get_target_sparsity_per_cell(
            sparsity_merfish_original, sorted_sparsity_merfish, mapped_sparsity_merfish
        )
    
        # 6. Adjust the data to match the target sparsity
        # Number of genes (features)
        num_genes = self.adata.X.shape[1]
    
        # Existing number of non-zero entries per cell
        if issparse(self.adata.X):
            existing_nonzero_per_cell = self.adata.X.getnnz(axis=1)
        else:
            existing_nonzero_per_cell = np.count_nonzero(self.adata.X, axis=1)
    
        # Target number of non-zero entries per cell
        target_nonzero_per_cell = np.ceil((1 - target_sparsity_per_cell) * num_genes).astype(int)
    
        # Compute the difference
        values_to_change_per_cell = target_nonzero_per_cell - existing_nonzero_per_cell
    
        # Calculate max_imputation_per_cell
        # We only consider positive differences (number of values to impute per cell)
        max_imputation_per_cell = np.clip(values_to_change_per_cell, 0, None)
    
        # Convert to PyTorch tensor and move to device
        max_imputation_per_cell_tensor = torch.tensor(max_imputation_per_cell, device=self.device, dtype=torch.int32)
    
        # Adjust the data matrix (if desired, optional)
        # Here, we proceed to feature propagation without adjusting the data directly
    
        ### End of OT implementation ###
    
        # Proceed with Feature Propagation using the original cell_data
    
        # Normalize the input data
        cell_data = torch.Tensor(cell_data).to(self.device)
        cell_data = cell_data / (torch.max(cell_data) + 1e-8)  # Prevent division by zero
    
        # Hard Feature Propagation
        print('Start Hard Feature Propagation ...!')
        edge_index, edge_weight = create_combined_graph(
            cell_data, self.adata, self.args.k, self.device, gcn_norm=False, sym=True
        )

        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
    
        self.model = FeaturePropagation(
            num_iterations=self.args.iter,
            adata=self.adata,
            mask=True,
            alpha=0.0,
            max_imputation_per_cell=max_imputation_per_cell_tensor,
            trainer=self
        )
        self.model = self.model.to(self.device)
    
        denoised_matrix = self.model(cell_data, edge_index, edge_weight)
        denoised_matrix_np = denoised_matrix.detach().cpu().numpy()
    
        # Soft Feature Propagation
        print('Start Soft Feature Propagation ...!')
        edge_index_new, edge_weight_new = create_combined_graph(
            denoised_matrix, self.adata, self.args.k, self.device, gcn_norm=False, sym=True
        )

        edge_index_new = edge_index_new.to(self.device)
        edge_weight_new = edge_weight_new.to(self.device)

        
        print('Alpha value:', self.args.alpha)
        self.model = FeaturePropagation(
            num_iterations=self.args.iter,
            adata=self.adata,
            mask=False,
            alpha=self.args.alpha,
            max_imputation_per_cell=max_imputation_per_cell_tensor,
            trainer=self
        )
        self.model = self.model.to(self.device)
    
        denoised_matrix = self.model(denoised_matrix, edge_index_new, edge_weight_new)
        print('Post soft feature propagation')

        
        # Dimensionality reduction
        pca = PCA(n_components=32)
        denoised_matrix_np = denoised_matrix.detach().cpu().numpy()
        reduced = pca.fit_transform(denoised_matrix_np)
    
        self.adata.obsm['denoised'] = denoised_matrix_np
        self.adata.obsm['reduced'] = reduced
    
        # Optionally save the processed data
        self.save_adata()
    
        return self.evaluate()

        
class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations, adata, mask, alpha=0.0, max_imputation_per_cell=None, early_stopping=True, patience=15, trainer=None):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.patience = patience
        self.adata = adata
        self.max_imputation_per_cell = max_imputation_per_cell 
        self.trainer = trainer 


    def forward(self, x, edge_index, edge_weight):
        device = self.trainer.device  # Assuming you passed 'trainer' when initializing FeaturePropagation
        x = x.to(device)
        print(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        
        original_x = x.clone()
        n_nodes, n_genes = x.shape
    
        # Create masks
        existing_nonzero_mask = (x != 0)  # Observed values
        imputation_mask = torch.zeros_like(x, dtype=torch.bool,  device=device)  # Track imputed values
    
        # Ensure max_imputation_per_cell is on the correct device
        max_imputation_per_cell = self.max_imputation_per_cell.to(device)

    
        # Residual term for the propagation formula
        res = (1 - self.alpha) * x
    
        # Create adjacency matrix
        adj = torch.sparse.FloatTensor(edge_index, edge_weight, size=(n_nodes, n_nodes)).to(device)
        adj = adj.float()
    
        for i in range(self.num_iterations):
            previous_x = x.clone()
            x = torch.sparse.mm(adj, x)  # Feature propagation step
    
            if self.mask:
                # Restore original observed values
                x[existing_nonzero_mask] = original_x[existing_nonzero_mask]
    
                # Identify newly imputed values
                newly_imputed = (x != 0) & ~existing_nonzero_mask & ~imputation_mask
    
                # Update imputation mask
                imputation_mask |= newly_imputed
    
                # Count total imputed values per cell
                total_imputed_per_cell = imputation_mask.sum(dim=1)
    
                # Determine cells that have exceeded their imputation limit
                over_imputed_cells = total_imputed_per_cell > max_imputation_per_cell
    
                if over_imputed_cells.any():
                    # For each over-imputed cell, reset excess imputed values
                    for cell_idx in over_imputed_cells.nonzero(as_tuple=True)[0]:
                        # Number of excess imputations
                        excess = (total_imputed_per_cell[cell_idx] - max_imputation_per_cell[cell_idx]).item()
    
                        # Indices of imputed genes in this cell
                        imputed_gene_indices = imputation_mask[cell_idx].nonzero(as_tuple=True)[0]
    
                        # Randomly select excess genes to reset
                        genes_to_reset = imputed_gene_indices[torch.randperm(len(imputed_gene_indices))[:excess]]
    
                        # Reset values and update masks
                        x[cell_idx, genes_to_reset] = 0
                        imputation_mask[cell_idx, genes_to_reset] = False
            else:
                x.mul_(self.alpha).add_(res)
    
        return x
    
    
