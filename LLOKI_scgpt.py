import os
import glob
import sys
import random
import datetime
import numpy as np
import pandas as pd
import scipy
import json
from scipy.sparse import csr_matrix, vstack
import matplotlib.pyplot as plt
import tqdm
import anndata as ad
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch_geometric.nn.inits import glorot, zeros
from pathlib import Path
from metrics import get_clustering_scores
from models import scFP_Trainer
from misc.utils import set_seed, set_filename, setup_logger
from argument import parse_args
import scgpt as scg
import argparse


def spatially_aware_splitting(adata, n_splits=2, method='spatial'):
    """ Split the AnnData object into subsets with a small overlap. """
    if adata.shape[0] <= 40000:
        print('no batching')
        return [adata]
        
    if method == 'spatial':
        # Spatial splitting
        coords = adata.obsm['spatial']
        sorted_indices = np.argsort(coords[:, 0])
        splits = np.array_split(sorted_indices, n_splits)
    elif method == 'kmeans':
        # Clustering-based splitting
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        splits = [adata.obs['leiden'] == cluster for cluster in np.unique(adata.obs['leiden'])]
    else:
        raise ValueError("Unsupported method for splitting.")
    
    return [adata[s] for s in splits]


def zero_lowest_expression(d, zero_threshold = .1):
    quantiles = np.quantile(d.X, zero_threshold, axis=1)
    print(quantiles)
    for cell, quantile in enumerate(quantiles):
        newline = d.X[cell,:]
        newline[newline < quantile] = 0
        d.X[cell,:]  = newline

def save_metrics(metrics, output_dir, filename_prefix):
    """ Save the metrics to a JSON file with a specific filename. """
    output_file = os.path.join(output_dir, f"{filename_prefix}_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_file}")


def process_with_scgpt(adata, model_dir):
    """ Apply zero_lowest_expression and scgpt embedding task on the data. """
    #print(f"Applying zero_lowest_expression with threshold={threshold}")
    adata_copy = adata.copy()   
    adata_copy.X = adata.obsm['denoised'].toarray()
    #zero_lowest_expression(adata, zero_threshold=threshold)
    
    adata_copy.var['gene_names'] = [s.upper() for s in adata_copy.var_names]
    print(adata_copy.var['gene_names'])
    
    # Example embedding task using scgpt, adjust model_dir if necessary
    d2 = scg.tasks.embed_data(adata_copy, model_dir, batch_size=64, gene_col='gene_names')
    
    # Return the modified AnnData
    return d2




def main(args_lst=None, data_dir=None, output_dir=None, model_dir=None):
    import torch
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")
    
    # h5ad_files = glob.glob(os.path.join(data_dir, '*.h5ad'))
    torch.set_num_threads(3)
    imputed_ari_list, imputed_nmi_list, imputed_ca_list = [], [], []
    reduced_ari_list, reduced_nmi_list, reduced_ca_list = [], [], []

    embedder_instance = None
    adata_scrna = sc.read_h5ad('/work/magroup/ehaber/UCE/data/MERFISH_BICCN/scref_full.h5ad')  # Update with your path    

    #loop over all files in directory
    for h5ad_file in h5ad_files:

        del embedder_instance  # Explicitly release memory after processing

        all_batches = []
        print(f"Processing file: {h5ad_file}")
        args.data_path = h5ad_file
        file = set_filename(args)

        adata = ad.read_h5ad(args.data_path)
        # if "xenium" in h5ad_file:  #### Just for testing
        #     adata=spatially_aware_splitting(adata,5)[0]
        if "subclass" not in adata.obs:
            if "Sub_molecular_cell_type" not in adata.obs:
                adata.obs["subclass"]=adata.obs["class"]   
            else:
                adata.obs["subclass"]=adata.obs["Sub_molecular_cell_type"]

        file_basename = os.path.splitext(os.path.basename(h5ad_file))[0]  # Get file name without extension

        for seed in range(0, 1):
            print(f'Seed: {seed}, Filename: {file}')
            set_seed(seed)
            args.seed = seed
    
            from models import scFP_Trainer
            embedder_instance = scFP_Trainer(args)
            embedder_instance.adata_scrna = adata_scrna
            for batch in spatially_aware_splitting(adata, n_splits=3 if "xenium" in h5ad_file else 2):
                print(len(adata))
                print('Processing batch of size: ', batch.shape[0])
                embedder_instance.adata = batch
                [imputed_ari, imputed_nmi, imputed_ca], [reduced_ari, reduced_nmi, reduced_ca] = embedder_instance.train()


                imputed_ari_list.append(imputed_ari)
                imputed_nmi_list.append(imputed_nmi)
                imputed_ca_list.append(imputed_ca)

                reduced_ari_list.append(reduced_ari)
                reduced_nmi_list.append(reduced_nmi)
                reduced_ca_list.append(reduced_ca)

                
                all_batches.append(embedder_instance.adata.copy())
                del embedder_instance.adata  # Explicitly release memory after processing
                torch.cuda.empty_cache()  # Clear GPU memory


  
            #save_metrics(info, output_dir, file_basename)
        
        adata = ad.concat(all_batches, join='outer', index_unique=None)
        adata.obsm['spatial'] = np.vstack([batch.obsm['spatial'] for batch in all_batches])

        print(adata.shape)

        print("Finished processing file: {h5ad_file}")
        print("Applying post-processing for file: {h5ad_file}")

        adata = adata[:, adata.var_names.isin(embedder_instance.adata_scrna.var_names)]                
        processed_data = process_with_scgpt(adata, model_dir)
        processed_adata_output_path = os.path.join(output_dir, f"{file_basename}_processed.h5ad")
        processed_data.write_h5ad(processed_adata_output_path)
        print(f"Processed AnnData object saved to {processed_adata_output_path}")
        print(f"Final score {get_clustering_scores(processed_data,'X_scGPT')}")

    print("All files processed and saved.")
        
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scGPT tasks")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory for input data")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory for saving output")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory for model")    
    parser.add_argument('--name', type=str, default='merfish1100', help="Name for this run")
    parser.add_argument('--k', type=int, default=40, help="K for KNN")
    parser.add_argument('--iter', type=int, default=40, help="Number of iterations")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha parameter")

    parser.add_argument('--device', type=int, default=0, help="CUDA device ID (default: 0)")


    args = parser.parse_args()
    args.n_runs=0
    args.drop_rate=0

    main(args, data_dir=args.data_dir, output_dir=args.output_dir, model_dir=args.model_dir)
