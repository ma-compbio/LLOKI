import os
import torch
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from argument import printConfig, config2string
from lloki.utils import drop_data
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.cluster import KMeans
from lloki.utils import imputation_error, cluster_acc
from sklearn.metrics.cluster import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder



class embedder:
    def __init__(self, args):
        self.args = args
        printConfig(args)
        self.config_str = config2string(args)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

        self.data_path = self.args.data_path
        #os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        self.result_path = f'result/{self.args.name}.txt'
        #os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

        self._init_dataset()

    def _init_dataset(self):

        self.adata = sc.read(self.data_path)
        if self.adata.obs['class'].dtype != int:
            self.label_encoding()

        self.preprocess()
        self.adata = drop_data(self.adata, rate=0)

    def label_encoding(self):
        self.adata.obs['original_class'] = self.adata.obs['class'].copy()
        
        label_encoder = LabelEncoder()
        celltype = self.adata.obs['class']
        celltype = label_encoder.fit_transform(celltype)
        self.adata.obs['class'] = celltype

    def preprocess(self, HVG=2000, size_factors=False, logtrans_input=True, normalize_input=False):
        sc.pp.filter_cells(self.adata, min_counts=1)
        sc.pp.filter_genes(self.adata, min_counts=1)
        
        self.adata.raw = self.adata.copy()

        if size_factors:
            sc.pp.normalize_per_cell(self.adata)
            self.adata.obs['size_factors'] = self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
        else:
            self.adata.obs['size_factors'] = 1.0

        if logtrans_input:
            sc.pp.log1p(self.adata)

        if normalize_input:
            sc.pp.scale(self.adata)

            
    def evaluate(self):
        if self.adata.obs['class'].dtype != int:
            self.label_encoding()

        X_imputed = self.adata.obsm['denoised']


        # clustering
        celltype = self.adata.obs['class'].values
        print(celltype)
        n_cluster = np.unique(celltype).shape[0]

        ### Imputed
        print('begin Kmeans')
        kmeans = KMeans(n_cluster, n_init=20, random_state=self.args.seed)
        y_pred = kmeans.fit_predict(X_imputed)

        imputed_ari = adjusted_rand_score(celltype, y_pred)
        imputed_nmi = normalized_mutual_info_score(celltype, y_pred)
        imputed_ca, imputed_ma_f1, imputed_mi_f1 = cluster_acc(celltype, y_pred)

        ### Reduced
        reduced = self.adata.obsm['reduced']
        kmeans = KMeans(n_cluster, n_init=20, random_state=self.args.seed)
        y_pred = kmeans.fit_predict(reduced)
        #reduced_silhouette = self.silhouette(self.adata, ref)
        reduced_ari = adjusted_rand_score(celltype, y_pred)
        reduced_nmi = normalized_mutual_info_score(celltype, y_pred)
        reduced_ca, reduced_ma_f1, reduced_mi_f1 = cluster_acc(celltype, y_pred)
        

        print(f"Dataset: {self.args.name}, Alpha: {self.args.alpha}")
        print()

        print("Imputed --> ARI : {:.4f} / NMI : {:.4f} / ca : {:.4f}\n".format(imputed_ari, imputed_nmi, imputed_ca))
        print("Reduced --> ARI : {:.4f} / NMI : {:.4f} / ca : {:.4f}\n".format(reduced_ari, reduced_nmi, reduced_ca))

     
        return [imputed_ari, imputed_nmi, imputed_ca], [reduced_ari, reduced_nmi, reduced_ca]
