import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
def leiden_cluster(dataset, use_rep, target_key="subclass", resolution=1.0, tolerance=0.01):
    sc.pp.neighbors(dataset, use_rep=use_rep)
    method = "leiden"
    sc.tl.leiden(dataset, resolution=resolution)
    target_clusters = len(dataset.obs[target_key].unique())
    num_clusters = len(dataset.obs[method].unique())
    lower_bound = 0.1 * resolution
    upper_bound = 10 * resolution
    effective_resolution = resolution
    while target_clusters and num_clusters != target_clusters and np.abs(lower_bound - upper_bound) > tolerance:
        effective_resolution = (lower_bound * upper_bound) ** 0.5
        sc.tl.leiden(dataset, resolution=effective_resolution)
        num_clusters = len(dataset.obs[method].unique())
        if num_clusters < target_clusters:
            lower_bound = effective_resolution
        elif num_clusters >= target_clusters:
            upper_bound = effective_resolution
    print(f"final number of clusters: {num_clusters}")
    print(f"final resolution: {effective_resolution}")

def get_clustering_scores(imputed, embedding_key, cell_type_color="class",
                          ct_threshold=0.01, keep_obs="subclass", starting_raw_resolution=1.0,
                         starting_imputed_resolution=1.0):
#     sc.pp.neighbors(raw, use_rep=embedding_key)
#     sc.tl.umap(raw)
#     sc.pl.umap(raw, color=cell_type_color, title=f"raw {cell_type_color} embeddings")
    sc.pp.neighbors(imputed, use_rep=embedding_key)
    sc.tl.umap(imputed)
#     sc.pl.umap(imputed, color=cell_type_color, title=f"imputed {cell_type_color} embeddings")
    xpercent = int(len(imputed.obs_names) * ct_threshold)
#     print(xpercent)
    kept_subclasses = imputed.obs[keep_obs].value_counts()[imputed.obs[keep_obs].value_counts() > xpercent]
#     raw_sub = raw[raw.obs[keep_obs].isin(kept_subclasses.index)].copy()
    imputed_sub = imputed[imputed.obs[keep_obs].isin(kept_subclasses.index)].copy()
#     leiden_cluster(raw_sub, use_rep=embedding_key, resolution=starting_raw_resolution)
    leiden_cluster(imputed_sub, use_rep=embedding_key, resolution=starting_imputed_resolution)
#     sc.pl.umap(raw_sub, color=[keep_obs,"leiden"], title=[f"raw {keep_obs}", "raw leiden"])
    sc.pl.umap(imputed_sub, color=[keep_obs,"leiden"], title=[f"imputed {keep_obs}", "imputed leiden"])
#     raw_ari = adjusted_rand_score(raw_sub.obs[keep_obs], raw_sub.obs["leiden"])
#     raw_nmi = normalized_mutual_info_score(raw_sub.obs[keep_obs], raw_sub.obs["leiden"])
#     raw_asw = silhouette_score(raw_sub.obsm[embedding_key], raw_sub.obs[keep_obs])
    imp_ari = adjusted_rand_score(imputed_sub.obs[keep_obs], imputed_sub.obs["leiden"])
    imp_nmi = normalized_mutual_info_score(imputed_sub.obs[keep_obs], imputed_sub.obs["leiden"])
    imp_asw = silhouette_score(imputed_sub.obsm[embedding_key], imputed_sub.obs[keep_obs])

    plt.savefig(f"output/imputed_umap_plot{str(imputed.shape[0])}.png")  # You can change the filename and format as needed

    print("imputed ari: ", imp_ari)
    print("imputed: nmi", imp_nmi)
    print("imputed: asw",imp_asw)
    print(f"imputed: ari:{imp_ari:.3f}& nmi:{imp_nmi:.3f}& asw:{imp_asw:.3f}&")

