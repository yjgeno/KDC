import scanpy as sc


def sim_adata(n_cells: int = 400,
              dose_key: bool = True, 
              split_key : bool = True):
    """
    Simulate AnnData based on pbmc3k dataset.
    """
    adata = sc.datasets.pbmc3k() # (2700, 32738)
    sc.pp.filter_cells(adata, min_counts=0)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]
    adata = adata[:n_cells, ].copy()
    adata.obs["condition"] = "drugA"
    adata.obs["condition"].values[:n_cells//4] = "control"
    adata.obs["condition"].values[n_cells//4:n_cells//2] = "drugB"
    if dose_key:
        adata.obs["dose_val"] = '0.3' # "drugA_0.3"
        adata.obs["dose_val"].values[:n_cells//4] = '0.1' # "control_0.1"
        adata.obs["dose_val"].values[n_cells//4:n_cells//2] = '0.5' # "drugB_0.5"
    if split_key:
        adata.obs["split"] = "train"
    return adata
