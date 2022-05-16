import scanpy as sc
import torch
import torch.nn.functional as F
import logging


def sim_adata(n_cells: int = 400,
              dose_key: bool = True, 
              split_key : bool = True):
    """
    Simulate AnnData based on pbmc3k dataset.
    """
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_counts=0)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]
    adata = adata[:n_cells, ].copy()
    adata.obs["condition"] = "drugA"
    adata.obs["condition"].values[:100] = "control"
    adata.obs["condition"].values[100:200] = "drugB"
    if dose_key:
        adata.obs["dose_val"] = '0.3' # "drugA_0.3"
        adata.obs["dose_val"].values[:100] = '0.1' # "control_0.1"
        adata.obs["dose_val"].values[100:200] = '0.5' # "drugB_0.5"
    if split_key:
        adata.obs["split"] = "train"
    return adata


if __name__ == "__main__":
    import sys
    import os
    path = os.path.dirname(os.path.realpath(__file__)) # test/
    sys.path.append(os.path.dirname(path))
    from kdc.data import Dataset
    from kdc.model import KDC

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler(os.path.join(path,'INFO_test.log'))
    logger.addHandler(output_file_handler)

    adata = sim_adata()
    logger.info("=== adata ===")
    logger.info(f"adata.shape:\n {adata.shape}")
    logger.info(f"adata.obs:\n {adata.obs.head()}")

    dataset = Dataset(
        adata, 
        perturbation_key="condition",
        control_label="control"
    )
    logger.info("\n=== Dataset ===")
    logger.info(dataset)
    # logger.info(f".drugs.shape:\n {dataset.drugs.shape}")

    model = KDC(
        dataset.num_genes,
        dataset.num_drugs,
        loss_ae="gauss"
    )
    logger.info("\n=== model ===")
    logger.info(model)
    logger.info("= hparams =")
    logger.info(model.hparams)
    
    # predict
    n_cells, n_genes, n_drugs = 1999, 2078, 3
    genes, drugs = (
        torch.rand(n_cells, n_genes),
        (F.one_hot(torch.arange(0, n_cells) % 3, num_classes=n_drugs)).float(),
    )
    logger.info("\n=== input drugs to predict ===")
    logger.info(drugs)
    logger.info(drugs.detach().numpy().shape)

    gene_reconstructions, latent_basal, latent_treated = model.predict(
        genes, drugs, return_latent=True
    )

    drug_embeddings = model.get_drug_embeddings(drugs)
    assert (
        latent_basal + drug_embeddings - latent_treated
    ).detach().numpy().sum() < 1e-6
    logger.info(
        f"gene_reconstructions, latent_treated = latent_basal + drug_embeddings:\n \
        {gene_reconstructions.detach().numpy().shape},\
        {latent_treated.detach().numpy().shape},\
        {latent_basal.detach().numpy().shape},\
        {drug_embeddings.detach().numpy().shape}"
    )