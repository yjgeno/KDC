import numpy as np
import scanpy as sc
from anndata import AnnData
import torch
import scipy
from sklearn.preprocessing import OneHotEncoder
import os
from typing import Union


class Dataset:
    def __init__(
        self,
        data: Union[AnnData, str],  # "pbmc3k_raw.h5ad"
        perturbation_key: str,
        dose_key: str = None,
        split_key: str = None, # "split"
        control_label: str = None,  # "control"
    ):
        if type(data) is str:
            path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "data", f"{data}"
            )
            print("read data from:", path)
            data = sc.read(path)
        self.genes = (
            torch.Tensor(data.X.A)
            if scipy.sparse.issparse(data.X)
            else torch.Tensor(data.X)
        )
        self.var_names = data.var_names

        # define col dose_key: str(float), if none then 1.0
        if not (perturbation_key in data.obs):
            raise AttributeError(
                f"Perturbation_key {perturbation_key} is missing in adata"
            )
        else:
            if dose_key is None:
                dose_key = "dose_val"
            if not (dose_key in data.obs):
                print(f"Creating a default value 1.0 for dose_key {dose_key}")
                dose_val = []
                for i in range(len(data)):
                    pert = data.obs[perturbation_key].values[i].split("+")  # drugs comb
                    dose_val.append("+".join(["1.0"] * len(pert)))
                data.obs[dose_key] = dose_val
            self.perturbation_key = perturbation_key
            self.dose_key = dose_key
            # self.perturbation = data.obs[perturbation_key]
            # self.dose = data.obs[dose_key]
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            # get unique drugs
            drugs_names_unique = set()
            for d in self.drugs_names:
                [drugs_names_unique.add(i) for i in d.split("+")]
            self.drugs_names_unique = np.array(list(drugs_names_unique))

            # save encoder for a comparison with Mo's model
            # later we need to remove this part
            encoder_drug = OneHotEncoder(sparse=False)
            encoder_drug.fit(self.drugs_names_unique.reshape(-1, 1))  # (# drugs, 1)
            self.encoder_drug = encoder_drug

            self.perts_dict = dict(
                zip(
                    self.drugs_names_unique,
                    encoder_drug.transform(self.drugs_names_unique.reshape(-1, 1)),
                )
            )

            # define col split_key: "train" & "test"
            if not (split_key in data.obs):
                split_key = "split"
                print("Performing automatic train-test split with 0.25 ratio")
                from sklearn.model_selection import train_test_split

                data.obs[split_key] = "train"
                idx_train, idx_test = train_test_split(
                    data.obs_names, test_size=0.25, random_state=42
                )
                data.obs.loc[idx_train, split_key] = "train"
                data.obs.loc[idx_test, split_key] = "test"
                self.split = data.obs[split_key]
            self.split_key = split_key

            # define col control: bool
            if "control" in data.obs:
                self.ctrl = data.obs["control"].values
            elif control_label in self.drugs_names_unique:
                print(f"Assigning 1 for {control_label}")
                data.obs["control"] = 0
                data.obs.loc[
                    (data.obs[perturbation_key] == control_label), "control"
                ] = 1
                self.ctrl = data.obs["control"].values
                print(f"Assigned {sum(self.ctrl)} control cells")
            else:
                raise KeyError(f"{control_label} is not in {perturbation_key}")
            self.control_key = "control"

            # get drug combinations
            drugs = []
            for i, comb in enumerate(self.drugs_names):
                drugs_combos = encoder_drug.transform(
                    np.array(comb.split("+")).reshape(-1, 1)
                )
                dose_combos = str(data.obs[dose_key].values[i]).split("+")
                for j, d in enumerate(dose_combos):
                    if j == 0:
                        drug_ohe = float(d) * drugs_combos[j]
                    else:
                        drug_ohe += float(d) * drugs_combos[j]
                drugs.append(drug_ohe)
            self.drugs = torch.Tensor(
                np.array(drugs)
            )  # (# cells, # drugs) with dose values

            atomic_ohe = encoder_drug.transform(self.drugs_names_unique.reshape(-1, 1))

            self.drug_dict = {}
            for idrug, drug in enumerate(self.drugs_names_unique):
                i = np.where(atomic_ohe[idrug] == 1)[0][0]
                self.drug_dict[i] = drug

        self.num_genes = self.genes.shape[1]
        self.num_drugs = len(self.drugs_names_unique) if self.drugs is not None else 0
        self.is_control = data.obs["control"].values.astype(bool)
        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs["control"] == 1)[0].tolist(),
            "treated": np.where(data.obs["control"] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist()
        }

    # def subset(self, split, condition="all"):
    #     idx = list(set(self.indices[split]) & set(self.indices[condition]))
    #     return SubDataset(self, idx)

    def __getitem__(self, i):  # for each cell i: [gene expression, drug]
        indx = lambda a, i: a[i] if a is not None else None
        return (
            self.genes[i],
            indx(self.drugs, i),
        )

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        message = f"""shape: {self.genes.shape}
perturbation_key: "{self.perturbation_key}", {self.drugs_names_unique}
dose_key: "{self.dose_key}"
split_key: "{self.split_key}", train:test= {len(self.indices["train"])}:{len(self.indices["test"])}
control_key: "{self.control_key}", control:treated= {len(self.indices["control"])}:{len(self.indices["treated"])}
encoder_drug: {self.perts_dict}"""
        return message
