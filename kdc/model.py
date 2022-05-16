import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class NBLoss(torch.nn.Module):
    """
    Negative binomial negative log-likelihood.
    """
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + y * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
        )
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + np.inf, res)
        return -torch.mean(res)


class MLP(torch.nn.Module):
    """
    A multilayer perceptron class.
    """
    def __init__(self, sizes, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KDC(torch.nn.Module):
    """
    KDC autoencoder model.
    """
    def __init__(
        self,
        num_genes,
        num_drugs,
        loss_ae="gauss"
    ):
        super(KDC, self).__init__()
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_ae = loss_ae
        # set hyperparameters
        self.set_hparams() # self.hparams

        # init models
        # 1. AE
        self.encoder = MLP(
            [num_genes]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [self.hparams["dim"]]
        )

        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["autoencoder_width"]] * self.hparams["autoencoder_depth"]
            + [num_genes * 2]
        )
        # AE loss
        if self.loss_ae == "nb":
            self.loss_autoencoder = NBLoss()
        elif self.loss_ae == "gauss":
            self.loss_autoencoder = nn.GaussianNLLLoss()
        else:
            raise KeyError("use either \"nb\" or \"gauss\" for loss_ae")

        # 2. Discriminator
        self.adversary_drugs = MLP(
            [self.hparams["dim"]]
            + [self.hparams["adversary_width"]] * self.hparams["adversary_depth"]
            + [num_drugs]
        )
        # Adversary loss
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()

        # 3. Doser projector f
        self.dosers = torch.nn.ModuleList()
        for _ in range(num_drugs):
            self.dosers.append(
                MLP(
                    [1]
                    + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                    + [1],
                    batch_norm=False,
                )
            )

        # 4. drug embedding projector: get V_perturbation
        self.drug_embeddings = torch.nn.Embedding(self.num_drugs, self.hparams["dim"])

        self.iteration = 0
        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        # optim 1
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs)
        )
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        # optim 2
        _parameters = get_params(self.adversary_drugs, has_drugs)
        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams["adversary_lr"],
            weight_decay=self.hparams["adversary_wd"],
        )
        # optim 3
        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["dosers_lr"],
                weight_decay=self.hparams["dosers_wd"],
            )

    def set_hparams(self):
        self._hparams = {
            "dim": 128,
            "dosers_width": 128,
            "dosers_depth": 2,
            "dosers_lr": 4e-3,
            "dosers_wd": 1e-7,
            "autoencoder_width": 128,
            "autoencoder_depth": 3,
            "adversary_width": 64,
            "adversary_depth": 2,
            "reg_adversary": 60,
            "penalty_adversary": 60,
            "autoencoder_lr": 3e-4,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "adversary_wd": 4e-7,
            "adversary_steps": 3,
        }
        return self._hparams

    def get_model_args(self):
        """
        Returns a list of arguments for KDC init.
        """
        return self.num_genes, self.num_drugs, self.loss_ae

    @property
    def hparams(self):
        """
        Returns a list of the hyper-parameters.
        """
        return self.set_hparams()

    def get_drug_embeddings(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response fun.
        """
        doses = []
        for d in range(drugs.size(1)):
            this_drug = drugs[:, d].view(-1, 1)
            doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
        return torch.cat(doses, 1) @ self.drug_embeddings.weight

    def predict(
        self,
        genes,
        drugs,
        return_latent=False,
    ):
        """
        Predict gene expression given drugs.
        """
        genes, drugs = genes.to(self.device), drugs.to(self.device)
        if self.loss_ae == "nb":
            genes = torch.log1p(genes)

        latent_basal = self.encoder(genes)
        latent_treated = latent_basal
        if self.num_drugs > 0:
            latent_treated = latent_treated + self.get_drug_embeddings(drugs)

        gene_reconstructions = self.decoder(latent_treated)
        if self.loss_ae == "gauss":
            # convert variance estimates to a positive value in [1e-3, inf)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = F.softplus(gene_reconstructions[:, dim:]).add(1e-3)

        if self.loss_ae == "nb":
            gene_means = F.softplus(gene_means).add(1e-3)
        gene_reconstructions = torch.cat([gene_means, gene_vars], dim=1)

        if return_latent:
            return gene_reconstructions, latent_basal, latent_treated
        return gene_reconstructions

    @staticmethod
    def compute_gradients(output, input):
        grads = torch.autograd.grad(output, input, create_graph=True)
        grads = grads[0].pow(2).mean()
        return grads

    def update(self, genes, drugs):
        """
        Update parameters given a minibatch of genes and drugs.
        """
        genes, drugs = genes.to(self.device), drugs.to(self.device)
        gene_reconstructions, latent_basal, _ = self.predict(
            genes,
            drugs,
            return_latent=True
        )

        dim = gene_reconstructions.size(1) // 2
        gene_means, gene_vars = gene_reconstructions[:, :dim], gene_reconstructions[:, dim:]
        reconstruction_loss = self.loss_autoencoder(gene_means, genes, gene_vars)
        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            adversary_drugs_loss = self.loss_adversary_drugs(
                adversary_drugs_predictions, drugs.gt(0).float()
            )

        adversary_drugs_penalty = torch.tensor([0.0], device=self.device)
        if self.iteration % self.hparams["adversary_steps"]:  # 1. adversary step
            if self.num_drugs > 0:
                adversary_drugs_penalty = KDC.compute_gradients(
                    adversary_drugs_predictions.sum(), latent_basal
                ) # gradient penalty for the discriminator, adversary_drugs_penalty: [1]
        # https://arxiv.org/pdf/1706.04156.pdf 3.4
        # https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944

            self.optimizer_adversaries.zero_grad()
            (
                adversary_drugs_loss
                + self.hparams["penalty_adversary"] * adversary_drugs_penalty
            ).backward()  # loss for adversary step
            self.optimizer_adversaries.step()
        
        else:  # 2. AE step
            self.optimizer_autoencoder.zero_grad()
            if self.num_drugs > 0:
                self.optimizer_dosers.zero_grad()
            (
                reconstruction_loss
                - self.hparams["reg_adversary"] * adversary_drugs_loss
            ).backward()  # loss for AE step
            self.optimizer_autoencoder.step()
            if self.num_drugs > 0:
                self.optimizer_dosers.step()
        
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item()
        }


def save_model(model, name: str = 'KDC'):
    from torch import save
    import os
    if isinstance(model, KDC):
        return save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(name: str = 'KDC'):
    from torch import load
    import os
    r = KDC()
    r.load_state_dict(load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.th'), map_location='cpu'))
    return r
