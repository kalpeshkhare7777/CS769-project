import logging
from typing import Any, List, Optional, Union, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Subset

from weasel.utils.optimization import get_loss
from weasel.models.encoder_models.encoder_MLP import MLPEncoder
from weasel.models.downstream_models.base_model import DownstreamBaseModel

log = logging.getLogger(__name__)

# ---------------- RETRIEVE Coreset Selection ----------------
def compute_gradient(x, model, loss_fn):
    x = x.detach().requires_grad_(True)
    y_pred = model(x)
    loss = loss_fn(y_pred, y_pred.detach())
    loss.backward()
    return x.grad.clone().detach()

def retrieve_coreset(unlabeled_dataset, model, loss_fn, k, alpha=1e-4):
    grads = []
    for idx in range(len(unlabeled_dataset)):
        L, X = unlabeled_dataset[idx]
        grad = compute_gradient(X.unsqueeze(0), model, loss_fn)
        grads.append((idx, grad))

    selected = set()
    while len(selected) < k:
        best_gain, best_idx = -float('inf'), None
        for idx, grad in grads:
            if idx in selected:
                continue
            score = (alpha * grad).norm().item()
            if score > best_gain:
                best_gain, best_idx = score, idx
        if best_idx is not None:
            selected.add(best_idx)

    return list(selected)

# ----------------- WeaSEL++ Model -----------------
class WeaselPlusPlus(LightningModule):
    def __init__(self, end_model: DownstreamBaseModel, num_LFs: int, n_classes: int,
                 loss_function='cross_entropy', class_conditional_accuracies=True,
                 temperature=2.0, use_aux_input=False,
                 coreset_k=64, retrieve_every=5):
        super().__init__()
        self.save_hyperparameters()
        self.end_model = end_model
        self.n_classes = n_classes
        self.coreset_k = coreset_k
        self.retrieve_every = retrieve_every
        self.use_aux_input = use_aux_input
        self.class_conditional_accuracies = class_conditional_accuracies

        output_dim = num_LFs * n_classes if class_conditional_accuracies else num_LFs
        self.encoder = MLPEncoder(num_LFs, None, output_dim)
        self.criterion = get_loss(loss_function)

        self.accuracy_func = nn.Softmax(dim=1)
        self.acc_scaler = np.sqrt(num_LFs * n_classes if class_conditional_accuracies else num_LFs)
        self.class_balance = torch.tensor([1. / n_classes] * n_classes)
        self.st_indices = None
        self.current_epoch_idx = 0

    def forward(self, X: Any):
        return self.end_model(X)

    def get_accuracy_scores(self, L, X):
        raw_acc = self.encoder(L, X if self.use_aux_input else None)
        if self.class_conditional_accuracies:
            raw_acc = raw_acc.reshape(-1, self.hparams.num_LFs, self.hparams.n_classes)
        raw_acc = raw_acc / self.hparams.temperature
        return self.acc_scaler * self.accuracy_func(raw_acc)

    def encode(self, L, X):
        accs = self.get_accuracy_scores(L, X)
        L_ind = self._create_L_ind(L)
        if self.class_conditional_accuracies:
            logits = (L_ind * accs).sum(dim=1)
        else:
            logits = (accs.unsqueeze(1) @ L_ind).squeeze(1)
        logits += torch.log(self.class_balance.to(logits.device))
        return logits, accs

    def _create_L_ind(self, L):
        n, m = L.shape
        L = L + 1
        L_ind = torch.zeros((n, m, self.n_classes), device=L.device)
        for c in range(1, self.n_classes + 1):
            L_ind[:, :, c - 1] = (L == c).float()
        return L_ind

    def training_step(self, batch, batch_idx):
        L, X = batch
        if self.st_indices is not None:
            L = L[self.st_indices]
            X = X[self.st_indices]

        logits_f = self.end_model(X)
        logits_e, _ = self.encode(L, X)
        probs_f = self.end_model.logits_to_probs(logits_f)
        probs_e = self.end_model.logits_to_probs(logits_e)

        loss_f = self.criterion(probs_f, probs_e.detach())
        loss_e = self.criterion(probs_e, probs_f.detach())
        loss = loss_f + loss_e

        self.log_dict({'train/loss': loss, 'train/loss_f': loss_f, 'train/loss_e': loss_e})
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch_idx % self.retrieve_every == 0:
            loader = self.trainer.datamodule.train_dataloader()
            dataset = loader.dataset
            selected_indices = retrieve_coreset(dataset, self.end_model, self.criterion, self.coreset_k)
            
            # Replace the dataloader's dataset with a subset
            self.trainer.datamodule._train_dataset = Subset(dataset, selected_indices)
        self.current_epoch_idx += 1


    def validation_step(self, batch, batch_idx):
        return self.end_model.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        # Let end_model compute stats WITHOUT logging
        stats = self.end_model.validation_epoch_end(outputs)

        # Manually log stats here from LightningModule
        if isinstance(stats, dict):
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                elif not isinstance(v, (float, int)):
                    continue  # skip non-numeric

                self.log(k, v, prog_bar=True)

        return stats




    def training_step(self, batch, batch_idx, optimizer_idx):
        L, X = batch

        if self.st_indices is not None:
            L = L[self.st_indices]
            X = X[self.st_indices]

        logits_f = self.end_model(X)
        logits_e, _ = self.encode(L, X)
        probs_f = self.end_model.logits_to_probs(logits_f)
        probs_e = self.end_model.logits_to_probs(logits_e)

        loss_f = self.criterion(probs_f, probs_e.detach())
        loss_e = self.criterion(probs_e, probs_f.detach())

        # Select loss based on which optimizer is running
        if optimizer_idx == 0:
            loss = loss_e
        elif optimizer_idx == 1:
            loss = loss_f

        self.log_dict({f'train/loss_opt{optimizer_idx}': loss}, prog_bar=True)
        return loss


    def test_epoch_end(self, outputs):
        stats = self.end_model.test_epoch_end(outputs)
        if isinstance(stats, dict):
            for k, v in stats.items():
                if isinstance(v, (float, int)):
                    self.log(k, v, prog_bar=True)
                elif isinstance(v, torch.Tensor):
                    self.log(k, v.item(), prog_bar=True)
        return stats


    def configure_optimizers(self):
        opt_e = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        opt_f = torch.optim.Adam(self.end_model.parameters(), lr=1e-4)
        return [opt_e, opt_f]
    def test_step(self, batch, batch_idx):
        return self.end_model.test_step(batch, batch_idx)

