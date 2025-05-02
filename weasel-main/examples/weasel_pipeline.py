import warnings
warnings.filterwarnings("ignore")
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from weasel.utils import utils
import hydra
from hydra.utils import instantiate as hydra_instantiate
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import types

# Seed & torch settings
_ = seed_everything(42, workers=True)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)

with hydra.initialize(config_path="configs/"):
    config = hydra.compose(
        config_name="profTeacher_simple.yaml",
        overrides=["end_model.adjust_thresh=False", "+trainer.gpus=1"]
    )

utils.print_config(config)

# Load DataModule
profTeacher_data_module = hydra_instantiate(config.datamodule)

# WeaSEL Model
MLP_end_model_wsl = hydra_instantiate(config.end_model)
weasel_model = hydra_instantiate(config.Weasel, end_model=MLP_end_model_wsl, recursive=False)
checkpoint_callback_wsl = ModelCheckpoint(monitor="Val/auc", mode="max")
trainer_wsl = pl.Trainer(callbacks=[checkpoint_callback_wsl], max_epochs=75, logger=False)
trainer_wsl.fit(model=weasel_model, datamodule=profTeacher_data_module)
test_stats_wsl = trainer_wsl.test(model=weasel_model, datamodule=profTeacher_data_module, ckpt_path='best')

# Snorkel Baseline
from snorkel.labeling.model.label_model import LabelModel
from weasel.datamodules.dataset_classes import BasicDownstreamDataset

label_matrix = np.array(profTeacher_data_module.ws_train_set.L)
snorkel_label_model = LabelModel(cardinality=config.datamodule.n_classes)
snorkel_label_model.fit(L_train=label_matrix)

profTeacher_snorkel_dm = hydra_instantiate(config.datamodule)
Y_probs_snorkel = snorkel_label_model.predict_proba(label_matrix)
profTeacher_snorkel_dm.ws_train_set = BasicDownstreamDataset(
    X=profTeacher_snorkel_dm.ws_train_set.X, Y=Y_probs_snorkel, filter_uncertains=True
)

MLP_end_model = hydra_instantiate(config.end_model)

def float32_evaluation_log(self, Y, preds, split='Val', verbose=True):
    if isinstance(Y, torch.Tensor) and Y.dtype == torch.float64:
        Y = Y.float()
    if isinstance(preds, torch.Tensor) and preds.dtype == torch.float64:
        preds = preds.float()
    stats = self._evaluation_log(Y, preds, split, verbose)
    if isinstance(stats, dict):
        for key, value in list(stats.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                stats[key] = value.float()
            elif isinstance(value, (int, float)):
                stats[key] = torch.tensor(float(value), dtype=torch.float32, device=self.device)
    return stats

MLP_end_model._evaluation_log = types.MethodType(float32_evaluation_log, MLP_end_model)

def force_float32_log(self, name, value, *args, **kwargs):
    if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
        value = value.float()
    elif isinstance(value, (int, float)):
        value = torch.tensor(float(value), dtype=torch.float32, device=self.device)
    original_log = super(type(self), self).log
    return original_log(name, value, *args, **kwargs)

MLP_end_model.log = types.MethodType(force_float32_log, MLP_end_model)

def force_float32_log_dict(self, dictionary, *args, **kwargs):
    float32_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            float32_dict[k] = v.float()
        elif isinstance(v, (int, float)):
            float32_dict[k] = torch.tensor(float(v), dtype=torch.float32, device=self.device)
        else:
            float32_dict[k] = v
    original_log_dict = super(type(self), self).log_dict
    return original_log_dict(float32_dict, *args, **kwargs)

MLP_end_model.log_dict = types.MethodType(force_float32_log_dict, MLP_end_model)

checkpoint_callback = ModelCheckpoint(monitor="Val/auc", mode="max")
trainer = hydra_instantiate(config.trainer, callbacks=checkpoint_callback, deterministic=True, max_epochs=75, precision=32)
trainer.fit(model=MLP_end_model, datamodule=profTeacher_snorkel_dm)
snorkel_test_stats = trainer.test(datamodule=profTeacher_snorkel_dm, ckpt_path='best')

# WeaSEL++
from weasel.models.weaselplusplus import WeaselPlusPlus

def compute_metrics(Y, preds, average="binary"):
    if isinstance(Y, torch.Tensor):
        y_true = Y.argmax(dim=1).cpu().numpy() if Y.ndim == 2 else Y.cpu().numpy()
    else:
        y_true = Y.argmax(axis=1) if Y.ndim == 2 else Y
    if isinstance(preds, torch.Tensor):
        y_pred = preds.argmax(dim=1).cpu().numpy()
        preds_np = preds.cpu().numpy()
    else:
        y_pred = preds.argmax(axis=1)
        preds_np = preds
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
    }
    try:
        if preds_np.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_true, preds_np[:, 1])
    except Exception:
        pass
    return metrics

MLP_end_model_pp = hydra_instantiate(config.end_model)

def patched_evaluation_log(self, Y, preds, split="Test", verbose=True):
    average = "binary" if preds.shape[1] == 2 else "micro"
    stats = compute_metrics(Y, preds, average=average)
    stats = {f"{split}/{k}": v for k, v in stats.items()}
    if hasattr(self, "get_decision_thresh"):
        stats["decision_thresh"] = self.get_decision_thresh()
    return stats

MLP_end_model_pp._evaluation_log = patched_evaluation_log.__get__(MLP_end_model_pp)

weaselpp_model = WeaselPlusPlus(
    end_model=MLP_end_model_pp,
    num_LFs=config.datamodule.num_LFs,
    n_classes=config.datamodule.n_classes,
    coreset_k=64,
    retrieve_every=5,
    temperature=2.0
)

checkpoint_callback_pp = ModelCheckpoint(monitor="Val/auc", mode="max")
trainer_pp = pl.Trainer(callbacks=[checkpoint_callback_pp], max_epochs=75, logger=False)
trainer_pp.fit(model=weaselpp_model, datamodule=profTeacher_data_module)
test_stats_pp = trainer_pp.test(model=weaselpp_model, datamodule=profTeacher_data_module, ckpt_path='best')

# Comparison plots
metrics_to_plot = ["Test/accuracy", "Test/f1", "Test/precision", "Test/recall", "Test/auc"]
metrics_wsl = test_stats_wsl[0]
metrics_pp = test_stats_pp[0]
metrics_snorkel = snorkel_test_stats[0]

def plot_comparison(values1, values2, label1, label2, title):
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, values1, width, label=label1)
    plt.bar(x + width/2, values2, width, label=label2)
    for i in range(len(x)):
        plt.text(x[i] - width/2, values1[i]+0.01, f"{values1[i]:.2f}", ha='center')
        plt.text(x[i] + width/2, values2[i]+0.01, f"{values2[i]:.2f}", ha='center')
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(ticks=x, labels=[m.split("/")[-1].upper() for m in metrics_to_plot])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

plot_comparison(
    [metrics_pp.get(k, 0.0) for k in metrics_to_plot],
    [metrics_snorkel.get(k, 0.0) for k in metrics_to_plot],
    "WeaSEL++", "Snorkel", "Comparison of WeaSEL++ vs Snorkel"
)

plot_comparison(
    [metrics_wsl.get(k, 0.0) for k in metrics_to_plot],
    [metrics_pp.get(k, 0.0) for k in metrics_to_plot],
    "WeaSEL", "WeaSEL++", "Comparison of WeaSEL vs WeaSEL++"
)
