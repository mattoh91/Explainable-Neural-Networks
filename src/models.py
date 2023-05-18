from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AveragePrecision, F1Score, MetricCollection
import torchvision
from typing import Dict, Tuple, Union
from transformers import AutoModel

from model_utils import createEvalPlots

class HFLitImageClassifier(LightningModule):
    def __init__(
        self,
        checkpoint: str,
        num_classes: int,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.body = AutoModel.from_pretrained(checkpoint)
        self.last_hidden_layer_size = list(self.body.children())[-1].normalized_shape[0]
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.last_hidden_layer_size, out_features=200),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=200, out_features=50),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=50, out_features=20),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=20, out_features=self.num_classes)
        )
        metrics = MetricCollection([
            Accuracy(task="multiclass", num_classes=self.num_classes),
            F1Score(task="multiclass", num_classes=self.num_classes),
            AveragePrecision(task="multiclass", num_classes=self.num_classes)
        ])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.val_step_labels = []
        self.val_step_preds = []
        self.test_step_labels = []
        self.test_step_preds = []

    def forward(
        self,
        inputs: Union[Dict[str, Union[int, torch.Tensor]], torch.Tensor]
        ) -> torch.Tensor:

        # If-else block to work with Captum downstream
        if isinstance(inputs, torch.Tensor):
            output = self.body(inputs)
        else:
            output = self.body(**inputs)
        logits = self.classifier(output.pooler_output)
        return logits
    
    def _shared_step(
        self,
        batch: Dict,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)
        # nn.CrossEntropyLoss not used to get loss and pred separately
        loss = F.cross_entropy(logits, labels)
        preds = torch.softmax(logits, dim=1)
        return loss, preds, labels
    
    def training_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        metrics = self.train_metrics(preds, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.val_step_labels.append(labels)
        self.val_step_preds.append(preds)
        metrics = self.valid_metrics(preds, labels)
        self.log("valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):

        labels = torch.cat(self.val_step_labels)
        preds = torch.cat(self.val_step_preds)
        fig_cm, fig_bar, fig_prc = createEvalPlots(preds, labels, num_classes=self.num_classes, normalize=None)
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(figure=fig_cm, artifact_file="val_confusion_matrix.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_bar, artifact_file="val_pos_neg_counts.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_prc, artifact_file="val_prc.png", run_id=self.logger.run_id)

    def test_step(self, batch: Dict, batch_idx: int) -> int:

        _, preds, labels = self._shared_step(batch, batch_idx)
        self.test_step_labels.append(labels)
        self.test_step_preds.append(preds)
        metrics = self.test_metrics(preds, labels)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_test_epoch_end(self):

        labels = torch.cat(self.test_step_labels)
        preds = torch.cat(self.test_step_preds)
        fig_cm, fig_bar, fig_prc = createEvalPlots(preds, labels, num_classes=self.num_classes, normalize=None)
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(figure=fig_cm, artifact_file="test_confusion_matrix.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_bar, artifact_file="test_pos_neg_counts.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_prc, artifact_file="test_prc.png", run_id=self.logger.run_id)

    def predict_step(self, batch: Dict,  batch_idx: int) -> torch.Tensor:

        pixel_values = batch["pixel_values"]
        return self(pixel_values)
    
    def configure_optimizers(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6, verbose=True,
            ),
            "monitor": "valid_loss",
        }
        return [self.optimizer], [self.scheduler]
    
    def fine_tune(self, fine_tune: bool = False):
        if fine_tune:
            for param in self.body.parameters():
                param.requires_grad = True
        else:
            for param in self.body.parameters():
                param.requires_grad = False

class ImageClassifier(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        model = torchvision.models.get_model(model_name, weights="DEFAULT")
        last_hidden_layer_size = list(model.children())[-1].in_features
        self.body = list(model.children())[:-1]
        self.body = nn.Sequential(*self.body)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=last_hidden_layer_size, out_features=200),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=200, out_features=50),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=50, out_features=20),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=20, out_features=self.num_classes)
        )
        metrics = MetricCollection([
            Accuracy(task="multiclass", num_classes=self.num_classes),
            F1Score(task="multiclass", num_classes=self.num_classes),
            AveragePrecision(task="multiclass", num_classes=self.num_classes)
        ])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.val_step_labels = []
        self.val_step_preds = []
        self.test_step_labels = []
        self.test_step_preds = []

    def forward(
        self,
        inputs: Union[Dict[str, Union[int, torch.Tensor]], torch.Tensor]
        ) -> torch.Tensor:

        output = torch.flatten(self.body(inputs), start_dim=1)
        logits = self.classifier(output)
        return logits
    
    def _shared_step(
        self,
        batch: Dict,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pixel_values, labels = batch
        logits = self(pixel_values)
        # nn.CrossEntropyLoss not used to get loss and pred separately
        loss = F.cross_entropy(logits, labels)
        preds = torch.softmax(logits, dim=1)
        return loss, preds, labels
    
    def training_step(self, batch: Dict, batch_idx: int) -> int:
        
        loss, preds, labels = self._shared_step(batch, batch_idx)
        metrics = self.train_metrics(preds, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.val_step_labels.append(labels)
        self.val_step_preds.append(preds)
        metrics = self.valid_metrics(preds, labels)
        self.log("valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        labels = torch.cat(self.val_step_labels)
        preds = torch.cat(self.val_step_preds)
        fig_cm, fig_bar, fig_prc = createEvalPlots(preds, labels, num_classes=self.num_classes, normalize=None)
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(figure=fig_cm, artifact_file="val_confusion_matrix.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_bar, artifact_file="val_pos_neg_counts.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_prc, artifact_file="val_prc.png", run_id=self.logger.run_id)

    def test_step(self, batch: Dict, batch_idx: int) -> int:

        _, preds, labels = self._shared_step(batch, batch_idx)
        self.test_step_labels.append(labels)
        self.test_step_preds.append(preds)
        metrics = self.test_metrics(preds, labels)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        labels = torch.cat(self.test_step_labels)
        preds = torch.cat(self.test_step_preds)
        fig_cm, fig_bar, fig_prc = createEvalPlots(preds, labels, num_classes=self.num_classes, normalize=None)
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(figure=fig_cm, artifact_file="test_confusion_matrix.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_bar, artifact_file="test_pos_neg_counts.png", run_id=self.logger.run_id)
        mlflow_logger.log_figure(figure=fig_prc, artifact_file="test_prc.png", run_id=self.logger.run_id)

    def predict_step(self, batch: Dict,  batch_idx: int) -> torch.Tensor:

        features = batch
        return self(features)
    
    def configure_optimizers(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6, verbose=True,
            ),
            "monitor": "valid_loss",
        }
        return [self.optimizer], [self.scheduler]
    
    def fine_tune(self, fine_tune: bool = False) -> None:
        if fine_tune:
            for param in self.body.parameters():
                param.requires_grad = True
        else:
            for param in self.body.parameters():
                param.requires_grad = False