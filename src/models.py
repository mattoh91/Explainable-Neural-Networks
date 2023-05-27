from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule
from model_utils import create_eval_plots
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from transformers import AutoModel


class HFLitImageClassifier(LightningModule):
    def __init__(
        self,
        checkpoint: str,
        num_classes: int,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        class_weights: List[float] = None,
        metric_avg: str = "weighted",
    ):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.metric_avg = metric_avg
        self.body = AutoModel.from_pretrained(checkpoint)
        self.last_layer_size = list(self.body.modules())[-1].normalized_shape[0]
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.last_layer_size, out_features=15),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=15, out_features=10),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=10, out_features=5),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=5, out_features=self.num_classes),
        )
        metrics = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
                F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
                Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
                Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.val_step_labels = []
        self.val_step_preds = []
        self.test_step_labels = []
        self.test_step_preds = []

    def forward(
        self, inputs: Union[Dict[str, Union[int, torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:

        # If-else block to work with Captum downstream
        if isinstance(inputs, torch.Tensor):
            output = self.body(inputs)
        else:
            output = self.body(**inputs)
        logits = self.classifier(output.pooler_output)
        return logits

    def _shared_step(
        self, batch: Dict, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, device=self.device)
        # nn.CrossEntropyLoss not used to get loss and pred separately
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = torch.softmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        metrics = self.train_metrics(preds, labels)
        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.val_step_labels.append(labels)
        self.val_step_preds.append(preds)
        metrics = self.valid_metrics(preds, labels)
        self.log(
            "valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):

        labels = torch.cat(self.val_step_labels)
        preds = torch.cat(self.val_step_preds)
        fig_cm, fig_bar, fig_prc = create_eval_plots(
            preds, labels, num_classes=self.num_classes, normalize=None
        )
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(
            figure=fig_cm,
            artifact_file="val_confusion_matrix.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_bar,
            artifact_file="val_pos_neg_counts.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_prc, artifact_file="val_prc.png", run_id=self.logger.run_id
        )

    def test_step(self, batch: Dict, batch_idx: int) -> int:

        _, preds, labels = self._shared_step(batch, batch_idx)
        self.test_step_labels.append(labels)
        self.test_step_preds.append(preds)
        metrics = self.test_metrics(preds, labels)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_test_epoch_end(self):

        labels = torch.cat(self.test_step_labels)
        preds = torch.cat(self.test_step_preds)
        fig_cm, fig_bar, fig_prc = create_eval_plots(
            preds, labels, num_classes=self.num_classes, normalize=None
        )
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(
            figure=fig_cm,
            artifact_file="test_confusion_matrix.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_bar,
            artifact_file="test_pos_neg_counts.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_prc, artifact_file="test_prc.png", run_id=self.logger.run_id
        )

    def predict_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:

        pixel_values = batch["pixel_values"]
        return self(pixel_values)

    def configure_optimizers(self) -> Tuple[List]:
        """Initializes the optimizer and learning rate scheduler.

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "valid_loss",
        }
        return [self.optimizer], [self.scheduler]


class ImageClassifier(LightningModule):
    def __init__(
        self,
        model_name: str = None,
        avgpool: bool = False,
        weights: str = "DEFAULT",
        num_classes: int = 3,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        class_weights: List[float] = None,
        metric_avg: str = "weighted",
    ):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.metric_avg = metric_avg
        model = torchvision.models.get_model(name=model_name, weights=weights)
        last_layer_size = list(model.modules())[-1].in_features
        self.body = list(model.children())[:-1]
        self.body = nn.Sequential(*self.body)
        self.avgpool = avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=last_layer_size, out_features=20),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=10, out_features=5),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=5, out_features=self.num_classes),
        )
        metrics = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
                F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
                Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
                Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=self.metric_avg,
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.val_step_labels = []
        self.val_step_preds = []
        self.test_step_labels = []
        self.test_step_preds = []

    def forward(
        self, inputs: Union[Dict[str, Union[int, torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:

        # If-else block to add average pooling if pretrain model excludes it
        if self.avgpool:
            output = F.adaptive_avg_pool2d(self.body(inputs), (1, 1))
            output = torch.flatten(output, start_dim=1)
        else:
            output = torch.flatten(self.body(inputs), start_dim=1)
        logits = self.classifier(output)
        return logits

    def _shared_step(
        self, batch: Dict, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pixel_values, labels = batch
        logits = self(pixel_values)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, device=self.device)
        # nn.CrossEntropyLoss not used to get loss and pred separately
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = torch.softmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        metrics = self.train_metrics(preds, labels)
        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> int:

        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.val_step_labels.append(labels)
        self.val_step_preds.append(preds)
        metrics = self.valid_metrics(preds, labels)
        self.log(
            "valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        labels = torch.cat(self.val_step_labels)
        preds = torch.cat(self.val_step_preds)
        fig_cm, fig_bar, fig_prc = create_eval_plots(
            preds, labels, num_classes=self.num_classes, normalize=None
        )
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(
            figure=fig_cm,
            artifact_file="val_confusion_matrix.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_bar,
            artifact_file="val_cm_count_plot.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_prc, artifact_file="val_prc.png", run_id=self.logger.run_id
        )

    def test_step(self, batch: Dict, batch_idx: int) -> int:

        _, preds, labels = self._shared_step(batch, batch_idx)
        self.test_step_labels.append(labels)
        self.test_step_preds.append(preds)
        metrics = self.test_metrics(preds, labels)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        labels = torch.cat(self.test_step_labels)
        preds = torch.cat(self.test_step_preds)
        fig_cm, fig_bar, fig_prc = create_eval_plots(
            preds, labels, num_classes=self.num_classes, normalize=None
        )
        mlflow_logger = self.logger.experiment
        mlflow_logger.log_figure(
            figure=fig_cm,
            artifact_file="test_confusion_matrix.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_bar,
            artifact_file="test_cm_count_plot.png",
            run_id=self.logger.run_id,
        )
        mlflow_logger.log_figure(
            figure=fig_prc, artifact_file="test_prc.png", run_id=self.logger.run_id
        )

    def predict_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:

        features = batch
        return self(features)

    def configure_optimizers(self) -> None:
        """Initializes the optimizer and learning rate scheduler.

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "valid_loss",
        }
        return [self.optimizer], [self.scheduler]
