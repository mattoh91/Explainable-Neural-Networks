from datetime import datetime
from pathlib import Path

import hydra
import torch
from datamodules import HFLitPneumoniaDataModule, PneumoniaDataModule
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from model_utils import fine_tune
from omegaconf import DictConfig

from models import HFLitImageClassifier, ImageClassifier

# Constants regarding Hydra config.
CONFIG_DIR = Path.cwd() / "conf/base"

# Main training function
# Update config_name via CLI using -cn option
@hydra.main(config_path=CONFIG_DIR, config_name=None)
def main(cfg: DictConfig) -> None:
    cwd = Path(hydra.utils.get_original_cwd())

    if cfg["model"]["framework"] == "huggingface":
        dm = HFLitPneumoniaDataModule(
            train_batch_size=cfg["train"]["params"]["batch_size"],
            eval_batch_size=cfg["eval"]["params"]["batch_size"],
            checkpoint=cfg["model"]["model_name"],
            data_dir=cwd / cfg["data"]["data_dir"],
        )
        dm.prepare_data()

        # If fine-tune, load saved feature extract model
        # Saved model will be entirely requires_grad==True
        if cfg["train"]["fine_tune"]["fine_tune"]:
            model = HFLitImageClassifier.load_from_checkpoint(
                cwd / cfg["train"]["fine_tune"]["model_filepath"],
                learning_rate=cfg["train"]["params"]["learning_rate"],
            )
        # Else feature extract using pre-train model
        else:
            model = HFLitImageClassifier(
                checkpoint=cfg["model"]["model_name"],
                num_classes=cfg["data"]["num_classes"],
                learning_rate=cfg["train"]["params"]["learning_rate"],
                class_weights=cfg["data"]["class_weights"],
            )

    elif cfg["model"]["framework"] == "torch":
        dm = PneumoniaDataModule(
            train_batch_size=cfg["train"]["params"]["batch_size"],
            eval_batch_size=cfg["eval"]["params"]["batch_size"],
            data_dir=cwd / cfg["data"]["data_dir"],
            image_size=cfg["train"]["transforms"]["image_size"],
        )

        # If fine-tune, load saved feature extract model
        # Saved model will be entirely requires_grad==True
        if cfg["train"]["fine_tune"]["fine_tune"]:
            model = ImageClassifier.load_from_checkpoint(
                cwd / cfg["train"]["fine_tune"]["model_filepath"],
                learning_rate=cfg["train"]["params"]["learning_rate"],
            )
        # Else feature extract using pre-train model
        else:
            model = ImageClassifier(
                model_name=cfg["model"]["model_name"],
                avgpool=bool(cfg["model"]["avgpool"]),
                weights=cfg["model"]["weights"],
                num_classes=cfg["data"]["num_classes"],
                learning_rate=cfg["train"]["params"]["learning_rate"],
                class_weights=cfg["data"]["class_weights"],
            )

    # If fine-tune, set requires_grad=False for layers up to num_layers in body
    fine_tune(
        model,
        fine_tune=cfg["train"]["fine_tune"]["fine_tune"],
        num_layers=cfg["train"]["fine_tune"]["num_layers"],
        normlayer_name=cfg["train"]["fine_tune"]["normlayer_name"],
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", patience=3, strict=False, verbose=False, mode="min"
    )

    lr_logger = LearningRateMonitor()

    curr_dt = datetime.now().strftime("%d%m%Y%H%M")

    checkpoint_callback = ModelCheckpoint(
        filename=f"{cfg['model']['save_name']}_{curr_dt}",
        dirpath=cwd / cfg["model"]["model_dir"],
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min",
    )

    mlflow_logger = pl_loggers.MLFlowLogger(
        experiment_name=cfg["logging"]["experiment_name"],
        run_name=f"{cfg['model']['save_name']}_{cfg['logging']['run_name']}_{curr_dt}",
        save_dir=cwd / cfg["logging"]["log_dir"],
        tracking_uri=f"file:{cwd/cfg['logging']['log_dir']}",
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, lr_logger],
        logger=[mlflow_logger],
        max_epochs=cfg["train"]["params"]["epochs"],
    )
    dm.setup(stage="fit")
    trainer.fit(model, dm)

    dm.setup(stage="test")
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
