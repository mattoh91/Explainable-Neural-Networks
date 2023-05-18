import logging

from datetime import datetime
import hydra
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig
from pathlib import Path

from models import HFLitImageClassifier, ImageClassifier
from datamodules import HFLitPneumoniaDataModule, PneumoniaDataModule, PredictImageDataset
import visualization as viz

CONFIG_DIR = Path.cwd()/"conf/base"
CONFIG_FILENAME = "pipelines_resnet.yaml"

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_FILENAME)
def main(cfg: DictConfig) -> None:
    cwd = Path(hydra.utils.get_original_cwd())

    if cfg["model"]["framework"] == "huggingface":
        dm = HFLitPneumoniaDataModule(
            train_batch_size=cfg["train"]["params"]["batch_size"],
            eval_batch_size=cfg["eval"]["params"]["batch_size"],
            checkpoint=cfg["model"]["model_name"],
            data_dir=cwd/cfg["data"]["data_dir"]
        )
        dm.prepare_data()

        if cfg["train"]["fine_tune"]:
            model = HFLitImageClassifier(
                checkpoint=cwd/cfg["model"]["model_filepath"],
                num_classes=cfg["data"]["num_classes"]
            )
        else:
            model = HFLitImageClassifier(
                checkpoint=cfg["model"]["model_name"],
                num_classes=cfg["data"]["num_classes"]
            )

    elif cfg["model"]["framework"] == "torch":
        dm = PneumoniaDataModule(
            train_batch_size=cfg["train"]["params"]["batch_size"],
            eval_batch_size=cfg["eval"]["params"]["batch_size"],
            data_dir=cwd/cfg["data"]["data_dir"],
            image_size=cfg["train"]["transforms"]["image_size"]
        )

        if cfg["train"]["fine_tune"]:
            model = ImageClassifier.load_from_checkpoint(
                cwd/cfg["model"]["model_filepath"]
            )
        else:
            model = ImageClassifier(
                model_name=cfg["model"]["model_name"],
                num_classes=cfg["data"]["num_classes"]
            )
    
    model.fine_tune(fine_tune=cfg["train"]["fine_tune"])
    
    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    lr_logger = LearningRateMonitor()

    curr_dt = datetime.now().strftime("%d%m%Y%H%M")

    checkpoint_callback = ModelCheckpoint(
        filename=f"{cfg['model']['save_name']}_{curr_dt}",
        dirpath=cwd/cfg["model"]["model_dir"],
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min"
    )
    
    mlflow_logger = pl_loggers.MLFlowLogger(
        experiment_name=cfg["logging"]["experiment_name"],
        run_name=f"{cfg['model']['save_name']}_{cfg['logging']['run_name']}_{curr_dt}",
        save_dir=cwd/cfg["logging"]["log_dir"],
        tracking_uri=f"file:{cwd/cfg['logging']['log_dir']}"
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, lr_logger],
        logger=[mlflow_logger],
        max_epochs = cfg["train"]["params"]["epochs"]
    )
    dm.setup(stage="fit")
    trainer.fit(model, dm)

    dm.setup(stage="test")
    trainer.test(model, dm)


if __name__ == "__main__":
    main()