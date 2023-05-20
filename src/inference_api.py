import io
import logging
from pathlib import Path
from typing import Annotated

import general_utils
import numpy as np
import torch
import visualization as viz
from captum.attr import IntegratedGradients
from datamodules import PneumoniaDataModule, PredictImageDataset
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import ToTensor

from models import ImageClassifier

CLASSES = ("bacteria", "normal", "virus")
LOG_FILEPATH = Path("../conf/base/logging.yaml").resolve()

# Add this dynamically using hydra in dashboard.py?
# Then send image and this as a request body
MODEL_FILEPATH = Path("../models/mobilenetv2_ft_200520231607.ckpt")

app = FastAPI()


class Results(BaseModel):
    tensor: list


# Setup logger
logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
general_utils.setup_logging(logging_config_path=LOG_FILEPATH)

logger.info("Loading model.")
model = ImageClassifier.load_from_checkpoint(MODEL_FILEPATH)
model.cpu()

# ### To Remove ###
# DATA_DIR = Path.cwd().parent / "data/processed"
# image_list = [DATA_DIR / "predict/IM-0022-0001.jpeg"]
# dm_test = PneumoniaDataModule(image_size=224)
# dm_test.setup("predict", pred_img_list=image_list)
### End ###


# Route requests to root over to swagger UI
@app.get("/")
def redirect_swagger():
    response = RedirectResponse("/docs")
    return response


# Inference
@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile) -> dict:
    logger.info("Opening image file.")
    ext = file.filename.split(".")[-1]
    if ext not in ("jpg", "jpeg", "png"):
        return "Image must be jpg or png format."
    image = io.BytesIO(await file.read())

    logger.info("Initialising datamodule.")
    dm = PneumoniaDataModule(image_size=224)

    logger.info("Initialising dataloader.")
    dm.setup(stage="predict", pred_img_list=[image])
    dl = dm.predict_dataloader()

    logger.info("Commencing prediction.")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            pred = model.predict_step(batch.cpu(), batch_idx)

    logger.info("Initialising explainability module.")
    prediction_score, pred_label_idx = torch.topk(pred, 1)
    # pred_class = CLASSES[pred_label_idx]
    integrated_gradients = IntegratedGradients(model)

    logger.info("Computing integrated gradients.")
    attributions_ig = integrated_gradients.attribute(
        batch, target=pred_label_idx.item(), n_steps=50
    )

    logger.info("Generating explainability image mask.")
    fig, ax = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(batch.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="blended_heat_map",
        show_colorbar=True,
        sign="all",
    )

    logger.info("Sending response.")
    fig.savefig(
        "test.png"
    )  # Stopped here; not sure if can put FileResponse in a Response Model

    return FileResponse("test.png")


def open_image(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image
