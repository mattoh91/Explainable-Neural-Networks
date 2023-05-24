import argparse
import io
import logging
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image

sys.path.append("./src")
import general_utils
import visualization as viz
from datamodules import PneumoniaDataModule

from models import ImageClassifier

# Use non-interactive backend to prevent figure pop-ups
matplotlib.use("Agg")


# Constants
CLASSES = ("bacteria", "normal", "virus")
CWD = Path.cwd().resolve()
LOG_CONFIG = CWD / "conf/base/logging.yaml"
MODEL_FILEPATH = CWD / "models" / os.getenv("MODEL_FILENAME")
# MODEL_FILEPATH = Path("./models")/"mobilenetv2_ft_200520232038.ckpt"

# Define custom responses
RESPONSES = {
    200: {
        "content": {"image/png": {}},
        "description": "Prediction image with explainability mask.",
    }
}

# Initialise fastapi app
app = FastAPI()

# Setup logger
logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
general_utils.setup_logging(logging_config_path=LOG_CONFIG)

# Load model
logger.info("Loading model.")
checkpoint = torch.load(MODEL_FILEPATH, map_location=torch.device("cpu"))
hyperparameters = checkpoint["hyper_parameters"]
weights = checkpoint["state_dict"]
model = ImageClassifier(**hyperparameters)
model.load_state_dict(weights)


# Route requests to root over to swagger UI
@app.get("/")
def redirect_swagger():
    response = RedirectResponse("/docs")
    return response


# Inference
@app.post("/predict", responses=RESPONSES)
async def predict(file: UploadFile) -> Response:
    logger.info("Opening image file.")
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
            logits = model.predict_step(batch.cpu(), batch_idx)

    logger.info("Initialising explainability module.")
    pred = torch.softmax(logits, dim=1)
    pred = torch.softmax(pred, dim=1)
    prediction_score, pred_label_idx = torch.topk(pred, 1)
    pred_class = CLASSES[pred_label_idx]
    integrated_gradients = IntegratedGradients(model)

    logger.info("Computing integrated gradients.")
    attributions_ig = integrated_gradients.attribute(
        batch, target=pred_label_idx.item(), n_steps=50
    )

    logger.info("Generating explainability image mask.")
    fig, _ = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(batch.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="blended_heat_map",
        show_colorbar=True,
        sign="all",
    )
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png")
    plt.close("all")
    # Reset image buffer to frame 0
    image_buffer.seek(0)

    logger.info("Sending response.")

    headers = {
        "predict_proba": str(prediction_score.item()),
        "predicted_class": pred_class,
    }

    return Response(
        content=image_buffer.getvalue(), media_type="image/png", headers=headers
    )


def open_image(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image
