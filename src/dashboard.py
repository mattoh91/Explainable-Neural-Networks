"""ONNX model streamlit dashboard This script spins up a streamlit dashboard
for an ONNX model trained using Spark MLLib."""

import logging
from io import BytesIO
from pathlib import Path

import general_utils
import requests
import streamlit as st
from PIL import Image

# Constants\
CWD = Path.cwd().resolve()
LOG_CONFIG = CWD / "conf/base/logging.yaml"


def main() -> None:

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    general_utils.setup_logging(LOG_CONFIG)

    logger.info("Initialising dashboard...")
    st.title("Explainable Pneumonia Classifier")

    uploaded_file = st.file_uploader(
        label="Upload a medical chest X-ray image", type=["jpg", "jpeg", "png"]
    )

    if st.button("Predict"):
        logger.info("Sending request to inference server.")

        # Obtain byte data from uploaded_file - child class of io.BytesIO
        files = {"file": uploaded_file.getvalue()}

        response = requests.post("http://fastapi-server:8080/predict", files=files)

        resp_image = Image.open(BytesIO(response.content))
        resp_proba = response.headers["predict_proba"]
        resp_predicted_class = response.headers["predicted_class"]

        logger.info("Inference has completed.")

        st.image(
            resp_image,
            caption="Red pixels contribute negatively to prediction; green pixels \
                contribute positively.",
        )
        st.write(f"Predicted class: {resp_predicted_class}")
        st.write(f"Predicted probability: {resp_proba}")


if __name__ == "__main__":
    main()
