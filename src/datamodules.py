import io
import pathlib
from pathlib import Path, PosixPath
from typing import Dict, List, Union

import PIL
import torch
import torchvision
from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor


class HFLitPneumoniaDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        checkpoint: str,
        data_dir: str,
        train_transforms: torchvision.transforms.Compose = None,
        eval_transforms: torchvision.transforms.Compose = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.checkpoint = checkpoint
        self.data_dir = data_dir
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.normalize = Normalize(
            mean=self.image_processor.image_mean, std=self.image_processor.image_std
        )
        self.imgsize = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
            )
        )
        if not train_transforms:
            self.train_transforms = Compose(
                [
                    Grayscale(3),
                    RandomResizedCrop(self.image_processor.size["shortest_edge"]),
                    RandomHorizontalFlip(),
                ]
            )
        if not eval_transforms:
            self.eval_transforms = Compose([Grayscale(3)])

    def prepare_data(self) -> None:
        # Create dataset
        self.dataset = load_dataset("imagefolder", data_dir=self.data_dir)

    def setup(
        self,
        stage: str,
        pred_img_list: List[Union[pathlib.PosixPath, io.BytesIO]] = None,
    ) -> None:
        if stage == "fit":
            self.train_ds = self.dataset["train"]
            self.val_ds = self.dataset["validation"]
            self.train_ds.set_transform(self.preprocess_train)
            self.val_ds.set_transform(self.preprocess_eval)

        elif stage == "test":
            self.test_ds = self.dataset["test"]
            self.test_ds.set_transform(self.preprocess_eval)

        elif stage == "predict":
            assert (
                pred_img_list is not None
            ), "'predict' stage requires a list of image inputs"
            self.pred_ds = PredictImageDataset(
                image_list=pred_img_list, transform=self.preprocess_predict
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.train_ds,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.train_batch_size,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.val_ds, collate_fn=self.collate_fn, batch_size=self.eval_batch_size
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.test_ds, collate_fn=self.collate_fn, batch_size=self.eval_batch_size
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.pred_ds,
            collate_fn=self.collate_fn_predict,
            batch_size=self.pred_ds.__len__(),
        )

    def preprocess_train(
        self, example_batch: Dict[str, List[Union[int, PIL.ImageFile.ImageFile]]]
    ) -> Dict[str, List[Union[int, PIL.ImageFile.ImageFile, torch.Tensor]]]:
        """Preprocesses a batch using a pretrained Huggingface image processor.

        Used as an arg to a Huggingface dataset.

        Args:
            example_batch: A batch dict of PIL image files and labels.

        Returns:
            Batch is returned with appended pixel values.
        """
        images = [
            self.train_transforms(img.convert("RGB")) for img in example_batch["image"]
        ]
        example_batch["pixel_values"] = self.image_processor(
            images, return_tensors="pt"
        )["pixel_values"]
        return example_batch

    def preprocess_eval(
        self, example_batch: Dict[str, List[Union[int, PIL.ImageFile.ImageFile]]]
    ) -> Dict[str, List[Union[int, PIL.ImageFile.ImageFile, torch.Tensor]]]:
        """Preprocesses a batch using a pretrained Huggingface image processor.

        Used as an arg to a Huggingface dataset.

        Args:
            example_batch: A batch dict of PIL image files and labels.

        Returns:
            Batch is returned with appended pixel values.
        """
        images = [
            self.eval_transforms(img.convert("RGB")) for img in example_batch["image"]
        ]
        example_batch["pixel_values"] = self.image_processor(
            images, return_tensors="pt"
        )["pixel_values"]
        return example_batch

    def preprocess_predict(self, image: PIL.Image.Image) -> torch.Tensor:
        """Preprocesses a single image using a pretrained Huggingface image
        processor.

        Used as an arg to a Torch dataset to facilitate decoupling of predict data input from datamodule.

        Args:
            image: A PIL image files with no labels.

        Returns:
            Pixel values of image is returned.
        """
        image = self.eval_transforms(image)
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze()

    @staticmethod
    def collate_fn(
        examples: List[Dict[str, Union[int, PIL.ImageFile.ImageFile, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Collates preprocessed inputs into a dict of batches suitable for
        input into a Huggingface model.

        Args:
            examples: List of dicts - 1 per example - comprising a PIL image file, label, and pixel value tensor.

        Returns:
            A dictionary of batches - 4d tensors containing pixel values and labels.
        """
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples]).type(
            torch.int64
        )
        return {"pixel_values": pixel_values, "labels": labels}

    @staticmethod
    def collate_fn_predict(examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Collates preprocessed inputs into a dict of batches suitable for
        input into a Huggingface model.

        Args:
            examples: List of tensors of pixel values - 1 per example.

        Returns:
            A dictionary of batches - 4d tensors containing pixel values with no labels.
        """
        pixel_values = torch.stack(examples)
        return {"pixel_values": pixel_values}


class PredictImageDataset(Dataset):
    def __init__(
        self,
        image_list: List[Union[str, io.BytesIO]],
        transform: torchvision.transforms.Compose = None,
        target_transform: torchvision.transforms.Compose = None,
    ):
        self.image_list = image_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw_image = self.image_list[idx]
        image = Image.open(raw_image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class PneumoniaDataModule(LightningDataModule):
    def __init__(
        self,
        image_size: int,
        train_batch_size: int = 200,
        eval_batch_size: int = 200,
        data_dir: pathlib.PosixPath = None,
        train_transforms: torchvision.transforms.Compose = None,
        eval_transforms: torchvision.transforms.Compose = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.image_size = image_size
        self.data_dir = data_dir
        if not train_transforms:
            self.train_transforms = Compose(
                [
                    Grayscale(3),
                    RandomResizedCrop(self.image_size, scale=(0.9, 1.05), ratio=(1, 1)),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        if not eval_transforms:
            self.eval_transforms = Compose(
                [
                    Grayscale(3),
                    Resize(self.image_size),
                    CenterCrop(self.image_size),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def prepare_data(self) -> None:
        pass

    def setup(
        self,
        stage: str,
        pred_img_list: List[Union[pathlib.PosixPath, io.BytesIO]] = None,
    ) -> None:
        if stage == "fit":
            self.train_ds = ImageFolder(
                self.data_dir / "train", transform=self.train_transforms
            )
            self.val_ds = ImageFolder(
                self.data_dir / "val", transform=self.eval_transforms
            )
        elif stage == "test":
            self.test_ds = ImageFolder(
                self.data_dir / "test", transform=self.eval_transforms
            )
        elif stage == "predict":
            assert (
                pred_img_list is not None
            ), "'predict' stage requires a list of image inputs"
            self.pred_ds = PredictImageDataset(
                image_list=pred_img_list, transform=self.eval_transforms
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.train_batch_size)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.val_ds, batch_size=self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.test_ds, batch_size=self.eval_batch_size)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.pred_ds, batch_size=self.pred_ds.__len__())
