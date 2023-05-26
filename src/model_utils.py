import functools
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from lightning.pytorch import LightningModule
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder


def create_eval_plots(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    normalize: Union[str, None] = None,
) -> Tuple[
    matplotlib.figure.Figure, matplotlib.figure.Figure, matplotlib.figure.Figure
]:

    preds = preds.detach().cpu()
    labels = labels.detach().cpu().numpy()
    classes = ("bacteria", "normal", "virus")
    cm = confusion_matrix(
        labels,
        torch.argmax(preds, dim=1).numpy(),
        labels=np.arange(0, num_classes, 1),
        normalize=normalize,
    )
    fig_cm = ConfusionMatrixDisplay(cm, display_labels=classes).plot().figure_

    fig_bar = cm_bar(cm=cm, classes=classes)

    fig_prc = multiclass_prc(
        num_classes=num_classes, labels=labels, classes=classes, preds=preds
    )

    # Configure plt to not show plots.
    plt.ioff()
    plt.close(fig_cm)
    plt.close(fig_bar)
    plt.close(fig_prc)
    return fig_cm, fig_bar, fig_prc


def cm_bar(
    cm: sklearn.metrics.confusion_matrix, classes: Tuple[str]
) -> matplotlib.figure.Figure:
    """Generates bar plot of counts of confusion matrix TP, FP, TN, FN values.

    Args:
        cm (sklearn.metrics.confusion_matrix): Confusion matrix object.
        classes (Tuple[str]): Tuple of strings of class names.

    Returns:
        matplotlib.figure.Figure: Bar plot figure object.
    """

    class_counts = np.sum(cm, axis=1)
    tp_counts = np.diagonal(cm)
    fp_counts = np.sum(cm, axis=0) - tp_counts
    fn_counts = class_counts - tp_counts

    x_ticks = np.arange(len(classes))
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(x_ticks, tp_counts, align="center", label="True Positive")
    ax_bar.bar(
        x_ticks, fp_counts, align="center", bottom=tp_counts, label="False Positive"
    )
    ax_bar.bar(
        x_ticks,
        fn_counts,
        align="center",
        bottom=tp_counts + fp_counts,
        label="False Negative",
    )
    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Count")
    ax_bar.set_xticks(x_ticks)
    ax_bar.set_xticklabels(classes)
    ax_bar.legend()

    # Configure plt to not show plots.
    plt.ioff()
    plt.close(fig_bar)
    return fig_bar


def multiclass_prc(
    num_classes: int, labels: np.array, classes: Tuple, preds: torch.Tensor
) -> matplotlib.figure.Figure:
    """Generates OneVsAll multiclass precision recall curve plot with
    microaverage precision recall.

    Args:
        num_classes (int): Number of classes.
        labels (np.array): Numerical labels for all examples (ground truth).
        classes (Tuple): Unique class names that can be mapped to numeric `labels`.
        preds (torch.Tensor): Predicted class probabilities for all examples.

    Returns:
        matplotlib.figure.Figure: Multiclass PRC figure object.
    """

    fig_prc, ax_prc = plt.subplots()
    enc = OneHotEncoder()
    enc.fit(np.arange(0, num_classes, 1).reshape(-1, 1))
    labels_ohe = enc.transform(np.array(labels).reshape(-1, 1)).toarray()
    # Per class precision-recall
    for i in range(len(classes)):
        PrecisionRecallDisplay.from_predictions(
            labels_ohe[:, i], preds.numpy()[:, i], name=classes[i], ax=ax_prc
        )
    # Ravel multiclass labels and preds into respective 1d arrays for microaveraging.
    PrecisionRecallDisplay.from_predictions(
        labels_ohe.ravel(), preds.numpy().ravel(), name="micro_avg", ax=ax_prc
    )
    ax_prc.set_xlabel("Recall")
    ax_prc.set_ylabel("Precision")

    # Configure plt to not show plots.
    plt.ioff()
    plt.close(fig_prc)
    return fig_prc


def fine_tune(
    model: LightningModule,
    fine_tune: bool = False,
    num_layers: Union[int, str] = None,
    normlayer_name: str = None,
) -> None:
    """Sets `requires_grad` of weight tensors for fine-tuning.

    Default behaviour is for feature extraction - no fine-tune.

    If `fine_tune` == True, default behaviour is to update gradients for all layers
    unless specified using `num_layers` param.

    Args:
        model (LightningModule): Model to finetune.

        fine_tune (bool, optional): Fine-tune or feature extract. Defaults to False.

        num_layers (int, optional): Int input specifies number of layers to maintain
        requires_grad=True. All other layers will be set to requires_grad=False since
        the default value is True after model is loaded.

        normlayer_name (str, optional): Name / Partial name of batch norm layers which will
        have their requires_grad set to False.
    """
    if fine_tune:
        # If num_layers not None and greater than 0
        if num_layers and num_layers > 0:
            # Set all layers up to num_layers idx to requires_grad=False
            for param in list(model.body.parameters())[:-num_layers]:
                param.requires_grad = False
            # Set batchnorm layers to requires_grad=False
            for name, param in model.body.named_parameters():
                if normlayer_name in name:
                    param.requires_grad = False

    else:
        for param in model.body.parameters():
            param.requires_grad = False


def rgetattr(obj: torch.nn.Module, attr: str, *args) -> torch.nn.Module:
    """Apply reduce + getattr to retrieve an attribute from a torch module
    dynamically using its attribute name in string. Reduce is used to access
    submodules that may be nested within nn.Sequential submodule groups.

    The intended use for this function is to retrieve the last submodule using a `attr`
    value from `list(obj.named_modules())[-1][0]`.

    Args:
        obj (torch.nn.Module): A Pytorch model.

        attr (str): Name of attribute to be returned.

    Returns:
        The attribute corresponding to the name specified as the `attr` arg.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: torch.nn.Module, attr: str, val: torch.nn.Module) -> torch.nn.Module:
    """Dynamically get then set new value to an attr using attr name in string.

    The intended use for this function is to replace the last classifier submodule of
    a pretrain model. However a more flexible approach would be to:

    (1) Remove the pretrained classifier head and save the body.

    (2) Create a new classifier head that is decoupled from the body.

    This enables setting requires_grad for body independent of the head.

    Args:
        obj (torch.nn.Module): A Pytorch model.

        attr (str): Name of attribute (submodule) to be replaced.

        val (torch.nn.Module): New submodule to replace old submodule with name `attr`

    Returns:
        The parent module `obj` with its `attr` replaced by the new `val` submodule.
    """
    pre, _, post = attr.rpartition(".")

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
