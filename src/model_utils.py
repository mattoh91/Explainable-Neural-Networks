import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import OneHotEncoder
import torch
from typing import Tuple, Union

def createEvalPlots(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    normalize: Union[str, None] = None) -> Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
        
    preds = preds.detach().cpu()
    labels = labels.detach().cpu().numpy()
    classes = ("bacteria", "normal", "virus")
    cm = confusion_matrix(
        labels,
        torch.argmax(preds, dim=1).numpy(),
        labels=np.arange(0,num_classes,1),
        normalize=normalize
    )
    fig_cm = ConfusionMatrixDisplay(cm, display_labels=classes).plot().figure_

    class_counts = np.sum(cm, axis=1)
    tp_counts = np.diagonal(cm)
    fp_counts = np.sum(cm, axis=0) - tp_counts
    fn_counts = class_counts - tp_counts

    x_ticks = np.arange(len(classes))
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(x_ticks, tp_counts, align='center', label='True Positive')
    ax_bar.bar(x_ticks, fp_counts, align='center', bottom=tp_counts, label='False Positive')
    ax_bar.bar(x_ticks, fn_counts, align='center', bottom=tp_counts+fp_counts, label='False Negative')
    ax_bar.set_xlabel('Class')
    ax_bar.set_ylabel('Count')
    ax_bar.set_xticks(x_ticks)
    ax_bar.set_xticklabels(classes)
    ax_bar.legend()

    fig_prc, ax_prc = plt.subplots()
    enc = OneHotEncoder()
    enc.fit(np.arange(0,num_classes,1).reshape(-1,1))
    labels_ohe = enc.transform(np.array(labels).reshape(-1,1)).toarray()
    for i in range(len(classes)):
        PrecisionRecallDisplay.from_predictions(labels_ohe[:,i], preds.numpy()[:,i], name=classes[i], ax=ax_prc)  
    PrecisionRecallDisplay.from_predictions(labels_ohe.ravel(), preds.numpy().ravel(), name="micro_avg", ax=ax_prc)
    ax_prc.set_xlabel("Recall")
    ax_prc.set_ylabel("Precision")
    
    plt.ioff()
    plt.close(fig_cm)
    plt.close(fig_bar)
    plt.close(fig_prc)
    return fig_cm, fig_bar, fig_prc