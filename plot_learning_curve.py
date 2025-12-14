"""
Utility plotting helpers that can be imported and called directly from
`tiny_shakespeare.py` (or any other training script).
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import matplotlib.pyplot as plt



def plot_learning_curve(
    training_losses: Sequence[float],
    validation_losses: Sequence[float],
    *,
    best_epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    early_stopping: Optional[bool] = None,
    path: Union[str, Path] = "learning_curves",
    title: Optional[str] = None,
    file_name: Optional[str] = None,
    show: bool = False,
) -> Path:
    """
    Plot and save training/validation loss curves.

    Args:
        training_losses: Sequence of training losses per epoch.
        validation_losses: Sequence of validation losses per epoch.
        best_epoch: 1-based epoch index considered best (used for early stopping).
        max_epochs: Total number of epochs; defaults to max length of the provided
            losses.
        early_stopping: Whether to split the plot at the best epoch. If None, it is
            inferred from whether best_epoch < max_epochs.
        path: Output directory for the saved plot (created if missing).
        title: Title to display on the plot.
        file_name: File stem for the saved image (defaults to a name derived from
            title).
        show: Whether to display the plot window.

    Returns:
        Path to the saved PNG file.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plot_learning_curve.")

    training_losses = list(training_losses)
    validation_losses = list(validation_losses)

    if max_epochs is None:
        max_epochs = max(len(training_losses), len(validation_losses))
    if best_epoch is None:
        best_epoch = max_epochs

    best_epoch = max(1, min(best_epoch, max_epochs))
    if early_stopping is None:
        early_stopping = best_epoch < max_epochs

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.7625))

    if early_stopping:
        if training_losses:
            ax.plot(
                range(1, min(best_epoch, len(training_losses)) + 1),
                training_losses[:best_epoch],
                label="Training",
                color="firebrick",
            )
            if best_epoch < len(training_losses):
                ax.plot(
                    range(best_epoch + 1, len(training_losses) + 1),
                    training_losses[best_epoch:],
                    linestyle="dotted",
                    color="firebrick",
                )

        if validation_losses:
            ax.plot(
                range(1, min(best_epoch, len(validation_losses)) + 1),
                validation_losses[:best_epoch],
                label="Validation",
                color="darkslategrey",
            )
            if best_epoch < len(validation_losses):
                ax.plot(
                    range(best_epoch + 1, len(validation_losses) + 1),
                    validation_losses[best_epoch:],
                    linestyle="dotted",
                    color="darkslategrey",
                )
    else:
        if training_losses:
            ax.plot(
                range(1, len(training_losses) + 1),
                training_losses,
                label="Training",
                color="firebrick",
            )
        if validation_losses:
            ax.plot(
                range(1, len(validation_losses) + 1),
                validation_losses,
                label="Validation",
                color="darkslategrey",
            )

    # Highlight key points
    if early_stopping and training_losses:
        epoch_idx = min(best_epoch, len(training_losses))
        ax.plot(
            epoch_idx,
            training_losses[epoch_idx - 1],
            marker=".",
            color="firebrick",
        )
    elif training_losses:
        ax.plot(
            len(training_losses),
            training_losses[-1],
            marker=".",
            color="firebrick",
        )

    if early_stopping and validation_losses:
        epoch_idx = min(best_epoch, len(validation_losses))
        ax.plot(
            epoch_idx,
            validation_losses[epoch_idx - 1],
            marker=".",
            color="darkslategrey",
        )
    elif validation_losses:
        ax.plot(
            len(validation_losses),
            validation_losses[-1],
            marker=".",
            color="darkslategrey",
        )

    ax.set_xlim([1, max_epochs])
    ax.set_title(title or "Learning Curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    if training_losses or validation_losses:
        ax.legend(loc="best")

    stem = file_name or title or "learning_curve"
    stem = stem.replace(" ", "_")
    output_file = output_dir / f"{stem}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return output_file
