from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(model, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    # ===== LOSS =====
    plt.figure()
    plt.plot(model.train_loss_history)
    plt.title("Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(output_dir / "train_loss.png")
    plt.close()

    # ===== METRICS =====
    for name, values in model.val_metrics_history.items():
        plt.figure()
        plt.plot(values)
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.savefig(output_dir / f"{name}.png")
        plt.close()
