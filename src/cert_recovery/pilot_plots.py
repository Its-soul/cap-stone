"""Standalone plots of measured pilot artifacts, embedded by the HTML writer."""
import json


def make_plots(run, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .pilot import WARNING
    paths = []
    def save(fig, name):
        fig.suptitle(WARNING, fontsize=8)
        path = run / (name + ".png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    for name, record in summary["models"].items():
        fig, ax = plt.subplots(figsize=(5,4))
        matrix = record["confusion_matrix"]
        ax.imshow(matrix, cmap="Blues")
        for y in (0,1):
            for x in (0,1):
                ax.text(x,y,str(matrix[y][x]),ha="center",va="center")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=record["matrix_order"],yticklabels=record["matrix_order"],
               xlabel="Predicted",ylabel="Actual",title=name)
        save(fig,name+"_confusion")
    if summary["system"]:
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        names = list(summary["system"])
        axes[0].bar(names,[summary["system"][n]["safe_commits"]/summary["system"][n]["tasks"] for n in names])
        axes[0].set(ylabel="Support-valid useful completion",ylim=(0,1))
        axes[1].bar(names,[summary["system"][n]["mean_total_seconds"]*1000 for n in names])
        axes[1].set(ylabel="Mean total wall time (ms), including setup")
        for ax in axes:
            ax.tick_params(axis="x",rotation=25,labelsize=8)
        save(fig,"workflow_comparison")
    selection = run / "checkpoints/selection.json"
    if selection.exists():
        logs = json.loads(selection.read_text(encoding="utf-8"))["log_history"]
        train = [x for x in logs if "loss" in x]
        validation = [x for x in logs if "eval_loss" in x]
        if train:
            fig, axes = plt.subplots(1,2,figsize=(10,4))
            axes[0].plot([x["step"] for x in train],[x["loss"] for x in train],label="training loss")
            if validation:
                axes[0].plot([x["step"] for x in validation],[x["eval_loss"] for x in validation],label="validation loss")
                axes[1].plot([x["step"] for x in validation],[x["eval_macro_f1"] for x in validation],label="validation macro F1")
            axes[0].legend()
            for ax in axes:
                ax.set(xlabel="Optimizer step")
            save(fig,"training_validation_curves")
    return paths
