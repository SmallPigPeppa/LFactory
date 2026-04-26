import json
import math
import os

from transformers.trainer import TRAINER_STATE_NAME

from . import logging

logger = logging.get_logger(__name__)


def smooth(values):
    if not values:
        return []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(values))) - 0.5)
    last = values[0]
    result = []
    for value in values:
        last = last * weight + (1 - weight) * value
        result.append(last)
    return result


def plot_loss(output_dir, keys=("loss",)):
    import matplotlib.pyplot as plt

    state_path = os.path.join(output_dir, TRAINER_STATE_NAME)
    with open(state_path, encoding="utf-8") as f:
        log_history = json.load(f).get("log_history", [])

    plt.switch_backend("agg")
    for key in keys:
        points = [(row["step"], row[key]) for row in log_history if key in row]
        if not points:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue
        steps, values = zip(*points)
        plt.figure()
        plt.plot(steps, values, alpha=0.4, label="original")
        plt.plot(steps, smooth(list(values)), label="smoothed")
        plt.title(f"training {key}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        path = os.path.join(output_dir, f"training_{key.replace('/', '_')}.png")
        plt.savefig(path, format="png", dpi=100)
        print("Figure saved at:", path)
