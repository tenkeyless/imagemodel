import argparse
import os
import time
from typing import Optional


def loss_coords(s):
    try:
        x, y = s.split(",")
        return (x, float(y))
    except:
        raise argparse.ArgumentTypeError("Loss must be x,y")


def model_option_coords(s):
    try:
        x, y = s.split("@")
        return x, y
    except:
        raise argparse.ArgumentTypeError(
            'Model option must be "[option name]@[option value]"'
        )


def setup_continuous_training(
    continuous_model_name: Optional[str], continuous_epoch: Optional[int]
) -> Optional[str]:
    training_id: Optional[str] = None
    # extract `training_id` from `continuous_model_name`
    if continuous_model_name is not None:
        continuous_run_id: str = os.path.basename(continuous_model_name)
        if continuous_run_id.find(".") != -1:
            continuous_run_id = continuous_run_id[: continuous_run_id.find(".")]
        training_id = continuous_run_id
    return training_id


def get_run_id() -> str:
    os.environ["TZ"] = "Asia/Seoul"
    time.tzset()
    return time.strftime("%Y%m%d_%H%M%S")
