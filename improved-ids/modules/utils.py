import json
import os
from typing import List, Tuple, Union


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


CHECKPOINT_DIR = './checkpoints'
PATH_TO_CHECKPOINTS = f'{CHECKPOINT_DIR}/feature_selection.json'


def save_checkpoint(
    split_id: int,
    features_data: List[Tuple[int, List[str], List[float]]],
    path: str = PATH_TO_CHECKPOINTS
) -> None:
    """Save checkpoint to local file

    Args:
        split_id: Current split index
        features_data: List of tuples containing:
            - split index (int)
            - feature names (list of strings)
            - feature weights (list of floats)
    """
    checkpoint_data = {
        'last_completed_split': split_id,
        'features_data': features_data
    }

    # Ensure directory exists (create parent directory only)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save with atomic write using temporary file
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f)
        if os.path.exists(path):
            os.remove(path)  # Remove existing file first
        os.rename(temp_path, path)  # Atomic operation
        print(bcolors.OKGREEN + f"Checkpoint saved for split {split_id + 1}" + bcolors.ENDC)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Failed to save checkpoint: {str(e)}")


def load_checkpoint(path: str = PATH_TO_CHECKPOINTS) -> Tuple[int, List[Tuple[int, List[str], List[float]]]]:
    """Load checkpoint from local file"""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data['last_completed_split'], checkpoint_data['features_data']
    except:
        pass
    return -1, []

def clear_checkpoint(path: str = PATH_TO_CHECKPOINTS) -> None:
    """Clear checkpoint file"""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(bcolors.OKGREEN + "Checkpoint cleared successfully" + bcolors.ENDC)
    except Exception as e:
        print(bcolors.WARNING + f"Failed to clear checkpoint: {str(e)}" + bcolors.ENDC)
