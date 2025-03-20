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


def save_checkpoint(split_id: int, features_data: List[Tuple[int, List[str], List[float]]]) -> None:
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
    
    os.makedirs('checkpoints', exist_ok=True)
    
    with open('checkpoints/feature_selection.json', 'w') as f:
        json.dump(checkpoint_data, f)
    print(bcolors.OKGREEN + f"Checkpoint saved for split {split_id + 1}" + bcolors.ENDC)

def load_checkpoint() -> Tuple[int, List[Tuple[int, List[str], List[float]]]]:
    """Load checkpoint from local file"""
    try:
        if os.path.exists('checkpoints/feature_selection.json'):
            with open('checkpoints/feature_selection.json', 'r') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data['last_completed_split'], checkpoint_data['features_data']
    except:
        pass
    return -1, []

def clear_checkpoint() -> None:
    """Clear checkpoint file"""
    try:
        if os.path.exists('checkpoints/feature_selection.json'):
            os.remove('checkpoints/feature_selection.json')
            print(bcolors.OKGREEN + "Checkpoint cleared successfully" + bcolors.ENDC)
    except Exception as e:
        print(bcolors.WARNING + f"Failed to clear checkpoint: {str(e)}" + bcolors.ENDC)
