import os
import random

import yaml

dataset_path = "data/scene_datasets/floorplanner/glb-arch-only"
splits_yaml_output_path = "data/scene_datasets/floorplanner/scene_splits.yaml"
train_split_ratio, test_split_ratio, val_split_ratio = 0.6, 0.3, 0.1


def generate_fp_dataset_splits():
    assert round(train_split_ratio + test_split_ratio + val_split_ratio, 3) == 1.0

    scenes = sorted(os.listdir(dataset_path))

    random.seed(0)
    random.shuffle(scenes)

    train_val_size = int(len(scenes) * (train_split_ratio + val_split_ratio))

    train_val_set = scenes[:train_val_size]
    test_set = scenes[train_val_size:]

    train_size = int(
        len(train_val_set) * train_split_ratio / (train_split_ratio + val_split_ratio)
    )

    train_set = train_val_set[:train_size]
    val_set = train_val_set[train_size:]

    val_size = len(val_set)
    test_size = len(test_set)

    scene_splits_dict = {"train": train_set, "test": test_set, "val": val_set}

    with open(splits_yaml_output_path, "w") as f:
        yaml.dump(scene_splits_dict, f, sort_keys=False)


if __name__ == "__main__":
    generate_fp_dataset_splits()
