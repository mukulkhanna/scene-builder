"""
This script (sourced from https://github.com/facebookresearch/habitat-lab/blob/main/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py) is for creating a PointNav dataset for Floor-planner scenes.
"""

import gzip
import json
import multiprocessing
import os
from itertools import repeat
from os import path as osp

import habitat
import habitat_sim
import tqdm
import yaml
from habitat.datasets.pointnav.pointnav_generator import \
    generate_pointnav_episode
from habitat_sim.nav import NavMeshSettings

NUM_EPISODES_PER_SCENE = int(100000)

glbs_path = "data/scene_datasets/floorplanner/glb-arch-only"
scenes_path = "/nethome/mkhanna37/flash1/proj-scene-builder/data/scene_datasets/floorplanner/hab-fp-dataset-no-doors/configs/stages"
splits_info_path = "data/scene_datasets/floorplanner/scene_splits.yaml"
dataset_config_path = "data/scene_datasets/floorplanner/hab-fp-dataset-no-doors/hab-fp.scene_dataset_config.json"

output_dataset_path = "data/datasets/pointnav/floorplanner-100k"


def _generate_fn(scene, split):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene.split(".")[0]
    cfg.SIMULATOR.SCENE_DATASET = dataset_config_path
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=False
    )
    if sim.pathfinder.is_loaded:
        navmesh_save_path = os.path.join(glbs_path, scene.split(".")[0] + ".navmesh")
        sim.pathfinder.save_nav_mesh(navmesh_save_path)
        print('Saved NavMesh to "' + navmesh_save_path + '"')
    else:
        print("NAVMESH NOT LOADED")

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE, is_gen_shortest_path=False
        )
    )

    for ep in dset.episodes:
        # ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]
        ep.scene_id = scene

    scene_key = scene.split(".glb")[0]

    out_file = os.path.join(
        output_dataset_path, split, "content", f"{scene_key}.json.gz"
    )
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    # global episodes
    # episodes += dset.episodes
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    sim.close()


def generate_fp_pointnav_dataset():
    # Load train / val statistics
    with open(splits_info_path, "r") as f:
        scene_splits = yaml.safe_load(f)

    os.makedirs(output_dataset_path, exist_ok=True)

    for split in scene_splits.keys():
        # if split != 'train':
        #     continue
        # global episodes
        # episodes = []
        scenes = scene_splits[split]

        # for scene in scenes:
        #     _generate_fn(scene, split)
        with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
            # for _ in pool.imap_unordered(_generate_fn, scenes):
            for _ in pool.starmap(_generate_fn, zip(scenes, repeat(split))):
                pbar.update()

        path = os.path.join(output_dataset_path, split, split + ".json.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # split_dataset = habitat.datasets.make_dataset("PointNav-v1")
        # split_dataset.episodes = episodes
        with gzip.open(path, "wt") as f:
            json.dump(dict(episodes=[]), f)
            # f.write(split_dataset.to_json())


if __name__ == "__main__":
    generate_fp_pointnav_dataset()
