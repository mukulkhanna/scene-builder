import os

import habitat
import habitat_sim
import numpy as np
from habitat_sim.nav import NavMeshSettings

dataset_config_path = "/nethome/mkhanna37/flash1/proj-scene-builder/data/scene_datasets/floorplanner/hab-fp-dataset-no-doors/hab-fp.scene_dataset_config.json"
scenes_path = "data/scene_datasets/floorplanner/hab-fp-dataset-no-doors/configs/stages"


def compute_fp_scene_size_stats():

    scenes = os.listdir(scenes_path)

    nav_areas = []
    for scene in scenes:
        cfg = habitat.get_config()
        cfg.defrost()
        cfg.SIMULATOR.SCENE_DATASET = dataset_config_path
        cfg.SIMULATOR.SCENE = scene.split(".")[0]
        cfg.freeze()

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(
            sim.pathfinder, navmesh_settings, include_static_objects=False
        )
        nav_areas.append(sim.pathfinder.navigable_area)
        sim.close()

    nav_areas = np.array(nav_areas)
    print("Total navigable area:", np.sum(nav_areas))
    print("Avg navigable area:", np.mean(nav_areas))


if __name__ == "__main__":
    compute_fp_scene_size_stats()
