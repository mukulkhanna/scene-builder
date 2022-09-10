import gzip
import json
import math
import os
import random

import cv2
import habitat
import habitat_sim
import magnum as mn
import numpy as np
from habitat.utils.visualizations import maps
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils import common as utils
from habitat_sim.utils.viz_utils import save_video

dataset_config_path = "/nethome/mkhanna37/flash1/proj-scene-builder/data/scene_datasets/floorplanner/hab-fp-dataset-no-doors/hab-fp.scene_dataset_config.json"
pointnav_dataset_path = "/nethome/mkhanna37/flash1/proj-scene-builder/data/datasets/pointnav/floorplanner-100k/train/content"
viz_output_path = "/nethome/mkhanna37/flash1/proj-scene-builder/data/datasets/pointnav/floorplanner-100k/viz"


def get_topdown_map_with_path(sim, start_pos, start_rot, goal_pos):
    topdown_map = maps.get_topdown_map(sim.pathfinder, height=start_pos[1])

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    topdown_map = recolor_map[topdown_map]
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])

    # convert world agent position to maps module grid point
    agent_grid_pos_source = maps.to_grid(
        start_pos[2], start_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )

    agent_grid_pos_target = maps.to_grid(
        goal_pos[2], goal_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )

    agent_forward = utils.quat_to_magnum(
        sim.agents[0].get_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))

    agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
    # draw the agent and trajectory on the map
    maps.draw_agent(
        topdown_map, agent_grid_pos_source, agent_orientation, agent_radius_px=24
    )

    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos
    found_path = sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    path_points = path.points
    # @markdown - Success, geodesic path length, and 3D points can be queried.
    # print("found_path : " + str(found_path))
    # print("geodesic_distance : " + str(geodesic_distance))
    # print("path_points : " + str(path_points))

    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path_points
    ]
    # draw the agent and trajectory on the map
    maps.draw_path(topdown_map, trajectory)

    return topdown_map, geodesic_distance, path_points


def visualize_fp_pointnav_dataset():

    os.makedirs(viz_output_path, exist_ok=True)

    scenes = os.listdir(pointnav_dataset_path)
    # scenes = random.sample(scenes, 2)

    path_lengths, geodesic_distances, nav_areas = [], [], []
    for scene in scenes:
        cfg = habitat.get_config()
        cfg.defrost()
        cfg.SIMULATOR.SCENE_DATASET = dataset_config_path
        cfg.SIMULATOR.SCENE = scene.split(".")[0]

        cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        cfg.freeze()

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(
            sim.pathfinder, navmesh_settings, include_static_objects=False
        )
        nav_areas.append(sim.pathfinder.navigable_area)
        dataset_path = os.path.join(
            pointnav_dataset_path, scene.split(".")[0] + ".json.gz"
        )
        with gzip.open(dataset_path, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))

        episodes = data["episodes"]
        episodes = random.sample(episodes, 4)

        for ep in episodes:
            start_pos = ep["start_position"]
            start_rot = ep["start_rotation"]
            goal_pos = ep["goals"][0]["position"]

            topdown_map, geodesic_distance, path_points = get_topdown_map_with_path(
                sim, start_pos, start_rot, goal_pos
            )

            imgs = []
            for point in path_points:
                sim.set_agent_state(point, start_rot)
                imgs.append(sim.render())

            save_video(
                os.path.join(
                    viz_output_path,
                    f'path_video_{scene.split(".")[0]}_{ep["episode_id"]}.mp4',
                ),
                imgs,
                fps=5,
            )

            path_lengths.append(len(path_points))
            geodesic_distances.append(geodesic_distance)

            cv2.imwrite(
                os.path.join(
                    viz_output_path,
                    f'topdown_map_{scene.split(".")[0]}_{ep["episode_id"]}.png',
                ),
                topdown_map,
            )

        sim.close()

    path_lengths = np.array(path_lengths)
    geodesic_distances = np.array(geodesic_distances)
    nav_areas = np.array(nav_areas)

    print("Avg path length:", np.mean(path_lengths))
    print("Avg geodesic_distance:", np.mean(geodesic_distances))
    print("Avg navigable area:", np.mean(nav_areas))


if __name__ == "__main__":
    visualize_fp_pointnav_dataset()
