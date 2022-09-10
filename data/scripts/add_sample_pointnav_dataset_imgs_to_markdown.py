import os

dataset_viz_path = "data/datasets/pointnav/floorplanner-100k/viz"
readme_path = os.path.join(os.path.dirname(dataset_viz_path), "README.md")

imgs = sorted(os.listdir(dataset_viz_path))
scene_dict = {}

for img in imgs:
    scene_id = img.replace("topdown_map_", "")
    scene_id = scene_id.replace(".png", "")
    episode_id = scene_id.split("_")[-1]
    scene_id = "_".join(scene_id.split("_")[:-1])

    if scene_id not in scene_dict.keys():
        scene_dict[scene_id] = [episode_id]
    else:
        scene_dict[scene_id].append(episode_id)

md_body_str = ""

for scene_id in scene_dict.keys():
    eps = scene_dict[scene_id]
    row_head_str = ""
    title_str = f"Scene: {scene_id}\n"
    for ep in eps[:3]:
        row_head_str += f"{ep}|"
    row_head_str += "\n"
    row_divider = ":-:|:-:|:-:\n"
    md_row_string = ""
    for ep in eps[:3]:
        img_file_name = f"topdown_map_{scene_id}_{ep}.png"
        img_path = os.path.join(dataset_viz_path, img_file_name)
        md_row_string += f"![](/{img_path})|"

    final_row_str = title_str + row_head_str + row_divider + md_row_string + "\n-----\n"

    md_body_str += final_row_str

with open(readme_path, "w") as f:
    f.write(md_body_str)
