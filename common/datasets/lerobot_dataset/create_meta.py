import json
import os
from tqdm import tqdm
from datasets import load_dataset
import cv2
import numpy as np
from task_config import TASKS_DICT

# from custom_scripts.common.constants import META_INFO_TEMPLATE, META_STATS_TEMPLATE

META_INFO_TEMPLATE = {
    "codebase_version": "v2.0",
    "robot_type": "piper",
    "total_episodes": 600,
    "total_frames": 35200,
    "total_tasks": 1,
    "total_videos": 600,
    "total_chunks": 12,
    "chunks_size": 50,
    "fps": 5,
    "splits": {
        "train": "0:600"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "gripper"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "gripper"
            ]
        },
        "observation.images.table": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 5.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 5.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": None
        }
    }
}
META_STATS_TEMPLATE = {
    "action": {
        "mean": [
            184750.00428977,
	        62906.23678977,
	        263574.93028409,
	        -56545.81463068,
            43992.02488636,
	        -80643.40144886,
	        60145.86931818
        ],
        "std": [
            102962.65069191,
	        113760.93033841,
	        92895.24751794,
	        148413.88694479,
	        29076.630872,
	        116143.05429282,
            13249.54133897
        ],
        "max": [
            448213,
            321698,
            574275,
            180000,
            90000,
            180000,
            72240
        ],
        "min": [
            -56501,
            -211263,
            0,
            -179969,
            -11012,
            -179968,
            -1470
        ]
    },
    "observation.state": {
        "mean": [
            184747.916875,
            62905.64642045,
            263577.69204545,
            -56504.12042614,
            43991.99284091,
            -80637.97610795,
            60145.88522727
        ],
        "std": [
            102962.28290167,
            113760.24414793,
            92895.45537357,
            148429.34473241,
            29076.48597257,
            116142.33207768,
            13249.43522619
        ],
        "max": [
            448213,
            321698,
            574275,
            180000,
            90000,
            180000,
            72240
        ],
        "min": [
            -56501,
            -211263,
            0,
            -179969,
            -11012,
            -179968,
            -1470
        ]
    },
    "timestamp": {
        "mean": [
            5.912466049194336
        ],
        "std": [
            3.6292903423309326
        ],
        "max": [
            16.799999237060547
        ],
        "min": [
            0.0
        ]
    },
    "frame_index": {
        "mean": [
            177.37397727272727
        ],
        "std": [
            108.87871008404305
        ],
        "max": [
            504
        ],
        "min": [
            0
        ]
    },
    "episode_index": {
        "mean": [
            288.24105113636364
        ],
        "std": [
            174.30801099730152
        ],
        "max": [
            599
        ],
        "min": [
            0
        ]
    },
    "task_index": {
        "mean": [
            2.0
        ],
        "std": [
            0.0
        ],
        "max": [
            2
        ],
        "min": [
            2
        ]
    },
    "index": {
        "mean": [
            17599.5
        ],
        "std": [
            10161.5090742796
        ],
        "max": [
            35199
        ],
        "min": [
            0
        ]
    },
    "observation.images.table": {
        "mean": [
            [
                [
                    91.68148143
                ]
            ],
            [
                [
                    95.98407115
                ]
            ],
            [
                [
                    99.49233975
                ]
            ]
        ],
        "std": [
            [
                [
                    42.90627931
                ]
            ],
            [
                [
                    46.07452878
                ]
            ],
            [
                [
                    47.25431926
                ]
            ]
        ],
        "max": [
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ]
        ],
        "min": [
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ]
        ]
    },
    "observation.images.wrist": {
        "mean": [
            [
                [
                    93.06738208
                ]
            ],
            [
                [
                    98.86193299
                ]
            ],
            [
                [
                    128.35764402
                ]
            ]
        ],
        "std": [
            [
                [
                    70.559473
                ]
            ],
            [
                [
                    75.98714874
                ]
            ],
            [
                [
                    68.42766891
                ]
            ]
        ],
        "max": [
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ],
            [
                [
                    255.0
                ]
            ]
        ],
        "min": [
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ],
            [
                [
                    0.0
                ]
            ]
        ]
    }
}

def create_meta(root_dir, episodes):
    ep = []
    tasks = []
    info = META_INFO_TEMPLATE
    stats = META_STATS_TEMPLATE

    print("dataset_root_dir: ", root_dir)

    tasks_dict = TASKS_DICT
    #     {
    #     '0' : "press the red button",
    #     '1' : "press the green button",
    #     '2' : "press the blue button",
    #     # '1' : "press the blue button two times",
    #     # '2' : "press the blue button three times",
    #     # '3' : "press the blue button four times",
    #     # '4' : "press the red, green, blue buttons in order",
    #     '5' : "wipe the window with towel",
    # }

    total_frames = 0

    action_df = []
    action_joint_df = []
    observation_state_df = []
    observation_state_joint_df = []
    timestamp_df = []
    frame_index_df = []
    episode_index_df = []
    task_index_df = []
    # index_df = []
    observation_images_exo_df = []
    observation_images_wrist_df = []
    observation_images_table_df = []

    # fetch data of each episode
    for i in tqdm(range(episodes)):
        parquet_file = os.path.join(root_dir, f"data/chunk-{i//50:03d}/episode_{i:06d}.parquet")
        exo_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.exo/episode_{i:06d}.mp4")
        wrist_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.wrist/episode_{i:06d}.mp4")
        table_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.table/episode_{i:06d}.mp4")

        parquet_data = load_dataset("parquet", data_files=parquet_file)['train']
        ep.append({
            "episode_index": i,
            "tasks": tasks_dict[str(parquet_data[0]['task_index'])],
            "length": len(parquet_data),
        })

        total_frames += len(parquet_data)
        exo_cap = cv2.VideoCapture(exo_file)
        wrist_cap = cv2.VideoCapture(wrist_file)
        table_cap = cv2.VideoCapture(table_file)

        for frame_idx in tqdm(range(len(parquet_data))):
            frame_data = parquet_data[frame_idx]

            action_df.append(frame_data['action'])
            #action_joint_df.append(frame_data['action_joint'])
            observation_state_df.append(frame_data['observation.state'])
            #observation_state_joint_df.append(frame_data['observation.state_joint'])
            timestamp_df.append(frame_data['timestamp'])
            frame_index_df.append(frame_data['frame_index'])
            episode_index_df.append(frame_data['episode_index'])
            task_index_df.append(frame_data['task_index'])
            # index_df.append(frame_data['index'])

            _, exo_frame = exo_cap.read()
            _, wrist_frame = wrist_cap.read()
            _, table_frame = table_cap.read()

            observation_images_exo_df.append(exo_frame)
            observation_images_wrist_df.append(wrist_frame)
            observation_images_table_df.append(table_frame)

    print("made episodes... saving...")

    # save meta/episodes.jsonl
    with open(os.path.join(root_dir, 'meta/episodes.jsonl'), "w") as f:
        for entry in ep:
            json.dump(entry, f)
            f.write("\n")

    # save meta/info.json
    info['total_episodes'] = episodes
    info['total_frames'] = total_frames
    info['total_tasks'] = len(tasks_dict)
    info['total_videos'] = episodes
    info['total_chunks'] = (episodes // info['chunks_size'])+1
    info['splits']['train'] = f"0:{episodes}"

    print("total_frames: ", total_frames)
    print("info made...")

    with open(os.path.join(root_dir, 'meta/info.json'), "w") as f:
        json.dump(info, f, indent=4)

    # save meta/tasks.jsonl
    for k,v in tasks_dict.items():
        tasks.append({
            "task_index": int(k),
            "task": v
        })

    print("task made... saving...")

    with open(os.path.join(root_dir, 'meta/tasks.jsonl'), "w") as f:
        for entry in tasks:
            json.dump(entry, f)
            f.write("\n")

    stacked_action = np.stack(action_df, axis=0)
    #stacked_action_joint = np.stack(action_joint_df, axis=0)
    stacked_observation_state = np.stack(observation_state_df, axis=0)
    #stacked_observation_state_joint = np.stack(observation_state_joint_df, axis=0)
    stacked_timestamp = np.stack(timestamp_df, axis=0)
    stacked_frame_index = np.stack(frame_index_df, axis=0)
    stacked_episode_index = np.stack(episode_index_df, axis=0)
    stacked_task_index = np.stack(task_index_df, axis=0)
    # stacked_index = np.stack(index_df, axis=0)

    stacked_exo = np.stack(observation_images_exo_df, axis=0)
    stacked_wrist = np.stack(observation_images_wrist_df, axis=0)
    stacked_table = np.stack(observation_images_table_df, axis=0)

    print("stacked data... starting to calculate")

    stats["action"]={
        "mean":stacked_action.mean(axis=0).tolist(),
        "std":stacked_action.std(axis=0).tolist(),
        "max":stacked_action.max(axis=0).tolist(),
        "min":stacked_action.min(axis=0).tolist(),
    }
    # stats["action_joint"]={
    #     "mean":stacked_action_joint.mean(axis=0).tolist(),
    #     "std":stacked_action_joint.std(axis=0).tolist(),
    #     "max":stacked_action_joint.max(axis=0).tolist(),
    #     "min":stacked_action_joint.min(axis=0).tolist(),
    # }
    stats["observation.state"]={
        "mean":stacked_observation_state.mean(axis=0).tolist(),
        "std":stacked_observation_state.std(axis=0).tolist(),
        "max":stacked_observation_state.max(axis=0).tolist(),
        "min":stacked_observation_state.min(axis=0).tolist(),
    }
    # stats["observation.state_joint"]={
    #     "mean":stacked_observation_state_joint.mean(axis=0).tolist(),
    #     "std":stacked_observation_state_joint.std(axis=0).tolist(),
    #     "max":stacked_observation_state_joint.max(axis=0).tolist(),
    #     "min":stacked_observation_state_joint.min(axis=0).tolist(),
    # }
    stats["timestamp"]={
        "mean":stacked_timestamp.mean(axis=0).reshape(1).tolist(),
        "std":stacked_timestamp.std(axis=0).reshape(1).tolist(),
        "max":stacked_timestamp.max(axis=0).reshape(1).tolist(),
        "min":stacked_timestamp.min(axis=0).reshape(1).tolist(),
    }
    stats["frame_index"]={
        "mean":stacked_frame_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_frame_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_frame_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_frame_index.min(axis=0).reshape(1).tolist(),
    }
    stats["episode_index"]={
        "mean":stacked_episode_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_episode_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_episode_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_episode_index.min(axis=0).reshape(1).tolist(),
    }
    stats["task_index"]={
        "mean":stacked_task_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_task_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_task_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_task_index.min(axis=0).reshape(1).tolist(),
    }
    # stats["index"]={
    #     "mean":stacked_index.mean(axis=0).reshape(1).tolist(),
    #     "std":stacked_index.std(axis=0).reshape(1).tolist(),
    #     "max":stacked_index.max(axis=0).reshape(1).tolist(),
    #     "min":stacked_index.min(axis=0).reshape(1).tolist(),
    # }
    stats["observation.images.exo"]={
        "mean":stacked_exo.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_exo.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_exo.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_exo.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    stats["observation.images.wrist"]={
        "mean":stacked_wrist.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_wrist.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_wrist.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_wrist.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    stats["observation.images.table"]={
        "mean":stacked_table.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_table.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_table.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_table.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }

    with open(os.path.join(root_dir,"meta/stats.json"),"w") as f:
        json.dump(stats,f,indent=4)

    print("created all meta files")


if __name__ == "__main__":
    print("creating meta dataset")
    #root_dir = "/data/ghkim/press the red button/lerobot_5hz"
    #episodes = 20
    #create_meta(root_dir, episodes)

    root_dir = "/data/ghkim/press_the_RGB_button_ep150"
    episodes = 150
    create_meta(root_dir, episodes)

    # root_dir = "/data/ghkim/press the green button/lerobot_5hz"
    # episodes = 20
    # create_meta(root_dir, episodes)
    #
    # root_dir = "/data/ghkim/press the red, green, blue button in order/lerobot_5hz"
    # episodes = 20
    # create_meta(root_dir, episodes)

    # root_dir = "/data/piper_grape0626_75"
    # episodes = 450
    # create_meta(root_dir, episodes)