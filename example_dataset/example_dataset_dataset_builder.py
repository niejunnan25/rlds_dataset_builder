from typing import Iterator, Tuple, Any

import os
import glob
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from utils import convert_ee_pos_quat_to_euler

IMAGE_RES = (1080, 1920)

class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 定义数据集的信息结构，包括样本字段、类型、形状、说明等
    def _info(self) -> tfds.core.DatasetInfo:

        """Dataset metadata (homepage, citation,...)."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'exterior_image_1_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 1 left viewpoint',
                        ),
                        'exterior_image_2_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 2 left viewpoint'
                        ),
                        'wrist_image_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB left viewpoint',
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Gripper position state',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Joint position state'
                        )
                    }),
                    'action_dict': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian position'
                        ),
                        'cartesian_velocity': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian velocity'
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper position'
                        ),
                        'gripper_velocity': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper velocity'
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint position'
                        ),
                        'joint_velocity': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint velocity'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [6x joint velocities, \
                            1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'recording_folderpath': tfds.features.Text(
                        doc='Path to the folder of recordings.'
                    )
                }),
            }))

    # 定义数据集的划分，如训练集与验证集
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples('data/train'),
            'val': self._generate_examples('data/val'),
        }


    def _generate_examples(self, base_path) -> Iterator[Tuple[str, Any]]:


        def _load_image(image_path):

            with Image.open(image_path) as img:

                # 更安全
                img = img.convert('RGB')
                return np.array(img)

        def _parse_episode(episode_dir, prompt : str):

            # episode_dir = 'data/train/prompt/episode_xxxxxxx'
            pkl_files = sorted(glob.glob(os.path.join(episode_dir, '*.pkl')))
            image_dir = os.path.join(episode_dir, 'image')
            wrist_image_dir = os.path.join(episode_dir, 'wrist_image')

            episode = []
            for index, pkl_file in enumerate(pkl_files):

                basename = os.path.basename(pkl_file)
                timestamp = os.path.splitext(basename)[0]

                with open(pkl_file, 'rb') as f:
                    cur_data = pickle.load(f)

                if index < len(pkl_files) - 1:
                    with open(pkl_files[index + 1], "rb") as f:
                        next_data = pickle.load(f)
                else:
                    with open(pkl_file, 'rb') as f:
                        next_data = pickle.load(f)

                image_path = os.path.join(image_dir, f'left_{timestamp}.png')
                wrist_image_path = os.path.join(wrist_image_dir, f'wrist_left_{timestamp}.png')

                image = _load_image(image_path)
                wrist_image = _load_image(wrist_image_path)

                height, width, channels = image.shape

                fake_image = np.zeros((height, width, channels), dtype=image.dtype)

                fake_array_1 = np.zeros((1,), dtype=np.float64)
                fake_array_6 = np.zeros((6,), dtype=np.float64)
                fake_array_7 = np.zeros((7,), dtype=np.float64)

                # TODO: 疑问？为什么要在这里转 BGR?
                # [..., ::-1]
                episode.append({
                    'observation': {
                        'exterior_image_1_left': image[..., ::-1],  # 转BGR
                        'exterior_image_2_left': fake_image[..., ::-1],
                        'wrist_image_left': wrist_image[..., ::-1],
                        'cartesian_position': convert_ee_pos_quat_to_euler(data=cur_data["ee_pos_quat"], degrees=False),
                        'joint_position': cur_data["joint_positions"][:7], 
                        'gripper_position': cur_data["gripper_position"],
                    },
                    'action_dict': {
                        'cartesian_position': convert_ee_pos_quat_to_euler(data=next_data["ee_pos_quat"], degrees=False),
                        'cartesian_velocity': fake_array_6,
                        'gripper_position': np.array([cur_data["control"][-1]], dtype=np.float64),
                        'gripper_velocity': fake_array_1,
                        'joint_position': cur_data["control"][:7],
                        'joint_velocity': next_data["joint_velocities"],
                    },
                    # 控制器，怎么控制
                    'action': np.concatenate([
                        cur_data["control"][:6],
                        np.array([cur_data["control"][-1]], dtype=np.float64)
                    ]),# next_step_ee_pos_quat + next_gripper_position, next_joint_velocities[:6] + next_gripper_position
                    'discount': 1.0,
                    'reward': float(i == (len(cur_data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(cur_data) - 1),
                    'is_terminal': i == (len(cur_data) - 1),
                    'language_instruction': prompt,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_dir,
                    'recording_folderpath': "test"
                }
            }

            return episode_dir, sample
        
        start_time = time.time()

        # 假设数据路径在: "data/train/this_is_your_prompt/episode_xxxxxx"
        # base_path = "data/train"
        prompt_list : list[str] = [prompt.replace('_', ' ').rstrip() + '.' for prompt in os.listdir(base_path)]
        
        for prompt_str in os.listdir(base_path):
            episode_dirs = [os.path.join(base_path, prompt_str, episode_name) for episode_name in os.listdir(os.path.join(base_path, prompt_str))]

        for i, episode_dir in enumerate(episode_dirs):
            yield _parse_episode(episode_dir, prompt_list[i])
        
        end_time = time.time()

        print(f"========================== {end_time - start_time} 秒 =================================")

        # （可选）大数据量可使用 Apache Beam 实现并行加载
        # 实际测试过, 没用
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
