# from typing import Iterator, Tuple, Any

# import os
# import glob
# import time
# import pickle
# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from PIL import Image

# from utils import convert_ee_pos_quat_to_euler

# LANGUAGE_INSTRUCTION = 'Do something'

# IMAGE_RES = (720, 1280)

# ########################################################################################################################

# # Notice:
# #        - 这个版本的实现依赖于我自己的理解， action_dict 字段是控制器的值
# #        - 这个图片的要求格式是 .jpeg, 要在 run_env_copy.py 中修改 .png -> .jpeg, cv2 的库会自动解析生成 .jpeg
# #           - 上面这一点, 0724-14:43 测试过了, 及时是丢 .png 图像进去也可以一样处理
# #        - TODO: 测试一下是不是一定需要 .jepg --------- 已经做完了, .png 也可以
# #        - TODO: 测试一下文件名的问题
# #        - TODO: 验证一下是否满足 RLDS 格式的要求
# #        - TODO: 尝试用测试的数据集进行 fine-tune
# #        - TODO: 采数据，

# ########################################################################################################################


# """
# 在进行数据转换之前，需要明确几件事情：

# 在观测 observation 字段下的所有字段其实是 robot 看到的“状态”，包含图像 + 状态量：
# 1. exterior_image_1_left	左侧外部相机图像1（jpeg）	RGB图像，大小为 IMAGE_RES，格式为 jpeg
# 2. exterior_image_2_left	左侧外部相机图像2（jpeg）	同上
# 3. wrist_image_left	机器人手腕相机的 RGB 图像（jpeg）	RGB图像，大小为 IMAGE_RES，格式为 jpeg
# 4. cartesian_position	机器人末端执行器的笛卡尔位置（位置+姿态）	6维向量：3维位置 + 3维姿态（如 XYZ + RollPitchYaw）
# 5. gripper_position	夹爪的开合位置	1维浮点数，表示张开程度
# 6. joint_position	机器人的关节角度	7维向量，对应机器人每个关节的角度或位置

# 在动作字典 action_dict 字段下的所有字段其实是机器人的控制指令，发给机器人的控制信号

# ** action_dict 记录的是发送给机械臂的命令，而不是遥操作设备本身的物理状态（例如手柄的 6D 位置或按键状态）**
# ** 遥操作设备的物理状态可能需要额外的处理（例如映射到机械臂的控制空间）才能成为 action_dict 的内容。**
# action_dict 仍然是处理后的控制命令，而不是遥操作设备的原始状态
# --- 遥操作设备与被控设备即便同构，只要不完全等长等位姿，就必须进行 FK → 空间变换 → IK 的流程，来保证控制语义的一致性。

# 1. cartesian_position	控制末端执行器位置	6维向量，末端的期望位置和姿态
# 2. cartesian_velocity	控制末端的速度	6维向量，笛卡尔空间的线速度 + 角速度
# 3. gripper_position	控制夹爪的位置	1维值，表示目标张开程度
# 4. gripper_velocity	控制夹爪的速度	1维值，张合速度
# 5. joint_position	控制各关节的角度	7维向量，对应目标关节角度
# 6. joint_velocity	控制各关节的速度	7维向量，关节速度

# action：精简动作（用于 RL 学习）
# [6x joint velocities, 1x gripper position]	机器人动作向量, 可以根据 action_dict 提取出来
# """

# class ExampleDataset(tfds.core.GeneratorBasedBuilder):
#     """DatasetBuilder for example dataset."""

#     VERSION = tfds.core.Version('1.0.0')
#     RELEASE_NOTES = {
#       '1.0.0': 'Initial release.',
#     }

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     # 定义数据集的信息结构，包括样本字段、类型、形状、说明等
#     def _info(self) -> tfds.core.DatasetInfo:

#         """Dataset metadata (homepage, citation,...)."""

#         return self.dataset_info_from_configs(
#             features=tfds.features.FeaturesDict({
#             'steps': tfds.features.Dataset({
#                     'observation': tfds.features.FeaturesDict({
#                         'exterior_image_1_left': tfds.features.Image(
#                             shape=(*IMAGE_RES, 3),
#                             dtype=np.uint8,
#                             encoding_format='jpeg',
#                             doc='Exterior camera 1 left viewpoint',
#                         ),
#                         'exterior_image_2_left': tfds.features.Image(
#                             shape=(*IMAGE_RES, 3),
#                             dtype=np.uint8,
#                             encoding_format='jpeg',
#                             doc='Exterior camera 2 left viewpoint'
#                         ),
#                         'wrist_image_left': tfds.features.Image(
#                             shape=(*IMAGE_RES, 3),
#                             dtype=np.uint8,
#                             encoding_format='jpeg',
#                             doc='Wrist camera RGB left viewpoint',
#                         ),
#                         'cartesian_position': tfds.features.Tensor(
#                             shape=(6,),
#                             dtype=np.float64,
#                             doc='Robot Cartesian state',
#                         ),
#                         'gripper_position': tfds.features.Tensor(
#                             shape=(1,),
#                             dtype=np.float64,
#                             doc='Gripper position state',
#                         ),
#                         'joint_position': tfds.features.Tensor(
#                             shape=(7,),
#                             dtype=np.float64,
#                             doc='Joint position state'
#                         )
#                     }),
#                     'action_dict': tfds.features.FeaturesDict({
#                         'cartesian_position': tfds.features.Tensor(
#                             shape=(6,),
#                             dtype=np.float64,
#                             doc='Commanded Cartesian position'
#                         ),
#                         'cartesian_velocity': tfds.features.Tensor(
#                             shape=(6,),
#                             dtype=np.float64,
#                             doc='Commanded Cartesian velocity'
#                         ),
#                         'gripper_position': tfds.features.Tensor(
#                             shape=(1,),
#                             dtype=np.float64,
#                             doc='Commanded gripper position'
#                         ),
#                         'gripper_velocity': tfds.features.Tensor(
#                             shape=(1,),
#                             dtype=np.float64,
#                             doc='Commanded gripper velocity'
#                         ),
#                         'joint_position': tfds.features.Tensor(
#                             shape=(7,),
#                             dtype=np.float64,
#                             doc='Commanded joint position'
#                         ),
#                         'joint_velocity': tfds.features.Tensor(
#                             shape=(7,),
#                             dtype=np.float64,
#                             doc='Commanded joint velocity'
#                         )
#                     }),
#                     'action': tfds.features.Tensor(
#                         shape=(7,),
#                         dtype=np.float64,
#                         doc='Robot action, consists of [6x joint velocities, \
#                             1x gripper position].',
#                     ),
#                     'discount': tfds.features.Scalar(
#                         dtype=np.float32,
#                         doc='Discount if provided, default to 1.'
#                     ),
#                     'reward': tfds.features.Scalar(
#                         dtype=np.float32,
#                         doc='Reward if provided, 1 on final step for demos.'
#                     ),
#                     'is_first': tfds.features.Scalar(
#                         dtype=np.bool_,
#                         doc='True on first step of the episode.'
#                     ),
#                     'is_last': tfds.features.Scalar(
#                         dtype=np.bool_,
#                         doc='True on last step of the episode.'
#                     ),
#                     'is_terminal': tfds.features.Scalar(
#                         dtype=np.bool_,
#                         doc='True on last step of the episode if it is a terminal step, True for demos.'
#                     ),
#                     'language_instruction': tfds.features.Text(
#                         doc='Language Instruction.'
#                     ),
#                 }),
#                 'episode_metadata': tfds.features.FeaturesDict({
#                     'file_path': tfds.features.Text(
#                         doc='Path to the original data file.'
#                     ),
#                     'recording_folderpath': tfds.features.Text(
#                         doc='Path to the folder of recordings.'
#                     )
#                 }),
#             }))

#     # 定义数据集的划分，如训练集与验证集
#     def _split_generators(self, dl_manager: tfds.download.DownloadManager):
#         return {
#             'train': self._generate_examples('data/train'),
#             'val': self._generate_examples('data/val'),
#         }


#     def _generate_examples(self, base_path) -> Iterator[Tuple[str, Any]]:


#         def _load_image(image_path):

#             with Image.open(image_path) as img:

#                 # 更安全
#                 img = img.convert('RGB')
#                 return np.array(img)

#         def _parse_episode(episode_dir):

#             # episode_dir = 'data/train/episode1'
#             pkl_files = sorted(glob.glob(os.path.join(episode_dir, '*.pkl')))
#             image_dir = os.path.join(episode_dir, 'image')
#             wrist_image_dir = os.path.join(episode_dir, 'wrist_image')

#             episode = []
#             for i, pkl_file in enumerate(pkl_files):

#                 basename = os.path.basename(pkl_file)
#                 timestamp = os.path.splitext(basename)[0]

#                 with open(pkl_file, 'rb') as f:
#                     data = pickle.load(f)

#                 image_path = os.path.join(image_dir, f'left_{timestamp}.png')
#                 wrist_image_path = os.path.join(wrist_image_dir, f'wrist_left_{timestamp}.png')

#                 image = _load_image(image_path)
#                 wrist_image = _load_image(wrist_image_path)

#                 height, width, channels = image.shape

#                 fake_image = np.zeros((height, width, channels), dtype=image.dtype)

#                 fake_array_1 = np.zeros((1,), dtype=np.float64)
#                 fake_array_6 = np.zeros((6,), dtype=np.float64)
#                 fake_array_7 = np.zeros((7,), dtype=np.float64)

#                 # TODO: 疑问？为什么要在这里转 BGR?
#                 # [..., ::-1]
#                 episode.append({
#                     'observation': {
#                         'exterior_image_1_left': image[..., ::-1],  # 转BGR
#                         'exterior_image_2_left': fake_image[..., ::-1],
#                         'wrist_image_left': wrist_image[..., ::-1],
#                         'cartesian_position': convert_ee_pos_quat_to_euler(data=data["ee_pos_quat"], degrees=False),
#                         'joint_position': data["joint_positions"][:7], 
#                         'gripper_position': data["gripper_position"],
#                     },
#                     'action_dict': {
#                         'cartesian_position': fake_array_6,
#                         'cartesian_velocity': fake_array_6,
#                         # 假定 data["control"].shape == (8,)
#                         'gripper_position': np.array([data["control"][-1]], dtype=np.float64),
#                         'gripper_velocity': fake_array_1,
#                         'joint_position': data["control"][:7],
#                         'joint_velocity': fake_array_7,
#                     },
#                     # 控制器，怎么控制
#                     'action': np.concatenate([
#                         data["control"][:6],
#                         np.array([data["control"][-1]], dtype=np.float64)
#                     ]),# next_step_ee_pos_quat + next_gripper_position, next_joint_velocities[:6] + next_gripper_position
#                     'discount': 1.0,
#                     'reward': float(i == (len(data) - 1)),
#                     'is_first': i == 0,
#                     'is_last': i == (len(data) - 1),
#                     'is_terminal': i == (len(data) - 1),
#                     'language_instruction': LANGUAGE_INSTRUCTION,
#                 })

#             sample = {
#                 'steps': episode,
#                 'episode_metadata': {
#                     'file_path': episode_dir,
#                     'recording_folderpath': "test"
#                 }
#             }

#             return episode_dir, sample
        
#         start_time = time.time()

#         episode_dirs = [os.path.join(base_path, episode_name) for episode_name in os.listdir(base_path)]

#         for episode_dir in episode_dirs:
#             yield _parse_episode(episode_dir)
        
#         end_time = time.time()

#         print(f"========================== {end_time - start_time} 秒 =================================")

#         # （可选）大数据量可使用 Apache Beam 实现并行加载
#         # 实际测试过, 没用
#         # beam = tfds.core.lazy_imports.apache_beam
#         # return (
#         #         beam.Create(episode_paths)
#         #         | beam.Map(_parse_example)
#         # )
