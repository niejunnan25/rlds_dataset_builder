import numpy as np

def quaternion_to_euler(q, degrees=False):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）。

    参数:
        q: array-like, 形状为 (4,)，四元数 [w, x, y, z]
        degrees: bool, 是否将输出转换为角度（True）或保留弧度（False）

    返回:
        tuple: (roll, pitch, yaw)，欧拉角，单位为弧度或角度
    """
    # 确保输入是 numpy 数组并归一化
    q = np.array(q, dtype=np.float64)
    q = q / np.linalg.norm(q)  # 归一化四元数

    # 提取 w, x, y, z
    w, x, y, z = q

    # 计算欧拉角（ZYX 顺序，即 yaw-pitch-roll）
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2.0  # 处理万向节锁
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

    return roll, pitch, yaw

def convert_ee_pos_quat_to_euler(data, degrees=False):
    """
    将 data["ee_pos_quat"] 的四元数转换为欧拉角，生成新的 NumPy 数组。

    参数:
        degrees: bool, 是否将欧拉角转换为角度（True）或保留弧度（False）

    返回:
        np.ndarray: 形状 (6,)，[x, y, z, roll, pitch, yaw]
    """
    # 在这里我们假设传入的 data 就已经是 numpy 数组了
    ee_pos_quat = data
    
    # 确保输入形状正确
    if ee_pos_quat.shape != (7,):
        raise ValueError("ee_pos_quat 数组的形状应为 (7,)，包含 [x, y, z, w, x_q, y_q, z_q]")

    # 提取位置 (x, y, z) 和四元数 (w, x_q, y_q, z_q)
    position = ee_pos_quat[:3]  # [x, y, z]
    quaternion = ee_pos_quat[3:]  # [w, x_q, y_q, z_q]

    # 转换为欧拉角
    roll, pitch, yaw = quaternion_to_euler(quaternion, degrees=degrees)

    result = np.array([position[0], position[1], position[2], roll, pitch, yaw], dtype=np.float64)

    return result

# 示例用法
if __name__ == "__main__":
    data = {
        "ee_pos_quat": np.array([0.5, 1.0, 1.5, 0.7071, 0.0, 0.7071, 0.0])  # [x, y, z, w, x_q, y_q, z_q]
    }
    
    # 转换为欧拉角（弧度）
    result = convert_ee_pos_quat_to_euler(data, degrees=False)
    print("Result (radians):", result)

    # 转换为欧拉角（角度）
    result_deg = convert_ee_pos_quat_to_euler(data, degrees=True)
    print("Result (degrees):", result_deg)