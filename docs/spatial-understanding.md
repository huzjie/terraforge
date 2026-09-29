# 空间理解

物理世界的符号底座，包含：

## 场景（scene）

`Scene` / `Object3D`：对象带 3D 位置、尺寸、类别；场景由确定性种子生成。

## 几何（geometry）

距离、相对位置、方位角、归一化、点积、叉积——纯 Python 实现。

## 深度 / 位姿 / 布局

- `depth.estimate_depth`：到相机原点的距离；
- `depth.depth_map`：粗粒度深度栅格；
- `pose.estimate_pose`：位置 + 朝向；
- `layout.predict_layout`：占用栅格。

## 场景图与推理

- `SceneGraph`：对象为节点，空间关系（left_of / right_of / above / below / in_front_of / behind / near）为边；
- `SpatialReasoner`：把自然语言空间问题解析为场景图查询。

## 确定性世界模型

`world_truth(scene_id, query)` 为任意场景+查询提供确定性真值，用于训练信号与基准评分。
