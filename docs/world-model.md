# 世界模型

## 潜状态动力学

`WorldModel.step(state, action)` 以衰减系数推进潜状态，动作注入确定性扰动。

## 下一状态预测

`NextStatePredictor` 带可训练残差（skill 门控），预测下一状态。

## 动作模型

`candidate_actions` + `select_action`：选择最接近目标状态的动作。

## 推演（rollout）

`rollout()` 沿动作序列推演多条轨迹。

## 情景记忆

`EpisodicMemory` 存储/检索历史场景嵌入，用于近邻检索。
