# 点云预处理与质量检查

`camera-check` 和 `AnyGraspBackend` 共用 `perception/preprocessing.py`。它只接收 RGB-D、目标 mask 和参数，不连接相机、机械臂，也不加载 AnyGrasp SDK。机械臂继续使用 `robots/actions.py` 中的 B601 动作。

## 模块边界

| 模块 | 负责内容 |
| --- | --- |
| `perception/pointcloud.py` | RGB-D 投影、统计离群点过滤、点与颜色及 mask 对齐 |
| `perception/spatial_filters.py` | 相机系工作范围、固定相机视角的夹爪遮罩 |
| `perception/cloud_quality.py` | 质量指标、阈值校验、通过或拒绝原因 |
| `perception/preprocessing.py` | 组合步骤、逐阶段统计；返回数据和报告 |
| `visualization/segmentation.py` | 检查图片、点云文件和窗口 |
| `grasping/backends/anygrasp.py` | 质量通过后调用 SDK，转换候选并复核工作范围 |

## 处理顺序

1. 按深度范围筛选，投影到相机光学坐标；X 向右、Y 向下、Z 向前，单位为米。
2. 用固定视角多边形和深度范围的交集排除可确认的自身夹爪点。
3. 用相机系三维包围盒约束目标可抓取点。**工作区域外的场景点仍保留给碰撞检测**，仅目标 `region_steering` 被限制。没有指定目标的全场景模式也采用工作区域 mask。
4. 运行统计离群点过滤，同步更新颜色和目标 mask。
5. 检查质量。未通过就返回原因，不调用 SDK 推理。目标或工作区域为空时不会退回无限制全场景。
6. 推理返回后，再检查统一指尖参考点是否位于相机工作范围；机器人基坐标范围与 IK 仍由原有规划流程检查。

工作区域是视觉处理范围，不等同于机械臂可达范围，也不是完整运动路径的碰撞检查。

## 当前现场参数

参数位于 `config/backends/anygrasp.yaml`：

- `workspace_camera`：XYZ 下界 `[-0.25, -0.20, 0.10]`，上界 `[0.25, 0.20, 0.60]`，单位米。
- `gripper_exclusion`：按现场 `camera-check-F9b4OCbE` 的固定 Gemini 336 视角配置保守的夹爪内部轮廓；只排除 0.05–0.19 米内的点。多边形坐标是归一化图像坐标。
- `outlier_filter`：20 个近邻，标准差倍率 2.0。
- `pointcloud_quality`：场景至少 1,000 点；目标至少 300 点；原始目标有效深度比例至少 60%；最终目标保留比例至少 50%；进入离群点步骤后，目标被该步骤剔除的比例至多 20%。

这些是可调整的工程初值，并非抓取成功率保证。夹爪遮罩是**保守的局部自身排除**，不覆盖所有手指位置，不是 CAD 自身碰撞模型。更换安装位置、相机裁剪/分辨率、夹爪形态后，应重新核对遮罩。分辨率不匹配会直接报错，禁止静默缩放；同分辨率下的安装变化仍需人工检查。遮罩深度范围外的物体点会保留。

三个可选过滤项各自设为 `null` 可关闭。`pointcloud_quality: null` 只关闭附加比例和数量阈值，仍拒绝空场景或空目标。不得把画面中未知区域随意扩入夹爪遮罩，以免移除真实物体或障碍物。

## 现场运行与文件

```bash
bash examples/camera_check/run.sh
```

仅采集、分割和检查点云，无需授权，不启动机械臂。`examples/camera_check/run.sh --input /absolute/path/frame.npz` 可回放一帧，不开启相机。

每次创建新的 `outputs/camera-check/run-*/result`，保存：

| 文件 | 内容 |
| --- | --- |
| `frame.npz` / `color.png` | 原始 RGB-D / 彩色图 |
| `segmentation.png` / `mask-*.png` | 原始检测实例，便于对照 |
| `preprocessing.png` | 绿色为过滤后目标；紫色为移除的自身点；灰色为工作区外或无有效深度；左上角为质量结果 |
| `gripper_exclusion.png` | 本帧实际命中夹爪轮廓及深度范围的像素 |
| `eligible_target.png` | 最终可用于目标引导的像素 |
| `pointcloud_raw.npz` / `pointcloud.npz` | 预处理前后点云；点、颜色、目标 mask 一一对应 |
| `inspection.json` | 每阶段数量、参数、质量阈值、指标和拒绝原因 |

检查窗口显示预处理结果，按 Q/Esc 后显示点云。退出码 0 表示通过质量阈值，2 表示检测/质量未通过，1 表示配置或运行错误。任何结果都不是 AnyGrasp 推理或真机抓取成功证明。

`inspection.json` 的 `preprocessing.counts` 区分夹爪移除、工作区排除与离群点剔除。顶层 `removed_selected_point_count` 是各步骤合计，不能全部理解为离群点数量。`valid_depth_ratio` 的分母是原始目标 mask 像素数；`retained_ratio` 的分母是原始目标有效深度点数；`outlier_ratio` 的分母是经过夹爪和工作区筛选、即将进入离群点过滤的目标点数。全场景模式下用全部图像像素作为原始区域。

实际推理保存的 `result.json` 包含同样的 `pointcloud_report`。质量失败状态为 `pointcloud_rejected`，后续不会生成或执行抓取计划。指定目标未检测到仍使用 `target_not_found`。

## 拒绝原因

| 代码 | 意义 |
| --- | --- |
| `target_not_found` | 未检测到指定目标 |
| `insufficient_scene_points` | 场景点数不足 |
| `insufficient_target_points` | 目标或可抓取区域点数不足 |
| `insufficient_valid_depth` | 原始目标有效深度比例不足 |
| `excessive_region_removal` | 过滤后目标保留比例不足 |
| `excessive_outliers` | 目标离群点比例过高 |

深度孔洞不自动补全。质量通过也可能存在局部边缘误差和遮挡，需要结合图像复核。
