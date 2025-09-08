# 点足机器人爬楼梯强化学习训练指南

## 项目修改总结

基于现有的 PointFoot Legged Gym 框架，我已经成功添加了专门的爬楼梯训练环境。以下是所有修改和新增的内容：

### 1. 新增文件

#### `legged_gym/envs/pointfoot_stairs/`
- `pointfoot_stairs.py` - 爬楼梯环境类
- `pointfoot_stairs_config.py` - 爬楼梯配置类  
- `__init__.py` - 包初始化文件

#### 文档
- `STAIRS_TRAINING_GUIDE.md` - 使用指南（本文件）

### 2. 修改文件

#### `legged_gym/envs/__init__.py`
- 添加了爬楼梯任务的注册
- 现在支持 `pointfoot_stairs` 任务

## 核心功能特性

### 🏗️ 地形配置优化
- **专注楼梯地形**: 70%楼梯 + 30%离散障碍
- **渐进式课程**: 从简单楼梯开始训练
- **精确高度测量**: 启用地形高度感知

### 🎯 爬楼梯特定奖励
1. **高度进展奖励** (`height_progress`): 鼓励向上爬行
2. **楼梯接触奖励** (`stair_contact`): 奖励正确踩踏楼梯
3. **平衡奖励** (`balance_on_stairs`): 在楼梯上保持稳定
4. **前进奖励** (`forward_progress`): 鼓励持续前进
5. **能效奖励** (`energy_efficiency`): 优化动作效率

### 📊 扩展观察空间
- **高度信息**: 当前高度相对于起点的进展
- **接触状态**: 左右脚与楼梯的接触情况
- **楼梯检测**: 是否在楼梯上的状态信息
- **前进距离**: 水平方向的移动进展

### ⚙️ 优化参数
- **降低移动速度**: 适应楼梯环境的安全速度
- **延长训练时长**: 30秒episode以完成爬楼梯
- **增加观察维度**: 35维观察空间（原30维+5维楼梯信息）

## 使用方法

### 1. 环境设置
```bash
# 设置机器人类型（支持所有PF系列）
export ROBOT_TYPE=PF_TRON1A
```

### 2. 开始训练
```bash
# 爬楼梯训练
python legged_gym/scripts/train.py --task=pointfoot_stairs --headless

# 可选参数
python legged_gym/scripts/train.py \
    --task=pointfoot_stairs \
    --headless \
    --num_envs=4096 \
    --max_iterations=2000
```

### 3. 测试已训练模型
```bash
python legged_gym/scripts/play.py \
    --task=pointfoot_stairs \
    --load_run=<your_model_folder> \
    --checkpoint=<checkpoint_number>
```

## 配置说明

### 主要配置参数

#### 地形设置
```python
terrain_proportions = [0.0, 0.0, 0.7, 0.0, 0.3]  # 70%楼梯
mesh_type = "trimesh"  # 支持复杂地形
measure_heights = True  # 启用高度测量
```

#### 奖励权重
```python
class scales:
    height_progress = 2.0      # 高度进展
    stair_contact = 0.5        # 楼梯接触  
    balance_on_stairs = 1.0    # 平衡控制
    forward_progress = 1.5     # 前进奖励
    energy_efficiency = -0.001 # 能效
```

#### 速度限制
```python
smooth_max_lin_vel_x = 1.5  # 降低前进速度
max_ang_vel_yaw = 2.0       # 降低转向速度
```

## 训练建议

### 1. 课程学习策略
- **阶段1**: 简单楼梯 (0-500 iterations)
- **阶段2**: 中等复杂度 (500-1000 iterations)  
- **阶段3**: 复杂楼梯环境 (1000+ iterations)

### 2. 超参数调优
```python
# 在 BipedCfgPPOStairs 中调整
learning_rate = 1.0e-3     # 学习率
num_learning_epochs = 5    # 每次更新的epoch数
num_mini_batches = 4       # 批次大小
```

### 3. 监控指标
- **高度进展**: 平均高度增益
- **楼梯接触率**: 足部正确接触比例
- **成功完成率**: 成功爬完楼梯的episode比例
- **能效比**: 动作平滑度和能耗

## 故障排除

### 常见问题

1. **训练不收敛**
   - 降低学习率到 `5e-4`
   - 增加 `num_learning_epochs` 到 8
   - 检查奖励权重平衡

2. **机器人摔倒频繁**
   - 增加 `balance_on_stairs` 奖励权重
   - 降低速度命令范围
   - 调整PD控制参数

3. **不爬楼梯**
   - 增加 `height_progress` 奖励权重
   - 检查地形生成是否正确
   - 确认楼梯检测逻辑

### 调试命令
```bash
# 可视化训练过程
python legged_gym/scripts/train.py --task=pointfoot_stairs --num_envs=64

# 检查环境配置
python -c "from legged_gym.envs import *; print('环境注册成功')"
```

## 扩展开发

### 添加新奖励函数
在 `pointfoot_stairs.py` 中添加：
```python
def _reward_your_custom_reward(self):
    # 自定义奖励逻辑
    return reward_tensor
```

在配置中注册：
```python
class scales:
    your_custom_reward = 1.0
```

### 修改楼梯难度
在 `pointfoot_stairs_config.py` 中调整：
```python
terrain_proportions = [0.0, 0.0, 0.9, 0.0, 0.1]  # 更多楼梯
max_init_terrain_level = 3  # 更高起始难度
```

## 实验结果预期

经过充分训练，点足机器人应该能够：
- ✅ 稳定爬上不同高度的楼梯
- ✅ 在楼梯上保持平衡
- ✅ 适应不同楼梯间距和高度
- ✅ 高效完成爬楼梯任务

训练通常需要 1000-1500 iterations 达到良好性能。

---

**注意**: 首次运行前请确保已正确安装 Isaac Gym 和所有依赖项。
