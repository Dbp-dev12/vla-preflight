# VLA Preflight Workbench

**把数据检查、可复现训练、机械臂上机准备和真实 rollout 证据连起来的本地 VLA 工作台。**

[English](README.md) · [完整工作流程](docs/workflows.md) · [验证记录](docs/validation.md)

v0.6.0 alpha 同时覆盖两类用户：没有机械臂时可用合成数据跑通实验；拥有 CUDA GPU 和机械臂时，可检查计算环境、生成带安全门禁的 LeRobot 上机方案，并统计真实或仿真 rollout 的成功率、不确定区间、人工干预和失败模式。工具不会静默连接或驱动机械臂。

## 这次能做什么

- **可视化检查**：轨迹列表、逐帧图像、动作曲线、预检发现、数值重复组与探索性时间偏移曲线。
- **数据准备**：按完整轨迹划分训练/验证集，重复数值轨迹保持同组，归一化只拟合训练集，记录文件指纹。
- **实际训练**：PyTorch 执行图像＋文本＋状态模型的梯度更新，支持动作块与 padding mask，保存真实采样 trace。
- **评估与续训**：验证集原始动作单位 MAE/RMSE、常数动作基线、训练曲线、最新/最佳检查点、断点续训、实验比较。
- **筛选导出**：明确排除轨迹，另建数据集，重排索引并重算数值统计量，原始文件保持不变。
- **大模型接入入口**：生成实验性的 SmolVLA 外部微调计划，物理导出训练集。
- **GPU 环境诊断**：报告 CUDA、显存、可选依赖和 LeRobot 命令可用性，不记录主机名、用户名和绝对路径。
- **机械臂上机门禁**：校验机械臂、遥操作器、相机、标定文件和安全声明，生成可审阅的遥操作与采集命令，但不执行。
- **真实 rollout 评估**：按任务和检查点统计成功率、Wilson 95% 区间、人工干预和失败模式，不用训练 loss 冒充成功率。

Tiny VLA 是随机初始化的小型参考模型，**不是预训练大模型**。LeRobot 环境契约与产物回收已有回归测试，但本版本不宣称完成了真实 SmolVLA/CUDA 训练。项目价值在于可检查、可复现的流程，不替代 LeRobot。

## 安装与启动

创建并激活 Python 3.10–3.13 虚拟环境，然后运行：

```shell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train]"
vla-preflight learning-demo demo-output/learning
vla-preflight studio demo-output/learning --workspace workbench-output/learning --open
```

纯数据预检只需 `pip install -e .`。图像需要 Pillow，视频帧读取需要 PyAV，均在 train 可选依赖中。

启动后依次进入：**轨迹检查 → 数据准备 → 用于训练 → 开始训练**。默认演示使用 CPU。24 条合成轨迹很少，可能过拟合；工具会保留实际结果并提示验证误差变差或未超过基线，不会把损失下降当成机械臂成功率。Windows 用户完成安装后也可以运行 `Run-Workbench.cmd`。

## 命令行训练

```shell
vla-preflight prepare demo-output/learning --output runs/prepared
vla-preflight train runs/prepared --output runs/first --steps 200
vla-preflight train runs/prepared --output runs/resumed --steps 400 --resume runs/first/checkpoint.pt
vla-preflight compare runs/first runs/resumed --output reports/comparison.json
```

续训的 400 是总步数，需使用新的输出目录。默认每 20 步评估保存。`checkpoint.pt` 是最新状态；`best.pt` 是本次运行已评估候选的最佳状态，包含初始模型。最佳模型依赖验证集选择，验证分数不是独立测试成绩。

源文件或划分/统计文件变化会阻止训练继续，需重新准备。输出不可放入源数据目录。

## 有 GPU 和机械臂时

```shell
vla-preflight doctor --output reports/doctor.json
python -m pip install -e ".[robot]"
vla-preflight robot-check robot.json --output reports/robot-check.json
vla-preflight robot-plan robot.json --preflight reports/robot-check.json --output runs/robot-plan
vla-preflight rollout-import rollouts.jsonl --protocol evaluation-protocol.json --robot-plan runs/robot-plan/plan.json --checkpoint model.safetensors --dataset-digest SHA256 --output runs/rollout-session
```

仓库中的机械臂示例故意保持阻塞状态。`robot-check` 会实际枚举串口并读取每个相机的三帧图像，但不会打开串口或接触电机。现场测试急停、清空工作区且所有检查通过后，`robot-plan` 才会生成有人监督的低速联调方案。完整说明见[硬件工作流程](docs/hardware.md)。

## 当前边界

支持本地 LeRobot v2.1/v3.0 Parquet 约定布局，不保证适配任意分支。机械臂控制和预训练大模型训练仍由外部 LeRobot 环境执行；本项目负责生成可审阅计划和核验产物。参考模型读取一个相机、32×32 图像和前 64 个 UTF-8 文本字节；默认预加载最多 50,000 帧。数值剖析会将数值序列放入内存，大数据请先做子集。

页面“排除”只影响导出，不直接影响当前数据集的准备。要训练清洗后数据，为导出目录另开工作台。共享视频原样复制，可能保留排除轨迹画面。

本地测试和远程 GitHub Actions 是两回事。结果、未验证内容与复现方法见验证记录。
