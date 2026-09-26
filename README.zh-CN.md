# VLA Preflight Workbench

**把数据检查、可复现准备、实际训练和实验比较连起来的本地 VLA 工作台。**

[English](README.md) · [完整工作流程](docs/workflows.md) · [验证记录](docs/validation.md)

v0.2.0 alpha 面向显存有限、没有机械臂、想先跑通机器人学习实验的人。内置合成图像＋语言＋动作数据，不需要购买硬件或提前准备数据。

## 这次能做什么

- **可视化检查**：轨迹列表、逐帧图像、动作曲线、预检发现、数值重复组与探索性时间偏移曲线。
- **数据准备**：按完整轨迹划分训练/验证集，重复数值轨迹保持同组，归一化只拟合训练集，记录文件指纹。
- **实际训练**：PyTorch 执行图像＋文本＋状态模型的梯度更新，支持动作块与 padding mask，保存真实采样 trace。
- **评估与续训**：验证集原始动作单位 MAE/RMSE、常数动作基线、训练曲线、最新/最佳检查点、断点续训、实验比较。
- **筛选导出**：明确排除轨迹，另建数据集，重排索引并重算数值统计量，原始文件保持不变。
- **大模型接入入口**：生成实验性的 SmolVLA 外部微调计划，物理导出训练集。

Tiny VLA 是随机初始化的小型参考模型，**不是预训练大模型**。SmolVLA 接口没有做真实模型训练或显存占用验证。项目价值在于可检查、可复现的流程，不宣称 GitHub 全球首创。

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

## 当前边界

支持本地 LeRobot v2.1/v3.0 Parquet 约定布局，不保证适配任意分支。没有机械臂控制、闭环仿真成功率或完整大模型训练实现。参考模型读取一个相机、32×32 图像和前 64 个 UTF-8 文本字节；默认预加载最多 50,000 帧。数值剖析会将数值序列放入内存，大数据请先做子集。

页面“排除”只影响导出，不直接影响当前数据集的准备。要训练清洗后数据，为导出目录另开工作台。共享视频原样复制，可能保留排除轨迹画面。

本地测试和远程 GitHub Actions 是两回事。结果、未验证内容与复现方法见验证记录。
