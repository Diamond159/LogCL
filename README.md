# DSPN-CL / Baseline: LogCL（Chen W, Wan H, Wu Y, et al. Local-global history-aware contrastive learning for temporal knowledge graph reasoning[C]. 2024 IEEE 40th International Conference on Data Engineering (ICDE). 2023: 733-746.）


## 1. 项目简介
本仓库用于复现与扩展时态知识图谱推理相关实验。核心模型为 **DSPN-CL**（Dual-Stream Prediction Network via Contrastive Learning），通过感知对比学习解决事件预测中的关键问题：

- **问题背景**：现有模型存在关系建模静态化、多时间尺度语义混叠、单一关系向量无法解耦长期语义与短期波动等挑战
- **技术方案**：并行构建实体中心的结构演化流与关系中心的独立交互流，实现全局-局部时序融合与关系动态捕捉
- **核心创新**：跨流交叉对比学习增强多时间尺度判别性，缓解语义混叠与历史依赖偏置，提升长时序事件演化预测能力


当前代码中包含：
- 主实验（ICEWS14、ICEWS18、ICEWS05-15、GDELT）
- 参数敏感性实验（边采样比率、温度系数）
- 消融实验
- 不同推理跳数与效率-性能权衡实验

实验组织目录主要在 `experiments/`，结果汇总在 `results/`。

## 2 系统环境

表 5.1 实验用服务器具体硬件参数：

| 硬件配置 | 实验室工作站 | 云服务器实例 |
| --- | --- | --- |
| CPU 型号 | Intel Core i9-13900K @ 3.00GHz | Intel Xeon Silver 4214R @ 2.40GHz |
| CPU 核心数 | 24 核心 (8P + 16E) | 12 vCPU |
| GPU 型号 | NVIDIA RTX 4090 | NVIDIA RTX 3090 |
| 显存 | 24 GB | 24 GB |
| 内存 | 64 GB | 90 GB |
| 磁盘 | 1 TB | 80 GB |

表 5.2 实验用服务器相关依赖库说明：

| 名称 | 版本 | 说明 |
| --- | --- | --- |
| PyTorch | 2.1.2 | 深度学习框架，支持 GPU 加速计算 |
| CUDA | 11.8 | 并行计算平台 |
| NumPy | 1.26.4 | 多维数组与数值计算支持 |
| scikit-learn | 1.5.2 | 数据预处理与模型评估 |
| DGL | 1.1.2 (cu118) | 图神经网络库 |

推荐运行环境：

- OS: Ubuntu 22.04 / Windows 10 或 11
- Python: 3.10
- CUDA: 11.8
- PyTorch: 2.1.2
- GPU: NVIDIA RTX 3090 24GB 及以上

## 3 依赖安装说明

**AUTODL服务器环境配置**：
服务器默认环境已预装 CUDA 11.8 + PyTorch 2.1.2，直接安装 DGL：

```bash
export DGLBACKEND=pytorch
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.1/cu118/repo.html
pip install -r requirement.txt
```

**本地环境配置**：

使用 conda 创建环境：

```bash
conda create -n logcl python=3.10
conda activate logcl
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

然后安装 DGL 和其他依赖：

```bash
export DGLBACKEND=pytorch
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.1/cu118/repo.html
pip install -r requirement.txt
```

> **注**：[requirement.txt](requirement.txt) 包含当前版本的完整依赖清单，确保 CUDA、PyTorch 和 DGL 版本与环境一致。


## 4. 数据说明
数据目录在 `data/`，目前仓库中已包含：
- ICEWS14
- ICEWS18
- ICEWS05-15
- GDELT

预处理脚本：
- `data/get_his_subg.py`：构建历史子图与查询字典
- `data/<dataset>/ent2word.py`：构建静态属性图相关词映射

建议预处理顺序：
```bash
cd data
python get_his_subg.py

cd ICEWS14
python ent2word.py

cd ../ICEWS18
python ent2word.py

cd ../ICEWS05-15
python ent2word.py

cd ../GDELT
python ent2word.py
```

## 5. 运行方法

### 5.1 单次主实验（以 ICEWS14 为例）
```bash
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl
```

### 5.2 按实验目录直接运行
项目里多数实验子目录都带有 `run.sh`，可直接进入对应目录执行：
```bash
bash run.sh
```

## 6. 复现实验（按论文表格）
下面目录已按表格拆好，可直接进入子目录运行 `run.sh`：

- 表5.4：`experiments/table5.4_主实验_ICEWS14_05-15/`
- 表5.5：`experiments/table5.5_主实验_ICEWS18_GDELT/`
- 表5.6：`experiments/table5.6_边采样比率敏感性分析/`
- 表5.7：`experiments/table5.7_温度系数参数分析/`
- 表5.8：`experiments/table5.8_消融实验_ICEWS14_18/`
- 表5.9：`experiments/table5.9_消融实验_0515_GDELT/`
- 表5.10：`experiments/table5.10_不同推理跳数对比/`
- 表5.11：`experiments/table5.11_计算效率与性能权衡/`

例如：
```bash
cd experiments/table5.7_温度系数参数分析/temperature_0.03
bash run.sh
```

## 7. 结果说明
默认会生成以下几类结果：

- 训练日志与中间输出：`checkpoints/`、各实验子目录下日志文件
- 模型参数：`models/`
- 结构化结果：`src/result/`、`results/`
- 表格汇总与可视化：`results/相关表格.md`、`results/第五章/`

论文实验映射关系，参考仓库里的 `paper_mapping.md`。


