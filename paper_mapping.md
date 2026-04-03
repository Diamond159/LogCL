# 论文-代码-脚本-结果映射说明（paper_mapping）

---

## 1. 总体流程映射

论文方法总体链路：
1. 时态四元组按时间切分快照，并构造历史窗口序列。
2. 将历史快照转为 DGL 子图（含反向关系边）作为时序输入。
3. 可选加载预处理历史子图/字典（his_graph_for, his_graph_inv, his_dict）增强历史检索。
4. RecurrentRGCN 编码历史图并进行实体/关系时序更新。
5. 融合静态图约束（可选）与对比学习损失（可选）。
6. 使用 ConvTransE/ConvTransR 解码并计算实体/关系预测得分。
7. 训练、验证、测试，保存模型到 models，并写测试结果到 result CSV。

代码主链路定位：
- 训练入口：src/main.py:242
- 程序入口：src/main.py:455
- 数据按时间切分：src/main.py:256, src/main.py:257, src/main.py:258, rgcn/utils.py:342
- 历史子图构建调用：src/main.py:381, src/main.py:389
- 模型主类：src/rrgcn.py:123
- 时序前向主流程：src/rrgcn.py:322
- 推理接口：src/rrgcn.py:418
- 损失接口：src/rrgcn.py:450
- 结果写出：src/main.py:216, src/main.py:218
- checkpoint 保存：src/main.py:438

---

## 2. 分节精确映射（对应论文 3.3.3.1 ~ 3.3.4）

说明：本节从“模块对应”升级为“符号-变量-代码位置”对应，便于论文公式、算法伪码与实现逐项核对。

### 2.1 3.3.3.1 静态属性图卷积层

论文核心：静态图卷积得到静态实体锚点 $E_s$，用于初始化与约束动态演化。

符号与变量映射：

| 论文符号/公式 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| $G_s$ | `static_graph` | src/main.py（构建），src/rrgcn.py `forward`（使用） | 训练时传入模型的静态图对象。 |
| $\tilde{E}^{(0)}$ | `dynamic_emb = self.get_dynamic_emb(t)`（`use_cl`时）或 `self.dynamic_emb` | src/rrgcn.py `forward` | 静态图卷积前的实体初始表示。 |
| $W$（词嵌入） | `self.words_emb` | src/rrgcn.py `__init__` | 静态词节点嵌入参数。 |
| $X_s=Concat(\tilde{E}^{(0)},W)$ | `torch.cat((dynamic_emb/self.dynamic_emb, self.words_emb), dim=0)` | src/rrgcn.py `forward` | 实体与词拼接后作为静态图节点输入。 |
| 静态RGCN聚合（式3.1） | `self.statci_rgcn_layer(static_graph, [])` | src/rrgcn.py `forward` | 在静态异构图上传播消息。 |
| $E_s$ | `static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]` | src/rrgcn.py `forward` | 取前 `num_ents` 行作为实体静态表示。 |
| $H_0$ 几何锚点 | `self.h = static_emb` | src/rrgcn.py `forward` | 作为后续历史演化主循环起点。 |

关键参数：
- `--add-static-graph`：启用静态图分支。
- `--layer-norm`：对 `static_emb` 做归一化。

### 2.2 3.3.3.2 全局历史建模层

论文核心：离线构建累计历史子图，在线首步提取全局历史上下文，再经查询调制得到 $H_{his}^q$。

离线阶段（预处理）映射：

| 论文步骤 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| 时间切分快照 | `split_by_time` | data/get_his_subg.py | 生成按时间顺序的历史快照序列。 |
| 累积历史 $\cup_{t<t_q}G_t$ | `his_list = all_list[:train_sample_num]` + `np.concatenate(his_list)` | data/get_his_subg.py | 形成查询时刻前的累计历史。 |
| 构建反向边 $r+|R|$ | `inverse_triples[:,1] += num_rels` | data/get_his_subg.py | 显式区分方向。 |
| 结果落盘 | `his_graph_for/*.npy`, `his_graph_inv/*.npy`, `his_dict/*.npy` | data/get_his_subg.py | 训练阶段直接读取。 |

在线阶段（模型）映射：

| 论文符号/公式 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| $G_{his}$ | `sub_graph` | src/main.py -> src/rrgcn.py `forward` | 每个时间步读取的历史子图输入。 |
| 线图 $L(G)$（式3.3, 3.4） | `lg = g.line_graph(backtracking=False)` | src/rrgcn.py `forward` | 事件-事件二阶交互图。 |
| 关联算子/矩阵 $P$（式3.6, 3.7） | `pm_pd = self.cal_pmpd(all_triples, num_nodes)` | src/rrgcn.py `cal_pmpd`, `forward` | +1/-1 入射关联矩阵。 |
| 全局历史编码（式3.8） | `self.his_ent, _ = self.all_GCN(self.h, sub_graph, use_cuda, pm_pd, lg)` | src/rrgcn.py `forward`, `all_GCN` | 历史图实体表示主干输出。 |
| 查询调制（式3.9） | `his_att = F.softmax(self.w5(query_mask + self.his_ent), dim=1)` | src/rrgcn.py `forward` | 每实体特征维门控权重。 |
| $H_{his}^q$（式3.10） | `his_emb = F.normalize(his_att * self.his_ent)` | src/rrgcn.py `forward` | 供解码器融合的全局历史通道。 |

注：实现里“全局历史编码”在 `g_list` 的首步（`i==0`）执行一次，随后沿用到该样本后续时间步。

### 2.3 3.3.3.3 关系建模层（算法3.1）

论文核心：从实体状态池化关系证据 $X_r$，再用门控将其与基础关系 $R_0$ 融合为动态关系 $R_i$。

| 论文符号/公式 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| $R_0$ | `self.emb_rel` | src/rrgcn.py `__init__` | 关系基础原型参数。 |
| $E_r$ | `temp_e = self.h[g.r_to_e]` | src/rrgcn.py `forward` | 关系关联实体特征集合。 |
| 关系均值池化（式3.11） | `for span, r_idx in zip(g.r_len, g.uniq_r): x_input[r_idx]=mean(...)` | src/rrgcn.py `forward` | 对每个关系聚合局部证据。 |
| $\tilde{R}_i = R_0 + X_r$（式3.12） | `x_input = self.emb_rel + x_input` | src/rrgcn.py `forward` | 关系先验与局部证据相加。 |
| $\Gamma_i=\sigma(\tilde{R}_iW_{tg}+b_{tg})$（式3.12） | `time_weight = sigmoid(mm(x_input, self.time_gate_weight)+self.time_gate_bias)` | src/rrgcn.py `forward` | 时间门权重。 |
| $R_i$（式3.13） | `self.hr = time_weight*x_input + (1-time_weight)*self.emb_rel` | src/rrgcn.py `forward` | 当前步动态关系表示。 |

### 2.4 3.3.3.4 实体建模层（算法3.2）

论文核心：时间编码注入 + RGCN 空间聚合 + GRU 时序写入 + 查询门控聚合。

| 论文符号/公式 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| 趋势+周期时间向量（式3.14） | `timevec = alpha*alpha_t*t + (1-alpha)*cos(2*pi*beta_t*t)` | src/rrgcn.py `get_dynamic_emb` | 初始化路径的时序编码。 |
| $\tilde{e}_e(t)$（式3.15） | `attn=cat(st_static_emb,timevec); mm(attn, temporal_w)` | src/rrgcn.py `get_dynamic_emb` | 静态基底与时间向量融合。 |
| 相对相位编码 $\tau_i$（式3.16） | `h_t = cos(weight_t2*t2 + bias_t2)` | src/rrgcn.py `forward` | 主循环中的相对时间距离编码。 |
| 时间融合（式3.17） | `self.h = self.w4(concat([self.h, h_t], dim=1))` | src/rrgcn.py `forward` | 将相位注入上一时刻实体状态。 |
| 空间聚合（式3.18, 3.19） | `current_h = self.rgcn.forward(g, self.h, [self.emb_rel,self.emb_rel], pm_pd, lg)` | src/rrgcn.py `forward` | 当前快照图上的RGCN消息传递。 |
| 查询门（式3.21） | `att_e = softmax(self.w2(query_mask + current_h), dim=1)` | src/rrgcn.py `forward` | 查询相关特征门控。 |
| GRU写入（式3.20） | `self.h_0 = self.entity_cell(current_h, self.h or self.h_0)` | src/rrgcn.py `forward` | 时序演化写入。 |
| $\tilde{H}_i$（式3.22） | `att_emb = att_e * self.h_0` | src/rrgcn.py `forward` | 门控后的历史状态。 |
| 最终实体表示（式3.23） | `att_ent = mean(concat(att_embs)); history_emb = att_ent + history_embs[-1]` | src/rrgcn.py `forward` | 与论文“均值 + 最新状态残差”一致。 |

### 2.5 3.3.3.5 静态属性图约束组件（式3.24）

论文核心：用逐时间步角度阈值约束动态表示与静态锚点的一致性，形成 $L_{cp}$。

| 论文符号/公式 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| 静态锚点 $e_{s,e}$ | `static_emb` | src/rrgcn.py `get_loss` | 来自静态图卷积输出。 |
| 动态状态 $h_{i,e}$ | `evolve_emb`（遍历 `history_embs`） | src/rrgcn.py `get_loss` | 每时间步实体演化输出。 |
| 阈值角 $\theta_i$ | `step = (self.angle * pi / 180) * (time_step + 1)` | src/rrgcn.py `get_loss` | 线性递增角度调度。 |
| 余弦相似度 | `sim_matrix = sum(static_emb*evolve_emb)/(||.|| ||.||)` | src/rrgcn.py `get_loss` | 显式计算余弦相似度。 |
| Hinge项 $max(0, cos\theta_i - sim)$ | `mask = (cos(step)-sim_matrix)>0` + `masked_select(...)` | src/rrgcn.py `get_loss` | 仅对低于阈值的样本惩罚。 |
| $\lambda L_{cp}$ | `self.static_alpha * self.weight * ...` | src/rrgcn.py `get_loss`；src/main.py `--weight` | 约束项权重缩放。 |

### 2.6 3.3.3.6 实体预测层（算法3.3）

论文核心：局部动态实体 + 全局历史通道融合，ConvTransE 解码并做全实体排序。

| 论文符号/公式 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| $H_e$ | `embedding` | src/rrgcn.py `predict/get_loss` | 来自主循环的局部历史实体表示。 |
| $H_{his}^q$ | `his_emb` | src/rrgcn.py `forward` -> `decoder_ob.forward` | 全局历史查询通道。 |
| 融合系数 $\rho$（式3.25） | `pre_weight` | src/main.py 参数；src/decoder.py `forward` | 控制局部/全局融合比例。 |
| 融合主语表示 $e_s'$ | `e1_embed = pre_weight*e1_embedded + (1-pre_weight)*e1_his_embedded` | src/decoder.py `forward` | 与论文双路径加权一致。 |
| 动态关系 $r_{i,r}$ | `rel_embedded = emb_rel[triplets[:,1]]` | src/decoder.py `forward` | 对应每个样本关系条件。 |
| 卷积交互 $z_{s,r}$（式3.26） | `conv1 -> fc -> relu` 的 `x` | src/decoder.py `forward` | ConvTransE特征模板。 |
| logits（式3.27） | `x = mm(x, e1_embedded_all.T)` | src/decoder.py `forward` | 对所有候选实体打分排序。 |

补充：关系预测分支对应 `ConvTransR`，在 `--relation-prediction` 开启时生效。

### 2.7 3.3.4 模型训练流程（算法3.4）

论文核心：按 epoch 与时间步训练，计算 $L_{ent}$ 与 $L_{cp}$（可叠加对比损失），验证集选择最优模型。

| 论文流程/符号 | 项目变量/函数 | 代码位置 | 对应说明 |
|---|---|---|---|
| 快照序列 $\{G_t\}$ | `train_list = split_by_time(train_data)` | src/main.py + rgcn/utils.py | 按时间切分快照。 |
| 历史窗口 $G_{t-m:t-1}$ | `input_list = train_list[...]` | src/main.py 训练循环 | 每个训练时刻取历史窗口。 |
| 查询掩码 $Q$ | `que_pair=e2r(...); query_mask` | src/main.py + src/rrgcn.py `get_loss` | 由当前批次关系上下文构造。 |
| 静态初始化 $E_s$ | `self.forward(...)->static_emb` | src/rrgcn.py | 进入动态循环前先得到静态锚点。 |
| 关系更新 $R_k$ | `self.hr` | src/rrgcn.py `forward` | 每个历史步门控更新。 |
| 实体更新 $H_k$ | `self.h_0 = self.entity_cell(...)` | src/rrgcn.py `forward` | 每步时序写入。 |
| 实体预测损失 $L_{ent}$（式3.29） | `loss_ent += self.loss_e(scores_ob, triples[:,2])` | src/rrgcn.py `get_loss` | 尾实体交叉熵损失。 |
| 静态一致性损失 $L_{cp}$（式3.30） | `loss_cp += ...` | src/rrgcn.py `get_loss` | 角度约束项。 |
| 总损失 $L$（式3.31） | `loss = loss_e + loss_cp + loss_cl` | src/main.py 训练循环 | 当前实现还加入 `loss_cl`（对比学习项）。 |
| 反传与更新 | `loss.backward(); clip_grad_norm_; optimizer.step(); zero_grad()` | src/main.py 训练循环 | 完整优化步骤。 |
| 验证与保存最优 | `test(...valid...)` + `torch.save(...)` | src/main.py | 依据 filtered MRR 早停/保存。 |

### 2.8 3.3.3~3.3.4 关键符号总对照（便于写论文时直接引用）

| 论文符号 | 代码变量 | 备注 |
|---|---|---|
| $H_0$ | `self.h`（静态图后） | 动态循环初始实体状态。 |
| $H_i$ | `self.h_0` | 第 i 步GRU输出。 |
| $\hat{H}_i$ | `current_h` | 当前快照RGCN聚合输出。 |
| $R_0$ | `self.emb_rel` | 基础关系原型。 |
| $R_i$ | `self.hr` | 门控后的动态关系。 |
| $A_i$ | `att_e` / `his_att` | 查询门控（局部/全局两处）。 |
| $H_{his}$ | `self.his_ent` | 全局历史图实体表示。 |
| $H_{his}^q$ | `his_emb` | 查询调制后的全局通道。 |
| $H_e$ | `history_emb` | 最终实体预测输入。 |
| $L_{ent}$ | `loss_ent` | 实体分类损失。 |
| $L_{cp}$ | `loss_cp` | 静态一致性约束。 |
| $\lambda$ | `self.weight`（乘上 `self.static_alpha`） | 静态约束强度。 |
| $\rho$ | `pre_weight` | 局部/全局融合系数。 |

### 2.9 DSPN-CL 创新点细化映射（针对当前论文表述）

对应论文创新描述：
- 问题1：关系建模静态化，单一关系向量难以表达关系随时间的动态漂移。
- 问题2：多时间尺度语义混叠，单流结构容易把长期趋势与短期扰动混在同一表示里。
- 问题3：演化阶段感知不足，近期历史依赖过强，导致长时序预测稳定性下降。

代码中的 DSPN-CL 机制拆解：

1. 双流并行建模（Dual-Stream）
- 实体中心结构演化流（Entity-centric stream）：
	- 入口：src/rrgcn.py:326
	- 关键实现：历史快照循环内 `self.rgcn.forward(...) + self.entity_cell(...)`
	- 作用：持续更新实体状态，保留事件结构在时间轴上的局部演化信息。
- 关系中心独立交互流（Relation-centric stream）：
	- 入口：src/rrgcn.py:326
	- 关键实现：`x_input` 按关系池化后，经 `time_gate_weight/time_gate_bias` 门控得到 `self.hr`
	- 作用：将关系演化与实体演化显式分离，减少“单一关系向量”造成的表达瓶颈。

2. 长期语义与短期波动的解耦
- 长期语义分量：
	- 入口：src/rrgcn.py:484
	- 关键实现：`all_GCN(...)` 在累计历史图上提取 `his_emb`（全局历史语义）。
- 短期波动分量：
	- 入口：src/rrgcn.py:358
	- 关键实现：逐快照 `RGCN + GRU` 更新得到 `history_embs`（局部时序动态）。
- 解耦融合：
	- 入口：src/decoder.py:87
	- 关键实现：`pre_weight * local + (1-pre_weight) * global`
	- 作用：避免把长期趋势和短期扰动压缩到同一通道，提升多尺度表达可分性。

3. 多时间尺度一致性增强（跨流交叉对比）
- 入口：src/rrgcn.py:460
- 关键实现：`get_loss_conv(x1, x2)` 对齐以下两类视角：
	- 全局历史视角（`self.his_ent + his_r_emb`）
	- 局部演化视角（`evolve_emb + his_rel_embs[id]`）
- 作用：在不同时间尺度与不同信息流之间建立判别一致性，缓解语义混叠与阶段偏置。

4. 演化阶段感知机制
- 入口：src/rrgcn.py:375
- 关键实现：`h_t = cos(weight_t2 * t2 + bias_t2)` 的相位编码 + 关系时间门控。
- 作用：让模型显式感知“距离当前时刻有多远”的阶段信息，而非仅依赖历史窗口位置。

5. 训练稳定性约束（静态锚点）
- 入口：src/rrgcn.py:473
- 关键实现：动态实体表示与静态图表示的角度约束（`loss_cp`）。
- 作用：在长时序训练中抑制表示漂移，减少预测震荡。

实验层参数落点（可直接复现上述创新）：
- `--pre-type all`：启用“局部流 + 全局流”联合建模。
- `--pre-weight`：控制长期语义与短期动态的融合比例。
- `--use-cl --temperature`：启用跨流对比学习与温度缩放。
- `--add-static-graph --weight --angle`：启用静态语义锚点与演化角度约束。

---

## 3. 主要实验章节映射（第5章，按表号排序）

| 论文位置 | 代码文件 | 运行脚本 | 输出 |
|----------|----------|----------|------|
| 表5.4（ICEWS14/ICEWS05-15 主结果） | src/main.py, src/rrgcn.py | 主实验命令（-d ICEWS14 / -d ICEWS05-15） | results/相关表格.md |
| 表5.5（ICEWS18/GDELT 主结果） | src/main.py, src/rrgcn.py | 主实验命令（-d ICEWS18 / -d GDELT） | results/相关表格.md |
| 表5.6（边采样比率） | src/main.py | 扫描 --weight=0.0/0.2/0.35/0.4/0.6/0.8/1.0 | results/相关表格.md, results/第五章/sampling_ratio_mrr_5.1.svg |
| 表5.7（温度系数） | src/main.py, results/第五章/plot_temperature_experiment.py | 扫描 --temperature=0.01~0.90 | results/相关表格.md |
| 表5.8（ICEWS14/ICEWS18 消融） | src/main.py | 组合开关：--add-static-graph / --use-cl / --weight 0 | results/相关表格.md |
| 表5.9（ICEWS05-15/GDELT 消融） | src/main.py | 同表5.8消融策略，替换数据集 | results/相关表格.md |
| 表5.10（多跳推理） | src/main.py | 主实验命令 + --multi-step | results/相关表格.md |
| 表5.11（效率-性能权衡） | src/main.py | 扫描 pre-weight 与 weight 组合 | results/相关表格.md |

---

## 4. 论文图形映射（按图号排序）

| 论文位置 | 代码文件 | 运行脚本 | 输出 |
|----------|----------|----------|------|
| 图3.1 第三章总体方案图 | results/第三章/3.1第三章总体方案图.drawio_page_1.svg | 静态文件直接引用 | results/第三章/3.1第三章总体方案图.drawio_page_1.svg |
| 图3.2 第三章模型结构图 | results/第三章/3.2第三章model.drawio.svg | 静态文件直接引用 | results/第三章/3.2第三章model.drawio.svg |
| 图3.3 全局历史交互图 | results/第三章/3.3全局历史信息.drawio_page_1.svg | 静态文件直接引用 | results/第三章/3.3全局历史信息.drawio_page_1.svg |
| 图5.1 边采样比率曲线图 | results/第五章/plot_sampling_ratio_5.1.py | conda run -n logcl python results/第五章/plot_sampling_ratio_5.1.py | results/第五章/sampling_ratio_mrr_5.1.svg |
| 图5.2 消融实验对比图 | results/第五章/plot_ablation_study_5.2.py | conda run -n logcl python results/第五章/plot_ablation_study_5.2.py | results/第五章/ablation_study_mrr_5.2.svg |

---

## 5. 统一复现入口

1. conda run -n logcl python data/get_his_subg.py
2. 执行主实验命令（ICEWS14、ICEWS18、ICEWS05-15、GDELT）
3. 执行参数扫描命令（weight, temperature, multi-step）
4. 运行画图脚本生成第5章图形

结果总汇：
- results/相关表格.md
- results/第五章/*.svg
- src/result/*.csv
