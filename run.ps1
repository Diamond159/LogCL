# LogCL 统一运行脚本（Windows PowerShell）
# 说明：每一段都可单独复制执行；按顺序可完成主要实验或全流程复现。

# 0) 进入项目根目录（请根据实际路径修改）
Set-Location "D:\Code\paperCode\LogCL"

# 输出说明：切换到仓库根目录，后续命令均在此路径下执行。

# 1) 生成历史子图（全流程复现必需，已生成过可跳过）
python data/get_his_subg.py

# 输出说明：生成 data/<dataset>/his_graph_for、his_graph_inv、his_dict 下的 .npy 文件。

# 2) 主实验（ICEWS14）
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl

# 输出说明：
# - 模型参数保存到 models/*.pt
# - 训练/测试日志输出到 experiments/logs/ICEWS14/（如存在）
# - 结果写入 src/result/ICEWS14.csv

# 3) 主实验（ICEWS05-15）
python src/main.py -d ICEWS05-15 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl

# 输出说明：
# - 模型参数保存到 models/*.pt
# - 结果写入 src/result/ICEWS05-15.csv

# 4) 主实验（ICEWS18）
python src/main.py -d ICEWS18 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl

# 输出说明：
# - 模型参数保存到 models/*.pt
# - 结果写入 src/result/ICEWS18.csv

# 5) 主实验（GDELT）
python src/main.py -d GDELT --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl

# 输出说明：
# - 模型参数保存到 models/*.pt
# - 结果写入 src/result/GDELT.csv

# 6) 边采样比率扫描（表5.6）
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.35 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl

# 输出说明：
# - 改动 weight 参数即可完成不同边采样比率实验
# - 结果汇总可手动写入 results/相关表格.md

# 7) 温度系数扫描（表5.7）
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03 --use-cl

# 输出说明：
# - 改动 temperature 参数即可完成温度敏感性实验

# 8) 消融实验示例（关闭对比学习）
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 1.0 --entity-prediction --angle 10 --discount 1 --pre-weight 0.9 --pre-type all --add-static-graph --temperature 0.03

# 输出说明：
# - 通过去掉 --use-cl、--add-static-graph 或调整 weight 即可完成不同消融组合
