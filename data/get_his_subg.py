import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import pickle
import pandas as pd

# 从历史图中采样，生成当前三元组的相关三元组。该函数处理正向和逆向三元组，并统计频率。
# TODO 在这里增加时间信息？
def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
    # q_to_sro = defaultdict(list)
    q_to_sro = set()
    # 反转三元组并更新关系索引
    inverse_triples = triples[:, [2, 1, 0, 3]]  # triples 的结构为 [头实体, 关系, 尾实体, 时间]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate([triples, inverse_triples])
    # ent_set = set(all_triples[:, 0])
    src_set = set(triples[:, 0])
    # TODO 原作者： dst_set = set(triples[:, 0])
    dst_set = set(triples[:, 2])

    # ----------------二阶邻居采样-----------------------
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # ent_list = list(ent_set)
    # rel_list = list(set(all_triples[:, 1]))

    inverse_subg = subg_arr[:, [2, 1, 0, 3]]  # 保留时间信息
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    # 创建 DataFrame，保留时间信息
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst', 'time'])
    # 整合重复三元组并统计频率，将频率作为第五列数据
    subg_df = df.groupby(['src', 'rel', 'dst', 'time']).size().reset_index(name='freq')
    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #将查询字段转化为pandas

    dst_df = df_dic.query('sr in @er_list')  #获取查询实体和关系的pandas
    dst_get = dst_df['dst'].values    #获取目标尾实体
    two_ent = set().union(*dst_get)   #将头实体与尾实体进行整合
    all_ent = list(src_set|two_ent)   
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')  #获取查询实体和关系的pandas
    dst_get_inv = dst_df_inv['dst'].values    #获取目标尾实体
    two_ent_inv = set().union(*dst_get_inv)   #将头实体与尾实体进行整合
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')

    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    return  q_tri,q_tri_inv

# 更新实体与关系的映射字典，维护源实体到关系-目标实体的映射。
def update_dict(subg_arr, s_to_sro, sr_to_sro,num_rels):
    # 根据输入的每一个时间的图来更新查询
    # 保留时间信息，假设 subg_arr 的结构为 [头实体, 关系, 尾实体, 时间]
    inverse_subg = subg_arr[:, [2, 1, 0, 3]]  # 反转三元组并保留时间
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst, _) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)

# 根据时间将输入的三元组数据分成多个快照（snapshot），每个快照对应一段时间内发生的三元组。统计并打印快照的节点数和关系数。
def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:4])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

# 从指定路径加载四元组数据，返回四元组数组和时间集合。
def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

dataset_list = ["ICEWS14", "ICEWS18","ICEWS05-15"]
#dataset_list = ["ICEWS14"]
for dataset in dataset_list:
    train_data, train_times = load_quadruples('./{}'.format(dataset), 'train.txt')
    num_nodes,num_rels= get_total_number('./{}'.format(dataset), 'stat.txt')
    print("the number of entity and relation", num_nodes,num_rels)

    train_list = split_by_time(train_data)
    id_list = [_ for _ in range(len(train_list))]
    sample_len = 3

# 该目录用于保存每个时间快照的正向（前向）历史图数据。文件名格式为 train_s_r_{index}.npy，其中 {index} 表示时间步的索引。
# 每个文件保存的是在当前时间步中，与目标实体及关系相关的正向三元组样本，用于捕获当前节点和关系的历史结构。
    save_dir_subg = './{}/his_graph_for/'.format(dataset)

# 该目录用于保存每个时间快照的反向历史图数据。文件名格式为 train_o_r_{index}.npy。
# 每个文件保存的是当前时间步中，与目标实体及关系相关的反向三元组样本。这些反向三元组将关系反向处理，用于获取额外的邻居信息，进一步扩展上下文。
    save_dir_obj = './{}/his_graph_inv/'.format(dataset)

# 该目录用于保存实体-关系到目标节点的字典（sr_to_sro）的数据，文件名为 train_s_r.npy。
# 这是一个全局字典，记录了每个实体-关系对可能连接的目标节点集合。这些信息可以用于生成特定实体和关系的所有潜在目标节点集合，便于在历史图的生成或后续查询时快速检索。
    save_dir_sub = './{}/his_dict/'.format(dataset)

    def mkdirs(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdirs(save_dir_obj)
    mkdirs(save_dir_sub)
    mkdirs(save_dir_subg)

    # f2 = open('../data/{}/copy_seq_graph/train_h_r_copy_seq.pkl'.format(args.dataset), 'rb')
    # que_subg = pickle.load(f2)
    sr_to_sro = defaultdict(set)
    s_to_sro = defaultdict(set)
    sro_to_fre = dict()
    subgraph_arr = []
    subgraph_arr_inv = []
    print("------------{}sample history graph-------------------------------------".format(dataset))
    all_list= train_list
    idx = [_ for _ in range(len(all_list))]
    for train_sample_num in tqdm(idx):
        if train_sample_num == 0: continue
        output = all_list[train_sample_num:train_sample_num+1]
        history_graph = all_list[train_sample_num-1:train_sample_num]
        update_dict(history_graph[0], s_to_sro, sr_to_sro, num_rels)
        if train_sample_num > 0:
            his_list = all_list[:train_sample_num]
            subg_arr = np.concatenate(his_list)
            sub_snap,sub_snap_inv = get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, output[0], num_nodes,num_rels)
        np.save('./{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num), sub_snap)
        np.save('./{}/his_graph_inv/train_o_r_{}.npy'.format(dataset, train_sample_num), sub_snap_inv)
    np.save('./{}/his_dict/train_s_r.npy'.format(dataset), sr_to_sro)
    # print(sub_snap)
    # arr = np.load('./{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num))
    # print(arr)
    # print(len(sr_to_sro.keys()))
    # sr_dic = np.load('./{}/his_dict/train_s_r.npy'.format(dataset), allow_pickle=True).item()
    # print(len(sr_dic.keys()))

    

# t1 = time.time()
# que_subg_list = defaultdict(list)
# que_subg_len = defaultdict(set)
# for id in tqdm(id_list):
#     triple = train_list[id:id+1]
#     sample_seq_graph = train_list[max(0, id-sample_len):min(id+sample_len, len(id_list))]
#     his_arr = np.concatenate(sample_seq_graph)
#     # que_subg = his_graph_sample1(his_arr, triple[0], num_r,que_subg_list, que_subg_len)
#     que_subg = get_sample_from_history_graph3(his_arr, triple[0], num_rels,que_subg_list, que_subg_len)
#     # with open('../data/{}/copy_seq_graph/train_h_r_copy_seq_{}.pkl'.format(args.dataset, id), 'wb') as f:
#     #     pickle.dump(que_subg, f)
# with open('./{}/copy_seq_graph/train_h_r_copy_seq.pkl'.format(dataset), 'wb') as f1:
#     pickle.dump(que_subg, f1)
# t2 = time.time() 