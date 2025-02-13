import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def filter_by_time_window(events, current_time, window_size=5):
    """
    根据时间窗口过滤事件，只保留当前时间戳附近的事件。
    :param events: 事件列表，每个事件包含 (head, rel, tail, time)
    :param current_time: 当前时间戳
    :param window_size: 时间窗口大小，表示当前时间附近的事件
    :return: 过滤后的事件列表
    """
    filtered_events = [event for event in events if abs(event[3] - current_time) <= window_size]
    return filtered_events


def filter_by_embedding_similarity(current_event, events, threshold=0.7):
    """
    根据嵌入相似度过滤事件，保留与当前事件嵌入相似度高的事件。
    :param current_event: 当前事件的嵌入
    :param events: 事件列表，包含待比较的所有事件的嵌入
    :param threshold: 相似度阈值
    :return: 过滤后的事件列表
    """
    similarities = cosine_similarity([current_event], events)
    filtered_events = [event for event, similarity in zip(events, similarities[0]) if similarity >= threshold]
    return filtered_events


def filter_by_frequency(events, sro_to_fre, threshold=1):
    """
    根据事件的频率过滤，只保留频率大于等于阈值的事件。
    :param events: 事件列表
    :param sro_to_fre: 一个字典，包含每个三元组的频率
    :param threshold: 频率阈值
    :return: 过滤后的事件列表
    """
    filtered_events = [event for event in events if sro_to_fre.get((event[0], event[1], event[2]), 0) >= threshold]
    return filtered_events


def filter_by_graph_neighbors(events, current_entity, graph, threshold=0.1):
    """
    基于图的邻居关系过滤事件，保留与当前实体在图中连接的事件。
    :param events: 事件列表
    :param current_entity: 当前实体（头实体）
    :param graph: 图的邻接矩阵
    :param threshold: 邻居关系的相似度阈值
    :return: 过滤后的事件列表
    """
    filtered_events = []
    for event in events:
        # 根据邻接矩阵检查该事件是否与当前实体有较强的连接
        neighbor_similarity = graph[current_entity, event[0]]  # 使用邻接矩阵计算相似度
        if neighbor_similarity >= threshold:
            filtered_events.append(event)
    return filtered_events


def filter_events(current_event, events, sro_to_fre, entity_embeds, rel_embeds, graph, current_time, time_window=5,
                  similarity_threshold=0.7, freq_threshold=1):
    """
    综合多种过滤方式，逐步筛选出与当前事件相关的事件。
    :param current_event: 当前事件
    :param events: 所有事件列表
    :param sro_to_fre: 三元组的频率字典
    :param entity_embeds: 实体嵌入矩阵
    :param rel_embeds: 关系嵌入矩阵
    :param graph: 图的邻接矩阵
    :param current_time: 当前时间戳
    :param time_window: 时间窗口大小
    :param similarity_threshold: 嵌入相似度阈值
    :param freq_threshold: 频率阈值
    :return: 经过筛选后的事件列表
    """
    # 先根据时间窗口过滤
    time_filtered_events = filter_by_time_window(events, current_time, time_window)

    # 再根据嵌入相似度过滤
    embedding_filtered_events = filter_by_embedding_similarity(current_event, time_filtered_events,
                                                               similarity_threshold)

    # 再根据频率过滤
    frequency_filtered_events = filter_by_frequency(embedding_filtered_events, sro_to_fre, freq_threshold)

    # 如果需要，可以根据图的邻居关系再过滤
    final_filtered_events = filter_by_graph_neighbors(frequency_filtered_events, current_event[0], graph)

    return final_filtered_events


def get_sample_from_history_graph(subg_arr, s_to_sro, sr_to_sro, sro_to_fre, triples, num_nodes, num_rels,
                                  current_time):
    # 获取历史图的样本并进行过滤
    inverse_triples = triples[:, [2, 1, 0, 3]]
    inverse_triples[:, 1] += num_rels
    all_triples = np.concatenate([triples, inverse_triples])

    # 根据当前事件进行过滤
    filtered_events = filter_events(current_event=triples[0], events=all_triples, sro_to_fre=sro_to_fre,
                                    entity_embeds=s_to_sro, rel_embeds=sr_to_sro, graph=sro_to_fre,
                                    current_time=current_time)

    return filtered_events, inverse_triples


def update_dict(subg_arr, s_to_sro, sr_to_sro,num_rels):
    # 根据输入的每一个时间的图来更新查询查询
    inverse_subg = subg_arr[:, [2, 1, 0, 3]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst, time) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)

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


def process_data(dataset_list):
    for dataset in dataset_list:
        train_data, train_times = load_quadruples('./{}'.format(dataset), 'train.txt')
        num_nodes, num_rels = get_total_number('./{}'.format(dataset), 'stat.txt')
        print("the number of entity and relation", num_nodes, num_rels)

        train_list = split_by_time(train_data)
        id_list = [_ for _ in range(len(train_list))]
        sample_len = 3

        save_dir_subg = './{}/his_graph_for/'.format(dataset)
        save_dir_obj = './{}/his_graph_inv/'.format(dataset)
        save_dir_sub = './{}/his_dict/'.format(dataset)

        def mkdirs(path):
            if not os.path.exists(path):
                os.makedirs(path)

        mkdirs(save_dir_obj)
        mkdirs(save_dir_sub)
        mkdirs(save_dir_subg)

        sr_to_sro = defaultdict(set)
        s_to_sro = defaultdict(set)
        sro_to_fre = dict()
        subgraph_arr = []
        subgraph_arr_inv = []

        print("------------{} sample history graph-------------------------------------".format(dataset))
        all_list = train_list
        idx = [_ for _ in range(len(all_list))]

        for train_sample_num in tqdm(idx):
            if train_sample_num == 0: continue
            output = all_list[train_sample_num:train_sample_num + 1]
            history_graph = all_list[train_sample_num - 1:train_sample_num]
            update_dict(history_graph[0], s_to_sro, sr_to_sro, num_rels)

            if train_sample_num > 0:
                his_list = all_list[:train_sample_num]
                subg_arr = np.concatenate(his_list)
                sub_snap, sub_snap_inv = get_sample_from_history_graph(subg_arr, s_to_sro, sr_to_sro, sro_to_fre,
                                                                       output[0], num_nodes, num_rels,
                                                                       current_time=train_times[train_sample_num])

            np.save('./{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num), sub_snap)
            np.save('./{}/his_graph_inv/train_o_r_{}.npy'.format(dataset, train_sample_num), sub_snap_inv)

        np.save('./{}/his_dict/train_s_r.npy'.format(dataset), sr_to_sro)

# dataset_list = ["ICEWS14", "ICEWS18","ICEWS05-15"]
dataset_list = ["ICEWS14"]
process_data(dataset_list)

    

# t1 = time.time()
# que_subg_list = defaultdict(list)
# que_subg_len = defaultdict(set)
# for id in tqdm(id_list):
#     triple = train_list[id:id+1]
#     sample_seq_graph = train_list[max(0, id-sample_len):min(id+sample_len, len(id_list))]
#     his_arr = np.concatenate(sample_seq_graph)
#     # que_subg = his_graph_sample1(his_arr, triple[0], num_r,que_subg_list, que_subg_len)
#     que_subg = get_sample_from_history_graph3(his_arr, triple[0], num_rels,que_subg_list, que_subg_len)
#     # with open('./data/{}/copy_seq_graph/train_h_r_copy_seq_{}.pkl'.format(args.dataset, id), 'wb') as f:
#     #     pickle.dump(que_subg, f)
# with open('./{}/copy_seq_graph/train_h_r_copy_seq.pkl'.format(dataset), 'wb') as f1:
#     pickle.dump(que_subg, f1)
# t2 = time.time() 