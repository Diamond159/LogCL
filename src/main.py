import csv
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append(".")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph, get_relhead_reltal, build_super_g
from src.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def append_entity_metrics_to_excel(row, excel_path='./result/entity_prediction_metrics.xlsx', sheet_name='entity_prediction'):
    """Append one experiment row to an Excel sheet for paper-ready filtering."""
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    row_df = pd.DataFrame([row])

    try:
        if os.path.exists(excel_path):
            try:
                history_df = pd.read_excel(excel_path, sheet_name=sheet_name)
            except ValueError:
                history_df = pd.DataFrame()
            out_df = pd.concat([history_df, row_df], ignore_index=True)
        else:
            out_df = row_df

        out_df.to_excel(excel_path, sheet_name=sheet_name, index=False)
        print("[Excel] Entity metrics appended to {}".format(excel_path))
    except ImportError:
        print("[Excel] Skip export: openpyxl is not installed. Please install openpyxl.")


def load_id_name_map(file_path):
    id_to_name = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            name, idx = parts
            id_to_name[int(idx)] = name
    return id_to_name


def relation_id_to_name(rel_id, num_rels, rel_id_to_name):
    if rel_id < num_rels:
        return rel_id_to_name.get(rel_id, str(rel_id))
    base_id = rel_id - num_rels
    return rel_id_to_name.get(base_id, str(base_id)) + "_inv"


def update_dict(subg_arr, s_to_sro, sr_to_sro,sro_to_fre, num_rels):
    # 维护历史查询索引: s->(s,r,o) 与 (s,r)->o。
    # 该字典在局部/全局历史采样阶段复用，用于快速构造候选历史边。
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)
        
def e2r(triplets, num_rels):
    # 将“实体->关系集合”整理成紧凑张量，供关系池化与查询门控使用。
    # 对应 paper_mapping 中关系建模层的关系统计池化输入。
    src, rel, dst = triplets.transpose()
    # get all relations
    # uniq_e = np.concatenate((src, dst))
    uniq_e = np.unique(src)
    # generate r2e
    e_to_r = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].add(rel)
        # e_to_r[dst].add(rel+num_rels)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx,idx+len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()
    return [uniq_e, r_len, r_idx]

def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,num_nodes, num_rels, use_cuda, gpu):
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 0])

    # ----------------二阶邻居采样-----------------------
    # 先按 (s,r) 取一跳候选尾实体，再把候选实体回查到历史子图中，形成二阶历史上下文。
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # 用字典计数替代 DataFrame.groupby，降低内存峰值并避免分组异常。
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])

    triple_freq = defaultdict(int)
    for s, r, d in subg_triples:
        triple_freq[(int(s), int(r), int(d))] += 1

    def collect_two_hop_entities(er_pairs):
        ents = set()
        for s, r in er_pairs:
            dsts = sr_to_sro.get((int(s), int(r)))
            if dsts:
                ents.update(dsts)
        return ents

    two_ent = collect_two_hop_entities(er_list)
    two_ent_inv = collect_two_hop_entities(er_list_inv)

    all_ent = src_set | two_ent
    all_ent_inv = dst_set | two_ent_inv

    result = [[s, r, d, c] for (s, r, d), c in triple_freq.items() if s in all_ent]
    result_inv = [[s, r, d, c] for (s, r, d), c in triple_freq.items() if s in all_ent_inv]
    #----------------二阶邻居采样-----------------------
    # result = subg_df.query('src in @src_set')
    q_tri = np.array(result, dtype=np.int64)
    q_tri_inv = np.array(result_inv, dtype=np.int64)

    if q_tri.ndim == 1:
        q_tri = q_tri.reshape(0, 4)
    if q_tri_inv.ndim == 1:
        q_tri_inv = q_tri_inv.reshape(0, 4)

    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu) 
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return  his_sub,his_sub_inv



def test(model ,history_len, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []
    ranks_raw_r_inv, ranks_filter_r_inv, mrr_raw_list_r_inv, mrr_filter_list_r_inv = [], [], [], []
    ranks_raw1, ranks_filter1 = [],[]
    detailed_rows = []

    entity_id_to_name = load_id_name_map('data/{}/entity2id.txt'.format(args.dataset))
    rel_id_to_name = load_id_name_map('data/{}/relation2id.txt'.format(args.dataset))

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        print("------------store_path----------------",model_name)
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # 测试阶段使用滑动历史窗口，逐快照外推。
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    start_time = len(history_list)

    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)
    sr_to_sro = np.load('data/{}/his_dict/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()

    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        tc = start_time + time_idx
        tlist = list(range(tc - args.train_history_len, tc))
        # tlist = [min(start_time-args.start_history_len-1,t) for t in tlist]
        tlist = torch.Tensor(tlist).cuda()

        # 构建局部历史图序列，对应实体建模层的历史快照输入。
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples =test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        que_pair =  e2r(test_snap, num_rels)
        que_pair_inv =  e2r(inverse_triples, num_rels)

        sub_snap,sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap , num_nodes,num_rels,use_cuda, args.gpu)


        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(
            inverse_triples)
        test_triples, final_score = model.predict(que_pair, tlist, sub_snap, time_idx, history_glist, num_rels,
                                                  static_graph, test_triples_input, input_list, num_nodes, use_cuda)
        inv_test_triples, inv_final_score = model.predict(que_pair_inv, tlist, sub_snap_inv, time_idx, history_glist,
                                                          num_rels, static_graph, test_triples_input_inv, input_list,
                                                          num_nodes, use_cuda)

        # 评估时同时统计正向与反向查询，最终按论文口径汇总 all_* 指标。

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
            # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)

        # 逐样本导出（raw/filter 双口径），保证与日志指标可逐列对齐复算。
        triples_np = test_triples.detach().cpu().numpy()
        triples_inv_np = inv_test_triples.detach().cpu().numpy()

        rank_raw_np = rank_raw.detach().cpu().numpy().astype(np.int64)
        rank_filter_np = rank_filter.detach().cpu().numpy().astype(np.int64)
        rank_raw_inv_np = rank_raw_inv.detach().cpu().numpy().astype(np.int64)
        rank_filter_inv_np = rank_filter_inv.detach().cpu().numpy().astype(np.int64)

        top1_raw = torch.argmax(final_score, dim=1).detach().cpu().numpy().astype(np.int64)
        top1_raw_inv = torch.argmax(inv_final_score, dim=1).detach().cpu().numpy().astype(np.int64)

        filter_score_forward = utils.filter_score(test_triples, final_score.detach().clone(), all_ans_list[time_idx])
        filter_score_inverse = utils.filter_score(inv_test_triples, inv_final_score.detach().clone(), all_ans_list[time_idx])
        top1_filter = torch.argmax(filter_score_forward, dim=1).detach().cpu().numpy().astype(np.int64)
        top1_filter_inv = torch.argmax(filter_score_inverse, dim=1).detach().cpu().numpy().astype(np.int64)

        for i, (h, r, t) in enumerate(triples_np):
            rr_raw = 1.0 / int(rank_raw_np[i])
            rr_filter = 1.0 / int(rank_filter_np[i])
            pred_raw = int(top1_raw[i])
            pred_filter = int(top1_filter[i])
            detailed_rows.append({
                'time_idx': int(time_idx),
                'direction': 'forward',
                'head_id': int(h),
                'rel_id': int(r),
                'tail_true_id': int(t),
                'pred_tail_top1_raw_id': pred_raw,
                'pred_tail_top1_filter_id': pred_filter,
                'head_name': entity_id_to_name.get(int(h), str(int(h))),
                'rel_name': relation_id_to_name(int(r), num_rels, rel_id_to_name),
                'tail_true_name': entity_id_to_name.get(int(t), str(int(t))),
                'pred_tail_top1_raw_name': entity_id_to_name.get(pred_raw, str(pred_raw)),
                'pred_tail_top1_filter_name': entity_id_to_name.get(pred_filter, str(pred_filter)),
                'raw_rank': int(rank_raw_np[i]),
                'filter_rank': int(rank_filter_np[i]),
                'rr_raw': rr_raw,
                'rr_filter': rr_filter,
                'mrr': rr_filter,
                'is_hit1_raw': 1 if int(rank_raw_np[i]) <= 1 else 0,
                'is_hit3_raw': 1 if int(rank_raw_np[i]) <= 3 else 0,
                'is_hit10_raw': 1 if int(rank_raw_np[i]) <= 10 else 0,
                'is_hit1_filter': 1 if int(rank_filter_np[i]) <= 1 else 0,
                'is_hit3_filter': 1 if int(rank_filter_np[i]) <= 3 else 0,
                'is_hit10_filter': 1 if int(rank_filter_np[i]) <= 10 else 0,
            })

        for i, (h, r, t) in enumerate(triples_inv_np):
            rr_raw = 1.0 / int(rank_raw_inv_np[i])
            rr_filter = 1.0 / int(rank_filter_inv_np[i])
            pred_raw = int(top1_raw_inv[i])
            pred_filter = int(top1_filter_inv[i])
            detailed_rows.append({
                'time_idx': int(time_idx),
                'direction': 'inverse',
                'head_id': int(h),
                'rel_id': int(r),
                'tail_true_id': int(t),
                'pred_tail_top1_raw_id': pred_raw,
                'pred_tail_top1_filter_id': pred_filter,
                'head_name': entity_id_to_name.get(int(h), str(int(h))),
                'rel_name': relation_id_to_name(int(r), num_rels, rel_id_to_name),
                'tail_true_name': entity_id_to_name.get(int(t), str(int(t))),
                'pred_tail_top1_raw_name': entity_id_to_name.get(pred_raw, str(pred_raw)),
                'pred_tail_top1_filter_name': entity_id_to_name.get(pred_filter, str(pred_filter)),
                'raw_rank': int(rank_raw_inv_np[i]),
                'filter_rank': int(rank_filter_inv_np[i]),
                'rr_raw': rr_raw,
                'rr_filter': rr_filter,
                'mrr': rr_filter,
                'is_hit1_raw': 1 if int(rank_raw_inv_np[i]) <= 1 else 0,
                'is_hit3_raw': 1 if int(rank_raw_inv_np[i]) <= 3 else 0,
                'is_hit10_raw': 1 if int(rank_raw_inv_np[i]) <= 10 else 0,
                'is_hit1_filter': 1 if int(rank_filter_inv_np[i]) <= 1 else 0,
                'is_hit3_filter': 1 if int(rank_filter_inv_np[i]) <= 3 else 0,
                'is_hit10_filter': 1 if int(rank_filter_inv_np[i]) <= 10 else 0,
            })
            # used to show slide results
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            # else:
            #     predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
            # subg_arr = np.concatenate([subg_arr,test_snap])
            # print(np.shape(subg_arr))
        idx += 1

    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter,hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv,hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv,hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")
    all_mrr_raw = (mrr_raw+mrr_raw_inv)/2
    all_mrr_filter = (mrr_filter+mrr_filter_inv)/2
    all_hit_raw, all_hit_filter,all_hit_raw_r, all_hit_filter_r = [],[],[],[]
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id]+hit_raw_inv[hit_id])/2)
        all_hit_filter.append((hit_filter[hit_id]+hit_filter_inv[hit_id])/2)
    print("(all_raw) MRR, Hits@ (1,3,10):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_raw.item(), all_hit_raw[0],all_hit_raw[1],all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,10):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_filter.item(), all_hit_filter[0],all_hit_filter[1],all_hit_filter[2]))

    entity_metrics = {
        'filter_MRR': float(mrr_filter),
        'filter_H@1': hit_filter[0],
        'filter_H@3': hit_filter[1],
        'filter_H@10': hit_filter[2],
        'filter_inv_MRR': float(mrr_filter_inv),
        'filter_inv_H@1': hit_filter_inv[0],
        'filter_inv_H@3': hit_filter_inv[1],
        'filter_inv_H@10': hit_filter_inv[2],
        'all_MRR': all_mrr_raw.item(),
        'all_H@1': all_hit_raw[0],
        'all_H@3': all_hit_raw[1],
        'all_H@10': all_hit_raw[2],
        'filter_all_MRR': all_mrr_filter.item(),
        'filter_all_H@1': all_hit_filter[0],
        'filter_all_H@3': all_hit_filter[1],
        'filter_all_H@10': all_hit_filter[2],
    }
    
    # 文件转储: 仅在 test 模式输出到 ./result/*.csv，便于实验表格复现。
    if mode == "test": # test模式写入，train模式忽略
        filename = './result/'+ args.dataset + ".csv"
        if os.path.isfile(filename) == False:# 如果文件不存在，则创建
            with open (filename,'w', newline='') as f:
                # 写入列名
                fieldnames=['encoder','opn','pre_type','use_static','use_cl','gpu','datetime','pre_weight',
                            'train_len','test_len','temperature','lr','n_hidden',
                            'filter_MRR','filter_H@1','filter_H@3','filter_H@10',
                            'filter_inv_MRR','filter_inv_H@1','filter_inv_H@3','filter_inv_H@10',
                            'all_MRR','all_H@1','all_H@3','all_H@10',
                            'filter_all_MRR','filter_all_H@1','filter_all_H@3','filter_all_H@10']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        # 写入数据
        with open (filename,'a', newline='') as f:
            writer = csv.writer(f)
            row={'encoder':args.encoder,'opn':args.opn,'pre_type':args.pre_type,'use_static':args.add_static_graph,'use_cl':args.use_cl,'gpu':args.gpu,'datetime':datetime.now(),'pre_weight':args.pre_weight,
                'train_len':args.train_history_len,'test_len':args.test_history_len,'temperature':args.temperature,'lr':args.lr,'n_hidden':args.n_hidden,
                'filter_MRR':float(mrr_filter),'filter_H@1':hit_filter[0],'filter_H@3':hit_filter[1],'filter_H@10':hit_filter[2],
                'filter_inv_MRR':float(mrr_filter_inv),'filter_inv_H@1':hit_filter_inv[0],'filter_inv_H@3':hit_filter_inv[1],'filter_inv_H@10':hit_filter_inv[2],
                'all_MRR':all_mrr_raw.item(),'all_H@1':all_hit_raw[0],'all_H@3':all_hit_raw[1],'all_H@10':all_hit_raw[2],
                'filter_all_MRR':all_mrr_filter.item(),'filter_all_H@1':all_hit_filter[0],'filter_all_H@3':all_hit_filter[1],'filter_all_H@10':all_hit_filter[2]}
            writer.writerow(row.values())

        detail_df = pd.DataFrame(detailed_rows)

        def summarize(df, scope, prefix):
            return {
                'scope': scope,
                'metric_type': prefix,
                'MRR': float(df['rr_{}'.format(prefix)].mean()),
                'Hits@1': float(df['is_hit1_{}'.format(prefix)].mean()),
                'Hits@3': float(df['is_hit3_{}'.format(prefix)].mean()),
                'Hits@10': float(df['is_hit10_{}'.format(prefix)].mean()),
            }

        df_forward = detail_df[detail_df['direction'] == 'forward']
        df_inverse = detail_df[detail_df['direction'] == 'inverse']
        summary_rows = [
            summarize(df_forward, 'forward', 'raw'),
            summarize(df_forward, 'forward', 'filter'),
            summarize(df_inverse, 'inverse', 'raw'),
            summarize(df_inverse, 'inverse', 'filter'),
            summarize(detail_df, 'all', 'raw'),
            summarize(detail_df, 'all', 'filter'),
        ]
        summary_df = pd.DataFrame(summary_rows)

        excel_output_dir = getattr(args, 'log_dir', './result')
        os.makedirs(excel_output_dir, exist_ok=True)
        detail_file = os.path.join(excel_output_dir, '{}_prediction_details_{}.xlsx'.format(args.dataset, int(time.time())))
        try:
            with pd.ExcelWriter(detail_file, engine='openpyxl') as writer:
                detail_df.to_excel(writer, sheet_name='detailed_predictions', index=False)
                summary_df.to_excel(writer, sheet_name='metrics_from_details', index=False)
            print('[Excel] Detailed prediction rows saved to {}'.format(detail_file))
        except ImportError:
            print("[Excel] Skip detailed export: openpyxl is not installed. Please install openpyxl.")

    return all_mrr_raw, all_mrr_filter, entity_metrics


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # 1) 数据准备: 动态快照切分 + filtered 评估答案字典。
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature,args.pre_weight, args.use_cl, args.pre_type,  args.n_hidden, args.encoder,str(time.time()))
    model_state_file = './models/' + model_name+ ".pt"
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # 2) 静态属性图: e-w-graph 作为语义锚点，供静态约束分支使用。
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None


    # 3) 构建 DSPN-CL 主模型:
    #    - 实体中心结构演化流: 历史快照 RGCN + GRU
    #    - 关系中心独立交互流: 关系池化 + 时间门控
    #    - 跨流对比学习: use_cl + temperature
    #    - 静态锚点约束: add_static_graph + weight + angle
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        pre_weight = args.pre_weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        pre_type = args.pre_type,
                        use_cl = args.use_cl,
                        temperature = args.temperature,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # 4) 优化器设置。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    mrr_raw, mrr_filter = None, None

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, test_metrics = test(model,
                                args.train_history_len,
                                train_list+valid_list, 
                                test_list, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                all_ans_list_test, 
                                all_ans_list_r_test, 
                                model_state_file, 
                                static_graph, 
                                "test")
        test_row = {
            'dataset': args.dataset,
            'result_split': 'test_final',
            'best_epoch': -1,
            'encoder': args.encoder,
            'decoder': args.decoder,
            'pre_type': args.pre_type,
            'use_static': args.add_static_graph,
            'use_cl': args.use_cl,
            'gpu': args.gpu,
            'datetime': datetime.now(),
            'pre_weight': args.pre_weight,
            'train_len': args.train_history_len,
            'test_len': args.test_history_len,
            'temperature': args.temperature,
            'lr': args.lr,
            'n_hidden': args.n_hidden,
            'weight': args.weight,
            'angle': args.angle,
            'discount': args.discount,
            'model_state_file': model_state_file,
        }
        test_row.update(test_metrics)
        metrics_excel_path = os.path.join(getattr(args, 'log_dir', './result'), 'entity_prediction_metrics.xlsx')
        append_entity_metrics_to_excel(test_row, excel_path=metrics_excel_path)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        his_best = 0
        best_valid_metrics = None
        best_epoch = -1
        # 5) 训练循环: 每个时间快照执行一次时序外推训练。
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_cp = []

            idx = [_ for _ in range(len(train_list))]

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                    tlist = torch.Tensor(list(range(len(input_list)))).cuda()
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                       train_sample_num]
                    tlist = torch.Tensor(list(range(train_sample_num - args.train_history_len, train_sample_num))).cuda()


                # 预处理历史子图: 提供局部实体流与关系流的结构证据输入。
                subgraph_arr = np.load('data/{}/his_graph_for/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                subgraph_arr_inv = np.load('data/{}/his_graph_inv/train_o_r_{}.npy'.format(args.dataset, train_sample_num))
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)   #取出采样子图
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                # que_pair/que_pair_inv 用于 query-aware 掩码，强调“当前待预测实体相关关系”信息。
                que_pair =  e2r(output[0], num_rels)
                que_pair_inv =  e2r(inverse_triples, num_rels)
                # 历史图序列: 对应 paper_mapping 中实体建模层的历史快照输入。
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]

                input_list_inv = np.array([
                    [triple[2], triple[1], triple[0]]  # 交换 head 和 tail，关系加上偏移量     + num_rels
                    for triple in input_list
                ])

                history_glist_inv = [
                    build_sub_graph(num_nodes, num_rels, snap_inv, use_cuda, args.gpu) for snap_inv in input_list_inv
                ]
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda()
                # 正/反向三元组联合训练，提升关系方向鲁棒性。
                for id in range(2):
                    if id % 2 ==0:
                        loss_e, loss_r, loss_cp, loss_cl = model.get_loss(que_pair, subg_snap, train_sample_num, history_glist, triples, static_graph, tlist,input_list,num_nodes, use_cuda)
                    else:
                        loss_e, loss_r, loss_cp, loss_cl = model.get_loss(que_pair_inv, subg_snap_inv, train_sample_num, history_glist_inv, inverse_triples,static_graph, tlist,input_list,num_nodes, use_cuda)

                    # 总损失 = 实体预测 + 静态锚点约束 + 跨流对比一致性。
                    loss = loss_e + loss_cp + loss_cl

                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_cp.append(loss_cp.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                # break
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_cp), best_mrr, model_name))

            # 6) 验证与早停: 以 filtered MRR 选择最佳模型。
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, valid_metrics = test(model,
                                    args.train_history_len,
                                    train_list, 
                                    valid_list, 
                                    num_rels, 
                                    num_nodes, 
                                    use_cuda, 
                                    all_ans_list_valid, 
                                    all_ans_list_r_valid, 
                                    model_state_file, 
                                    static_graph, 
                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs:
                            break
                        if his_best>=5:
                            break
                    else:
                        his_best=0
                        best_mrr = mrr_filter
                        best_valid_metrics = valid_metrics
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            torch.cuda.empty_cache()

        # 当训练轮数过少（如 n_epochs=1）未触发验证保存时，兜底保存当前参数供测试加载。
        if not os.path.exists(model_state_file):
            fallback_epoch = max(args.n_epochs - 1, 0)
            torch.save({'state_dict': model.state_dict(), 'epoch': fallback_epoch}, model_state_file)
            if best_epoch < 0:
                best_epoch = fallback_epoch

        mrr_raw, mrr_filter, test_metrics = test(model,
                            args.train_history_len,
                            train_list+valid_list,
                            test_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            all_ans_list_test, 
                            all_ans_list_r_test, 
                            model_state_file, 
                            static_graph, 
                            mode="test")

        # 训练结束后导出“最终验证(best valid)”与“最终测试(final test)”的实体预测指标到 Excel。
        excel_common = {
            'dataset': args.dataset,
            'encoder': args.encoder,
            'decoder': args.decoder,
            'pre_type': args.pre_type,
            'use_static': args.add_static_graph,
            'use_cl': args.use_cl,
            'gpu': args.gpu,
            'datetime': datetime.now(),
            'pre_weight': args.pre_weight,
            'train_len': args.train_history_len,
            'test_len': args.test_history_len,
            'temperature': args.temperature,
            'lr': args.lr,
            'n_hidden': args.n_hidden,
            'weight': args.weight,
            'angle': args.angle,
            'discount': args.discount,
            'model_state_file': model_state_file,
        }
        if best_valid_metrics is not None:
            valid_row = dict(excel_common)
            valid_row['result_split'] = 'valid_best'
            valid_row['best_epoch'] = best_epoch
            valid_row.update(best_valid_metrics)
            metrics_excel_path = os.path.join(getattr(args, 'log_dir', './result'), 'entity_prediction_metrics.xlsx')
            append_entity_metrics_to_excel(valid_row, excel_path=metrics_excel_path)

        test_row = dict(excel_common)
        test_row['result_split'] = 'test_final'
        test_row['best_epoch'] = best_epoch
        test_row.update(test_metrics)
        metrics_excel_path = os.path.join(getattr(args, 'log_dir', './result'), 'entity_prediction_metrics.xlsx')
        append_entity_metrics_to_excel(test_row, excel_path=metrics_excel_path)

        return mrr_raw, mrr_filter
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogCL')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="GDELT",
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type",  type=str, default="short",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=False,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="the temperature of cl")
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn", # {uvrgcn,kbat,compgcn}
                        help="method of encoder")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", default=True,
                        help="dilate history graph")
    parser.add_argument("--add-pm-pd", action='store_true', default=True,
                        help="是否添加pm_pd")


    args = parser.parse_args()

    # === 设置日志和参数保存目录 ===
    # 采用 checkpoints/YYYY/M/D/H/M 结构，保留完整实验追踪信息。
    now = datetime.now()
    log_dir = os.path.join(
        "checkpoints", 
        str(now.year), 
        str(now.month), 
        str(now.day), 
        str(now.hour), 
        str(now.minute)
    )
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    # 日志文件路径
    log_file = os.path.join(log_dir, "experiment.log")
    args_file = os.path.join(log_dir, "args.log")
    
    # 替换系统输出同时保存到文件和终端
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout  # 错误信息也一起捕获

    # === 结束设置 ===


    # 若未传入命令行参数，则使用项目内置默认实验配置。
    if len(sys.argv) == 1:
        args.dataset = "ICEWS14"  # 数据集名称
        args.train_history_len = 7  # 训练历史长度
        args.test_history_len = 7  # 测试历史长度
        args.dilate_len = 1  # 扩张历史图长度
        args.lr = 0.001  # 学习率
        args.n_layers = 2  # 传播层数
        args.evaluate_every = 1  # 每n轮进行评估
        args.gpu = 0  # GPU ID
        args.n_hidden = 200  # 隐藏单元数量
        args.self_loop = True  # 是否使用自环
        args.decoder = "convtranse"  # 解码器类型
        args.encoder = "uvrgcn"  # 编码器类型
        args.layer_norm = True  # 是否使用层归一化
        args.weight = 0.5  # 静态约束权重
        args.entity_prediction = True  # 是否添加实体预测损失
        args.angle = 10  # 演变速度
        args.discount = 1  # 静态约束的折扣
        args.pre_weight = 0.9  # 实体预测任务的权重
        args.pre_type = "all"  # 预训练类型
        args.add_static_graph = True  # 是否使用静态图信息
        args.temperature = 0.03  # 对比学习的温度
        args.batch_size = 32  # 批量大小
        args.use_cl = True  # 是否使用显示时间对比学习

    print(args)
    args.__dict__["test_history_len"] = args.__dict__["train_history_len"]

    # ======= 把所有参数信息记录到 args.log ========
    with open(args_file, "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    # ==========================================

    run_experiment(args)



