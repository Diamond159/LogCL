import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, RGAT, UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from collections import defaultdict
from scipy.sparse import coo_matrix

class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb, pm_pd, lg):
        # 局部历史图编码器: 在每个时间步更新实体表示。
        if self.encoder_name == "uvrgcn" or self.encoder_name == "kbat" or self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, pm_pd, r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb, pm_pd, lg):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, pm_pd, r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')




class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, 
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1,pre_weight=0.7, discount=0, angle=0, use_static=False, pre_type = 'short', 
                 use_cl= False, temperature=0.007, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, inference_hops=1, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.static_alpha = 1e-5
        # pre_weight 用于局部演化流与全局历史流的融合权重，是双流解耦后的关键控制参数。
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_static = use_static
        # pre_type=all 时启用“局部+全局”双流预测路径。
        self.pre_type = pre_type
        # use_cl 控制跨流对比学习，提升多时间尺度表示一致性。
        self.use_cl = use_cl
        self.temp =temperature
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        # 推理跳数元信息，仅用于记录实验配置，不参与计算图与梯度更新。
        self.inference_hops = int(inference_hops)

        # 查询构造层: 将实体自身表示与关系池化表示拼接为 query-guided mask。
        self.w1 = nn.Linear(self.h_dim*2, self.h_dim)
        # 查询门控与时间编码相关参数。
        
        self.w2 = nn.Linear(self.h_dim, self.h_dim)
        self.w3 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim*2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.w6 = nn.Linear(self.h_dim,self.h_dim)
        self.w7 = nn.Linear(self.h_dim, self.h_dim)
        self.w_cl = nn.Linear(self.h_dim*2, self.h_dim)

        # 时间相位参数: 用于构造演化阶段感知编码 cos(w*t+b)。
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        self.weight_1 = nn.Linear(self.h_dim*2, self.h_dim)
        self.weight_2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        # 关系/实体打分门相关参数，保留扩展空间（当前主要使用 w1/w2/w4/w5/w_cl）。
        self.weight_3 = nn.Linear(self.h_dim, 1)
        self.weight_4 = nn.Linear(self.h_dim, 1)
        self.bias_r = nn.Parameter(torch.zeros(1))

        # 基础关系原型向量，后续在每个时间步通过 time gate 生成动态关系表示 hr。
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        # 动态实体初始表示，将被历史图流递推更新。
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            # 静态属性图分支: 对应 paper_mapping 3.3.3.1。
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)
        self.projection_model = MLPLinear(self.h_dim, self.h_dim)

        # 时间门控参数: 把关系局部统计 x_input 与关系原型 emb_rel 做动态融合。
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)   

        # 预留的预测门控参数，可扩展到更细粒度的多流融合策略。
        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))
        # self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim))
        # nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))                      

        # 实体/关系双通道时间递推单元，对应 DSPN-CL 中结构演化流与关系交互流。
        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell = nn.GRUCell(self.h_dim, self.h_dim)
        # GRU 分别承担实体状态与关系状态的时间递推。

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.decoder_ob1 = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

        self.alpha = 0.5
        self.pi = 3.14159265358979323846
        self.alpha_t = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        self.beta_t = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        self.temporal_w = torch.nn.Parameter(torch.Tensor(self.h_dim * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.alpha_t)
        torch.nn.init.normal_(self.beta_t)
        torch.nn.init.normal_(self.temporal_w)

        self.st_static_emb = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.st_static_emb)
    def get_dynamic_emb(self,t):
        # 趋势项 + 周期项时间编码: 让实体初态具备“阶段位置”与“周期扰动”双信息。
        timevec = self.alpha * self.alpha_t*t + (1-self.alpha) * torch.cos(2 * self.pi * self.beta_t*t)
        attn = torch.cat([self.st_static_emb,timevec],1)
        return torch.mm(attn, self.temporal_w)

    def sparse2th(self, mat, shape):
        value = mat.data
        indices = torch.LongTensor([mat.row, mat.col])
        tensor = torch.sparse.FloatTensor(indices, torch.from_numpy(value).float(), shape)
        return tensor

    def change_edges(self, edges):
        edges_list = []
        node_id_dic = {}

        i = 0
        for line in edges:
            head = line[1]
            tail = line[0]
            rel = line[2]

            if head not in node_id_dic:
                node_id_dic[head] = i
                i = i + 1
            if tail not in node_id_dic:
                node_id_dic[tail] = i
                i = i + 1
            edges_list.append([node_id_dic[head], rel, node_id_dic[tail]])
        edges_list = np.array(edges_list)
        return edges_list

    def cal_pmpd(self, edges, num_nodes):
        use_edges = self.change_edges(edges)
        src, rel, dst = use_edges.transpose()
        coo_rows = []
        coo_cols = []
        coo_data = []

        for index, data in enumerate(src):
            coo_rows.append(data)
            coo_cols.append(index)
            coo_data.append(1)

        for index, data in enumerate(dst):
            coo_rows.append(data)
            coo_cols.append(index)
            coo_data.append(-1)

        coo_rows = np.array(coo_rows)
        coo_cols = np.array(coo_cols)
        coo_data = np.array(coo_data)

        data = coo_matrix((coo_data, (coo_rows, coo_cols)))

        data = self.sparse2th(data, (num_nodes, len(edges)))

        return data

    def forward(self,sub_graph,T_idx, query_mask, g_list, static_graph ,t ,input_list,num_nodes, use_cuda):
        if self.use_static:
            # 1) 静态属性图卷积: 生成静态语义锚点 static_emb。
            static_graph = static_graph.to(self.gpu)
            if self.use_cl:
                dynamic_emb = self.get_dynamic_emb(t)
                static_graph.ndata['h'] = torch.cat((dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            else:
                static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else: 
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None


        # input = [F.normalize(self.get_dynamic_emb(static_emb,t))]
        # self.h = input[-1]

        # #-----------------全局历史建模-------------------------------------        # 改到glist循环内, 让局部和全局历史信息都增加线图
        # self.his_ent, subg_index = self.all_GCN(self.h, sub_graph,use_cuda)     # 全局历史实体嵌入his_ent
        # his_r_emb = F.normalize(self.emb_rel)  # 全局历史关系嵌入his_r_emb
        # his_att = F.softmax(self.w5(query_mask+ self.his_ent),dim=1)
        # his_emb = his_att*self.his_ent
        # his_emb = F.normalize(his_emb)

        # history_embs: 局部实体演化流；his_rel_embs: 局部关系演化流；his_emb/his_r_emb: 全局历史流。
        history_embs = []
        att_embs = []
        his_temp_embs =[]
        his_rel_embs =[]
        if self.pre_type=="all":
            for i, g in enumerate(g_list):
                # 2) 局部历史建模: 每个历史快照构建 pm_pd 与 line graph。
                g_trilist = input_list[i]
                inverse_test_triplets = g_trilist[:, [2, 1, 0]]
                inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels  #
                all_triples = torch.cat((torch.from_numpy(g_trilist), torch.from_numpy(inverse_test_triplets)))
                pm_pd = self.cal_pmpd(all_triples, num_nodes)

                g = g.to(self.gpu)

                lg = g.line_graph(backtracking=False)

                if i == 0:
                    # 3) 全局历史建模: 首步对累计历史做编码，对应 3.3.3.2。
                    self.his_ent, subg_index = self.all_GCN(self.h, sub_graph, use_cuda, pm_pd, lg)  # 全局历史实体嵌入his_ent
                    his_r_emb = F.normalize(self.emb_rel)  # 全局历史关系嵌入his_r_emb
                    his_att = F.softmax(self.w5(query_mask + self.his_ent), dim=1)
                    his_emb = his_att * self.his_ent
                    his_emb = F.normalize(his_emb)

                # t2 表示距离当前时刻的相对距离，值越大表示更早历史。
                t2 = len(g_list)-i+1
                # 4) 相对时间相位注入: 将时间距离编码注入实体状态。
                h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents,1)
                self.h =self.w4(torch.concat([self.h,h_t],dim=1))
                temp_e = self.h[g.r_to_e]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1],:]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean
                # 5) 关系门控更新: 将局部统计(x_input)与关系原型(emb_rel)融合，得到阶段感知关系表示 hr。
                x_input = self.emb_rel + x_input
                current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel], pm_pd, lg)
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                # current_h1 = F.sigmoid(self.w6(current_h))   # 让相应的维度大小早）0~1之间，通过mask矩阵获取query time 出现的实体，其他实体设置为0
                att_e = F.softmax(self.w2(query_mask+current_h),dim=1)

                if i == 0:
                    self.h_0 = self.entity_cell(current_h, self.h)    # 第1层输入
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                    # self.hr = self.relation_cell(x_input, self.emb_rel)    # 第1层输入
                    # self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                else:
                    self.h_0 = self.entity_cell(current_h, self.h_0)  # 第2层输出==下一时刻第一层输入
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                    # self.hr = self.relation_cell(x_input, self.hr)  # 第2层输出==下一时刻第一层输入
                    # self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                # time_weight 是关系流的核心“解耦阀门”: 保留长期先验 or 放大短期波动。
                time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                self.hr = time_weight * x_input + (1-time_weight) * self.emb_rel
                self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                history_embs.append(self.h_0)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h_0)
                self.h = self.h_0
                att_emb = att_e*self.h_0
                att_embs.append(att_emb.unsqueeze(0))
            # 对多时间步局部实体流做查询引导聚合，避免无关历史主导预测。
            att_ent = torch.mean(torch.concat(att_embs,dim=0),dim=0)
            # 6) 跨步聚合: 查询门控后的历史平均 + 最新状态残差。
            att_ent = F.normalize(att_ent)
            history_emb = att_ent+history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            self.hr = None
            history_emb = None

        return history_emb, static_emb, self.hr, his_emb, his_r_emb,his_temp_embs,his_rel_embs,history_embs


    def predict(self,que_pair, tlist, sub_graph,T_id, test_graph, num_rels, static_graph, test_triplets,input_list,num_nodes, use_cuda):
        with torch.no_grad():
            all_triples = test_triplets

            # 查询侧关系池化: 从 (s, r_set) 构造 query mask。
            uniq_e = que_pair[0]
            r_len = que_pair[1]
            r_idx = que_pair[2]
            temp_r = self.emb_rel[r_idx]
            e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean

            query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            e1_emb = self.dynamic_emb[uniq_e]
            rel_emb = e_input[uniq_e] #实体所连的所有关系池化
            query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
            query_mask[uniq_e] = query_emb

            embedding, _, r_emb, his_emb, his_r_emb,_,_,_ = self.forward(sub_graph,T_id, query_mask,test_graph, static_graph, tlist[0],input_list,num_nodes, use_cuda)

            if self.pre_type == "all":
                # 解码阶段融合局部实体流与全局历史流。

                scores_ob, _= self.decoder_ob.forward(embedding, r_emb, all_triples,  his_emb, self.pre_weight, self.pre_type)
                score_seq = F.softmax(scores_ob, dim=1)
                score_en = score_seq
            scores_en = torch.log(score_en)
            return all_triples, scores_en


    def get_loss(self,que_pair, sub_graph,T_idx, glist, triples, static_graph, tlist,input_list,num_nodes, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cp = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        all_triples = triples

        # 查询构造与门控输入准备。
        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        temp_r = self.emb_rel[r_idx]
        e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]:span[1],:]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean

        query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        t1 = torch.tensor(T_idx).cuda().to(self.gpu)
        q_t = torch.cos(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents,1)
        qe_emb = self.w4(torch.concat([self.dynamic_emb,q_t],dim=1))

        e1_emb = qe_emb[uniq_e]

        rel_emb = e_input[uniq_e]
        query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
        query_mask[uniq_e] = query_emb

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs, history_embs = self.forward(sub_graph, T_idx, query_mask, glist, static_graph, tlist[0] ,input_list ,num_nodes , use_cuda)



        if self.pre_type == "all":
            # 双流融合后的实体预测主损失: 局部流 embedding + 全局流 his_emb。
            scores_ob, _= self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb,self.pre_weight, self.pre_type)
            # score_seq = F.softmax(scores_ob, dim=1)
            # score_en = score_seq
            loss_ent += self.loss_e(scores_ob, triples[:, 2])


        # scores_en = torch.log(score_en)


        if self.relation_prediction:
            score_rel = self.rdecoder.forward(embedding,r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_cl and self.pre_type=="all":
            # 跨流交叉对比学习: 对齐“全局历史流”与“局部演化流”，缓解多时间尺度语义混叠。
            for id, evolve_emb in enumerate(his_temp_embs):
                t3 = len(his_temp_embs)-id+1
                # query/query2 分别来自全局视角与局部视角，包含实体与关系联合语义。
                query = torch.concat([self.his_ent[all_triples[:, 0]],his_r_emb[all_triples[:, 1]]],dim=1)
                query2 = torch.concat([evolve_emb[all_triples[:, 0]], his_rel_embs[id][all_triples[:, 1]]],dim=1)
                x1 = self.w_cl(query)
                x2 = self.w_cl(query2)
                loss_cl += self.get_loss_conv(x1, x2)

            for time_step, evolve_emb in enumerate(history_embs):
                # 静态角度约束: 限制表示漂移速度，保证长时序训练稳定性。
                angle = 90 // len(history_embs)
                # step = (self.angle * math.pi / 180) * (time_step + 1)
                step = (self.angle * math.pi / 180) * (time_step + 1)
                if self.layer_norm:
                    sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                else:
                    sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                    c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                    sim_matrix = sim_matrix / c
                mask = (math.cos(step) - sim_matrix) > 0
                loss_cp += self.static_alpha * self.weight * torch.sum(
                    torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_cp, loss_cl

    def all_GCN(self,ent_emb, sub_graph, use_cuda, pm_pd, lg):
        # 全局历史图编码器: 累计历史语义主干，用于与局部流形成双流互补。
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb 
        his_emb = self.his_rgcn_layer.forward(sub_graph, ent_emb, [self.emb_rel, self.emb_rel], pm_pd, lg)
        subg_index = torch.masked_select(
                torch.arange(0, sub_graph.number_of_nodes(), dtype=torch.long).cuda(),
                (sub_graph.in_degrees(range(sub_graph.number_of_nodes())) > 0))
        return F.normalize(his_emb),subg_index
    
    def get_loss_conv(self, ent1_emb, ent2_emb):
        # SimCLR 风格对比损失（双向 + 同视角）:
        # 1) pred1/pred2 对齐跨流同一样本;
        # 2) pred3/pred4 约束各自流内部结构;
        # 3) 通过 temperature 控制对齐难度，避免过硬匹配导致训练不稳。

        loss_fn = nn.CrossEntropyLoss().to(self.gpu)
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.gpu)
        # train_cl_loss =(loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)) / 2
        train_cl_loss =(loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)+loss_fn(pred3 / self.temp, labels) + loss_fn(pred4 / self.temp, labels)) / 4
        return train_cl_loss