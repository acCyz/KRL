import logging
import os
import torch.nn.functional as F
import numpy as np
import torch
from scipy import sparse
from collections import defaultdict


class NegSampler:

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, dsn):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation

        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.dsn = dsn.split('/')[1]  # dataset name

    # 该函数返回KG的邻接矩阵
    def _get_adj_mat(self):
        a_mat = sparse.dok_matrix((self.nentity, self.nentity))

        for (h, _, t) in self.triples:
            a_mat[t, h] = 1
            a_mat[h, t] = 1

        a_mat = a_mat.tocsr()
        return a_mat

    def build_k_hop(self, k_hop, dataset_name):
        """
        显式计算k_hop邻域
        """
        if k_hop == 0:
            return None

        save_path = f'cached_matrices\\matrix_{dataset_name}_k{k_hop}_nrw0.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat
        else:
            logging.info(f'begin build k_hop')

        # 论文中的显式计算
        _a_mat = self._get_adj_mat()
        _k_mat = _a_mat ** (k_hop - 1)
        k_mat = _k_mat * _a_mat + _k_mat

        sparse.save_npz(save_path, k_mat)

        return k_mat

    def build_k_rw(self, n_rw, k_hop, dataset_name):

        if n_rw == 0 or k_hop == 0:
            return None

        save_path = f'cached_matrices\\matrix_{dataset_name}_k{k_hop}_nrw{n_rw}.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat
        else:
            logging.info(f'begin build k_rw')

        a_mat = self._get_adj_mat()  # 邻接矩阵
        k_mat = sparse.dok_matrix((self.nentity, self.nentity))  # 初始空K跳矩阵

        randomly_sampled = 0

        for i in range(0, self.nentity):
            neighbors = a_mat[i]
            print(i, self.nentity)
            if len(neighbors.indices) == 0:
                randomly_sampled = randomly_sampled + 1
                walker = np.random.randint(self.nentity, size=n_rw)
                k_mat[i, walker] = 1  # 其实相当于随机采样了
            else:
                for _ in range(0, n_rw):
                    walker = i
                    for _ in range(0, k_hop):
                        idx = np.random.randint(len(neighbors.indices))
                        walker = neighbors.indices[idx]
                        neighbors = a_mat[walker]
                    k_mat[i, walker] += 1
        logging.info(f'randomly_sampled: {randomly_sampled}')
        k_mat = k_mat.tocsr()

        sparse.save_npz(save_path, k_mat)

        return k_mat

    #  -------------------------------------------- nscahing -----------------------------------------
    def get_cache_list(self):
        head_cache = {}  # 头索引
        tail_cache = {}
        head_pos = []  # 头缓存
        tail_pos = []
        head_idx = []
        tail_idx = []
        count_h = 0
        count_t = 0
        """
        zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
        然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
        """
        for h, r, t in self.triples:
            if not (t, r) in head_cache:
                head_cache[(t, r)] = count_h
                head_pos.append([h])
                count_h += 1
            else:
                head_pos[head_cache[(t, r)]].append(h)  # head_pos[head_cache[(t,r)]]表示的是以（t，r）为索引的头实体缓存

            if not (h, r) in tail_cache:
                tail_cache[(h, r)] = count_t  # count_h代表索引尾缓存的个数
                tail_pos.append([t])
                count_t += 1
            else:
                tail_pos[tail_cache[(h, r)]].append(t)

            head_idx.append(head_cache[(t, r)])
            tail_idx.append(tail_cache[(h, r)])

        head_idx = np.array(head_idx, dtype=int)
        tail_idx = np.array(tail_idx, dtype=int)

        return head_idx, tail_idx

    def nsc_sample(self, head, rela, tail, head_idx, tail_idx, cache_size, sample='unif', loss='pair'):

        if sample == 'bern':  # Bernoulli
            n = head_idx.shape[0]  # n表示 头索引(r,t) 的数量,这里是等于训练集非重复三元组的总数
            h_idx = np.random.randint(low=0, high=self.nentity, size=(n, self.negative_sample_size))
            t_idx = np.random.randint(low=0, high=self.nentity, size=(n, self.negative_sample_size))

        elif sample == 'unif':  # NSCaching + uniform    N1：cache_size
            # randint 是一维np数组
            randint = np.random.randint(low=0, high=cache_size,
                                        size=min(cache_size, self.negative_sample_size))

            h_idx = self.head_cache[head_idx, randint]
            t_idx = self.tail_cache[tail_idx, randint]

        h_rand = h_idx
        t_rand = t_idx
        return h_rand, t_rand

    #  -------------------------------------------- nscahing -----------------------------------------
    # 采样——dataset类必须实现
    def sample(self, positive_sample, nsm, neg_args):

        # print("正三元组：",positive_sample)

        head, relation, tail = positive_sample

        negative_sample_list = []
        sample_size = 0
        corrupter_mode = self.mode

        if nsm == 'uniform':
            while sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2).astype(np.int64)
                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                sample_size += negative_sample.size
        elif nsm == 'sans':
            k_hop = neg_args[0]
            n_rw = neg_args[1]

            if n_rw == 0:
                k_neighbors = self.build_k_hop(k_hop, dataset_name=self.dsn)
            else:
                k_neighbors = self.build_k_rw(n_rw=n_rw, k_hop=k_hop, dataset_name=self.dsn)
            k_hop_flag = True
            while sample_size < self.negative_sample_size:
                if k_neighbors is not None and k_hop_flag:
                    if self.mode == 'head-batch':
                        khop = k_neighbors[tail].indices
                    elif self.mode == 'tail-batch':
                        khop = k_neighbors[head].indices
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = khop[np.random.randint(len(khop), size=self.negative_sample_size * 2)].astype(
                        np.int64)
                else:
                    logging.info('**************')
                    negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)

                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True  # 该参数为true，对整个返回的数组里的元素依次取反值
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)

                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                if negative_sample.size == 0:
                    k_hop_flag = False
                sample_size += negative_sample.size
        elif nsm == 'nscaching':
            cache_size = neg_args[0]
            subset_size = neg_args[1]

            self.head_cache = np.load(f'nscache/{self.dsn}_head_{cache_size}n1_{subset_size}n2.npy')
            self.tail_cache = np.load(f'nscache/{self.dsn}_tail_{cache_size}n1_{subset_size}n2.npy')
            while sample_size < self.negative_sample_size:
                h_idx = self.triples.index(positive_sample)
                t_idx = self.triples.index(positive_sample)

                head_idx, tail_idx = self.get_cache_list()

                h_idx = head_idx[h_idx]  # idx
                t_idx = tail_idx[t_idx]  # idx
                h_rand, t_rand = self.nsc_sample(head, relation, tail, h_idx, t_idx, cache_size, 'unif')

                if self.mode == 'head-batch':
                    mask = np.in1d(
                        h_rand,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True  # 该参数为true，对整个返回的数组里的元素依次取反值
                    )
                    negative_sample = h_rand[mask]
                    negative_sample_list.append(negative_sample)  # 在候选头实体序列中取一个负样本
                    sample_size += negative_sample.size
                else:
                    mask = np.in1d(
                        t_rand,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                    negative_sample = t_rand[mask]
                    negative_sample_list.append(negative_sample)  # 在候选尾实体序列中取一个负样本
                    sample_size += negative_sample.size
        else:
            raise ValueError('negative_sampling_mode %s not supported' % nsm)

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        return negative_sample

    @staticmethod
    def count_frequency(triples, start=4):

        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


# NSCaching
class BernCorrupter:
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = self.get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def corrupt(self, head, rela, tail):
        prob = self.bern_prob[rela]
        selection = torch.bernoulli(prob).numpy().astype('int64')
        ent_random = np.random.choice(self.n_ent, len(head))
        head_out = (1 - selection) * head.numpy() + selection * ent_random
        tail_out = selection * tail.numpy() + (1 - selection) * ent_random
        return torch.from_numpy(head_out), torch.from_numpy(tail_out)

    def get_bern_prob(self, data, n_ent, n_rel):
        # 特殊字典，当键值不存在，默认返回空set（），而不是报错
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for h, r, t in data:
            edges[r][h].add(t)  # 嵌套字典
            rev_edges[r][t].add(h)
        bern_prob = torch.zeros(n_rel)  # 每一个关系对应一个概率
        for k in edges.keys():
            right = sum(len(tails) for tails in edges[k].values()) / len(edges[k])
            left = sum(len(heads) for heads in rev_edges[k].values()) / len(rev_edges[k])
            bern_prob[k] = right / (right + left)
        return bern_prob
