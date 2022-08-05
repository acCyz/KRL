import logging
import os
import time
import torch.nn.functional as F
import numpy as np
import torch
from scipy import sparse
from collections import defaultdict
from torch.utils.data import Dataset


def time_it(fn):  # 函数装饰器，在fn函数执行前记录时间，执行后记录时间，打印耗时
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        end = time.time()
        logging.info(f'Time: {end - start}')
        return ret

    return wrapper


# 训练集类 继承自父类Dataset
class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, nsm, negative_sample_size,
                 mode, k_hop, n_rw,
                 cache_size, nscaching_subset_size, cache_update_mode,
                 dsn):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation

        self.negative_sampling_mode = nsm
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.dsn = dsn.split('/')[1]

        if self.negative_sampling_mode == 'sans':
            if n_rw == 0:
                self.k_neighbors = self.build_k_hop(k_hop, dataset_name=self.dsn)
            else:
                self.k_neighbors = self.build_k_rw(n_rw=n_rw, k_hop=k_hop, dataset_name=self.dsn)
        elif self.negative_sampling_mode == 'nscaching':
            self.cache_size = cache_size
            self.subset_size = nscaching_subset_size
            self.update = cache_update_mode
            self.corrupter = BernCorrupter(self.triples, self.nentity, self.nrelation)
            self.head_idx, self.tail_idx, self.head_cache, self.tail_cache, \
            self.head_pos, self.tail_pos = self.get_cache_list()
            self.entity_embedding = None
            self.relation_embedding = None

    # 该函数返回KG的邻接矩阵
    def _get_adj_mat(self):
        a_mat = sparse.dok_matrix((self.nentity, self.nentity))
        for (h, _, t) in self.triples:
            a_mat[t, h] = 1
            a_mat[h, t] = 1

        a_mat = a_mat.tocsr()
        return a_mat

    @time_it
    def build_k_hop(self, k_hop, dataset_name):

        if k_hop == 0:
            return None

        save_path = f'cached_matrices\\matrix_{dataset_name}_k{k_hop}_nrw0.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat
        else:
            logging.info(f'begin build k_hop')

        _a_mat = self._get_adj_mat()
        _k_mat = _a_mat ** (k_hop - 1)
        k_mat = _k_mat * _a_mat + _k_mat

        sparse.save_npz(save_path, k_mat)

        return k_mat

    # NSCaching 中更新缓存算法需要用到embedding
    def send_embed_for_update(self, entity_embedding, relation_embedding):
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

    @time_it
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

        a_mat = self._get_adj_mat()
        k_mat = sparse.dok_matrix((self.nentity, self.nentity))

        randomly_sampled = 0

        for i in range(0, self.nentity):
            neighbors = a_mat[i]  # i实体的邻接列表
            print(i, self.nentity)
            if len(neighbors.indices) == 0:
                randomly_sampled += 1
                walker = np.random.randint(self.nentity, size=n_rw)
                k_mat[i, walker] = 1  # 其实相当于随机采样了
            else:
                for _ in range(0, n_rw):
                    walker = i
                    for _ in range(0, k_hop):
                        idx = np.random.randint(len(neighbors.indices))
                        walker = neighbors.indices[idx]
                        neighbors = a_mat[walker]  # 更新下一次的邻居列表
                    k_mat[i, walker] += 1
        logging.info(f'randomly_sampled: {randomly_sampled}')
        k_mat = k_mat.tocsr()

        sparse.save_npz(save_path, k_mat)

        return k_mat

    #  -------------------------------------------- nscahing -----------------------------------------
    def save_NSCaching(self):
        np.save(f'nscache/{self.dsn}_head_{self.cache_size}n1_{self.subset_size}n2', self.head_cache)
        np.save(f'nscache/{self.dsn}_tail_{self.cache_size}n1_{self.subset_size}n2', self.tail_cache)

    def get_cache_list(self):
        head_cache = {}
        tail_cache = {}
        head_pos = []
        tail_pos = []
        head_idx = []
        tail_idx = []
        count_h = 0
        count_t = 0

        for h, r, t in self.triples:
            if not (t, r) in head_cache:
                head_cache[(t, r)] = count_h
                head_pos.append([h])
                count_h += 1
            else:
                head_pos[head_cache[(t, r)]].append(h)

            if not (h, r) in tail_cache:
                tail_cache[(h, r)] = count_t
                tail_pos.append([t])
                count_t += 1
            else:
                tail_pos[tail_cache[(h, r)]].append(t)

            head_idx.append(head_cache[(t, r)])
            tail_idx.append(tail_cache[(h, r)])

        head_idx = np.array(head_idx, dtype=int)
        tail_idx = np.array(tail_idx, dtype=int)

        # 把头尾采样缓存随机初始化，格式是count×负采样数量的array矩阵
        head_cache = np.random.randint(low=0, high=self.nentity, size=(count_h, self.cache_size))
        tail_cache = np.random.randint(low=0, high=self.nentity, size=(count_t, self.cache_size))
        print('head/tail_idx: head/tail_cache', len(head_idx), len(tail_idx), head_cache.shape, tail_cache.shape,
              len(head_pos), len(tail_pos))
        return head_idx, tail_idx, head_cache, tail_cache, head_pos, tail_pos

    def remove_positive(self, remove=True):
        length_h = len(self.head_pos)  # 来自train
        length_t = len(self.tail_pos)  # 来自train
        length = length_h + length_t
        self.count_pos = 0  # 自建

        def head_remove(arr):
            idx = arr[0]
            mark = np.isin(self.head_cache[idx], self.head_pos[idx])
            if remove == True:
                rand = np.random.choice(self.nentity, size=(self.cache_size,), replace=False)
                self.head_cache[idx][mark] = rand[mark]
            self.count_pos += np.sum(mark)

        def tail_remove(arr):
            idx = arr[0]
            mark = np.isin(self.tail_cache[idx], self.tail_pos[idx])
            if remove == True:
                rand = np.random.choice(self.nentity, size=(self.cache_size,), replace=False)
                self.tail_cache[idx][mark] = rand[mark]
            self.count_pos += np.sum(mark)

        head_idx = np.expand_dims(np.array(range(length_h), dtype='int'), 1)
        tail_idx = np.expand_dims(np.array(range(length_t), dtype='int'), 1)

        np.apply_along_axis(head_remove, 1, head_idx)
        np.apply_along_axis(tail_remove, 1, tail_idx)

        return self.count_pos / length

    def prob(self, head, rela, tail):
        with torch.no_grad():
            head_embed = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head.squeeze()
            )

            rela_embed = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=rela.squeeze()
            )

            tail_embed = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail.squeeze()
            )
            '''
            print("head向量：", head_embed)
            print("head向量size：",head_embed.size())
            print("tail向量size：", tail_embed.size())
            print("rela向量size：", rela_embed.size())
            '''

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                raise ValueError('NSCaching only support TransE now')

            score = -torch.norm(tail_embed - head_embed - rela_embed, p=2, dim=-1)
            score = score.view(head.squeeze().size())
            score = score / 2.0

        return F.softmax(score, dim=-1)

    def update_cache(self, head, rela, tail, head_idx, tail_idx):

        head_idx, head_uniq = np.unique(head_idx, return_index=True)
        tail_idx, tail_uniq = np.unique(tail_idx, return_index=True)

        tail_h = tail[head_uniq]
        rela_h = rela[head_uniq]

        rela_t = rela[tail_uniq]
        head_t = head[tail_uniq]

        h_cache = self.head_cache[head_idx]
        t_cache = self.tail_cache[tail_idx]
        h_cand = np.concatenate([h_cache, np.random.choice(self.nentity, (len(head_idx), self.subset_size))], 1)
        t_cand = np.concatenate([t_cache, np.random.choice(self.nentity, (len(tail_idx), self.subset_size))], 1)
        h_cand = torch.from_numpy(h_cand).type(torch.LongTensor)
        t_cand = torch.from_numpy(t_cand).type(torch.LongTensor)

        rela_h = rela_h.unsqueeze(1).expand(-1, self.cache_size + self.subset_size)
        tail_h = tail_h.unsqueeze(1).expand(-1, self.cache_size + self.subset_size)
        head_t = head_t.unsqueeze(1).expand(-1, self.cache_size + self.subset_size)
        rela_t = rela_t.unsqueeze(1).expand(-1, self.cache_size + self.subset_size)

        h_probs = self.prob(h_cand, rela_h, tail_h)
        t_probs = self.prob(head_t, rela_t, t_cand)

        if self.update == 'IS':

            h_new = torch.multinomial(h_probs, self.cache_size, replacement=False)
            t_new = torch.multinomial(t_probs, self.cache_size, replacement=False)
        elif self.update == 'top':
            _, h_new = torch.topk(h_probs, k=self.cache_size, dim=-1)
            _, t_new = torch.topk(t_probs, k=self.cache_size, dim=-1)

        h_idx = torch.arange(0, len(head_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.cache_size)
        t_idx = torch.arange(0, len(tail_idx)).type(torch.LongTensor).unsqueeze(1).expand(-1, self.cache_size)
        h_rep = h_cand[h_idx, h_new]
        t_rep = t_cand[t_idx, t_new]

        self.head_cache[head_idx] = h_rep.cpu().numpy()
        self.tail_cache[tail_idx] = t_rep.cpu().numpy()

    def neg_sample(self, head, rela, tail, head_idx, tail_idx, sample='unif', loss='pair'):

        if sample == 'bern':  # Bernoulli
            n = head_idx.shape[0]  # n表示 头索引(r,t) 的数量,这里应该是等于训练集非重复三元组的总数
            h_idx = np.random.randint(low=0, high=self.nentity, size=(n, self.negative_sample_size))
            t_idx = np.random.randint(low=0, high=self.nentity, size=(n, self.negative_sample_size))

        elif sample == 'unif':  # NSCaching + uniform    N1：cache_size
            # randint含head.shape[0]个元素
            randint = np.random.randint(low=0, high=self.cache_size,
                                        size=min(self.cache_size, self.negative_sample_size))

            h_idx = self.head_cache[head_idx, randint]
            t_idx = self.tail_cache[tail_idx, randint]

        h_rand = h_idx
        t_rand = t_idx
        return h_rand, t_rand

    #  -------------------------------------------- nscahing -----------------------------------------

    #  返回元组数——dataset类必须实现
    def __len__(self):
        return self.len

    # 采样——dataset类必须实现
    def __getitem__(self, idx):
        '''
        由于不同文献采取的loss函数不同，根据loss函数对正样本进行负采样，因此：
            uniform和sans： 正负样本采样的个数比例为 1 : self.negative_sample_size
                           格式为 （正三元组，打碎的一边）
            nscahing ：    正负样本采样个数比例为 1:1
                          格式为 （正三元组，负三元组）
        '''

        positive_sample = self.triples[idx]

        # print("正三元组：",positive_sample)

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # sqrt对tensor里每个分量开根号

        negative_sample_list = []
        negative_sample_size = 0
        corrupter_mode = self.mode

        if self.negative_sampling_mode == 'uniform':
            while negative_sample_size < self.negative_sample_size:
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
                negative_sample_size += negative_sample.size
        elif self.negative_sampling_mode == 'sans':
            k_hop_flag = True
            while negative_sample_size < self.negative_sample_size:
                if self.k_neighbors is not None and k_hop_flag:
                    if self.mode == 'head-batch':
                        khop = self.k_neighbors[tail].indices
                    elif self.mode == 'tail-batch':
                        khop = self.k_neighbors[head].indices
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = khop[np.random.randint(len(khop), size=self.negative_sample_size * 2)].astype(
                        np.int64)
                else:
                    negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)

                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],  # true_head为字典类型，用(relation, tail)索引出正确的tail
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
                if negative_sample.size == 0:
                    k_hop_flag = False
                negative_sample_size += negative_sample.size
        elif self.negative_sampling_mode == 'nscaching':
            h_idx = self.head_idx[idx]
            t_idx = self.tail_idx[idx]
            h_rand, t_rand = self.neg_sample(head, relation, tail, h_idx, t_idx, 'unif')
            # h_rand, t_rand都含self.negative_sample_size个候选
            prob = self.corrupter.bern_prob[relation]
            # print(prob)
            selection = torch.bernoulli(prob).item()
            if selection == 1:  # 使用NSCahing原文的方法来决定，而不是根据self.mode
                mask = np.in1d(
                    h_rand,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = h_rand[mask]
                negative_sample_list.append((negative_sample[0], relation, tail))  # 在候选头实体序列中取一个负样本
            else:
                mask = np.in1d(
                    t_rand,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = t_rand[mask]
                negative_sample_list.append((head, relation, negative_sample[0]))

            if not ('nsmode' == 'bern'):
                with torch.no_grad():
                    self.update_cache(torch.tensor([head]), torch.tensor([relation]),
                                      torch.tensor([tail]), [h_idx], [t_idx])
            '''
            # remove false negative
            if self.args.remove:
                self.remove_positive(self.args.remove)
            '''
        else:
            raise ValueError('negative_sampling_mode %s not supported' % self.negative_sampling_mode)

        if self.negative_sampling_mode == 'nscaching':
            negative_sample = np.concatenate(negative_sample_list).astype(np.int64)
        else:

            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)  # 创建张量
        positive_sample = torch.LongTensor(positive_sample)  # 创建张量

        return positive_sample, negative_sample, subsampling_weight, corrupter_mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)  # 拼接结果多一维
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)  # 拼接后维数和原来一致
        corrupter_mode = data[0][3]
        # 对于同一TrainDataloader，corrupter_mode都是一样的
        return positive_sample, negative_sample, subsample_weight, corrupter_mode

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


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloaders):
        self.num_of_dataloaders = len(dataloaders)
        if self.num_of_dataloaders > 1:
            self.iterator_head = self.one_shot_iterator(dataloaders[0])
            self.iterator_tail = self.one_shot_iterator(dataloaders[1])
        else:
            self.iterator_nscaching = self.one_shot_iterator(dataloaders[0])
        self.step = 0

    def __next__(self):
        if self.num_of_dataloaders > 1:
            self.step += 1
            if self.step % 2 == 0:
                data = next(self.iterator_head)
            else:
                data = next(self.iterator_tail)
        else:
            data = next(self.iterator_nscaching)

        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        此函数可看作数据集迭代器，每调用一次返回下一个数据
        '''
        while True:
            for data in dataloader:
                yield data


# NSCaching
class BernCorrupter:
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = self.get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    # 对传进来的head和tail进行打碎
    def corrupt(self, head, rela, tail):
        prob = self.bern_prob[rela]
        selection = torch.bernoulli(prob).numpy().astype('int64')
        ent_random = np.random.choice(self.n_ent, len(head))
        head_out = (1 - selection) * head.numpy() + selection * ent_random
        tail_out = selection * tail.numpy() + (1 - selection) * ent_random
        return torch.from_numpy(head_out), torch.from_numpy(tail_out)

    def get_bern_prob(self, data, n_ent, n_rel):
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for h, r, t in data:
            edges[r][h].add(t)
            rev_edges[r][t].add(h)
        bern_prob = torch.zeros(n_rel)  # 每一个关系对应一个概率
        for k in edges.keys():
            right = sum(len(tails) for tails in edges[k].values()) / len(edges[k])
            left = sum(len(heads) for heads in rev_edges[k].values()) / len(rev_edges[k])
            bern_prob[k] = right / (right + left)
        return bern_prob
