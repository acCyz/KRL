import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TestDataset, BidirectionalOneShotIterator

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.train_dataloader = None
        self.train_iterator = None

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False  # 不需要跟踪梯度计算
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        # 初始化实体向量
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # print("entity_embedding szie:",self.entity_embedding.size())
        # 初始化关系向量
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'TransD']:
            raise ValueError('model %s not supported' % model_name)

    def set_train_dataloader(self, dataloaders):
        self.train_dataloader = dataloaders
        if len(dataloaders) == 1:
            self.train_dataloader[0].dataset.send_embed_for_update(self.entity_embedding,
                                                                   self.relation_embedding)
        self.train_iterator = BidirectionalOneShotIterator(dataloaders)

    def forward(self, sample, mode='single'):
        # torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch。
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            # 返回的tensor和原tensor并不使用相同的内存
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)  # 在第二维增加一个维度

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=sample[:, 0]
                ).view(batch_size, negative_sample_size, -1)

                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)
            else:
                head_t = None
                relation_t = None
                tail_t = None

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)  # view（-1）转换成一维的tensor
            ).view(batch_size, negative_sample_size, -1)  # 再转换成三维的，其中第三维根据前两维确定

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
            else:
                head_t = None
                relation_t = None
                tail_t = None

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,  # dim=0 按行索引； =1按列索引
                index=tail_part.view(-1)  # 转一维
            ).view(batch_size, negative_sample_size, -1)

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)

                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            else:
                head_t = None
                relation_t = None
                tail_t = None

        elif mode == 'nscaching_batch':
            # 返回的向量tensor和原tensor并不使用相同的内存
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)  # 在第二维增加一个维度

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            # print("head向量size：",head.size())

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                raise ValueError('NSCaching only support TransE now')
            else:
                head_t = None
                relation_t = None
                tail_t = None
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, head_t, tail_t, relation_t, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, head_t, tail_t, relation_t, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)  # 求在dim维度求p范数
        return score

    # @staticmethod
    def train_step(self, model, optimizer, args):

        """
        Dropout: 训练过程中，为防止模型过拟合，增加其泛化性，会随机屏蔽掉一些神经元，相当于输入每次走过不同的“模型”。
        测试模式时，所有神经元共同作用，类似于boosting。
        BN: 训练过程中，模型每次处理一个minibatch数据，BN根据一个minibatch来计算mean和std后做归一化处理，
        这也是为什么模型的性能和minibatch的大小关系很大（后续也有系列文章来解决BN在
        小minibatch下表现不佳的问题）。测试时，BN会利用训练时得到的参数来处理测试数据。
        如果不设置model.eval()，输入单张图像，会报错。
        """
        """
         在model.train（）的情况下，模型知道它必须学习层，当我们使用model.eval（）时，
         它指示模型不需要学习任何新的内容，并且模型用于测试
        """
        if (self.train_dataloader is None) or (self.train_iterator is None) :
            raise ValueError('train_iterator haven\'t set in this model')

        model.train()  # 启用Batch Normalization和Drop out，也即模型的训练阶段。

        # 在torch.optim中实现大多数的优化方法，例如RMSProp、Adam、SGD等
        optimizer.zero_grad()  # 梯度清零

        # 从dataset->dataloader中取数据  --- 先通过dataset中接口 __getitem__()取单个样本，
        # 再通过next(iter(dataloader))凑齐返回总数batch_size的批样本
        # 以sans负采样为例：这里返回的positive_sample形状是[batch_size, 3]，
        #                        negative_sample形状是[batch_size, negative_sample_size]，
        # 并且注意train_iterator包含两个迭代器，他们的self.mode不同（nscahing方法则不用管这个self.mode）
        # 所以调用一次train_step，迭代器1返回一批batch_size，下次再轮到迭代器2
        positive_sample, negative_sample, subsampling_weight, mode = next(self.train_iterator)

        # GPU加速
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        if args.negative_sampling_mode == 'nscaching':
            # NSCaching 方法下 ，取样得到的positive_sample,negative_sample形状都是[batch_size,3]

            # 更新缓存
            if len(self.train_dataloader) > 1:
                raise ValueError('NSCaching 方法只支持一个迭代器')
            else:
                self.train_dataloader[0]. dataset. send_embed_for_update(self.entity_embedding,
                                                                       self.relation_embedding)
            '''
            negative_h, negative_t = negative_sample.chunk(2, dim=1)  # 头尾分割变成形状[batch_size,1]
            negative_h, negative_t = negative_h.squeeze(), negative_t.squeeze() # 变成一维形状[batch_size]
            
            #print("负尾：", negative_t)
            #print(negative_t.size())

            positive_h, positive_r, positive_t = positive_sample.chunk(3, dim=1)  # 提取出h,r,t,形状是[batch_size,1]
            positive_h, positive_r, positive_t = positive_h.squeeze(), positive_r.squeeze(), positive_t.squeeze()
            '''
            # 对接forward函数，参数mode='nscaching_batch'
            positive_score = model(positive_sample, mode=mode)
            negative_score = model(negative_sample, mode=mode)

            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # 删除第1维

            positive_sample_loss = -positive_score.mean()  # 所有元素的平均值
            negative_sample_loss = -negative_score.mean()

            # loss = torch.sum(F.relu(args.gamma + positive_score - negative_score))
            loss = (positive_sample_loss + negative_sample_loss) / 2

        else:
            # 对接forward函数
            negative_score = model((positive_sample, negative_sample), mode=mode)

            # 自对抗版本
            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)  # 第一维求平均（每行求一个平均值 ）

            # 对接forward函数，默认参数mode='single'
            positive_score = model(positive_sample)

            # 激活函数
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # 删除第1维

            if args.uni_weight:
                positive_sample_loss = - positive_score.mean()  # 所有元素的平均值
                negative_sample_loss = - negative_score.mean()
            else:
                positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()  # 反向传播

        optimizer.step()  # 更新参数

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        # log = ''
        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()


        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0, 0), #args.cpu_num // 2
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0, 0),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1


                        ranking = 1 + ranking.item()
                        logs.append({
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
