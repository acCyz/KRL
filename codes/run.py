import argparse
import pandas as pd
import json
import os
import streamlit as st
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import KGEModel
from dataloader import TrainDataset


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    # action='store_true' 表示当命令行参数中包含定义的参数名，则将该参数存储为true
    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-nsm', '--negative_sampling_mode', default='uniform', type=str)
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=0.5, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-khop', '--negative_k_hop_sampling', default=0, type=int)
    parser.add_argument('-nrw', '--negative_n_random_walks', default=0, type=int)

    parser.add_argument('-ncs', '--nscaching_cache_size', default=50, type=int, help='N1')
    parser.add_argument('-nss', '--nscaching_subset_size', default=50, type=int, help='N2,random subset size')
    parser.add_argument('-cum', '--cache_update_mode', default='IS', type=str, help='cache_update_mode')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=5, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.max_steps = 100

    return parser.parse_args(args)


def override_config(args):  # 从checkpoint继承训练进度
    """
    Override model and data configuration
    """

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.save_path = argparse_dict['save_path']
    args.model = argparse_dict['model']
    # args.negative_sampling_mode = argparse_dict['negative_sampling_mode']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_filename = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_filename = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    # logging.basicConfig(
    # format='%(asctime)s %(levelname)-8s %(message)s',
    # level=logging.INFO,
    # datefmt='%Y-%m-%d %H:%M:%S',
    # filename=log_filename,
    # filemode='w'
    # )
    # 不能用logging.basicConfig配置，因为import的streamlit库里用过了，import之后再自行logging.basicConfig，会无效
    # 打印日志到流
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)

    # 打印日志到文件
    logfile = logging.FileHandler(log_filename, mode='w')
    logfile.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logfile.setFormatter(formatter)
    logging.getLogger('').addHandler(logfile)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


# 读已经缓存好的模型测试结果
def read_cached_test(cached_path):
    result = {
        'HITS@1': 0,
        'HITS@3': 0,
        'HITS@10': 0,
        'MRR': 0,
        'MR': 0,
    }
    with open(os.path.join(cached_path, 'test.log')) as fin:
        for line in fin:
            if "Test MRR" in line:
                temp = line.strip().split(':')
                result['MRR'] = float(temp[-1])
            if "Test MR" in line:
                temp = line.strip().split(':')
                result['MR'] = float(temp[-1])
            if "Test HITS@1" in line and "HITS@10" not in line:
                temp = line.strip().split(':')
                result['HITS@1'] = float(temp[-1])
            if "Test HITS@3" in line:
                temp = line.strip().split(':')
                result['HITS@3'] = float(temp[-1])
            if "Test HITS@10" in line:
                temp = line.strip().split(':')
                result['HITS@10'] = float(temp[-1])

    return result


def merge_dict(dict_list):
    dic = {}
    for d in dict_list:
        for k, v in d.items():
            dic.setdefault(k, []).append(v)

    return dic


def read_data(data_path):
    all_symbol_triples = []
    entity2id = dict()  # 创建字典
    relation2id = dict()
    train_triples = []
    valid_triples = []
    test_triples = []

    with open(os.path.join(data_path, 'entities.dict'), encoding='utf-8') as fin:
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict'), encoding='utf-8') as fin:
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    with open(os.path.join(data_path, 'train.txt'), encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            train_triples.append((entity2id[h], relation2id[r], entity2id[t]))
            all_symbol_triples.append((h, r, t))

    with open(os.path.join(data_path, 'valid.txt'), encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            valid_triples.append((entity2id[h], relation2id[r], entity2id[t]))
            all_symbol_triples.append((h, r, t))

    with open(os.path.join(data_path, 'test.txt'), encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            test_triples.append((entity2id[h], relation2id[r], entity2id[t]))
            all_symbol_triples.append((h, r, t))

    return entity2id, relation2id, train_triples, valid_triples, test_triples, all_symbol_triples


def run_model(args, task=None, ui=None):
    if task == 'train':
        last_rows1 = ui[0]
        chart1 = ui[1]
        progress_bar1 = ui[2]
        status_text1 = ui[3]
        if args.do_valid:
            valid_chart = ui[4]
        if args.do_test:
            test_table = ui[5]
    else:
        test_table = ui[5]

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console 日志打印配置
    set_logger(args)

    entity2id, relation2id, train_triples, valid_triples, test_triples, all_symbol = read_data(args.data_path)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    # train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    # valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    # test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()  # 将模型加载到GPU上训练

    if args.do_train:
        # Set training dataloader iterator
        if args.negative_sampling_mode == 'nscaching':
            train_dataloader = DataLoader(
                TrainDataset(train_triples,
                             nentity,
                             nrelation,
                             args.negative_sampling_mode,
                             args.negative_sample_size,
                             'nscaching_batch',
                             args.negative_k_hop_sampling,
                             args.negative_n_random_walks,
                             args.nscaching_cache_size,
                             args.nscaching_subset_size,
                             args.cache_update_mode,
                             dsn=args.data_path),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,  # max(1, args.cpu_num//2)
                collate_fn=TrainDataset.collate_fn
            )
            train_dataloader = [train_dataloader]
        else:
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples,
                             nentity,
                             nrelation,
                             args.negative_sampling_mode,
                             args.negative_sample_size,
                             'head-batch',
                             args.negative_k_hop_sampling,
                             args.negative_n_random_walks,
                             args.nscaching_cache_size,
                             args.nscaching_subset_size,
                             args.cache_update_mode,
                             dsn=args.data_path),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,  # max(1, args.cpu_num//2)
                collate_fn=TrainDataset.collate_fn
            )

            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples,
                             nentity,
                             nrelation,
                             args.negative_sampling_mode,
                             args.negative_sample_size,
                             'tail-batch',
                             args.negative_k_hop_sampling,
                             args.negative_n_random_walks,
                             args.nscaching_cache_size,
                             args.nscaching_subset_size,
                             args.cache_update_mode,
                             dsn=args.data_path),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,  # max(1, args.cpu_num//2)
                collate_fn=TrainDataset.collate_fn
            )
            train_dataloader = [train_dataloader_head, train_dataloader_tail]

        # 将dataloader传入训练模型
        kge_model.set_train_dataloader(train_dataloader)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate  # ,weight_decay=self.weight_decay
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    # st.write('Start Training...')
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_sampling_mode = %s' % args.negative_sampling_mode)
    if args.negative_sampling_mode == 'sans':
        logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
        logging.info('negative_k_hop_sampling = %d' % args.negative_k_hop_sampling)
        logging.info('negative_n_random_walks = %d' % args.negative_n_random_walks)
        logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
        if args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, args)

            if task:
                percent = step * 100 // args.max_steps
                progress_bar1.progress(percent)
                status_text1.write('训练进度: %s%% ' % percent)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                if args.negative_sampling_mode == 'nscaching':
                    for i in train_dataloader:
                        i.dataset.save_NSCaching()

            if step % args.log_steps == 0:
                if task:
                    cur_loss = [log['loss']]
                    chart1.add_rows(cur_loss)

                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and step != 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)

                if task:
                    valid_metrics_temp = metrics.copy()
                    valid_metrics_temp.pop('MR')
                    without_MR = pd.DataFrame.from_dict({step: valid_metrics_temp}, orient='index')
                    valid_chart.add_rows(without_MR)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

        if task:
            st.write(metrics)
            table_metrics = {f'{args.save_path.split("/")[1]}': metrics}
            table_metrics = pd.DataFrame.from_dict(table_metrics, orient='index')
            # last_rows1.dataframe(table_metrics)
            st.write('模型测试结果指标如下：')
            test_table.table()
            test_table.add_rows(table_metrics)

    if args.evaluate_train:
        logging.info('Evaluating  on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


def create_cache(triples):
    h_r = {}
    hr_t = {}
    for h, r, t in triples:
        if h not in h_r:
            h_r[h] = []
        h_r[h].append(r)
        hr_t[(h, r)] = t
    return h_r, hr_t


def tran2mean(list, dic):
    result = []
    for eid in list:
        result.append(dic[eid])
    return result


if __name__ == '__main__':
    run_model(parse_args())
