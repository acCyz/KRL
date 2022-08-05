import pandas as pd
import os
import streamlit as st
from PIL import Image
import numpy as np
import torch
import run

from onlysampling import NegSampler


def get_dir_list(path='./models/'):
    dir_list = os.listdir(path)
    return dir_list


def highlight_best(s):  # 表格高亮最佳数据
    """
    highlight the maximum in a Series yellow.
    highlight_max(subset=['HITS@1','HITS@3','HITS@10','MRR'],
                                               color='red',
                                               axis=0)
    """
    if s.name == 'MR':
        is_list = s == s.min()
    else:
        is_list = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_list]


def highlight_min(s):
    """
    highlight the maximum in a Series yellow.
    highlight_max(subset=['HITS@1','HITS@3','HITS@10','MRR'],
                                               color='red',
                                               axis=0)
    """
    is_min = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_min]


st.set_page_config(
    page_title="easy KRL",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# cyz. This is an *extremely* cool app!"
    }
)


info = st.empty()
info1, info2 = info.columns([8, 1])
image = Image.open('./codes/头部背景1.jpeg')
TopHeader = st.container()
with TopHeader:
    st.title('Hi,欢迎使用Easy KRL 🎉🥇')
    st.title('一键图谱嵌入工具')
    st.image(image, use_column_width=True)

tips = st.empty()
with tips.container():
    st.write("📢在这里你可以轻松的调配训练知识表示向量空间,我们准备了以下功能:")
    st.markdown('''
                    - 上传或选择已有数据集进行训练.
                    - 我们实现了TransE作为主要训练模型.
                    - 在TransE中适配了Uniform Sampling, SANS 和 NSCaching 三种Sampling method.
                    - 支持训练实时反馈和负采样分析以及模型指标对比.
                ''')

    col0, col1, col2 = st.columns([1, 1, 1])

    col0.write('🚀__如何开始__')
    col0.markdown('''
                        - &#x1F448 菜单栏选择数据集
                        - 我们已经为你准备了
                        - 📗FB15k-237.
                        - 📗wn18
                        - 📗wn18rr
                        - 📗yago3-10
                        - 或者上传你自己的数据集.
                    ''')
    col0.image("https://static.streamlit.io/examples/cat.jpg")

    col1.write('🚀__负采样分析__')
    col1.markdown('''
                            - 我们实现了以下负采样方法
                            - 🎈Uniform
                            - 🎈SANS 
                            - 🎈NSCaching
                            - 🎈....
                            - 你可以基于不同数据集进行负采样.
                        ''')
    col1.image("https://static.streamlit.io/examples/cat.jpg")

    col2.write("🚀__模型训练和对比__")
    col2.markdown('''
                         - 目前训练模型支持TransE
                         - 你可以配置不同参数训练，并支持以下功能
                         - 🏄实时训练反馈
                         - 🏃多模型指标对比
                         - 🚴实体链接预测
                         - 🏃更多功能有待开发
                            ''')
    col2.image("https://static.streamlit.io/examples/dog.jpg")

# st.image(Image.open('D:\\KRL\\codes\\框架.png'), use_column_width=True)

# st.sidebar.image(image, caption='Sunrise by the mountains', use_column_width=True)
# st.sidebar.title("菜单")

dataset_path = './data/'
models_path = './models/'
dataset_list = get_dir_list(dataset_path)
st.sidebar.header("选择数据集和你要执行的任务")
dataset_list.insert(0, '上传数据集')
dataset = st.sidebar.selectbox("选择数据集", dataset_list)

if dataset == '上传数据集':
    with st.sidebar.expander('一次性上传（批量上传所有文件）', expanded=True):
        file_list = st.file_uploader('请按e2id/r2id/训练集/验证集/测试集顺序选择文件', accept_multiple_files=True)
    with st.sidebar.expander('分步上传（单独上传每一个文件）'):
        e2id = st.file_uploader('上传e2id文件(必须)')
        r2id = st.file_uploader('上传r2id文件(必须)')
        train_data = st.file_uploader('上传训练集(必须)')
        valid_data = st.file_uploader('上传验证集')
        test_data = st.file_uploader('上传测试集')
    st.sidebar.write("")
    dataset_name = st.sidebar.text_input("输入上传的数据集名称")
    upload = st.sidebar.button("确认上传")
    if upload:
        if len(file_list) < 3 or (e2id and r2id and train_data):
            info1.error("未完整上传必须的e2id/r2id/训练集，请重新上传！")
        elif os.path.exists(os.path.join(dataset_path, dataset_name)):
            info1.warning(f'名为{dataset_name}的数据集已存在！')
        else:
            os.makedirs(os.path.join(dataset_path, dataset_name))
            info1.success("上传成功，数据集列表已更新，请重新选择！")
        info2.button('确认')
    st.stop()

entity2id, relation2id, train_triples, valid_triples, test_triples, all_symbol_triples \
    = run.read_data(f'data/{dataset}')

h_r, hr_t = run.create_cache(all_symbol_triples)

menu = st.sidebar.selectbox("选择执行任务", ['None', '上传数据集', '负采样分析', '模型训练', '模型测试', '实体预测'])
if menu == 'None':
    st.stop()
elif menu == '上传数据集':
    with st.sidebar.expander('一次性上传（批量上传所有文件）', expanded=True):
        file_list = st.file_uploader('请按e2id/r2id/训练集/验证集/测试集顺序选择文件', accept_multiple_files=True)
    with st.sidebar.expander('分步上传（单独上传每一个文件）'):
        e2id = st.file_uploader('上传e2id文件(必须)')
        r2id = st.file_uploader('上传r2id文件(必须)')
        train_data = st.file_uploader('上传训练集(必须)')
        valid_data = st.file_uploader('上传验证集')
        test_data = st.file_uploader('上传测试集')
    st.sidebar.write("")
    dataset_name = st.sidebar.text_input("输入上传的数据集名称")
    upload = st.sidebar.button("确认上传")
    if upload:
        if len(file_list) < 3 or (e2id and r2id and train_data):
            info1.error("未完整上传必须的e2id/r2id/训练集，请重新上传！")
        elif os.path.exists(os.path.join(dataset_path, dataset_name)):
            info1.warning(f'名为{dataset_name}的数据集已存在！')
        else:
            os.makedirs(os.path.join(dataset_path, 'data/', dataset_name))
            info1.success("上传成功，数据集列表已更新，请重新选择！")
        info2.button('确认')
    st.stop()
elif menu == '负采样分析':
    # form = st.sidebar.form("submit_form")
    sampling_mode = st.sidebar.multiselect("选择负采样方法", ['Uniform', 'uniform-SANS', 'rw-SANS', 'NSCaching'])
    nsample = st.sidebar.number_input("选择负样本生成数量", 1, 30, value=10)
    useCache = st.sidebar.checkbox("使用缓存库", value=True)
    start_sampling = st.sidebar.button("Go!")
    # sampling_container = st.beta_container()
    with st.expander(f'👉点击展开，配置{[i for i in sampling_mode]}负采样具体参数'):
        if 'Uniform' in sampling_mode:
            param = st.empty()
        if 'uniform-SANS' in sampling_mode:
            st.write("uniform-SANS")
            col1, col2 = st.columns(2)
            with col1:
                usans_k = st.number_input("选择uniform-SANS邻域k", 1, 10, value=3)
            with col2:
                usans_w = st.number_input("选择随机漫步次数w", 0, 0)
        if 'rw-SANS' in sampling_mode:
            st.write("rw-SANS")
            col1, col2 = st.columns(2)
            with col1:
                rsans_k = st.number_input("选择rw-SANS邻域k", 1, 10, value=3)
            with col2:
                rsans_w = st.number_input("选择随机漫步次数w", 0, 5000, value=2000)
        if 'NSCaching' in sampling_mode:
            st.write("NSCaching")
            col1, col2 = st.columns(2)
            with col1:
                nsc_n1 = st.number_input("选择缓存大小N1", 10, max(100, nsample), value=30)
            with col2:
                nsc_n2 = st.number_input("选择更新子集大小N2", 10, 100, value=30)

    st.write('')
    st.write('👇从图谱中选择要生成负样本的三元组：')

    h, r, t, mode = st.columns(4)
    h_list = list(entity2id.keys())
    # h_list = tran2mean(h_list, english)
    h_selected = h.selectbox("选择头实体", h_list)

    r_list = h_r[h_selected]
    r_selected = r.selectbox("选择关系", r_list)

    t_list = hr_t[(h_selected, r_selected)]
    # t_list = tran2mean(t_list, english)
    t_selected = t.selectbox("选择尾实体", [t_list])

    mode = mode.selectbox("选择打碎的位置", ['头实体', '尾实体', '关系'])

    testinfo = st.empty()

    sampler = NegSampler(train_triples, len(entity2id), len(relation2id),
                         nsample, 'head-batch', f'data/{dataset}')
    # 校验参数合法性
    if start_sampling:
        negative_sample_list = {}
        testinfo.info("负采样中........")
        for nsm in sampling_mode:
            arg = []
            nsm_ = ''
            if nsm == 'uniform-SANS':
                k = usans_k
                rw = usans_w
                nsm_ = 'sans'
                arg.append(k)
                arg.append(rw)
            if nsm == 'rw-SANS':
                k = rsans_k
                rw = rsans_w
                nsm_ = 'sans'
                arg.append(k)
                arg.append(rw)
            if nsm == 'Uniform':
                nsm_ = 'uniform'
            if nsm == 'NSCaching':
                nsm_ = 'nscaching'
                arg.append(nsc_n1)
                arg.append(nsc_n2)
                if not os.path.exists(f'nscache/{dataset}_head_{nsc_n1}n1_{nsc_n2}n2.npy'):
                    st.error("由于NSCAching为动态负采样，因此目前只支持从以往训练的模型缓存中展示，您当前选择的NSCaching采样参数找不到缓存结果，请更改参数或先使用该参数进行模型训练")
                    continue
            negative_sample = sampler.sample(
                (entity2id[h_selected], relation2id[r_selected], entity2id[t_selected]),
                nsm_, arg)
            negative_sample = [list(entity2id.keys())[list(entity2id.values()).index(v)] for v in negative_sample]
            negative_sample_list[nsm] = negative_sample

        # st.write(negative_sample_list)
        st.write('👌各负采样方法生成负样本如下：')

        sample_result_table = st.table(negative_sample_list)
        testinfo.success("负采样完成！")
        # 运行结果
        st.balloons()
        x = b'aaaaaaaaa'
        st.download_button(
            label='下载',
            data = x,
            file_name= "a.png"
        )

    st.stop()

elif menu == '模型训练':
    form = st.sidebar.form("submit_form")
    train_args = [['--do_train', '--data_path', f'data/{dataset}']]
    with form:
        train_model = st.selectbox("选择训练模型", ['TransE'])
        train_args.append(['--model', f'{train_model}'])
        with form.expander(f'点击展开配置{train_model}训练参数'):
            epoch = st.number_input("输入epoch", 1000, step=1000, format='%d', value=int(6000))
            lr = st.number_input("输入learn rate", 0.00001, format='%f', value=0.001)
            gamma = st.number_input("输入优化距离", 1.0, 12.0, format='%f', step=1.0)
            batch_size = st.number_input("输入batch_size", 100, 2000, format='%d', value=500)
            train_args.append(['--max_steps', f'{epoch}',
                               '-lr', f'{lr}',
                               '-g', f'{gamma}',
                               '-b', f'{batch_size}'])
            if train_model == 'TransE':
                dim = st.number_input("输入嵌入维度", 100, 1000, format='%d')
                train_args.append(['-d', f'{dim}'])
        form.write("")
        sampling_mode = st.selectbox("选择负采样方法", ['uniform', 'sans', 'nscaching'])
        train_args.append(['-nsm', f'{sampling_mode}'])
        with form.expander(f'点击展开配置{sampling_mode}负采样参数'):
            nsample = st.number_input("选择负样本生成数量", 10, 1024, format='%d')
            train_args.append(['-n', f'{nsample}'])
            if 'uniform' == sampling_mode:
                param = st.empty()
            if 'sans' == sampling_mode:
                sans_k = st.number_input("选择SANS邻域k", 1, 10, format='%d')
                sans_rw = st.number_input("选择SANS随机游走nrw", 0, 5000, format='%d')
                train_args.append(['-khop', f'{sans_k}',
                                   '-nrw', f'{sans_rw}'])
            if 'nscaching' == sampling_mode:
                nsc_n1 = st.number_input("选择缓存大小N1", 10, max(100, nsample), format='%d', value=30)
                nsc_n2 = st.number_input("选择更新子集大小N2", 10, 100, format='%d', value=30)
                train_args.append(['-ncs', f'{nsc_n1}',
                                   '-nss', f'{nsc_n2}'])

        train_args.append(['-save', f'models/{train_model}_{dataset}_{sampling_mode}_test'])

        st.write("")
        valid = st.checkbox("训练同时在验证集上验证", value=True)
        if valid:
            valid_steps = st.number_input("选择验证周期", 100, 1000, format='%d', value=200)
            train_args.append(['--do_valid', '--valid_steps', f'{valid_steps}'])

        test = st.checkbox("训练结束后测试", value=True)
        if test:
            test_size = st.number_input("选择测试时的batch大小", 4, 100, format='%d', value=16)
            train_args.append(['--do_test', '--test_batch_size', f'{test_size}'])

        col1, col2 = st.columns([1, 5])
        start_train = col1.form_submit_button("Go!")
        useGPU = col2.checkbox("use GPU")

    # 校验参数合法性
    train_args = [j for i in train_args for j in i]
    # st.write(train_args)
    if start_train:

        tips.empty()

        # 运行结果
        # st.write(train_args)
        # print(train_args) #parse_args(train_args)
        last_rows = [0.0]  # np.random.randn(1, 1)
        # st.write(last_rows)
        if not valid:
            st.write('训练过程实时loss曲线：[X-epoch / Y-loss]')
            loss_chart = st.line_chart()
            valid_chart = None
        else:
            chart1, chart2 = st.columns([3, 2])
            with chart1:
                st.write('训练过程实时loss曲线：[X-epoch / Y-loss]')
                loss_chart = st.line_chart()
            with chart2:
                st.write('训练过程在验证集上指标变化')
                valid_chart = st.line_chart()
            dic = {0: {'HITS@1': 0.0,
                       'HITS@3': 0.0,
                       'HITS@10': 0.0,
                       'MRR': 0.0}}
            valid_chart.add_rows(pd.DataFrame.from_dict(dic, orient='index'))

        progress_bar = st.progress(0)
        status_text = st.empty()
        test_table = st.empty()

        with st.spinner(f'模型训练中：数据集[{dataset}]  |  翻译模型[{train_model}]  |  负采样算法[{sampling_mode}]'):
            run.run_model(run.parse_args(train_args), 'train',
                      [last_rows, loss_chart, progress_bar, status_text, valid_chart, test_table])
        status_text.write("训练进度:100%")
        st.success(f'训练完成，模型已保存至: models/{train_model}_{dataset}_{sampling_mode}_test')
        st.balloons()
    st.stop()


elif menu == '模型测试':
    test_args = [['--do_test']]

    isall_model = st.sidebar.checkbox("显示全部模型（不限数据集）")
    model_list = get_dir_list(models_path)
    if not isall_model:
        model_list = [x for x in model_list if dataset in x.split('_')]

    form = st.sidebar.form("submit_form")
    with form:
        # 检查测试集完整性
        test_models = st.multiselect("选择测试模型", model_list)

        test_batch_size = st.number_input("输入测试batch_size", 16, 1000, format='%d')
        test_args.append(['--test_batch_size', f'{test_batch_size}'])

        start_test = st.form_submit_button("Go!")

    test_info = st.empty()
    st.write("👇overall 指标对比结果如下，每项指标的最优值已经高亮：")

    test_result_table = st.dataframe()
    test_result_chart = st.bar_chart()
    # bar_chart组件会报UserWarning: I don't know how to infer vegalite type from 'empty'.  Defaulting to nominal.
    # 因为未开始执行的时候还没有值，表是空的，可以忽略这个警告
    index = ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']

    HITS1_index = st.expander('👉HITS@1对比')
    HITS1_index1, HITS1_index2 = HITS1_index.columns([1, 1])
    HITS1_index_chart = HITS1_index1.bar_chart()
    HITS1_index_table = HITS1_index2.dataframe()

    HTIS3_index = st.expander('👉HITS@3对比')
    HITS3_index1, HITS3_index2 = HTIS3_index.columns([1, 1])
    HITS3_index_chart = HITS3_index1.bar_chart()
    HITS3_index_table = HITS3_index2.dataframe()

    HITS10_index = st.expander('👉HITS@10对比')
    HITS10_index1, HITS10_index2 = HITS10_index.columns([1, 1])
    HITS10_index_chart = HITS10_index1.bar_chart()
    HITS10_index_table = HITS10_index2.dataframe()

    MRR_index = st.expander('👉MRR对比')
    MRR_index1, MRR_index2 = MRR_index.columns([1, 1])
    MRR_index_chart = MRR_index1.bar_chart()
    MRR_index_table = MRR_index2.dataframe()

    MR_index = st.expander('👉MR对比')
    MR_index1, MR_index2 = MR_index.columns([1, 1])
    MR_index_chart = MR_index1.bar_chart()
    MR_index_table = MR_index2.dataframe()

    test_args = [j for i in test_args for j in i]
    # 校验模型完整性
    if start_test:
        tips.empty()

        test_args.append('-init')
        table_metrics_all = pd.DataFrame()
        for model in test_models:
            init_path = os.path.join('models/', model)
            if os.path.exists(os.path.join(init_path, 'test.log')):
                test_info.info(f'模型{model}使用缓存测试结果......')
                test_result = run.read_cached_test(init_path)

                table_metrics = {f'{model}': test_result}
                table_metrics = pd.DataFrame.from_dict(table_metrics, orient='index')
                table_metrics_all = pd.concat([table_metrics_all, table_metrics])
                # table_metrics.style.highlight_max(subset=['HITS@10'], color='red', axis=0)

                test_result_temp = test_result.copy()
                test_result_temp.pop('MR')
                without_MR = {f'{model}': test_result_temp}
                chart_metrics = pd.DataFrame.from_dict(without_MR, orient='columns')  # columns

                test_result_table.add_rows(table_metrics)
                test_result_chart.add_rows(chart_metrics)
                # dict([(key, base[key]) for key in subkey])
                HITS1_index_chart.add_rows(pd.DataFrame.from_dict({model: {'hits@1': test_result['HITS@1']}},
                                                                  orient='index'))
                HITS1_index_table.add_rows(pd.DataFrame.from_dict({model: {'HITS@1': test_result['HITS@1']}},
                                                                  orient='index'))
                HITS3_index_chart.add_rows(pd.DataFrame.from_dict({model: {'hits@3': test_result['HITS@3']}},
                                                                  orient='index'))
                HITS3_index_table.add_rows(pd.DataFrame.from_dict({model: {'HITS@3': test_result['HITS@3']}},
                                                                  orient='index'))
                HITS10_index_chart.add_rows(pd.DataFrame.from_dict({model: {'hits@10': test_result['HITS@10']}},
                                                                   orient='index'))
                HITS10_index_table.add_rows(pd.DataFrame.from_dict({model: {'HITS@10': test_result['HITS@10']}},
                                                                   orient='index'))
                MRR_index_chart.add_rows(pd.DataFrame.from_dict({model: {'MRR': test_result['MRR']}},
                                                                orient='index'))
                MRR_index_table.add_rows(pd.DataFrame.from_dict({model: {'MRR': test_result['MRR']}},
                                                                orient='index'))
                MR_index_chart.add_rows(pd.DataFrame.from_dict({model: {'MR': test_result['MR']}},
                                                               orient='index'))
                MR_index_table.add_rows(pd.DataFrame.from_dict({model: {'MR': test_result['MR']}},
                                                               orient='index'))

            else:
                cur_test_args = test_args + [init_path]
                # st.write(cur_test_args)
                test_info.info(f'模型{model}在数据集{model.split("_")[1]}上测试中......')
                run.run_model(run.parse_args(cur_test_args), 'test', [0, 0, 0, 0, 0, test_result_table])

        # table_metrics_all = merge_dict(table_metrics_all)
        # table_metrics_all = pd.DataFrame.from_dict(table_metrics_all, orient='index')
        # st.write(table_metrics_all)
        test_result_table.dataframe(table_metrics_all.style.apply(highlight_best))

        # 运行结果
        test_info.info("所有模型测试完成!")
        st.balloons()
    st.stop()

elif menu == '实体预测':
    model_list = get_dir_list(models_path)
    model_list = [x for x in model_list if dataset in x.split('_')]
    test_model = st.sidebar.multiselect("选择模型部署", model_list)
    predict = st.sidebar.selectbox("选择补全方式", ['关系预测', '头实体预测', '尾实体预测'])
    # useless_predict = st.sidebar.checkbox("")
    start = st.sidebar.button("GO!")
    e_list = list(entity2id.keys())
    r_list = list(relation2id.keys())
    col1, col2 = st.columns(2)
    if predict == '关系预测':
        a = col1.selectbox("头实体", e_list)
        b = col2.selectbox("尾实体", e_list)
    elif predict == '头实体预测':
        a = col1.selectbox("关系", r_list)
        b = col2.selectbox("尾实体", e_list)
    else:
        a = col1.selectbox("头实体", e_list)
        r_list = h_r[a]
        b = col2.selectbox("关系", r_list)

    embed_list = []
    # r_embed_list = []
    for model in test_model:
        e_embed = np.load(os.path.join(models_path, model, 'entity_embedding.npy'))
        r_embed = np.load(os.path.join(models_path, 'models/', model, 'relation_embedding.npy'))
        e_embed = torch.from_numpy(e_embed)
        r_embed = torch.from_numpy(r_embed)
        embed_list.append([e_embed, r_embed])
        # r_embed_list.append(r_embed)
    # st.write(e_embed)
    id2entity = {int(v): k for k, v in entity2id.items()}
    id2relation = {int(v): k for k, v in relation2id.items()}

    if start:
        score_list = []
        result_list = []
        for embed in embed_list:
            if predict == '关系预测':
                score = (embed[0][entity2id[a]] - embed[0][entity2id[b]]) + embed[1]
            elif predict == '头实体预测':
                score = embed[0] + (embed[1][relation2id[a]] - embed[0][entity2id[b]])
            else:
                score = (embed[0][entity2id[a]] + embed[1][relation2id[b]]) - embed[0]
            score = torch.norm(score, p=1, dim=-1)
            topk, index = torch.topk(score, 10, largest=False)
            topk = [x.item() for x in topk]
            if predict == '关系预测':
                result = [id2relation[x.item()] for x in index]
            else:
                result = [id2entity[x.item()] for x in index]
            score_list.append(topk)
            result_list.append(result)

        i = 0
        over_all = {}
        for model in test_model:
            over_all[model] = result_list[i]
            i = i + 1
        st.write("👇各模型的预测前十结果如下，按得分从高到低排序：")
        st.dataframe(over_all)

        i = 0
        for model in test_model:
            with st.expander('👉点击展开 ' + model + ' 详细预测得分情况'):
                # st.write(score_list[i])
                # st.write(result_list[i])
                data = {'预测结果': result_list[i], '得分': score_list[i]}
                st.table(data)

            i = i + 1

        # st.write(score_list, result_list)

        st.stop()
        # st.write(score)
        # st.write(score.size())
        # st.write(score[115],score[116],score[117])
        # st.write(min(score))

# run_model(parse_args())
# streamlit run D:/testcode/KGtest/codes/run.py