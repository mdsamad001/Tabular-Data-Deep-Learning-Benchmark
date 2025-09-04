#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helpers.persistence import save_var, load_var
from helpers.progress_bar import ProgressBar
from helpers.openml_data_v2 import get_data1, openml_cc18_list, hard_list
from helpers.openml_data import tabular_id_list


# In[2]:


import numpy as np
from tqdm import tqdm
import scipy


# In[3]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked


# In[4]:


from NPT.run import main
from NPT.npt.configs import build_parser


# In[5]:


# npt uses sklearn to do CV and splitting and uses the same random state = 42 and has same test_index
# problem is that somewhere in trainer or deeper the data rows are shuffled and that is the y_preds assertion fails


# In[6]:


# dataset_name = 23

# parser = build_parser()
# args = parser.parse_args([
#     '--data_set', f'custom__{dataset_name}', 
#     '--custom_data_set', f'{dataset_name}', 
#     '--exp_test_perc', '0.2',
#     '--exp_val_perc', '0.1',
#     '--exp_patience', '30',
#     '--exp_n_runs', '1',
#     '--exp_num_total_steps', '10',
#     '--exp_batch_size', '128',
#     # '--exp_disable_cuda',
#     # '--data_set_on_cuda', 'True',
#     '--exp_full_batch_gd',
# ])


# fold_preds, fold_trues = main(args)
# # dataset, _ = main(args)
# # args


# In[7]:


def get_cv_results(test_preds, test_trues):
    
    scores = []
    y_trues = []
    y_preds = []
    
    
    # pbar.add_prefix('starting 10-fold cv')

    for fold_index, (y_test, y_pred) in enumerate(zip(test_preds, test_trues)):    
        
        score = f1_score(y_test, y_pred, average='weighted')
        
        # acc = ((y_test == y_pred).sum() / y_test.shape[0])
        # print('acc', acc)
        
        scores.append(score)
        y_preds.append(y_pred)
        y_trues.append(y_test)
        
    return scores, y_preds, y_trues


# scores, _, _ = get_cv_results(fold_preds, fold_trues)
# scores


# In[8]:


save_path = './saved_vars/test-npt.pkl'
dataset_results = load_var(save_path) or {}

cache_path = './saved_vars/test-npt-outputs.pkl'
outputs = load_var(cache_path) or {}

# dataset_results, outputs = {}, {}


# In[ ]:


# pbar = ProgressBar(data_loaders.items())
pbar = hard_list
pbar = [ 1063,  1510,  1464,   469,   458,  1494,  1068,  1049,    23,
        1050, 40975, 40982,  1067,  1487,  1485,  4134, 40701,  1497,
        1475,  4538]

pbar = [ 458, 1050, 1475, 1485, 1487, 1497, 4134, 4538][::2]

skip_list = [1050, 1487] #outOfMemory error

pbar = [i for i in pbar if i not in [458]]
# pbar = [('breast cancer', data_loaders['breast cancer'])]
failed_list = []

for dataset_name in pbar:
    
    if dataset_name in skip_list:
        continue
    
    try:
        df, y, cat = get_data1(dataset_name)
    except Exception as e:
        print('couldnt do', dataset_name)
        # raise e
        failed_list.append(dataset_name)
        continue
        
    if df.shape[1] > 100:
        continue
        
    batch_size = 128
    # NPT takes max steps as input and calculates epochs from there.
    n_batches = int(np.ceil(df.shape[0]*0.8/batch_size)) if batch_size > 0 else 1
    n_steps = n_batches * 1000
    # print(n_steps)

    parser = build_parser()
    args = parser.parse_args([
        '--data_set', f'custom__{dataset_name}', 
        '--custom_data_set', f'{dataset_name}', 
        '--exp_test_perc', '0.2',
        '--exp_val_perc', '0.1',
        '--exp_patience', '-1',
        '--exp_n_runs', '30',
        '--exp_bootstrap', '30',
        '--exp_num_total_steps', f'{n_steps}',
        '--exp_batch_size', f'{batch_size}',
        # '--exp_disable_cuda',
    ])
    
    print('Trying ', dataset_name)
    
    fold_preds, fold_trues = [], []
    
    if dataset_name in outputs.keys():
        fold_preds, fold_trues = outputs[dataset_name]
        print(f'NPT output for {dataset_name} found, skipping model train & eval')
    else:
        fold_preds, fold_trues = main(args)
        outputs[dataset_name] = fold_preds, fold_trues
        save_var(outputs, cache_path)
    
    try:
        if dataset_name in dataset_results.keys():
            print(f'NPT results for {dataset_name} found, skipping loop')
            continue
            
        scores = get_cv_results(fold_preds, fold_trues)
        dataset_results[dataset_name] = scores
        save_var(dataset_results, save_path)

    except Exception as e:
        print('couldnt do', dataset_name)
        # raise e
        failed_list.append(dataset_name)
    
    # print(scores)


# In[ ]:


failed_list


# In[ ]:


outputs.keys(), dataset_name


# In[ ]:


outputs[dataset_name] = fold_preds, fold_trues
save_var(outputs, cache_path)


# In[ ]:


np.mean(dataset_results[23][0])

# save_var(dataset_results, './saved_vars/expt-MI-1-gcn.pkl')

cols = ['dataset', 
        # 'n_samples', 'n_features', 'n_classes', 
        'similarity', 'binary', 'cutoff', 'fold', 'val_score', 'test_score']
rows = []

for k,v in dataset_results.items():
    common_data = k.replace('mutual_information', "mutual").split('_')
    common_data[-1] = np.round(float(common_data[-1]), 1)
    # print(common_data)
    
    for i in range(10):
        row = common_data + [i, v[0][i], v[1][i]]
        # print(row)
        
        rows.append(row)
        
df = pd.DataFrame(rows, columns=cols)
# df.to_csv('./exports/expt-1-gat-10x2.csv', index=0)
# dataset_results['wine_mutual_information_True_0.0']df['model'] = 'GCN' + df.binary.apply(lambda x: '_binary' if x=="True" else '') + '_' + df.similarity# df.pivot_table(index=['dataset','model','lr','layer1','layer2','layer3'], columns='fold', values='val_score')
best_rows = df.groupby(['dataset','model','fold']).val_score.idxmax()
df.loc[best_rows][['dataset','model','fold','test_score']].to_csv(export_path, index=0)# df.loc[best_rows].groupby('dataset').similarity.value_counts()# df.loc[best_rows][['dataset','model','fold','test_score']].dataset.unique()df.loc[best_rows].pivot_table(columns=['similarity','binary'], index=['dataset','fold'], values='cutoff')
# In[ ]:




