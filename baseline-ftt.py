#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helpers.openml_data_v2 import openml_cc18_list, hard_list, get_data1
from helpers.openml_data import tabular_id_list
from helpers.persistence import save_var, load_var

from sklearn.preprocessing import StandardScaler, label_binarize

import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse


# In[2]:


from typing import Any, Dict

import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import delu


# In[3]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# In[4]:


# rtdl.FTTransformer.make_default(
#             n_num_features=30,
#             cat_cardinalities=None,
#             last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
#             d_out=2,
#         ).get_default_transformer_config()


# In[5]:


# rtdl.FTTransformer.make_default(
#             n_num_features=34,
#             cat_cardinalities=None,
#             last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
#             d_out=6,
#         ).get_default_transformer_config()


# In[6]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# Docs: https://yura52.github.io/delu/0.0.4/reference/api/zero.improve_reproducibility.html
delu.random.seed(0)


# In[7]:


[1]*5 + [0]*5


# In[8]:


sklearn.model_selection.train_test_split(
            list(range(10)), test_size=1/8, stratify=[1]*5 + [0]*5,
            random_state=42
        )


# In[9]:


def do_kfold_cv(dataset_id, patience=100, max_epochs=1000, use_pbar = False, parent_pbar = False, key='Doing X'):
    df, target, cats = get_data1(dataset_id)
    cols = df.columns.values
    
    df_num = df[cols[~cats]]
    df_cat = df[cols[cats]]
    
    X_all_num = df_num.values.astype('float32')
    X_all_cat = df_cat.values.astype('int64')
    
    has_num = len(cols[~cats])>0
    has_cat = len(cols[cats])>0
    cat_cardinalities = [df[x].unique().shape[0] for x in cols[cats]]
    # print(cat_cardinalities)
    
    y_all = target.astype('int64')
    
    task_type = 'multiclass'
    if task_type != 'regression':
        y_all = sklearn.preprocessing.LabelEncoder().fit_transform(y_all).astype('int64')
    n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None
    
    
    val_scores = []
    test_scores = []
    y_preds = []
    y_trues = []
    
    index = np.arange(X_all_num.shape[0])

    for fold_index in range(30):
        prefix = f'{key} | fold: {fold_index+1}/30'
        X_num = {}
        X_cat = {}
        y = {}
        
        train_index, test_index = sklearn.model_selection.train_test_split(
            index, test_size=2/10, stratify=y_all, 
            random_state=fold_index
        )
        
        train_index, val_index = sklearn.model_selection.train_test_split(
            train_index, test_size=1/8, stratify=y_all[train_index], 
            random_state=42
        )
        
        
        X_num['train'], X_num['val'], X_num['test'] = X_all_num[train_index], X_all_num[val_index], X_all_num[test_index]
        X_cat['train'], X_cat['val'], X_cat['test'] = X_all_cat[train_index], X_all_cat[val_index], X_all_cat[test_index]
        y['train'], y['val'], y['test'] = y_all[train_index], y_all[val_index], y_all[test_index]
        
        if has_num:
            scaler = StandardScaler()
            scaler.fit(X_num['train'])
            X_num = {
                k: torch.tensor(scaler.transform(v), device=device)
                for k, v in X_num.items()
            }
        else:
            X_num = {
                k: torch.tensor(v, device=device)
                for k, v in X_num.items()
            }
        
        X_cat = {
            k: torch.tensor(v, device=device)
            for k, v in X_cat.items()
        }
        y = {k: torch.tensor(v, device=device) for k, v in y.items()}

        if task_type != 'multiclass':
            y = {k: v.float() for k, v in y.items()}
        
        d_out = n_classes or 1

        model = rtdl.FTTransformer.make_default(
            n_num_features=X_all_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=d_out,
        )
        
        model.to(device)
        optimizer = (
            model.make_default_optimizer()
            if isinstance(model, rtdl.FTTransformer)
            else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        )
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if task_type == 'binclass'
            else F.cross_entropy
            if task_type == 'multiclass'
            else F.mse_loss
        )
        
        def apply_model(x_num, x_cat=None):
            if isinstance(model, rtdl.FTTransformer):
                return model(x_num, x_cat)
            elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
                assert x_cat is None
                return model(x_num)
            else:
                raise NotImplementedError(
                    f'Looks like you are using a custom model: {type(model)}.'
                    ' Then you have to implement this branch first.'
                )


        @torch.no_grad()
        def evaluate(part, final=False):
            model.eval()
            prediction = []
            for x_num_batch, x_cat_batch in delu.iter_batches((X_num[part],X_cat[part]), 128):
                x_num_batch = x_num_batch if has_num else None
                x_cat_batch = x_cat_batch if has_cat else None
                prediction.append(apply_model(x_num_batch, x_cat_batch))
                
            loss = loss_fn(torch.cat(prediction).squeeze(1), y[part])
            prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
            target = y[part].cpu().numpy()
            
            prediction = prediction.argmax(1)
            score = f1_score(target, prediction, average='weighted') 
            
            return (score, target, prediction) if final else (score, loss)
        
        # Create a dataloader for batches of indices
        # Docs: https://yura52.github.io/delu/reference/api/zero.data.IndexLoader.html
        batch_size = 128 #256 #1024
        train_loader = delu.data.IndexLoader(len(X_num['train']), batch_size, device=device)

        # Create a progress tracker for early stopping
        # Docs: https://yura52.github.io/delu/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=patience)
        
        n_epochs = max_epochs
        report_frequency = 10 # len(X['train']) // batch_size // 5

        pbar = parent_pbar 
        best_val =  -1.00
        min_val_loss = np.inf
        val_score = test_score = 0.0
        val_loss = np.inf
        best_model_state = -1
        
        score, y_test, y_pred = 0, 0, 0
        
        for epoch in range(1, n_epochs+1):
            for iteration, batch_idx in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                x_num_batch = X_num['train'][batch_idx]
                x_cat_batch = X_cat['train'][batch_idx] 
                
                x_num_batch = x_num_batch if has_num else None
                x_cat_batch = x_cat_batch if has_cat else None
                
                y_batch = y['train'][batch_idx]
                loss = loss_fn(apply_model(x_num_batch, x_cat_batch).squeeze(1), y_batch)
                loss.backward()
                optimizer.step()
                # if iteration % report_frequency == 0:
                #     # use_pbar and pbar.set_description(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f} | val_score: {val_score:.4f} (best={best_val:.4f})')
                #     use_pbar and pbar.set_description(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f} | val_loss: {val_loss:.4f} (best={min_val_loss:.4f}/{best_val:.4f})')

            val_score, val_loss = evaluate('val')
            # test_score = evaluate('test')
            
            use_pbar and pbar.set_description((
                f'{prefix} | '
             f'(epoch) {epoch} | '
             f'(loss) {loss.item():.4f} | '
             f'val_loss: {val_loss:.4f} '
             f'(best={min_val_loss:.4f}/{best_val:.4f})'
            ))

            # progress.update(1 * val_score)
            progress.update(-1 * val_loss)
            if progress.success:
                best_val = val_score
                min_val_loss = val_loss
                best_model_state = model.state_dict()  
                # if best_val==1:
                #     break
            if progress.fail:
                # print('Best epoch = ', best_epoch)
                break
        
        model.load_state_dict(best_model_state)
        test_score, y_true, y_pred = evaluate('test', final=True)
        # print(test_score)
        
        val_scores.append(best_val)
        test_scores.append(test_score)
        y_preds.append(y_pred)
        y_trues.append(y_test)
        
        del model
        
    return val_scores, test_scores, y_preds, y_trues
        
        


# In[10]:


save_path = './saved_vars/test-ftt.pkl'
dataset_results = load_var(save_path) or {}
# dataset_results = {}


# In[11]:


names = tabular_id_list[10:20] + hard_list

names = sorted(list(set(names)))
# names = [151, 1461] + [3]
# names = [40994, 1480, 11, 37, 54, 50, 31, 1462] # easy 8
names = [40975]

pbar = tqdm(names)
i = 0
for dataset_name in pbar:    
    pbar.set_description(f'preparing dataset={dataset_name}')

    key = f"{dataset_name}_ftt"
    pbar.set_description(f'Doing {key}')

    if key in dataset_results.keys():# or dataset_name in [4134]:
        continue

    try:
        val_scores, test_scores, y_preds, y_trues = do_kfold_cv(
            dataset_name, max_epochs = 1000, patience=9999,
            key = key,
            use_pbar=True, parent_pbar = pbar)
        dataset_results[key] = {
            'val_scores':val_scores, 
            'test_scores': test_scores, 
            'y_preds': y_preds, 
            'y_trues': y_trues
        }
        save_var(dataset_results, save_path)
        
    except Exception as e:
        print(f'Could not do {dataset_name}')
        # print(e)
        raise e
    
    # break


# In[12]:


# dataset_results.keys()
# dataset_results

# y_test
# np.mean(scores[0])
for k in dataset_results.keys():
    print(k, len(dataset_results[k]['test_scores']))


# In[13]:


rows = []
cols = ['dataset', 'corruption', 'fold', 'val_score','test_score']

for k,v in dataset_results.items():
    k = k.split('_')
    
    for i in range(len(v['val_scores'])):
        rows.append([k[0], k[1], i, v['val_scores'][i], v['test_scores'][i]])
                     
df = pd.DataFrame(rows, columns=cols)
df.to_csv('./exports/ftt.csv', index=0)


# In[14]:


df.dataset.unique()


# In[15]:


df.groupby(['dataset','corruption']).test_score.agg('mean')


# In[ ]:




