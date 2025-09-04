import numpy as np
import pandas as pd
from scipy import stats


def is_a_better(a, b):
    mean_a, std_a = np.mean(a), np.std(a)
    mean_b, std_b = np.mean(b), np.std(b)
    
    # compare mean results of A and B
    if mean_a > mean_b:
        return True
    # if they are both equal, results with lower stdev wins
    elif mean_a == mean_b and std_a < std_b:
        return True
    
    return False


def raw_win_ratio(results_A, results_B, dataset_names, method=None, debug=False):
    # Calculate the win ratio between two models just mean performance
    # print(np.array(model_A).shape, np.array(model_B).shape)
    assert len(results_A)==len(results_B), 'no. of datasets not equal'
    
    winners = []
    stat_diff = []
    
    for i, (a, b) in enumerate(zip(results_A, results_B)):
        stat_diff.append(dataset_names[i])

        a_is_better = is_a_better(a,b)
        if a_is_better: winners.append(dataset_names[i])

        debug and print(f'a_is_better = {a_is_better}')

        
    return winners, stat_diff


def win_ratio(results_A, results_B, dataset_names, method='welch', debug=False):
    # Calculate the win ratio between two models
    # print(np.array(model_A).shape, np.array(model_B).shape)
    assert len(results_A)==len(results_B), 'no. of datasets not equal'
    
    winners = []
    stat_diff = []
    
    for i, (a, b) in enumerate(zip(results_A, results_B)):
        
        if len(a)<30 or len(b)<30:
            print(f'{dataset_names[i]} is not done 30 times ({len(a)} vs {len(b)})')
            
        if debug:
            mean_a, std_a = np.mean(a), np.std(a)
            mean_b, std_b = np.mean(b), np.std(b)
            print (f'dataset:{dataset_names[i]}; a = {mean_a:.3f}({std_a:.3f}); b = {mean_b:.3f}({std_b:.3f});')
        
        # Use a two-sample t-test to determine if there is a significant difference between the accuracy scores of the two models
        if method == 'welch':
            test, p = stats.ttest_ind(a, b, equal_var=False, random_state=0)
            debug and print('welch pvalue=', p)
        elif method == 'wilcoxon':
            diff = np.array(a) - np.array(b)
            test, p = stats.wilcoxon(a, b) # returns sum of rank (>=0) and pvalue
            debug and print(f'wilcoxon pvalue={p}')
            
        if p < 0.05:
            stat_diff.append(dataset_names[i])
            
            a_is_better = is_a_better(a,b)
            if a_is_better: winners.append(dataset_names[i])
            
            debug and print(f'a_is_better = {a_is_better}')

        

    return winners, stat_diff


def generator_old(results, models, dataset_names, method='welch', debug=False):
    ''' 
    do not use 
    assuming 2D matrix 
    (each row shows results across datasets for each model) e.g. [30_model1_results, 30_model2_results, ...]
    '''
    
    # Generate accuracy scores for each dataset and model
    # Calculate the win ratio between each pair of models and store the results in a matrix
    cols, rows = len(results), len(results)
    
    output = [[0 for i in range(cols)] for j in range(rows)]
    debug_rows = []
    
    for i, results_A in enumerate(results):
        for j, results_B in enumerate(results):
            if i==j:
                debug_rows.append([models[i], models[j], [], [], '-'])
                continue
                
            assert ~np.array_equal(results_A, results_B), f'(i,j)=({i},{j}); results of {models[i]} and {models[j]} should not be same but got np.array_equals = {np.array_equal(results_A, results_B)}'
            
            winners, competitors = list(win_ratio(results_A, results_B, dataset_names, method=method, debug=debug))
            output[i][j] = f"{len(winners)}/{len(competitors)}"
            
            debug_rows.append([models[i], models[j], winners, competitors, output[i][j]])

    df = pd.DataFrame(output, index=models, columns=models)
    debug_df = pd.DataFrame(debug_rows, columns=['model','baseline','good_on','different_on','score'])
    
    return df, debug_df


def get_test_scores(df, method, dataset):
    r = df.query(f'corruption=="{method}" and dataset=={dataset}').test_score.values
    if len(r)<30:
        print(method, dataset, f'has {len(r)} simulations')
        
    return r

def generator(summary_df, df, test_with='welch', debug=False, statistical=True):
    methods = summary_df.columns
    n_methods = methods.shape[0]
    
    output = [[0 for i in range(n_methods)] for j in range(n_methods)]
    debug_rows = []
    

    for i, method in enumerate(methods):
        method_datasets = summary_df.index[(~summary_df[method].isna())].values
        # print(method_datasets)


        for j, baseline in enumerate(methods):
            if method == baseline: 
                debug_rows.append([method, baseline, [], [], '-'])
                continue

            baseline_datasets = summary_df.index[(~summary_df[baseline].isna())].values
            
            # debug and print(method, baseline, baseline_datasets)
            
            common_datasets = np.intersect1d(method_datasets, baseline_datasets)
            debug and print(f'n_datasets for {methods[i]} vs {methods[j]}', common_datasets, len(common_datasets))
            # output[i][j] = len(common_datasets)
            
            # TODO: use one list comprehension to move a loop
            method_results = [get_test_scores(df, method, d) for d in common_datasets]
            baseline_results = [get_test_scores(df, baseline, d) for d in common_datasets]
            
            
            winners, competitors = list(win_ratio(method_results, baseline_results, common_datasets, method=test_with, debug=debug))
            output[i][j] = f"{len(winners)}/{len(competitors)}"
            
            debug_rows.append([method, baseline, winners, competitors, output[i][j]])

    df = pd.DataFrame(output, index=methods, columns=methods)
    debug_df = pd.DataFrame(debug_rows, columns=['model','baseline','good_on','different_on','score'])
    
    return df, debug_df