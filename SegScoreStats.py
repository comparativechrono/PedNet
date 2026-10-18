import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro
from os import makedirs


#------------------------------------------------
# OPTIMIZATION EFFICACY STATISTICAL COMPARISON
#------------------------------------------------
def run_tests(unoptimized_metric, optimized_metric, label):
    '''
    Compares unoptimized versus optimized segregation scoring results by a given metric.
    Runs appropriate paired mean difference stistical comparison depending on normality between trail-paired performance metrics
    Checks for normality of mean differences. If normal, runs paired tptest. If not normal, runs wilcoxon rank sum test.

    PARAMETERS:
    -----------
    unoptimized_metric (numpy.array): array of performance metrics (one for each optimization trial) of the unoptimized scoring scheme
    optimized_metric (numpy.array): array of performance metrics (one for each optimization trial) of the optimized scoring scheme
    label (string): the performance metric to be used in comparison (label of the column in the optimization trial output CSV)

    RETURN:
    -------
    stats test results (dict): dictionary of the results of the statistical tests run including normality shaprio testing and mean difference comparison
    '''
    diff = optimized_metric - unoptimized_metric
    
    shapiro_stat, shapiro_p = shapiro(diff)
    normal = shapiro_p > 0.05

    if normal:
        stat, p_value = ttest_rel(optimized_metric, unoptimized_metric)
        test_used = 'Paired t-test'
    else:
        stat, p_value = wilcoxon(optimized_metric, unoptimized_metric)
        test_used = 'Wilcoxon Signed-Rank'
    
    return {
        "metric": label,
        "mean_unoptimized": unoptimized_metric.mean(),
        "mean_optimized": optimized_metric.mean(),
        "mean_diff": diff.mean(),
        "shapiro_p": shapiro_p,
        "normality_pass": normal,
        "test_used": test_used,
        "statistic": stat,
        "p_value": p_value
    }

#------------------------------------------------------------------------------
# EXECUTION OF STATISTICAL COMPARISON AND VISUALIZATION OF PERFORMANCE METRICS
#------------------------------------------------------------------------------
def run_stats_analysis(result_log_path, output_dir, mode):
    '''
    Runs trial-paired mean difference comparison between unotpimized and optimized performance for all available performance metrics (Top@1, Inverse Rank, and DCG)
    Also visualizes these comaprisons using violin plots for performance metrics and bar graph for comparison of optimized weights (saved as PNG images)
    Stats testing results saved as CSV

    PARAMETERS:
    -----------
    result_log_path (string): location of optimization trial results CSV to be used in statistical comparison (see SegScoreBenchmark.py)
    output_dir (path): desired directory path for data visualization plots and stats testing results
    mode (string): mode of the pedigrees in the generated pedigree sets used in optimizaqtion trials ['AD','AR']
    '''
    agg_results_df = pd.read_csv(result_log_path)
    
    performance_metrics = ['Top1','IR','DCG']
    
    performance_results = []
    for perf_metric in performance_metrics:
        performance_results.append(run_tests(
                                        unoptimized_metric= agg_results_df[f'AvgManual{perf_metric}'],
                                        optimized_metric= agg_results_df[f'AvgOpt{perf_metric}'],
                                        label= perf_metric))
    
    makedirs(output_dir, exist_ok=True)
    performance_results_df = pd.DataFrame(performance_results)
    performance_results_df.to_csv(f'{output_dir}/statistical_tests.csv', index= False)

    average_optimized_weights = {}
    for weight in ['EdgeWeight', 'GenWeight', 'BetweenessWeight']:
        average_optimized_weights[weight] = [np.mean(agg_results_df[weight])]
    pd.DataFrame(average_optimized_weights).to_csv(f'{output_dir}/optimized_weights.csv', index= False)


    # -----------------------------
    # Visualizations
    # -----------------------------
    metric_colors = {
        "IR": "#1f77b4",     # blue
        "DCG": "#2ca02c",   # green
        "Top1": "#d62728"   # red
    }   
    
    avg_edge_weight = np.mean(agg_results_df['EdgeWeight'])
    avg_gen_weight = np.mean(agg_results_df['GenWeight'])
    avg_bet_weight = np.mean(agg_results_df['BetweenessWeight'])
    
    edge_weights = agg_results_df['EdgeWeight']
    gen_weights = agg_results_df['GenWeight']
    bet_weights = agg_results_df['BetweenessWeight']
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)

    sns.barplot(data=[edge_weights,gen_weights,bet_weights], estimator='mean', errorbar='ci', palette=plt.color_sequences['tab20'], legend = False)

    plt.title(f'{mode} Optimized Weights')
    plt.ylim(0,1)
    ax.set_xticklabels(['Edge Consistency', 'Generation Continuity', 'Carrier Betweenness'])
    makedirs(f'{output_dir}/WeightBarPlots', exist_ok= True)
    plt.savefig(f'{output_dir}/WeightBarPlots/WeightsPlot.png')
    plt.close()

    color_seq = plt.color_sequences['tab20'][:2] if mode == 'AD' else plt.color_sequences['tab20b'][:2]
    #paired violin
    for perf_metric in performance_metrics:
        unopt_metric = agg_results_df[f'AvgManual{perf_metric}']
        opt_metric = agg_results_df[f'AvgOpt{perf_metric}']
        
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        #color = metric_colors[perf_metric]

        sns.violinplot(data= [unopt_metric, opt_metric], legend= False, palette=color_seq)

        # Add significance marker
        y_max = np.concatenate((unopt_metric,opt_metric)).max()
        y_sig = y_max*0.02 + y_max
        x1, x2 = 0, 1
        plt.plot([x1, x1, x2, x2], [y_sig, y_sig+0.005, y_sig+0.005, y_sig], color='black')
        
        for _, row in performance_results_df.iterrows():
            if row['metric'] == perf_metric:
                p_value = row['p_value']

        if p_value < 0.001:
            sig_label = "***"
        elif p_value < 0.01:
            sig_label = "**"
        elif p_value < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"
            
        plt.text((x1+x2)/2, y_sig+0.008, sig_label, ha='center', va='bottom', fontsize=12)
        title_metric = 'Inverse Rank' if perf_metric == 'IR' else perf_metric
        plt.title(f'Average {title_metric} Across {mode} Trials')
        ax.set_xticklabels(['Unoptimize', 'Optimized'])
        plt.ylim(top = 1)
        makedirs(f'{output_dir}/ViolinPlots', exist_ok= True)
        plt.savefig(f'{output_dir}/ViolinPlots/{perf_metric}_ViolinPlot.png')
        plt.close()


#---------------------------------
# Main Statical Analysis Execution
#---------------------------------
if __name__ == '__main__':
    '''
    Run statistical analysis described above on existing optimization trial data, separate trials sets by mode of inheritance (AD and AR)
    '''
    for mode in ['AD','AR']:
        run_stats_analysis(
            result_log_path= f'data/Segregation_Scoring_Trial_Results/{mode}_trial_results.csv',
            output_dir= f'data/Segregation_Scoring_Trial_Results/{mode}_results',
            mode= mode
        )

    #make a bar plot of the default weights for comaprison to the optimized weights 
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)

    sns.barplot(data=[0.6,0.2,0.2], palette=plt.color_sequences['tab20'], legend = False)

    plt.title(f'Default Weights')
    ax.set_xticklabels(['Edge Consistency', 'Generation Continuity', 'Carrier Betweenness'])
    plt.ylim(0,1)
    makedirs(f'data/Segregation_Scoring_Trial_Results/', exist_ok= True)
    plt.savefig(f'data/Segregation_Scoring_Trial_Results/DefaultWeightsPlot.png')
    plt.close()


