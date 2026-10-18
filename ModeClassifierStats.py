import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, f_oneway, ttest_ind, kruskal, mannwhitneyu, chi2_contingency
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os


#------------------------------
# Distribution Normality Check
#------------------------------
def check_normality(groups, alpha=0.05):
    '''
    Checks a given set of distibution for normality based on shapiro testing with a default alpha value of 0.05
    If any of the given distributions fault normality testing, all groups labeled collectively as non-normal

    PARAMETERS:
    -----------
    groups(list[list()]): list of list where the inner list is a set of values to be considered as distribution,
        distributions to be considered for further comparison (should be related)
    alpha(float): desired alpha value for shapiro testing, default = 0.05
    
    RETURN:
    -------
    bool: whether all pedigrees passed normaily by shaprio testing (True if all deemed normal, False if at least one fails)
    '''
    for g in groups:
        if len(g) < 3: #group too small for normality
            return False
        stat, p_value = shapiro(g)
        if p_value < alpha:
            return False
    return True

#----------------------------------------------------------
# Numerical Statistical Testing between Pedigree Groupings
#----------------------------------------------------------
def run_numeric_tests(results_df, value_column, group_column):
    '''
    Tests for statistical testing between pedigrees based on a given grouping category

    PARAMETERS:
    -----------
    results_df(pandas.DataFrame): dataframe containing testing pedigree set results from threshold determination
    value_column(string): the name of the column containing the values to be used in given groupings for comparison
    group_column(string): the name of the column containing flag values for use in grouping values in value_column

    RETURN:
    -------
    Tuple: primary stats test used, statistic value, p-value, posthoc output object (object type depends on type of posthoc run)
    '''
    groups = [results_df.loc[results_df[group_column] == size, value_column].values
                for size in results_df[group_column].unique()]
    
    #if less than two possible grouping levels found in grouping column, no comparison can be made
    if len(groups) < 2:
        return None, None, None, None
    
    normality = check_normality(groups)

    if normality:
        #Parametric Test = ANOVA (difference between pedigree sizes)
        stat, p_value = f_oneway(*groups)
        posthoc = None
        #posthoc in the case the ANOVA is significant, difference between pedigree sizes
        if p_value < 0.05 and len(groups) > 2:
            posthoc = pairwise_tukeyhsd(endog= results_df[value_column], groups= results_df[group_column], alpha=0.05)
        return 'ANOVA', stat, p_value, posthoc

    else:
        #Nonparametric Test = Kruskal-Wallis
        stat, p_value = kruskal(*groups)
        posthoc= None
        if p_value < 0.05 and len(groups) > 2:
            posthoc = sp.posthoc_dunn(results_df, val_col=value_column, group_col=group_column, p_adjust='bonferroni')
        return "Kruskal-Wallis", stat, p_value, posthoc



#------------------------------------------------------------
# Statical Comparison of MOI Classification by Pedigree Size
#------------------------------------------------------------
def run_stats_analysis(results_log_path, generation_counts, output_dir='data/MoI_Benchmarking_Results'):
    '''
    Compare Mode of Inheritance Classification outcomes and performance between pedigree size groupings
    in initial generated pedigree set

    PARAMETERS:
    -----------
    results_log_path(string): path to directory containing benchmark results to use for statistical testing
    generation_counts(list[int]): list of the pedigree sizes to include in grouping comparisons
    output_dir(string): path to directory for desired statistical results and plots

    OUTPUT:
    -------
    Saved to desired output directory:
        - Gen Threshold Results: CSV with threshold values and classification performance per benchmarking trial (one file per pedigree size)
        - Averaged Thresold Results: CSV with thresold values, direction and AUC for each metric for each tested pedigree size (in one file)
        - Post Hoc Results: CSV with details from posthoc analysis if primary comaprison statistic was significant (one per significant comparison)
        - Threshold Bar Plots: png images depicting optimal thresholds by pedigree size
        - Classification Performance Violin Plots: png images depticting classifcation performance by pedigree size using a variety of accuracy metrics

    RETURN:
    -------
    None
    '''

    all_results_df = pd.DataFrame()
    metrics = ['ratio_aff_parent', 'sibling_aff_ratio', 'gen_cov', 'avg_bet_unaff',]

    averaged_thresholds =[]
    for gen_count in generation_counts:
        ss_results_df = pd.read_csv(results_log_path + f'/{gen_count}GenResults/{gen_count}Gen_Threshold_Results.csv')

        for metric in metrics:
           averaged_thresholds.append({
                'size':gen_count,
                'metric':metric,
                'threshold':np.mean(ss_results_df[f'{metric} threshold']),
                'direction':ss_results_df[f'{metric} direction'][0],
                'auc':np.mean(ss_results_df[f'{metric} auc'])
            })
        all_results_df = pd.concat(objs= [all_results_df, ss_results_df], ignore_index=True)


    os.makedirs(output_dir, exist_ok=True)

    #Exporting Averaged Thresholds
    averaged_thresholds_df = pd.DataFrame(averaged_thresholds)
    averaged_thresholds_df.to_csv(f'{output_dir}/averaged_thresholds.csv', index= False)

    
    metrics = ['ratio_aff_parent', 'sibling_aff_ratio', 'gen_cov', 'avg_bet_unaff',] 

    stats_results = []

    #Threshold values
    for metric in metrics:
        test_name, stat, p_value, posthoc = run_numeric_tests(
                                                results_df= all_results_df,
                                                value_column= f'{metric} threshold',
                                                group_column= 'PedigreeSize')
        
        if test_name:
            stats_results.append({
                            'test_type': test_name,
                            'metric': metric,
                            'statistic': stat,
                            'p_value':p_value
                            })
            if posthoc is not None:
                #Tukey's HSD returns a ruslts object with summary attribute
                if hasattr(posthoc, "summary"):
                    metric_posthoc_df = pd.DataFrame(
                                    data=posthoc.summary().data[1:], 
                                    columns=posthoc.summary().data[0]
                                    )
                elif isinstance(posthoc, pd.DataFrame):
                    metric_posthoc_df = posthoc
                else:
                    raise TypeError('Unexpected post hoc results')

                metric_posthoc_df.to_csv(f'{output_dir}/MetricPostHocSummaries/post_hoc_threshold_{metric}.csv')
    

    performance_metrics = ['accuracy', 'certainty', 'precision', 'recall', 'F1']
    for perf_metric in performance_metrics:
        test_name, stat, p_value, posthoc = run_numeric_tests(
                                                results_df= all_results_df,
                                                value_column= perf_metric,
                                                group_column= 'PedigreeSize')
        if test_name:
            stats_results.append({
                'test_type': test_name,
                'metric': perf_metric,
                'statistic': stat,
                'p_value':p_value
                })
            if posthoc is not None:
                #Tukey's HSD returns a ruslts object with summary attribute
                if hasattr(posthoc, "summary"):
                    perf_posthoc_df = pd.DataFrame(
                                    data=posthoc.summary().data[1:], 
                                    columns=posthoc.summary().data[0]
                                    )
                elif isinstance(posthoc, pd.DataFrame):
                    perf_posthoc_df = posthoc
                else:
                    raise TypeError('Unexpected post hoc results')

                perf_posthoc_df.to_csv(f'{output_dir}/PerformancePostHocSummaries/post_hoc_perfomance_{perf_metric}.csv')
    
    stats_results_df = pd.DataFrame(stats_results)
    stats_results_df.to_csv(f'{output_dir}/statistical_tests.csv', index=False)

    # -----------------------------
    # Visualizations
    # -----------------------------
    sns.set(style='whitegrid')
    metric_name_dict = {
        'ratio_aff_parent': 'Affected Parent-Child Pairs', 
        'sibling_aff_ratio': 'Affected Sibling Pairs', 
        'gen_cov': 'Generational Coverage', 
        'avg_bet_unaff': 'Average Betweenness Unaffected', 
        'aff_gen_clustering': 'Affected Generation Clustering'
    }

    perf_name_dict = {
        'accuracy': 'Accuracy', 
        'certainty': 'Certainty', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'F1': 'F1'
    }

    #Threshold value bar plots
    metric_color_series = plt.color_sequences['tab20c']
    color_rotator = 0
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.barplot(data= all_results_df, x= 'PedigreeSize', y= f'{metric} threshold', estimator='mean', errorbar='sd', palette= metric_color_series[color_rotator:])

        plt.title(f'{metric_name_dict[metric]} Threshold Value by Pedigree Size')
        plt.ylabel(f'{metric_name_dict[metric]} Threshold')
        plt.xlabel('Pedigree Size')
        os.makedirs(f'{output_dir}/MetricBarPlots', exist_ok= True)
        plt.savefig(f'{output_dir}/MetricBarPlots/{metric}_theshold_by_size.png')
        plt.close()
        color_rotator += 4

    #Performance value violin plots
    perf_color_series = plt.color_sequences['tab20c'][8:]
    for perf_metric in performance_metrics:
        plt.figure(figsize=(10,6))
        sns.violinplot(data= all_results_df, x= 'PedigreeSize', y= perf_metric, palette=perf_color_series)
        plt.title(f'Mode of Inheritance {perf_name_dict[perf_metric]} by Pedigree Size')
        plt.ylabel(f'{perf_name_dict[perf_metric]}')
        plt.xlabel('Pedigree Size')
        plt.ylim(top=1)
        plt.savefig(f'{output_dir}/PerformanceViolinPlots/{perf_metric}_by_size.png')
        plt.close()
    

    print(f'Analysis complete. Results save to {output_dir}')


#---------------------------------
# Main Statical Analysis Execution
#---------------------------------
if __name__ == "__main__":
    '''
    Run statistical analysis described above on exisiting benchmarking data, grouped by pedigree size (3,4,5)
    '''
    run_stats_analysis(
            results_log_path= 'data/MoI_Benchmarking_Results',
            generation_counts= [3,4,5])
