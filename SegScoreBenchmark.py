import pandas as pd
from SegScoreMain import trial_based_segregation_scoring_weight_optimization
from pprint import pprint
import math
import matplotlib.pyplot as plt
from os import makedirs
from scipy.stats import ttest_rel, wilcoxon
# ---------------------------------------------------------------------
# BENCHMARKING CONFIGURATIONS
# ---------------------------------------------------------------------
NUMBER_TRIALS = 40
NUMBER_PEDIGREES = 1000
ACC_METRICS = ['Top1', 'IR', 'DCG']

MODES_OF_INHERITANCE = ['AD', 'AR']
Scoring_Methods = ['Original', 'Extended']
OPTIMIZATION_METHOD = 'Rank'

# ---------------------------------------------------------------------
# TRIAL ACCURACY METRICS
# ---------------------------------------------------------------------
'''
A set of helper functions to calculate accuracy metrics to use in evaluating segregation scoring performance.
Metrics include:
    - Hit at 1: binary outcome stating whether the linked variant is ranked with the highest segregation score (0.0 for false and 1.0 for true)
    - Inverse Rank (IR): continuous variable meant to represent the degree to which a linked variant is ranked (1/linked variant rank)
    - Discounted Cummulative Gain (DCG): another attempt to represent linked variant ranking through continuous variable with a smoother distribution (1/log2(linked variant rank + 1))
'''
##### Hit at 1 #####
def calc_hit_at_1(rank_of_linked):
    '''
    Returns binary if the linked variant was correctly ranked at position 1 or not
    '''
    return 1.0 if rank_of_linked == 1 else 0.0

##### Average Precision #####
def calc_inverse_rank(rank_of_linked):
    '''
    Returns average precision based on the ranking of the linked variant
    '''
    return 1.0 / rank_of_linked

##### Normalized Discounted Cummulative Gain #####
def calc_DCG(rank_of_linked):
    '''
    Returns NDCG based on the ranking of the linked variant
    '''
    return 1.0 / math.log2(rank_of_linked + 1)

##### Accuracy Metrics Calculation Wrapper #####
def compute_acc_metrics(rank_of_linked):
    return {
        'Rank' : rank_of_linked,
        'Top1' : calc_hit_at_1(rank_of_linked),
        'IR' : calc_inverse_rank(rank_of_linked),
        'DCG' : calc_DCG(rank_of_linked)
    }


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # MAIN BENCHMARKING LOOP
    # ---------------------------------------------------------------------
    '''
    Benchmarks segregation scoring performance in both AD and AR generated pedigree sets. 
    Also compares optimized versus unoptimized weighting performance to assess optimization effectivity.
    For each of the two MOIs:
        Generates 1000 pedigree which is split 8:2 testing:training
        Optimized weights determined using training set
        Optimized weights used to score the testing set
        Unoptimized default weights used to score the same testing set (scores stored separately)
        Performance metrics calculated based on testing set scoring performance for both optimized and unoptimized weightings
        Export performance results for both optimized and unoptimized weightings, with optimized weight values as CSVs (one entry per benchmarking trial)
    '''
    accuracy_results = {
        'AD' : [],
        'AR' : []
    }
    weights_results = {
        'AD' : [],
        'AR' : []
    }
    for mode in MODES_OF_INHERITANCE:
        for trial in range(NUMBER_TRIALS):
            print(f'Currently Running: {mode} Trial #{trial+1}')

            #Run weight optimization trail on generated pedigree set as described in SegScoreMain.py
            test_results_dict, optimized_weights = trial_based_segregation_scoring_weight_optimization(
                                                                    pedigree_count= NUMBER_PEDIGREES,
                                                                    Scoring_Method= 'Original',
                                                                    Optimization_Method= OPTIMIZATION_METHOD,
                                                                    Mode= mode,
                                                                    generation_range= (3,4)
            )

            #caclulate and store performance metrics averaged over benchmarking trial
            manual_Top1s = []
            manual_IRs = []
            manual_DCGs = []

            opt_Top1s = []
            opt_IRs = []
            opt_DCGs = []

            for PedigreeID in test_results_dict.keys():
                manual_linked_score = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedScore']
                manual_linked_rank = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedRank']
                manual_linked_margin = test_results_dict[PedigreeID]['Original'][f'UnoptimizedMargin']
                manual_accuracy_metrics = compute_acc_metrics(rank_of_linked= manual_linked_rank)

                manual_Top1s.append(manual_accuracy_metrics['Top1'])
                manual_IRs.append(manual_accuracy_metrics['IR'])
                manual_DCGs.append(manual_accuracy_metrics['DCG'])
                

                opt_linked_score = test_results_dict[PedigreeID]['Original'][f'{OPTIMIZATION_METHOD}LinkedScore']
                opt_linked_rank = test_results_dict[PedigreeID]['Original'][f'{OPTIMIZATION_METHOD}LinkedRank']
                opt_linked_margin = test_results_dict[PedigreeID]['Original'][f'{OPTIMIZATION_METHOD}Margin']
                opt_accuracy_metrics = compute_acc_metrics(rank_of_linked= opt_linked_rank)

                opt_Top1s.append(opt_accuracy_metrics['Top1'])
                opt_IRs.append(opt_accuracy_metrics['IR'])
                opt_DCGs.append(opt_accuracy_metrics['DCG']) 


            weights_results[mode].append({
                'TrialID': trial,

                'EdgeWeight': optimized_weights['w_edge'],
                'GenWeight' : optimized_weights['w_gen'],
                'BetweenessWeight': optimized_weights['w_bet'],

                'AvgManualTop1' : sum(manual_Top1s)/len(manual_Top1s),
                'AvgManualIR': sum(manual_IRs)/len(manual_IRs),
                'AvgManualDCG': sum(manual_DCGs)/len(manual_DCGs),

                'AvgOptTop1' : sum(opt_Top1s)/len(opt_Top1s),
                'AvgOptIR': sum(opt_IRs)/len(opt_IRs),
                'AvgOptDCG': sum(opt_DCGs)/len(opt_DCGs)
                })

        #Export performance stats
        weights_df = pd.DataFrame(weights_results[mode])
        trial_results_dir = 'data/Segregation_Scoring_Trial_Results'
        makedirs(trial_results_dir, exist_ok= True)
        weights_df.to_csv(f'{trial_results_dir}/{mode}_trial_results.csv', index= False)
        print(f'{mode} scoring trial results saved:') 
        print(f'{trial_results_dir}/{mode}_trial_results.csv\n')



