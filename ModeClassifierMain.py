import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, auc, roc_curve
import numpy as np
from PedigreeDAGAnalysis import generations, aff, construct_pedigree_graph, calc_pedigree_metrics, longest_path_length, aff_child_with_unaff_parents
from PedigreeDataGeneration import pedigree_generator

############### GLOBAL DEFAULT PEDIGREE PARAMETERS ##################
PEDIGREE_COUNT = 500
MAX_CHILDREN = 5
GENERATION_RANGE = (3,5)
ALT_FREQ_RANGE = (5, 25)
BACKPROP_LIKELIHOOD_RANGE = (25, 75)
SPOUSE_LIKELIHOOD_RANGE = (25, 75)
AFFECTED_SPOUSE = True

############### GLOBAL DEFAULT MoI PARAMETERS #####################
ACCURACY_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.75
AUC_THRESHOLD = 0.7


#---------------------------------------
# Classification Metric ROC Analysis
#---------------------------------------
def metric_thresholds_determination(
                pedigree_count= PEDIGREE_COUNT,

                #Pedigree Parameters
                generation_range= GENERATION_RANGE, 
                max_children= MAX_CHILDREN,
                alt_freq_range= ALT_FREQ_RANGE,
                BackpropLikelihoodRange = BACKPROP_LIKELIHOOD_RANGE,
                SpouseLikelihoodRange= SPOUSE_LIKELIHOOD_RANGE,
                AffectedSpouse = AFFECTED_SPOUSE,

                #MoI Classification Parameters
                auc_threshold = AUC_THRESHOLD,

                #Display Parameters
                roc_display = False
                ):
    '''
    Determines optimal the optimial thresholds for a generated 
    pedigree training sample set. Accuracy of classification assessed through
    application of optimized thresholds to a generated pedigree testing
    sample set.
    
    PARAMETERS:
    -----------
    Pedigree Parameters:
    pedigree_count(int): total number of pedgrees for be generated in pedgree sample set (split 8:2 training-testing)

    generation_range((int,int)): duple of integer values indicating the range of generational sizes to be included in generated pedigree set (inclusive),
        randomly chosen per pedgree from the given range

    max_children(int): the maximum number of children that can be generated for each spousal pair in each pedigree in generated sample set

    alt_freq_range((int,int)): duple of integer values indicating the range of alternate allele frequencies (as percentage) to be used in pedigree generation (inclusive),
        ranomly chosen per pedigree from the given range

    BackpropLikelihoodRange((int,int)): duple of integer values indicating the range of backpropigation likelihoods to be used in pedigree generation (inclusive)

    SpouseLikelihoodRange((int,int)): duple of integer values indicating the range of reporductive likelihoods to be used in pedigree generation (inclusive)

    AffectedSpouse(bool): indicating whether non-founder individuals should be be considered as potential founders in pedigree generation,
        recommended this always remain True and affected likelihood be modulated through alternate allele frequency


    ROC Parameter:
    auc_threshold(float): the cutoff in terms of AUC score that deems any particular pedigree metric as sufficiently informative to be used in threhsold determination


    Display Parameter:
    roc_display(bool): option to have multi-metric ROC plot displayed.,
        ROC based on training pedigree set,
        displays all metrics, including those deemed insufficiently informative by AUC threshold (see above)

    
    RETURN:
    -------
    classification_results(list[dict{}]): a list of dictionaries, one dictionary per pedigree in testing pedigree set,
        items in each dict include PedigreeID, TrueMode, PredictedMode (by optimal testing set thresholds), PedigreeSize, and all calculated metric values,
        to be used to assess the performance of classifciation in the testing set

    threshold_results(list[dict{}]): a list of dictionaries, one dictionary per pedigree graph metric trialed in training pedigree set,
        items in each dict= {threshold:(float), 'direction':(string, ['HIGH->AR', 'HIGH-AD']),  'auc':(float)},
        to be used in applying optimized threshold in classification
    '''

    
    #Training/Testing Split
    training_pedigree_set_size = int(0.8*pedigree_count)
    testing_pedigree_set_size = pedigree_count - training_pedigree_set_size

    #TRAINING
    #Generate Training Pedigree Data
    training_trial_metrics_df = pd.DataFrame()
    for pedigree_num in range(training_pedigree_set_size):

        FamilyID = 'Fam' + str(pedigree_num+1)
        true_mode = random.choice(['AD', 'AR'])

        pedigree_df = pedigree_generator(
                                    FamilyID= FamilyID,
                                    mode= true_mode,
                                    max_children= max_children,
                                    generation_range= generation_range,
                                    alt_freq_range= alt_freq_range,
                                    SpouseLikelihoodRange= SpouseLikelihoodRange,
                                    BackpropLikelihoodRange= BackpropLikelihoodRange,
                                    AffectedSpouse= AffectedSpouse)



        pedigree_dg = construct_pedigree_graph(pedigree_df)
        pedigree_metrics = calc_pedigree_metrics(pedigree_dg)
        pedigree_metrics['true_mode'] = true_mode
        pedigree_metrics_df = pd.DataFrame(pedigree_metrics, index= [0])

        training_trial_metrics_df = pd.concat(objs= [training_trial_metrics_df, pedigree_metrics_df], ignore_index=True)

    #List of available metrics for testing
    metric_list = [metric for metric in training_trial_metrics_df.columns.to_list() if metric != 'true_mode']
    
    #Extracting true mode values as separate list and storing indices for each MoI
    Y_true_mode = training_trial_metrics_df['true_mode'].values
    
    AD_indeces, AR_indeces = [], []
    for i in range(len(Y_true_mode)):
        if Y_true_mode[i] == 'AD':
            AD_indeces.append(i)
        elif Y_true_mode[i] == 'AR':
            AR_indeces.append(i)

    #Prepping all-metric ROC figure depending on display options
    if roc_display:
        plt.figure(figsize=(7,7))

    #ROC Analysis for Training Data
    threshold_results = {}
    for metric in metric_list:
        X_metric = training_trial_metrics_df[metric].values

        #Threshold Direction Determination for ROC
        avg_metric_AD = np.mean([X_metric[i] for i in AD_indeces])
        avg_metric_AR = np.mean([X_metric[i] for i in AR_indeces])
        direction = 'HIGH->AD' if avg_metric_AD > avg_metric_AR else 'HIGH->AR'
        positive_label = 'AD' if avg_metric_AD > avg_metric_AR else 'AR'

        #Calculate ROC
        fpr, tpr, thresh = roc_curve(Y_true_mode, X_metric, pos_label=positive_label)

        #Calculate AUC based on ROC
        auc_score = auc(fpr, tpr)

        #Best Threshold Determination from ROC
        Youden_J_values = tpr - fpr # works because tpr and fpr are numpy arrays
        best_index = 0
        best_YJ = 0
        for i in range(len(Youden_J_values)):
            if Youden_J_values[i] > best_YJ:
                best_index = i
        best_threshold = thresh[best_index]
        

        threshold_results[metric] = {
            'threshold': best_threshold,
            'direction': direction,
            'auc': auc_score
        }

        #Plotting ROC depending on display options
        if roc_display:
            plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {auc_score:.2f})')
    
    #Filtering metrics down to those deemed accurate enough on this trial to use on testing data
    selected_metrics = {}
    for metric in threshold_results.keys():
        if threshold_results[metric]['auc'] >= auc_threshold:
            selected_metrics[metric] = threshold_results[metric]
    
    #Displaying ROC plot depending on display paramters
    if roc_display:
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("MOI Classificaiton Metric ROC Curves", fontsize=20)
        plt.legend(loc="lower right", fontsize="small")
        plt.show()


    #TESTING
    classification_results = []
    for pedigree_num in range(testing_pedigree_set_size):
        
        PedigreeID = 'Fam' + str(pedigree_num+1)
        true_mode = random.choice(['AD', 'AR'])

        pedigree_df = pedigree_generator(
                                    FamilyID= PedigreeID,
                                    mode= true_mode,
                                    max_children= max_children,
                                    generation_range= generation_range,
                                    alt_freq_range= alt_freq_range,
                                    SpouseLikelihoodRange= SpouseLikelihoodRange,
                                    BackpropLikelihoodRange= BackpropLikelihoodRange,
                                    AffectedSpouse= AffectedSpouse)
        
        pedigree_dg = construct_pedigree_graph(pedigree_df)
        predicted_mode = MoI_classification(
                                G= pedigree_dg,
                                thresholds_dict= selected_metrics)
        pedigree_metric_values = calc_pedigree_metrics(pedigree_dg)

        pedigree_results = {}
        pedigree_results['PedigreeID'] = PedigreeID
        pedigree_results['TrueMode'] = true_mode
        pedigree_results['PredictedMode'] = predicted_mode
        pedigree_results['PedigreeSize'] = (max(generations(pedigree_dg).values())+1)
        for metric, value in pedigree_metric_values.items():
            pedigree_results[metric] = value

        classification_results.append(pedigree_results)


        
    return classification_results, threshold_results





#---------------------------------------------
# Pedigree Mode of Inheritance Classificaiton
#---------------------------------------------
def MoI_classification(
            G,
            thresholds_dict,
            confidence_threshold= CONFIDENCE_THRESHOLD,
            ) -> str:
    '''
    Classifies a given pedigree (given as DAG object) by mode of inheritence into one of three catagories:
    Autosomal Dominant(AD), Autosomal Recessive(AR), or Uncertain based on confidence voting between multiple pedigree metrics

    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree for classfication

    threshold_results(list[dict{}]): a list of dictionaries, one dictionary per pedigree graph metric trialed in training pedigree set,
        items in each dict= {threshold:(float), 'direction':(string, ['HIGH->AR', 'HIGH-AD']),  'auc':(float)},
        calculated through metric_threshold_determination()

    confidence_threshold(float): fraction of metrics (given as a float) that must vote in favor of a mode for a confident classification to be made,
        if this fraction of total votes is not met, pedigree is classified as 'Uncertain'
    
    '''
    

    AD_votes= 0
    AR_votes= 0
    total= 0
    
    total +=1
    if aff_child_with_unaff_parents(G):
        AR_votes += 2
    else:
        AD_votes += 1

    #threshold-based vote tabulation
    sample_metrics = calc_pedigree_metrics(G)

    for metric, descriptors in thresholds_dict.items():

        threshold = descriptors['threshold']
        direction = descriptors['direction']

        metric_value = sample_metrics[metric]

        total += 1
        if direction == 'HIGH->AD':
            if metric_value >= threshold:
                AD_votes += 1
            else:
                AR_votes += 1
        elif direction == 'HIGH->AR':
            if metric_value >= threshold:
                AR_votes += 1
            else:
                AD_votes += 1

    if total == 0:
        return 'Uncertain'
    elif AD_votes/total >= confidence_threshold:
        return 'AD'
    elif AR_votes/total >= confidence_threshold:
        return 'AR'
    else:
        return 'Uncertain'


if __name__ == "__main__":
    #---------------------------------------------
    # MOI Classification Function Test
    #---------------------------------------------
    classification_results, threshold_results = metric_thresholds_determination(
                                                            pedigree_count= PEDIGREE_COUNT,
                                                            generation_range= GENERATION_RANGE,
                                                            max_children= MAX_CHILDREN,
                                                            alt_freq_range= ALT_FREQ_RANGE,
                                                            BackpropLikelihoodRange = BACKPROP_LIKELIHOOD_RANGE,
                                                            SpouseLikelihoodRange= SPOUSE_LIKELIHOOD_RANGE,
                                                            AffectedSpouse = AFFECTED_SPOUSE,
                                                            #MoI Classification Parameters
                                                            auc_threshold = AUC_THRESHOLD,
                                                            #Display Parameters
                                                            roc_display = True)
    
    #tallying classification results
    total_correct = 0
    total_certain = 0
    test_set_size = int(PEDIGREE_COUNT * 0.2)
    
    for pedigree in classification_results:
        if pedigree['PredictedMode'] != 'Uncertain':
            total_certain += 1
            if pedigree['PredictedMode'] == pedigree['TrueMode']:
                total_correct += 1
    
    print(f'Overall Classification Accuracy: {total_correct/test_set_size}')
    print(f'Number Certain Results: {total_certain} ({(total_certain/test_set_size)*100}%)')
    print(f'Certain Classification Accuracy: {total_correct/total_certain}')