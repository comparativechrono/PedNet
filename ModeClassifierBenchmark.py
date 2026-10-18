from InheritanceModeAnalysis import metric_thresholds_determination
from pprint import pprint
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# ---------------------------------------------------------------------
# BENCHMARKING CONFIGURATIONS
# ---------------------------------------------------------------------
NUMBER_TRIALS = 40
NUMBER_PEDIGREES = 1000
PEDIGREE_SIZES = [3,4,5]

# ---------------------------------------------------------------------
# PREDICTION EVALUATION
# ---------------------------------------------------------------------
def evaluate_predictions(true_mode_array, predicted_mode_array):
    '''
    Evaluates the classification performance in a labeled pedigreee test set

    PARAMETERS:
    -----------
    true_mode_array(numpy.array): array containing the ground truth classifications of the pedigree test set
    predicted_mode_array(numpy.array): array containing the predicted classifications as dictated by benchmarking trial on same pedigree test set

    RETURN:
    -------
    dictionary{'performance metric':value}: dictionary of performance metrics calculated from benchmarking pedigree test set classification predictions
    '''
    certain_prediction_count = len([prediction for prediction in predicted_mode_array if prediction != 'Uncertain'])

    return {
      #General Accuracy Evaluation
      'accuracy' : accuracy_score(true_mode_array, predicted_mode_array),
      'certainty' : certain_prediction_count / len(predicted_mode_array),
      'precision' : precision_score(true_mode_array, predicted_mode_array, labels = ['AD', 'AR'], average= 'micro'),
      'recall': recall_score(true_mode_array, predicted_mode_array, labels = ['AD', 'AR'], average= 'micro'),
      'F1': f1_score(true_mode_array, predicted_mode_array, labels = ['AD', 'AR'], average= 'micro'),

      #Mode-specific Precision Analysis
      'precision_AD' : precision_score(true_mode_array, predicted_mode_array, labels= ['AD'], average='micro'),
      'precision_AR' : precision_score(true_mode_array, predicted_mode_array, labels= ['AR'], average='micro')
    }


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # MAIN BENCHMARKING LOOP
    # ---------------------------------------------------------------------
    '''
    Loops through given set of pedigree sizes to be tested.
    Nestwed loop goes through given number of pedigree size-specific trials (new generated pedigree set with each trial)
    Record the threshold values and performance of each pedigree size-specifc trial
    '''
    for pedigree_size in PEDIGREE_SIZES:

      trial_classification_results = []
      trial_evaluated_threshold_results = []

      for trial in range(NUMBER_TRIALS):
          classification_results, threshold_results = metric_thresholds_determination(
                                                          pedigree_count= NUMBER_PEDIGREES,
                                                          generation_range= (pedigree_size,pedigree_size))
          classification_results_df = pd.DataFrame(classification_results)
          trial_prediction_evals = evaluate_predictions(
                                        true_mode_array= classification_results_df['TrueMode'],
                                        predicted_mode_array = classification_results_df['PredictedMode']
          )
        
          evaluated_threshold_results = {**{'TrialID':trial, 'PedigreeSize':pedigree_size}, **trial_prediction_evals}
          #expanding threshold descriptor terms from ROC into fields for evaluated threshold result table construction
          for metric, descriptors in threshold_results.items():
            for descriptor in descriptors.keys():
              evaluated_threshold_results[f'{metric} {descriptor}'] = threshold_results[metric][descriptor]
          
          trial_evaluated_threshold_results.append(evaluated_threshold_results)
          
          
          for pedigree_entry in classification_results:
            pedigree_entry['TrialID'] = trial
          trial_classification_results.extend(classification_results)
          


      # ---------------------------------------------------------------------
      # RESULT EXPORT
      # ---------------------------------------------------------------------
      '''
      Export threshold results and performance metrics to CSV with one entry per benchmarking trial (one file per pedigree size level tested)
      '''
      benchmark_dir_path = f'data/MoI_Benchmarking_Results/{pedigree_size}GenResults'
      os.makedirs(benchmark_dir_path, exist_ok=True)

      classification_df = pd.DataFrame(trial_classification_results)
      classification_df.to_csv(benchmark_dir_path + f'/{pedigree_size}Gen_Pedigree_MoIs.csv', index= False)
      print(f'\n{pedigree_size} Generation Test Pedigree MoI Classification benchmark results saved to:')
      print(benchmark_dir_path + f'{pedigree_size}Gen_Pedigree_MoIs.csv')

      threshold_df = pd.DataFrame(trial_evaluated_threshold_results)
      threshold_df.to_csv(benchmark_dir_path + f'/{pedigree_size}Gen_Threshold_Results.csv', index= False)
      print(f'{pedigree_size} Generation Optimal MoI Threshold results saved to:') 
      print(benchmark_dir_path + f'{pedigree_size}Gen_Threshold_Results.csv\n')

