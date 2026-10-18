import CVPedigreeAnalysis as cvped
from PedigreeDAGAnalysis import pedfile_readin, construct_pedigree_graph, longest_path_length, plot_pedigree_tree
from ModeClassifierMain import MoI_classification
from PedigreeDataGeneration import VarTableBackpadding
from SegScoreMain import pedigree_segregation_scoring

#TODO integrate arg parser
import cv2
import csv
from pprint import pprint
from os import listdir,makedirs
import pandas as pd
import networkx as nx

#----------------------------------
# COMPILE FAMILY ID LIST
#----------------------------------
'''
Determine which families are available for analysis based on the file name of the available pedigree images

input:
    image directory = ./data/Pedigree_Images/Raw_Images
output:
    FamilyIDs (list): list of family IDs
'''
FamilyIDs = []
avail_ped_images = listdir('data/Pedigree_Images/Raw_Images')
for ped_image in avail_ped_images:
    if ped_image.endswith('.png'):
        FamilyIDs.append(ped_image[:-4])
print('------------------------------------------------\n')



#--------------------------------------
# PROCESS PEDIGREE IMAGES TO PED FILES
#--------------------------------------
'''
Translate relational and phenotype information in pedigree image file to ped files

input: 
    image directory = ./data/Pedigree_Images/Raw_Images
output:
    PED file directory = ./data/PedFiles/Automatic_Pedigrees
'''
print('IMAGE PROCESSING:')
for FamilyID in FamilyIDs:
    print(f' Processing {FamilyID}')
    redacted_img, nodeless_img, line_img = cvped.pedigree_processing(FamilyID)
    #showing relational line image for visual inspection
    cv2.imshow(f'{FamilyID} lines', line_img)
k = cv2.waitKey(0)
cv2.destroyAllWindows
print('------------------------------------------------\n')



#--------------------------------------
# PED FILE IMPORT AND DAG TRANSLATION
#--------------------------------------
'''
Import the PED files constructed through CV processing and translate them to digraph objects
Pedigree graphs stored as digraph objects in pedigree data dictionary for further data analysis

input:
    PED file directory = ./data/PedFiles/Automatic_Pedigrees
ouput:
    ped_data_dict (dict): dictionary with pedigree graph stored under FamilyID entry in 'PedGraph'
        size of pedigree in generations is also stored under FamilyID entry in 'Size'
'''
ped_data_dict = {}
ped_size_counts = {}
for FamilyID in FamilyIDs:
    ped_data_dict[FamilyID] = {}
    family_df = pedfile_readin(f'data/PedFiles/Automatic_Pedigrees/{FamilyID}auto.ped')
    family_dg = construct_pedigree_graph(family_df)
    ped_data_dict[FamilyID]['PedGraph'] = family_dg

    size = longest_path_length(family_dg)+1
    ped_data_dict[FamilyID]['Size'] = size
    
    if size not in ped_size_counts.keys():
        ped_size_counts[size] = 0
    else:
        ped_size_counts[size] += 1
print('Pedigree Size Distribution:')
for size, count in ped_size_counts.items():
    print(f'{size} Generations: {count}')
print('------------------------------------------------\n')



#--------------------------------------
# IMPORT FAMILIAL GENOTYPE DATA
#--------------------------------------
'''
Imports formatted familial genotype for those families where it is available.
Stores this genotype data as a multidimensional dictionary VariantxProbandDict[probandID][geneID][nodeAttribute]:genotype

input:
    familial genotype data = .data/ProbandInfo/GenotypeData.txt
output:
    VariantxProbandDict (dict): dictionary storing all the information from the familial genotype data according to probandID
'''
with open('data/ProbandInfo/GenotypeData.txt', 'r') as gtData:
    ListxProband = ([line.strip().split(' - ') for line in gtData])
    VariantxProbandDict = {}
    for proband_entry in ListxProband:
        proband = proband_entry[0]
        VariantxProbandDict[proband] = {}
        gene_list = proband_entry[1].split(', ')
        for gene_entry in gene_list:
            gene_name, attributes = gene_entry.split('=')
            VariantxProbandDict[proband][gene_name] = {}
            attribute_list = attributes.split(' ')
            for attribute in attribute_list:
                attribute_name, attribute_value = attribute.split(':')
                VariantxProbandDict[proband][gene_name][attribute_name] = int(attribute_value)
        
print('KNOWN VARIANT DATA:')
pprint(VariantxProbandDict)
print('------------------------------------------------\n')



#----------------------------------------------------------------
# ATTACH FAMILIAL GENOTYPE DATA AND GENERATE VARIANT BACKGROUND
#----------------------------------------------------------------
'''
Attach the imported genotype data to the translated pedigree data in the form of a variant table (backfilled with generated variant entries as background)
Variant background between 4 and 9 variants long, chosen at random per pedigree.

input:
    porband list = ./data/ProbandInfo/ProbandList.txt
    ped_data_dict
    VariantxProbandDict
output:
    updated ped_data_dict: now has 'VarTable' and 'Proband' entry for every FamilyID
'''
with open('data/ProbandInfo/ProbandList.txt', 'r') as pl:
    ProbandList = ([line.strip() for line in pl])
print('FamilyID to ProbandID Connection')
for FamilyID in FamilyIDs:
    family_members = ped_data_dict[FamilyID]['PedGraph'].nodes()
    proband_found = False
    for proband in ProbandList:
        if int(proband) in family_members:
            ped_data_dict[FamilyID]['Proband'] = proband
            proband_found = True
            print(f' {FamilyID}: {proband}')
            
            #generate VarTable if we have the known variant data
            if proband in VariantxProbandDict.keys():
                ped_data_dict[FamilyID]['VarTable'] = VarTableBackpadding(PedGraph= ped_data_dict[FamilyID]['PedGraph'],
                                                                          VariantInfoDict= VariantxProbandDict[proband],
                                                                          VariantBackgroundRange= (4,9),
                                                                          alt_freq_range= (2,25))
            
            break
    if not proband_found:
        ped_data_dict[FamilyID]['Proband'] = 'None'
    print(f'{FamilyID}: {ped_data_dict[FamilyID]["Proband"]}')
print('------------------------------------------------\n')



#----------------------------------------------------------------
# MODE OF INHERITANCE CLASSIFICATION
#----------------------------------------------------------------
'''
Classify all available pedigrees based on mode of inheritance using metric-based classifier. 
Optimal thresholds taken to be averages from benchmarking efforts (see ModeClassifierBenchmark.py)

input:
    optimal thresholds file = ./data/MoI_Benchmarking_Results/averaged_thresholds.csv
    ped_data_dict
output:
    updated ped_data_dict: now has 'PredMode' entry for each FamilyID with the predicted mode of inheritance
'''
print('Mode of Inheritance Classifications')
average_weights_df = pd.read_csv('data/MoI_Benchmarking_Results/averaged_thresholds.csv')
moi_thresholds_dict = {
    3:{},
    4:{},
    5:{}
}
for _, row in average_weights_df.iterrows():
    if float(row['auc']) > 0.7:
        moi_thresholds_dict[int(row['size'])][row['metric']] = {
            'threshold':float(row['threshold']),
            'direction':row['direction'],
            'auc':float(row['auc'])
        }
pprint(moi_thresholds_dict)

for FamilyID in FamilyIDs:
    family_dg = ped_data_dict[FamilyID]['PedGraph']
    family_dg_size = ped_data_dict[FamilyID]['Size']
    ped_data_dict[FamilyID]['PredMode'] = MoI_classification(G= family_dg,
                                                            thresholds_dict= moi_thresholds_dict[family_dg_size])
    print(f"{FamilyID} {ped_data_dict[FamilyID]['Proband']}: {ped_data_dict[FamilyID]['PredMode']}")
print('------------------------------------------------\n')



#----------------------
# SEGREGATION SCORING
#----------------------
'''
Performing segregation scoring on pedigrees with available familial genotype data available (backpadded with generated variant entries)
Standard scoring scheme used.
Default weights used given failed attempt at weights optimization.
    w_edge  =   0.6
    w_gen   =   0.2
    w_bet   =   0.2

input:
    ped_data_dict
output:
    sequenced_ped_data_dict (dict): dictionary mirroring the structure ped_data_dict that only contains pedigree data for the pedigrees with attached familial genotype data
        this dictionary also contains segregation scoring results for each pedigree stored under 'Original'
'''
sequenced_ped_data_dict = {}
for FamilyID in FamilyIDs:
    if 'VarTable' in ped_data_dict[FamilyID].keys():
        sequenced_ped_data_dict[FamilyID] = ped_data_dict[FamilyID]

for FamilyID in sequenced_ped_data_dict.keys():
    PedDict = sequenced_ped_data_dict[FamilyID]
    PredMode = PedDict['PredMode'] 
    if PredMode == 'Uncertain':
        PredMode = 'AR'
    Size = PedDict['Size']
    sequenced_ped_data_dict[FamilyID] = pedigree_segregation_scoring(
                                            Ped_Dict= PedDict,
                                            Scoring_Method= 'Original',
                                            Mode = PredMode,
                                            Weights= { #find way to make this automatically import from optimized results
                                                'w_edge':0.6,
                                                'w_gen': 0.2,
                                                'w_bet': 0.2
                                            },
                                            )


#--------------------------------------------
# SEGREGATION SCORING PERFORMANCE ASSESSMENT
#--------------------------------------------
'''
Evaluate the performance of segregation scoring performance compared to ground truth linked variant where familial genotype data is available.
Performance valuated by the fraction of pedigrees where the linked variant is the top scoring variant.

input:
    sequenced_ped_data_dict
ouput:
    N/A
'''
total = 0
correct = 0
incorrect = []
not_scored = []
print('CORRECTLY SCORED PEDIGREES')
for FamilyID in sequenced_ped_data_dict.keys():
    if 'Original' in sequenced_ped_data_dict[FamilyID].keys():
        total += 1
        proband = sequenced_ped_data_dict[FamilyID]['Proband']
        LinkedVariant = list(VariantxProbandDict[proband].keys())[0]
        scores = sequenced_ped_data_dict[FamilyID]['Original']
        if max(scores, key= scores.get) == LinkedVariant:
            correct += 1
            print(f'\t{FamilyID}/{proband}')
        else:
            incorrect.append(FamilyID)
    else:
        not_scored.append(FamilyID)

print('\nINCORRECTLY SCORED PEDIGREES')
for FamilyID in incorrect:
    print(f'\t{FamilyID}/{sequenced_ped_data_dict[FamilyID]["Proband"]}')
print(f'\nSCORING TOP RANK FRACTION: {correct/total}')
print('\nSCORING RESULTS NOT FOUND:')
for FamilyID in not_scored:
    print(f'\t {FamilyID}/{sequenced_ped_data_dict[FamilyID]["Proband"]}')
print('------------------------------------------------\n')