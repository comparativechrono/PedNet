import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize
from PedigreeDAGAnalysis import generations, parents, aff, unaff, founder_influence, longest_path_length
from PedigreeDataGeneration import PedGraph_VarTable_generator
from IPython.display import display

############### GLOBAL DEFAULT PEDIGREE PARAMETERS ##################
PEDIGREE_COUNT = 1000
MAX_CHILDREN = 5
GENERATION_RANGE = (3,4)
ALT_FREQ_RANGE = (2, 25)
BACKPROP_LIKELIHOOD_RANGE = (25, 75)
SPOUSE_LIKELIHOOD_RANGE = (25, 75)
AFFECTED_SPOUSE = True

############### GLOBAL DEFAULT VARTABLE PARAMETERS #####################
SEQUENCE_COVERAGE_RANGE = (20, 100)
VARIANT_BACKGROUND_RANGE = (4, 9)


#################### MODULAR VARIANT SCORING METRIC CALCULATION ####################
'''
Categorical scores to be used in segregation scoring schemes (standard and extended)

Current List of Scores:
-------------------------
Edge Consistency (standard and extended)
Generation Continuity (standard and extended)
Carrier Betweenness (standard and extended)
Average Founder Influence (extended only)
Alternate Allele Depth (extended only)


'''
#----------------------------------------------------------------------
# 1. Edge Consistency
#----------------------------------------------------------------------
def edge_consistency(G, gt):
    """
    Fraction of parent→child edges whose genotype transition is Mendelian-
    compatible under the specified inheritance mode.
    
    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree
    gt(dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant

    RETURN:
    -------
    edge consistency ratio(float): the fraction of edges found in the given pedigree graph that follow mendelian inheritance for the given varient

    """
    #partental genotype (pg,mg) | (mg,pg) --> possible child genotypes {}
    BOTH_PARENT_ALLOWED_INHERITENCE = {
        (0,0):{0}, (1,0):{0,1}, (1,1):{0,1,2}, (2,1):{1,2}, (2,0):{1}, (2,2):{2}
    }
    SINGLE_PARENT_ALLOWED_INHERITENCE = {
        0:{0,1}, 1:{0,1,2}, 2:{1,2}
    }

    sequenced_samples = gt.keys()

    good=0; total=0
    for child in G.nodes:
        if child not in sequenced_samples:
            continue

        prnts=parents(G,child)
        sequenced_prnts = [p for p in prnts if p in sequenced_samples]


        if len(sequenced_prnts) == 1:
            if gt[child] in SINGLE_PARENT_ALLOWED_INHERITENCE.get(gt.get(sequenced_prnts[0]), {}):
                good+=1
            total+=1
        elif len(sequenced_prnts) == 2:
            gp,gm=[gt[p] for p in sequenced_prnts]
            par_gt = (gp,gm) if (gp,gm) in BOTH_PARENT_ALLOWED_INHERITENCE.keys() else (gm,gp)
            if gt[child] in BOTH_PARENT_ALLOWED_INHERITENCE.get(par_gt, {}):
                good+=2
            total+=2
        # If neither parent is sequenced, we cannot check Mendelian consistency for this child, so we skip.


    return good/total if total > 0 else 0

# ---------------------------------------------------------------------
# 2. Generation Continuity
# ---------------------------------------------------------------------
def generation_continuity(G, gt):
    """
    Calculates the fraction of generations with carriers (by genotype)

    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree
    gt(dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant

    RETURN:
    -------
    generation continuit ratio(float): the fraction of the total generations represented in the pedigree with at least one carrier (individual with genotype >=1)    
    """
    gen = generations(G)
    gens_total = max(gen.values())+1
    sequenced_samples = gt.keys()
    alt_gens = {gen[n] for n in sequenced_samples if gt[n]>0}
    sequenced_gens = {gen[n] for n in sequenced_samples}
    return len(alt_gens)/len(sequenced_gens) if sequenced_gens else 0



# ---------------------------------------------------------------------
# 3. Betweeness of Carriers in Affected+Carrier Subgraph
# ---------------------------------------------------------------------
def carrier_betweenness(G, gt):
    '''
    Calcculates the average betweenness of carriers when individually subgraphed with all affected individuals

    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree
    gt(dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant

    RETURN:
    -------
    averaged carrier betweenness(float): average of all carrier betweenness measures (without normalization), when calculated as individual subgraphs
        containing the carrier node in question (unaffected with genotype >=1) and all affected nodes
    '''
    aff_nodes = aff(G)
    unaff_nodes = unaff(G)
    sequenced_samples = gt.keys()
    sequenced_unaff_nodes = list(set(unaff_nodes) & set(sequenced_samples))
    sequenced_aff_nodes = list(set(unaff_nodes) & set(sequenced_samples))
    carrier_nodes = [n for n in sequenced_unaff_nodes if gt[n] == 1]
    carrier_aff_subgraph = G.subgraph(aff_nodes+carrier_nodes)
    subgraph_bet = nx.betweenness_centrality(carrier_aff_subgraph, normalized= False)

    avg_carrier_betweenness = sum(subgraph_bet[n] for n in carrier_nodes)/len(carrier_nodes) if len(carrier_nodes) > 0 else 0

    return avg_carrier_betweenness

# ---------------------------------------------------------------------
# 4. Average Founder Influence (extended scoring)
# ---------------------------------------------------------------------
def avg_founder_influence(G, gt):
    '''
    Calculates the average founder influence for all affected founders in pedgree graph.
    Founder influence here defined as the fraction of possible paths starting at a founder 
    that ends at an affected node divided by the total number of nodes reachable from the founder

    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree
    gt(dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant

    RETURN:
    -------
    average founder influence(float)
    '''
    sequenced_samples = gt.keys()
    founders = [n for n in G if G.in_degree(n)==0 and n in sequenced_samples and gt[n]>0]
    if founders:
        fi = founder_influence(G)
        avg_fi = sum(fi[f] for f in founders) / len(founders)
    else:
        avg_fi = 0
    return avg_fi


# ---------------------------------------------------------------------
# 5. Alternate Allelic Depth (extended scoring)
# ---------------------------------------------------------------------
def alt_depth_ratio(G, gt): 
    '''
    Calculates the the alternate allele depth ratio as the longest path extending between a founder (with genotype >= 1)
    and the most distant carrier descendant (genotype >= 1)

    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree
    gt(dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant

    RETURN:
    -------
    alternate allele depth ratio(float)
    '''
    depth = longest_path_length(G)
    sequenced_samples = gt.keys()
    alt_nodes = [n for n in sequenced_samples if gt[n]>0]
    alt_depth = 0
    founders = [n for n in G if G.in_degree(n)==0 and n in sequenced_samples and gt[n]>0]
    if depth and founders and alt_nodes:
        # shortest founder→alt path for each pair that is connected
        lengths = []
        for f in founders:
            for a in alt_nodes:
                if nx.has_path(G, f, a):
                    lengths.append(nx.shortest_path_length(G, f, a))
        if lengths:
            alt_depth = max(lengths)
    return alt_depth / depth


# ---------------------------------------------------------------------
# VARIANT CATEGORICAL SCORING WRAPPER
# ---------------------------------------------------------------------
def raw_categorical_scoring(G, gt):
    '''
    A mode agnostics categorical scoring wrapper to calculate and store individual scores for repeat segregation scoring
    for use in optimization without requiring repeat calculations

    PARAMETERS:
    -----------
    G(networkx.DiGraph): directed acyclic graph representation of pedigree
    gt(dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant

    RETURN:
    -------
    categorical scores(dict{'score category name':value})  
    '''
    return {
        'edge_consistency': edge_consistency(G, gt),
        'generation_continuity': generation_continuity(G, gt),
        'carrier_betweenness': carrier_betweenness(G, gt), 
        'founder_influence' : avg_founder_influence(G,gt),
        'alt_depth' : alt_depth_ratio(G,gt)
    }






#################### MAIN SEGREGATION SCORE CALCULATION ####################

# ---------------------------------------------------------------------
# SEGREGATION SCORING
# ---------------------------------------------------------------------
def segregation_network_score(  PedGraph, 
                                VariantEntry, 
                                mode, 
                                Scoring_Method= 'Original', 
                                categorical_scores= 0, 
                                weights={'w_edge':0.6,'w_gen':0.2,'w_bet':0.2}, 
                                verbose= False):
    '''
    Calculates the segregation score for a given pedigree and variant entry from variant table

    PARAMETERS:
    -----------
    PedGraph (networkx.DiGraph): directed acyclic graph representation of pedigree
    VariantEntry (dict{'individualID':genotype(0,1,2)}): dictionary of sequenced individuals in the pedigree and their genotypes for a given variant
    mode (string): mode of inheritance classification for given pedigreee ['AD','AR']
    Scoring_Method (string): the chosen scoring scheme to be used, dictating the set of categorical scores used ['standard', 'extended']
    categorical_scores (dict): dictionary of the categorical scores calculated over the the given PedGraph, if none given will be calculated using categorical score wrapper
    weights (dict): dictionary of the weightings to be applied to the categorical scores in segregation scoring.
        If none given will be set to defaults accoridnign to scoring scheme:
        defaults for standard scoring= w_edge:0.6, w_gen:0.2, w_bet:0.2
        defaults for extended scoring= w_edge:0.6, w_gen:0.1, w_bet:0.1, w_found:0.1, w_depth:0.1
    verbose (bool): choice of outputing relevant scoring information including categorical scores and weighted segregation score

    RETURN:
    -------
    total_score(float): weighted segregation score, ranges from 0 to 1
    '''

    #Categorical Score Calculation
    if not categorical_scores:

        categorical_scores = raw_categorical_scoring(PedGraph,VariantEntry)

    edge_score = categorical_scores['edge_consistency']
    
    gen_score = max(0,min(categorical_scores['generation_continuity'],1))
    
    bet_score = categorical_scores['carrier_betweenness'] if mode=='AR' else 1-categorical_scores['carrier_betweenness']
    bet_score = max(0,min(bet_score,1))


    #Extended Categorical Scores
    if Scoring_Method == 'Extended':
        
        #when incorrect number of weights are given for extended scores, resort to default weights
        if len(weights.keys()) < 5:
            weights = {'w_edge': 0.6, 'w_gen': 0.1, 'w_bet': 0.1, 'w_found': 0.1, 'w_depth': 0.1}
        
        founder_score = categorical_scores['founder_influence']
        depth_score = categorical_scores['alt_depth']


    #Weighted Score Calculation
    total_score = (weights['w_edge'] * edge_score) + (weights['w_gen'] * gen_score) + (weights['w_bet'] * bet_score)
    if Scoring_Method == 'Extended':
        total_score += (weights['w_found'] * founder_score) + (weights['w_depth'] * depth_score)

    '''
    Current Scoring Metrics:
        Original:
            edge_score
            gen_score
            bet_score
        Extended:
            found_score
            depth_score
    '''
    if verbose:
        print(f"Edge Score: {edge_score}; Generational Score: {gen_score}; Betweeness Score: {bet_score}")
        if Scoring_Method == 'Extended':
            print(f"Founder Score: {founder_score}; Depth Score: {depth_score}")
        print(f'Segregation Score: {total_score}')
    return total_score








############## SEGREGATION SCORING WEIGHT OPTIMIZATION ##################

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def max_score_highlighter(s):
    is_max = s == s.max()
    return [
        'background-color: green' if max_score and varID == 'chr1:100000_A>T'
        else 'background-color: red' if max_score
        else ''
        for varID, max_score in zip(s.index, is_max)]

def pprint_weights(weights_dict):
    for weight_name, weight_value in weights_dict.items():
        weight_value = round(weight_value, 3)
        print(f'{weight_name}: {weight_value}')
    print()

# ---------------------------------------------------------------------
# MARGIN-BASED WEIGHTS OPTIMIZATION OBJECTIVE
# ---------------------------------------------------------------------
def margin_weight_optimization_objective(weights_lst, weight_names, Multi_Ped_Dict, linked_variant,  Scoring_Method, mode):
    '''
    Calculates the performance of a given set of segregation scoring weights based on margin between linked variant score and top-scoring unlinked variant
    over a set of pedigrees with associated variant tables

    PARAMETERS:
    -----------
    weights_lst(list[float]): list of scoring weights in predefined order 
    weight_names(list[string]): list of scoring weight names in predefined order
    Multi_Ped_Dict(dict): dictionary of pedigrees and associated variant tables, including categorical scores for each variant in variant table
    linked_variant(string): variant ID for linked variant
    Scoring_Method(string): the chosen scoring scheme to be used, dictating the set of categorical scores used ['standard', 'extended']
    mode: mode of inheritance classification for given set of pedigrees ['AD','AR'], should be the same for all pedigrees used in optimization given difference in scoring between MOI

    RETURN:
    -------
    averaged margin score: a measure of segregation scoring weight set performance (ranging from 0 to 2) with 2 being the worst scoring performance (maximal negative margin) ,
        and 0 being the best possible margin (positive margin of 1)
    '''
    weights_dict = {weight_names[i]: weights_lst[i] for i in range(len(weight_names))}
    margins = []
    for FamilyID, FamilyData in Multi_Ped_Dict.items():
        PedGraph, VarTable = FamilyData['PedGraph'], FamilyData['VarTable']
        CategoricalScores = FamilyData['CategoricalScores'][linked_variant]
        linked_score = segregation_network_score(PedGraph= PedGraph,
                                                 VariantEntry= VarTable[linked_variant],
                                                 mode= mode,
                                                 Scoring_Method= Scoring_Method,
                                                 weights= weights_dict,
                                                 categorical_scores= CategoricalScores)

        unlinked_scores = []
        for VarID, gt in VarTable.items():
            if VarID != linked_variant:
                CategoricalScores = FamilyData['CategoricalScores'][VarID]
                unlinked_score = segregation_network_score(PedGraph= PedGraph,
                                                            VariantEntry= VarTable[VarID],
                                                            mode= mode,
                                                            Scoring_Method= Scoring_Method,
                                                            weights= weights_dict,
                                                            categorical_scores= CategoricalScores)
                unlinked_scores.append(unlinked_score)

        max_unlinked_score = max(unlinked_scores)

        margin = linked_score - max_unlinked_score
        margins.append(margin)

    avg_margin = np.mean(margins)

    return 1 - avg_margin


# ---------------------------------------------------------------------
# RANK-BASED WEIGHTS OPTIMIZATION OBJECTIVE
# ---------------------------------------------------------------------
def rank_weight_optimization_objective(weights_lst, Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode):
    '''
    Calculates the performance of a given set of segregation scoring weights based on margin between linked variant score and top-scoring unlinked variant
    and rank of linked variant amongst ordered list of segregation scores
    over a set of pedigrees with associated variant tables

    PARAMETERS:
    -----------
    weights_lst(list[float]): list of scoring weights in predefined order 
    weight_names(list[string]): list of scoring weight names in predefined order
    Multi_Ped_Dict(dict): dictionary of pedigrees and associated variant tables, including categorical scores for each variant in variant table
    linked_variant(string): variant ID for linked variant
    Scoring_Method(string): the chosen scoring scheme to be used, dictating the set of categorical scores used ['standard', 'extended']
    mode: mode of inheritance classification for given set of pedigrees ['AD','AR'], should be the same for all pedigrees used in optimization given difference in scoring between MOI

    RETURN:
    -------
    averaged margin score: a measure of segregation scoring weight set performance (ranging from 0 to 2*length of variant table) with 0 being the best possible margin (positive margin of 1) and rank of 1
    '''
    weights_dict = {weight_names[i]: weights_lst[i] for i in range(len(weight_names))}


    ranked_margins = []
    for FamilyID, FamilyData in Multi_Ped_Dict.items():
        PedGraph, VarTable = FamilyData['PedGraph'], FamilyData['VarTable']
        CategoricalScores = FamilyData['CategoricalScores'][linked_variant]
        linked_score = segregation_network_score(PedGraph= PedGraph,
                                                 VariantEntry= VarTable[linked_variant],
                                                 mode= mode,
                                                 Scoring_Method= Scoring_Method,
                                                 weights= weights_dict,
                                                 categorical_scores= CategoricalScores)

        unlinked_scores = []
        for VarID, gt in VarTable.items():
            if VarID != linked_variant:
                CategoricalScores = FamilyData['CategoricalScores'][VarID]
                unlinked_score  = segregation_network_score(PedGraph= PedGraph,
                                                            VariantEntry= VarTable[VarID],
                                                            mode= mode,
                                                            Scoring_Method= Scoring_Method,
                                                            weights= weights_dict,
                                                            categorical_scores= CategoricalScores)
                unlinked_scores.append(unlinked_score)

        all_scores = unlinked_scores + [linked_score]
        max_unlinked_score = max(unlinked_scores)

        all_scores.sort(reverse=True)
        linked_score_rank = all_scores.index(linked_score) + 1

        margin = linked_score - max_unlinked_score


        ranked_margins.append(linked_score_rank*margin)

    avg_ranked_margin = np.mean(ranked_margins)

    return len(VarTable) - avg_ranked_margin


# ---------------------------------------------------------------------
# WEIGHTS OPTIMIZATION OPERATIVE FUNCTION
# ---------------------------------------------------------------------
def weights_optimization(Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, Optimization_Method, initial_guess, mode):
    '''
    Optimizes the segregation scoring weight set according to one of the two optimization metrics definied above

    PARAMETERS:
    -----------
    weight_names(list[string]): list of scoring weight names in predefined order
    Multi_Ped_Dict(dict): dictionary of pedigrees and associated variant tables, including categorical scores for each variant in variant table
    linked_variant(string): variant ID for linked variant
    Scoring_Method(string): the chosen scoring scheme to be used, dictating the set of categorical scores used ['standard', 'extended']
    Optimization_Method(string): chosen optimization metric to use to assess given weight set scorings performance for optimization ['Margin','Rank']
    initial_guess (list[float]): list of weights values to start optimization effort from (given in same order as weight_names)
    mode: mode of inheritance classification for given set of pedigrees ['AD','AR'], should be the same for all pedigrees used in optimization given difference in scoring between MOI

    RETURN:
    -------
    optimized_weights(dict): dictionary containing the optimal weights determined through minimization of the chosen scoring performance loss function
    '''
    n_weights = len(weight_names)
    bounds = [(0.001,1)]*n_weights
    constraints = {'type': 'eq',
                  #figure out how this function is working
                  'fun': lambda w: np.sum(w)-1}

    if Optimization_Method == 'Margin':
        results = minimize(fun= margin_weight_optimization_objective,
                          x0= initial_guess,
                          args= (Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode),
                          bounds= bounds,
                          constraints= constraints)
    elif Optimization_Method == 'Rank':
        results = minimize(fun= rank_weight_optimization_objective,
                          x0= initial_guess,
                          args= (Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode),
                          bounds= bounds,
                          constraints= constraints)

    #Directly attaching weights with their names as dictionary for ease of use is scoring wrapper function
    optimized_weights = {weight_names[i]: results.x[i] for i in range(len(weight_names))}

    return optimized_weights






############## SEGREGATION SCORING WEIGHTS OPTIMIZATION ##################

# ---------------------------------------------------------------------
# TRIAL-BASED WEIGHTS OPTIMIZATION
# ---------------------------------------------------------------------
def trial_based_segregation_scoring_weight_optimization(
                                                    #PedGraph Parameters
                                                    Mode,
                                                    pedigree_count= PEDIGREE_COUNT,
                                                    generation_range = GENERATION_RANGE,
                                                    max_children = MAX_CHILDREN,
                                                    BackpropLikelihoodRange = BACKPROP_LIKELIHOOD_RANGE,
                                                    SpouseLikelihoodRange = SPOUSE_LIKELIHOOD_RANGE,
                                                    AffectedSpouse= AFFECTED_SPOUSE,

                                                    #Segregation Scoring Parameters
                                                    Scoring_Method= 'Original',
                                                    weights= 0,
                                                    Optimization_Method= 'Rank',
                                                    Verbose= False,

                                                    #VarTable Parameters
                                                    sequencing_coverage_range = SEQUENCE_COVERAGE_RANGE,
                                                    variant_background_range = VARIANT_BACKGROUND_RANGE,

                                                    #PedGraph and VarTable Parameters
                                                    alt_freq_range= ALT_FREQ_RANGE,
                                                    ):
    '''
    Generates a pedigree set of a specified size, each with an associated variant table and ground truth linked variant,
    optimizes segregation scoring weights based on a trainging subset of generated pedigrees,
    uses those optimized weights to score the remaining pedigrees in the testing subset of pedigrees

    PARAMETERS:
    -----------
    Pedigree Parameters:
    Mode (string): mode of inhertitance to be used in pedigree generation ['AD', 'AR']
    
    pedigree_count(int): total number of pedgrees for be generated in pedgree sample set (split 8:2 training-testing)
    
    generation_range((int,int)): duple of integer values indicating the range of generational sizes to be included in generated pedigree set (inclusive),
        randomly chosen per pedgree from the given range
    
    max_children(int): the maximum number of children that can be generated for each spousal pair in each pedigree in generated sample set
    
    BackpropLikelihoodRange((int,int)): duple of integer values indicating the range of backpropigation likelihoods to be used in pedigree generation (inclusive)
    
    SpouseLikelihoodRange((int,int)): duple of integer values indicating the range of reporductive likelihoods to be used in pedigree generation (inclusive)
   
    AffectedSpouse (bool): indicating whether non-founder individuals should be be considered as potential founders in pedigree generation,
        recommended this always remain True and affected likelihood be modulated through alternate allele frequency


    Variant Table Parameters:
    sequencing_coverage_range ((int,int)): duple of integer values indicating the range of sequencing coverage franctions to be used in variant table generation (inclusive)
    
    variant_background_range ((int,int)): duple of integers values indicating the number of background, unlinked variants to be generated as a part of variant table generation (inclusive)
    
    alt_freq_range((int,int)): duple of integer values indicating the range of alternate allele frequencies (as percentage) to be used in pedigree generation (inclusive),
        ranomly chosen per pedigree from the given range


    Segregation Scoring Parameters:
    Scoring_Method (string): the chosen scoring scheme to be used, dictating the set of categorical scores used ['standard', 'extended']
    
    weights (dict{'weight name':weight value(float)}): dictionary of the weights to be used in segregation scoring if no optimization, used as initial guess if optimization is attempted
    
    Optimization_Method (string): chosen optimization metric to use to assess given weight set scorings performance for optimization ['Margin','Rank']
    
    Verbose (bool): meant to display a gride of the final segregation scores of all variants across all testing pedigrees (not currently working)



    RETURN:
    -------
    test_Multi_Ped_Dict(dict): dictionary of all pedigrees in the testing pedigree set with attached segregation scoring results including linked variant rank and margin between linked variant and highest scroing unlinked variant
    optimized_weights(dict): dictionary of the optimized weights as determined through optimization over training pedigree set and used in scoring of testing pedigree set
    '''
    Multi_Ped_Dict = PedGraph_VarTable_generator(
                                            pedigree_count= pedigree_count,

                                            #PedGraph Parameters
                                            mode= Mode,
                                            max_children= max_children,
                                            generation_range= generation_range,
                                            BackpropLikelihoodRange= BackpropLikelihoodRange,
                                            SpouseLikelihoodRange= SpouseLikelihoodRange,
                                            AffectedSpouse= AffectedSpouse,
                                            
                                            #VarTable Parameters
                                            sequencing_coverage_range= sequencing_coverage_range,
                                            variant_background_range= variant_background_range,
                                            
                                            #PedGraph and VarTable Parameters
                                            alt_freq_range= alt_freq_range,)
    
    #Pre-calculate Categorical Scores to reduce computation time
    for FamID in Multi_Ped_Dict.keys():
        PedGraph = Multi_Ped_Dict[FamID]['PedGraph']
        VarTable = Multi_Ped_Dict[FamID]['VarTable']
        cat_score_dict = {}
        for VarID in VarTable.keys():
            cat_score_dict[VarID] = raw_categorical_scoring(G= PedGraph,
                                                            gt= VarTable[VarID])
        Multi_Ped_Dict[FamID]['CategoricalScores'] = cat_score_dict

    if Scoring_Method == 'Original':
        weight_names = ['w_edge', 'w_gen', 'w_bet']
    elif Scoring_Method == 'Extended':
        weight_names = ['w_edge', 'w_gen', 'w_bet', 'w_found', 'w_depth']

    #manually assignment of weights if no weights given if using original scoring
    if not weights:
        if Scoring_Method == 'Original':
            weights= {
                'w_edge': 0.6,
                'w_gen': 0.2,
                'w_bet': 0.2
            }
        elif Scoring_Method == 'Extended':
            weights= {
                'w_edge': 0.6,
                'w_gen': 0.1,
                'w_bet': 0.1,
                'w_found': 0.1,
                'w_depth': 0.1,
            }
    
    #Training and Testing Splits
    training_Multi_Ped_Dict = {}
    test_Multi_Ped_Dict = {}
    tt_split = 0.8
    for FamilyID in Multi_Ped_Dict.keys():
        if int(FamilyID[3:]) <= int(tt_split*len(Multi_Ped_Dict)):
            training_Multi_Ped_Dict[FamilyID] = Multi_Ped_Dict[FamilyID]
        else:
            test_Multi_Ped_Dict[FamilyID] = Multi_Ped_Dict[FamilyID]


    #Optimization of weights
    if Optimization_Method == 'Margin' or Optimization_Method == 'Rank':
        initial_guess = []
        for weight_name in weight_names:
            initial_guess.append(weights[weight_name])

        #Downsize the original multiple pedigree dict to the testing data now that we have done training testing split
        Multi_Ped_Dict = test_Multi_Ped_Dict
        optimized_weights= weights_optimization(Multi_Ped_Dict= training_Multi_Ped_Dict,
                                        linked_variant= 'chr1:100000_A>T',
                                        weight_names= weight_names,
                                        Scoring_Method= Scoring_Method,
                                        Optimization_Method= Optimization_Method,
                                        mode= Mode,
                                        initial_guess= initial_guess)
    else:
        raise ValueError(f'Optimization Method: {Optimization_Method} not valid')


    


    test_Optimized_Scores_df = pd.DataFrame(columns=test_Multi_Ped_Dict.keys())
    test_Unoptimized_Scores_df = pd.DataFrame(columns=test_Multi_Ped_Dict.keys())
    for FamilyID in test_Multi_Ped_Dict.keys():

        PedGraph, VarTable = test_Multi_Ped_Dict[FamilyID]['PedGraph'], test_Multi_Ped_Dict[FamilyID]['VarTable']
        CategoricalScores = test_Multi_Ped_Dict[FamilyID]['CategoricalScores']


        Multi_Ped_Dict[FamilyID][Scoring_Method] = {
                Optimization_Method : [],
                'Unoptimized' : []
        }

        for VarID in VarTable.keys():


            optimized_score = segregation_network_score(
                                            PedGraph= PedGraph,
                                            VariantEntry= VarTable[VarID],
                                            mode= Mode,
                                            Scoring_Method= Scoring_Method,
                                            weights= optimized_weights,
                                            categorical_scores= CategoricalScores[VarID])

            unoptimized_score = segregation_network_score(
                                            PedGraph= PedGraph,
                                            VariantEntry= VarTable[VarID],
                                            mode= Mode,
                                            Scoring_Method= Scoring_Method,
                                            weights= weights,
                                            categorical_scores= CategoricalScores[VarID])

            #storing scores at a list of duples (VarID, Score) for ease of sorting for rank extraction
            test_Multi_Ped_Dict[FamilyID][Scoring_Method][Optimization_Method].append((VarID, optimized_score))
            test_Multi_Ped_Dict[FamilyID][Scoring_Method]['Unoptimized'].append((VarID, unoptimized_score))

        #Pre-sorting Scores and determining linked variant rank
        for opt in [Optimization_Method, 'Unoptimized']:
            score_list = test_Multi_Ped_Dict[FamilyID][Scoring_Method][opt]
            sorted_score_list = sorted(score_list, key=lambda score: score[1], reverse=True)

            #storing the whole list of scores as list of tuples (VarID, score)
            test_Multi_Ped_Dict[FamilyID][Scoring_Method][opt] = sorted_score_list

            #storing linked variant score
            score_dict = dict(sorted_score_list)
            linked_score = score_dict['chr1:100000_A>T']
            test_Multi_Ped_Dict[FamilyID][Scoring_Method][f'{opt}LinkedScore'] = linked_score

            #storing margin
            unlinked_scores = []
            for VarID, score in score_dict.items():
                if VarID != 'chr1:100000_A>T':
                    unlinked_scores.append(score)
            margin = linked_score - max(unlinked_scores)
            test_Multi_Ped_Dict[FamilyID][Scoring_Method][f'{opt}Margin'] = margin

            #storing linked rank
            ranks = [Var[0] for Var in sorted_score_list]
            linked_rank = ranks.index('chr1:100000_A>T') + 1
            test_Multi_Ped_Dict[FamilyID][Scoring_Method][f'{opt}LinkedRank'] = linked_rank
        


        #TODO fix variant scoring table visualizaiton
        # Data Frame Construction does not work with list of duple strucutre and vairable number of background variants
        # test_Optimized_Scores_df[FamilyID] = test_Multi_Ped_Dict[FamilyID][Scoring_Method][Optimization_Method]
        # test_Unoptimized_Scores_df[FamilyID] = test_Multi_Ped_Dict[FamilyID][Scoring_Method]['Unoptimized']


    #TODO figure out how to display variant score table in normal python script run
    if Verbose:
        print(f'{Scoring_Method} Segregation Scoring Results:')
        styled_test_Optimized_Scores_df = test_Optimized_Scores_df.style.apply(max_score_highlighter, axis=0)
        styled_test_Optimized_Scores_df
        
        print('Unoptimzied Segregation Scoring Results:')
        styled_test_Unoptimized_Scores_df = test_Unoptimized_Scores_df.style.apply(max_score_highlighter, axis=0)
        styled_test_Unoptimized_Scores_df

        print('Default Weights:')
        pprint_weights(weights)
        print('Optimized Weights:')
        pprint_weights(optimized_weights)


    return test_Multi_Ped_Dict, optimized_weights










######################### SEGREGATION SCORING ##########################
# ---------------------------------------------------------------------
# PEDIGREE SEGREGATION SCORING
# ---------------------------------------------------------------------
def pedigree_segregation_scoring(Ped_Dict, Scoring_Method, Mode, Weights):
    '''
    Provides a means of performaning segregation scoring for a single pedigree and associated variant table (stored as a pedigree dictionary) using predefined weights

    PARAMETERS:
    -----------
    Ped_Dict(dict): dictionary of pedigree with associate variant table
    Scoring_Method(string): scoring scheme to be used ['standard','extended']
    Mode (string): the mode of the pedigree ['AR', 'AD']
    Weights (dictionary): the set of categorical weightings to be used in segregation scoring

    RETURN:
    -------
    Ped_Dict (dict): updated version of given parameter Ped_Dict that now includes entry with segregation scoring results
    '''
    PedGraph = Ped_Dict['PedGraph']
    VarTable = Ped_Dict['VarTable']

    CategoricalScores = {}
    for VarID in VarTable.keys():
        CategoricalScores[VarID] = raw_categorical_scoring(G= PedGraph,
                                                        gt= VarTable[VarID])
    Ped_Dict['CategoricalScores'] = CategoricalScores



    Ped_Dict[Scoring_Method] = {}
    for VarID in VarTable.keys():
        score = segregation_network_score(PedGraph= PedGraph,
                                                VariantEntry= VarTable[VarID],
                                                mode= Mode,
                                                Scoring_Method= Scoring_Method,
                                                weights= Weights,
                                                categorical_scores= CategoricalScores[VarID])
        Ped_Dict[Scoring_Method][VarID] = score

    return Ped_Dict




if __name__ == "__main__":
        #-------------------------------------
        # Segregation Scoring Function Test
        #-------------------------------------
        for mode in ['AD', 'AR']:
            scoring_results, optimized_weights = trial_based_segregation_scoring_weight_optimization(
                #Pedigree Parameters
                Mode=mode,
                pedigree_count= PEDIGREE_COUNT,
                generation_range= GENERATION_RANGE,
                max_children= MAX_CHILDREN,
                BackpropLikelihoodRange= BACKPROP_LIKELIHOOD_RANGE,
                SpouseLikelihoodRange= SPOUSE_LIKELIHOOD_RANGE,
                AffectedSpouse= AFFECTED_SPOUSE,

                #Variant Table Parameters
                sequencing_coverage_range= SEQUENCE_COVERAGE_RANGE,
                variant_background_range= VARIANT_BACKGROUND_RANGE,
                alt_freq_range= ALT_FREQ_RANGE,
                
                #Scoring Parameters
                Scoring_Method= 'Original',
                Optimization_Method= 'Rank'
            )

            #tallying scoring results
            total_top1 = 0
            total_IR = 0
            test_set_size = int(PEDIGREE_COUNT*0.2)

            for FamID in scoring_results.keys():
                linked_rank = scoring_results[FamID]['Original']['RankLinkedRank']
                
                #tallying top1
                if linked_rank == 1:
                    total_top1 += 1
                
                #tallying inverse rank
                total_IR += 1/linked_rank

            
            print(f'\n{mode} SEGREGATION SCORING RESULTS:')
            print('-------------------------------')
            print(f'  Linked Variant at Rank 1: {total_top1} of {test_set_size} pedigrees ({(total_top1/test_set_size)*100}%)')
            print(f'  Average Inverse Linked Variant Rank: {total_IR/test_set_size: .3f}\n')
