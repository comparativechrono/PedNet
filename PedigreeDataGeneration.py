import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc
from PedigreeDAGAnalysis import aff, unaff, construct_pedigree_graph, plot_pedigree_tree


#################### Inheritance Weights ####################
def inheritance_weights(p, q):
    '''
    Genotype probabilities for passing the alternate allele down the family (forward), and for choosing the parents
    of someone who married in (reverse). p and q are the reference and alternate allele frequencies.
    Forward, a child's genotype follows Mendel's laws. Reverse, each pair of parental genotypes is weighted by its
    Hardy-Weinberg frequency times the chance that it gives the child's genotype (Bayes' theorem), so ancestors
    chosen this way are in Hardy-Weinberg proportions, as people in the population are.
    '''
    hardy_weinberg = {0: p**2, 1: 2*p*q, 2: q**2}
    passes_alt = {0: 0.0, 1: 0.5, 2: 1.0}  # chance that a parent with this genotype passes on the alternate allele

    inheritance_patterns = {'forward_genotypes': {}, 'forward_weights': {},
                            'reverse_genotypes': {2: [], 1: [], 0: []}, 'reverse_weights': {2: [], 1: [], 0: []}}
    for paternal in (2, 1, 0):
        for maternal in (2, 1, 0):
            a, b = passes_alt[paternal], passes_alt[maternal]
            child = {2: a*b, 1: a*(1-b) + (1-a)*b, 0: (1-a)*(1-b)}
            possible = [g for g in (2, 1, 0) if child[g] > 0]
            #(paternal genotype, maternal genotype) -> [possible child genotypes]
            inheritance_patterns['forward_genotypes'][(paternal, maternal)] = possible
            inheritance_patterns['forward_weights'][(paternal, maternal)] = [child[g] for g in possible]
            #child genotype -> [possible (paternal,maternal) genotypes]
            for g in possible:
                inheritance_patterns['reverse_genotypes'][g].append((paternal, maternal))
                inheritance_patterns['reverse_weights'][g].append(hardy_weinberg[paternal] * hardy_weinberg[maternal] * child[g])

    return inheritance_patterns


#################### Pedigree Generator ####################
'''
Generates a pedigree based on given metrics

'''
def pedigree_generator(
            FamilyID,
            mode, 
            max_children,  
            generation_range,
            alt_freq_range, 
            SpouseLikelihoodRange, 
            BackpropLikelihoodRange,
            AffectedSpouse,
            Max_QC_attempts = 100):
                
        #-------------------------------------------
        # Range-based Parameter Determination
        #-------------------------------------------
        gen_min, gen_max = generation_range
        generation_count = random.randint(gen_min, gen_max)
        
        alt_freq_min, alt_freq_max = alt_freq_range
        alt_freq = random.randint(alt_freq_min, alt_freq_max)/100

        backprop_min, backprop_max = BackpropLikelihoodRange
        BackpropLikelihood = random.randint(backprop_min, backprop_max)/100

        spouse_min, spouse_max = SpouseLikelihoodRange
        SpouseLikelihood = random.randint(spouse_min, spouse_max)/100


        #-------------------------------------------
        # Helper Functions for Pedigree Propigation
        #-------------------------------------------

        '''
        Basic helper function to add new entry to pedigree dataframe
        '''
        def entry_generator(IndividualID, PaternalID, MaternalID, Sex, Phenotype, Genotype):
            nonlocal family_df
            family_df.loc[IndividualID] = [FamilyID, PaternalID, MaternalID, Sex, Phenotype, Genotype]

        '''
        Helper function to translate between genotype and phenotype
        Dependant on the mode of inheritance
        Input: genotype(int(0,1,2))
        Output: phenotype(int(1,2))
        '''
        def genotype_interpreter(genotype):
            if mode == 'AR':
                phenotype = 2 if genotype == 2 else 1
            if mode == 'AD':
                phenotype = 2 if genotype == 2 or genotype == 1 else 1
            return phenotype

        '''
        Wrapper function that generates the primary founder of the pedigree
        By default, this individual is affected
        If AD, 20% chance homozygous, 80% chance heterozygous.
        If AR, 100% chance homozygous.
        Input:
        Output:
        '''
        def primary_founder_generator():
            nonlocal family_df

            if mode == 'AD':
                Genotype = random.choices(population= [1,2],
                                          weights= (0.8, 0.2))[0]
            elif mode == 'AR':
                Genotype= 2

            entry_generator(IndividualID= 1,
                            PaternalID= 0,
                            MaternalID= 0,
                            Sex= random.randint(1,2),
                            Phenotype= 2,
                            Genotype= Genotype)
        '''
        Wrapper function that generates spouses unrelated to primary founder
        Spouse sex dependent on the relative of primary founder.
        Genotype and phenotype dependent on the mode of inheritance and affected spouse paramter.
        Input: relativeID(int)
        Ouput: n/a
        '''
        def spouse_generator(RelativeAnchorID):
            nonlocal family_df, alt_freq, ref_freq

            pp = ref_freq**2
            pq2 = 2*ref_freq*alt_freq
            qq = alt_freq**2

            Sex= 1 if family_df.loc[RelativeAnchorID]['Sex'] == 2 else 2

            if AffectedSpouse:
                Genotype= random.choices(population= [0,1,2],
                                          weights= (pp, pq2, qq),
                                          k=1)[0]

            else:
                Genotype = 0

            entry_generator(IndividualID= len(family_df)+1,
                            PaternalID= 0,
                            MaternalID= 0,
                            Sex= Sex,
                            Phenotype= genotype_interpreter(Genotype),
                            Genotype= Genotype)
        '''
        Wrapper function that generates an entry for the child of two given individuals.
        Child's genotype is chosen from list of allowed gentypes given parents genotypes with equal likelihood.
        Input: PaternalID(int), MaternalID(int)
        Output: n/a
        '''
        def child_generator(PaternalID, MaternalID):
            nonlocal family_df, inheritance_patterns

            parentalGenotype = (int(family_df.loc[PaternalID]['Genotype']), int(family_df.loc[MaternalID]['Genotype']))

            Genotype = random.choices(population= inheritance_patterns['forward_genotypes'][parentalGenotype],
                                      weights= inheritance_patterns['forward_weights'][parentalGenotype],
                                      k=1)[0]

            entry_generator(IndividualID= len(family_df)+1,
                            PaternalID= PaternalID,
                            MaternalID= MaternalID,
                            Sex= random.randint(1,2),
                            Phenotype= genotype_interpreter(Genotype),
                            Genotype= Genotype)
        #---------------------------------------
        # Primary Pedigree Contruction Functions
        #---------------------------------------
        '''
        Function that recursively constructs pedigree in backward direction.
        Infers ancestors of individuals unrelated to primary founder as they are added.
        Input: current_generation(int), RealativeAnchorID(int)
        Output: n/a
        '''
        def recursive_history_backprop(current_generation, RelativeAnchorID):
            nonlocal family_df, generation_count, inheritance_patterns, BackpropLikelihood

            BackpropRNG = random.randint(1,100)/100

            if current_generation > 0 and BackpropRNG <= BackpropLikelihood:

                GenotypeTup = random.choices(population= inheritance_patterns['reverse_genotypes'][family_df.loc[RelativeAnchorID]['Genotype']],
                                                    weights= inheritance_patterns['reverse_weights'][family_df.loc[RelativeAnchorID]['Genotype']],
                                                    k=1)[0]

                ID_list = ['PaternalID', 'MaternalID']

                for i in range(2):
                    entry_generator(IndividualID= len(family_df)+1,
                                    PaternalID= 0,
                                    MaternalID= 0,
                                    Sex= 1 + i,
                                    Phenotype= genotype_interpreter(GenotypeTup[i]),
                                    Genotype= GenotypeTup[i])
                    family_df.at[RelativeAnchorID, ID_list[i]] = len(family_df)
                    recursive_history_backprop(current_generation-1, len(family_df))

        '''
        Function that recursively constructs pedigree in forward direction.
        Input: current_generation(int), RelativeAnchorID(int)
        Output: n/a
        '''
        def recursive_pedigree_construction(current_generation, RelativeAnchorID):
            nonlocal family_df, max_children, generation_count

            if current_generation < generation_count-1:

                spouse_generator(RelativeAnchorID= RelativeAnchorID)

                #Determining Parental Sex for next generation
                if family_df.loc[RelativeAnchorID]['Sex'] == 1:
                    PaternalID = RelativeAnchorID
                    MaternalID = len(family_df)
                else:
                    PaternalID = len(family_df)
                    MaternalID = RelativeAnchorID

                if BackpropLikelihood:
                    recursive_history_backprop(current_generation, len(family_df))

                for child in range(random.randint(1, max_children)):
                    child_generator(PaternalID= PaternalID, MaternalID= MaternalID)
                    reproduction_rng = random.randint(1,100)/100
                    #the first child always has a partner and children, so that the pedigree has as many generations as asked for
                    if reproduction_rng <= SpouseLikelihood or child == 0:
                        recursive_pedigree_construction(current_generation+1, len(family_df))

        #Quality assurance to ensure the generated pedigree has sufficient phenotypic information (>1 affected individuals)
        QC_attempts = 0
        QC_pass = False
        while not QC_pass and QC_attempts < Max_QC_attempts:
            #-------------------------------------
            # 1. Construct the empty data frame
            #-------------------------------------
            pedigree_construction_columns = ['FamilyID', 'IndividualID', 'PaternalID', 'MaternalID', 'Sex', 'Phenotype', 'Genotype']
            family_df = pd.DataFrame(columns= pedigree_construction_columns)
            family_df.set_index('IndividualID', inplace=True)

            #-------------------------------------
            # 2. Generating Primary Founder
            #-------------------------------------
            primary_founder_generator()

            #--------------------------------------------
            # 3. Construct Inheritence Pattern Dictionary
            #--------------------------------------------
            ref_freq = 1 - alt_freq
            inheritance_patterns = inheritance_weights(ref_freq, alt_freq)

            #----------------------------------------
            # 4. Generating Pedigree
            #----------------------------------------
            recursive_pedigree_construction(current_generation= 0, RelativeAnchorID= 1)

            #-------------------------------
            # 5. Resetign Standard Indexing
            #-------------------------------
            family_df.reset_index(inplace= True)

            QC_attempts += 1
            affected_sum = family_df['Phenotype'].sum() - len(family_df.index)
            if affected_sum > 1:
                QC_pass = True
            elif QC_attempts >= Max_QC_attempts:
                print(f'Maximum Phenotypic of {Max_QC_attempts} Coverage Attempts Reached for {FamilyID}. Included to prioritize runtime.')
                QC_pass = True

        return family_df



#################### Background Variants ####################
def gene_drop(parents_of, q):
    '''
    Genotypes for one variant passed down a pedigree by Mendel's laws.

    PARAMETERS:
    -----------
    parents_of (dict): each person's parents, as IndividualID: (parent ID, parent ID), with 0 for a parent who isn't in the pedigree
    q (float): the variant's alternate allele frequency in the population

    RETURN:
    -------
    genotypes (dict): IndividualID: genotype (0, 1 or 2). People with no parents in the pedigree are drawn from the population,
        in Hardy-Weinberg proportions, and each child gets one allele from each parent at random
        (an allele from a parent who isn't in the pedigree is drawn from the population)
    '''
    p = 1 - q
    genotypes = {}
    def genotype(n):
        if n not in genotypes:
            known = [parent for parent in parents_of[n] if parent]
            if not known:
                genotypes[n] = random.choices([0, 1, 2], [p**2, 2*p*q, q**2])[0]
            else:
                alleles = [random.random() < genotype(parent)/2 for parent in known]
                alleles += [random.random() < q for _ in range(2 - len(known))]
                genotypes[n] = sum(alleles)
        return genotypes[n]
    for n in parents_of:
        genotype(n)
    return genotypes


#################### Variant Table Generator ####################
'''
Generates a variant table with linked variant based on genotype taken from a given pedigree DAG

inherited_background (bool): if True, the background (unlinked) variants are passed down the pedigree by Mendel's laws,
    as real variants are, so that relatives share them. If False (the default, as before), each person's genotype for a
    background variant is drawn independently from the population, which makes the linked variant, the only one shared
    within the family, easier to pick out than it would be in real data
'''
def simulate_variant_table(family_df, sequencing_coverage_range, variant_background_range, alt_freq_range, linked_variant = 'chr1:100000_A>T',
                           inherited_background = False):
    family_df.set_index('IndividualID', inplace=True)
    samples = [int(x) for x in family_df.index.values]

    #Selecting randomized range parameters
    sequence_cov_min, sequence_cov_max = sequencing_coverage_range
    sequencing_coverage = random.randint(sequence_cov_min,sequence_cov_max)/100

    variant_bg_min, variant_bg_max = variant_background_range
    n_bg = random.randint(variant_bg_min, variant_bg_max)

    #to account for imcomplete sequencing coverage across a pedigree
    sequenced_samples = []
    for sample in samples:
        sequencing_rng = random.randint(1,100)/100
        if sequencing_rng <= sequencing_coverage:
            sequenced_samples.append(sample)
    
    VarTable = {}

    #filling out linked variant table entry based on generated genotype data
    VarTable[linked_variant] = {}
    for sample in sequenced_samples:
        VarTable[linked_variant][sample] = family_df.loc[sample]['Genotype']

    #filling in unlinked variant table entries using randomly generated gentypes
    min_alt_freq, max_alt_freq = alt_freq_range
    if inherited_background:
        parents_of = {int(n): (int(family_df.loc[n]['PaternalID']), int(family_df.loc[n]['MaternalID'])) for n in family_df.index}
    for i in range(n_bg):
        VarID = f'chr1:{100200+i}_G>C'
        q = random.randint(min_alt_freq, max_alt_freq)/100
        p = 1 - q
        if inherited_background:
            genotypes = gene_drop(parents_of, q)
            VarTable[VarID] = {sample: genotypes[sample] for sample in sequenced_samples}
        else:
            VarTable[VarID] = {sample : random.choices([0,1,2],[p**2, 2*p*q, q**2])[0] for sample in sequenced_samples}
    
    family_df.reset_index(inplace=True)

    return VarTable


#################### Mass Pedigree and Variant Table Data Generator####################
'''
Generates a dictionary of families with each entry containing the generated pedigree DAG graph,
and variant table data (randomly generated with one linked variant) 
'''
def PedGraph_VarTable_generator(
            #PedGraph Parameters
            pedigree_count, 
            mode, 
            max_children, 
            generation_range,
            BackpropLikelihoodRange,
            SpouseLikelihoodRange, 
            AffectedSpouse,
            
            #VarTable Parameters
            variant_background_range, 
            sequencing_coverage_range, 
            
            #PedGraph and VarTable Parameters
            alt_freq_range,

            #Background variants passed down the pedigree (see simulate_variant_table)
            inherited_background = False
            ):

    Fam_Data_Dict = {}

    for Family_Num in range(1, pedigree_count+1):
        FamilyID = f'FAM{Family_Num}'

        ped_df = pedigree_generator(
                                FamilyID= FamilyID,
                                max_children= max_children,
                                mode= mode,
                                generation_range= generation_range,
                                alt_freq_range= alt_freq_range,
                                BackpropLikelihoodRange= BackpropLikelihoodRange,
                                SpouseLikelihoodRange= SpouseLikelihoodRange,
                                AffectedSpouse= AffectedSpouse
                                )
        ped_dg = construct_pedigree_graph(ped_df)
        affected_nodes = aff(ped_dg)


        #QC check for variant table
        var_QC_checks = 0
        var_QC_pass = False
        while not var_QC_pass:
            var_QC_checks += 1
            var_dict = simulate_variant_table(
                                            family_df= ped_df,
                                            alt_freq_range = alt_freq_range,
                                            sequencing_coverage_range= sequencing_coverage_range,
                                            variant_background_range= variant_background_range,
                                            inherited_background= inherited_background
                                            )

            #checking to make sure there more than 2 seuqneced individuals in the pedigree
            #selects first varID from the list given they should have the same number of entries (number of sequenced samples)
            if len(var_dict[list(var_dict.keys())[0]]) > 2:
                var_QC_pass = True
            elif var_QC_checks > 50:
                print(f'{mode} {FamilyID}: Failed VarTable QC checks, included despite QC failure to prioritize futher operations')
                var_QC_pass = True

        #consider moving the categorical scoring to outside data generation
        #means you dont have to do unneccesary computation when generating multi-pedigree data sets outside of segregation scoring analysis


        Fam_Data_Dict[FamilyID] = {'PedGraph': ped_dg, 'VarTable': var_dict}

    return Fam_Data_Dict




#################### Variant Table Backpadding ####################
'''
Builds a variant table for a real pedigree from known genotypes (VariantInfoDict), padded with simulated background variants.
inherited_background (bool): pass the background variants down the pedigree by Mendel's laws (see simulate_variant_table)
'''
def VarTableBackpadding(PedGraph, VariantInfoDict, VariantBackgroundRange, alt_freq_range, inherited_background = False):
    affected_nodes = aff(G=PedGraph)
    unaffected_nodes = unaff(G=PedGraph)

    VarTable = {}
    for gene, attributes in VariantInfoDict.items():
        VarTable[gene] = {}
        for group, genotype in attributes.items():
            if group == 'aff':
                for node in affected_nodes:
                    VarTable[gene][node] = genotype
            elif group == 'unaff':
                for node in unaffected_nodes:
                    VarTable[gene][node] = genotype
            elif group == 'all':
                for node in PedGraph.nodes():
                    VarTable[gene][node] = genotype
            elif int(group) in PedGraph.nodes():
                VarTable[gene][int(group)] = genotype
            else:
                print(f'{group}:{genotype} is not a valid entry for genotype data')
    
    
    SequencedNodes = set()
    for Variant, NodeGenotypes in VarTable.items():
        SequencedNodes = SequencedNodes.union(set(NodeGenotypes.keys()))

    #filling in unlinked variant table entries using randomly generated gentypes
    var_background_min, var_background_max = VariantBackgroundRange
    PaddingCount = random.randint(var_background_min,var_background_max)
    min_alt_freq, max_alt_freq = alt_freq_range
    parents_of = {n: tuple(PedGraph.predecessors(n)) for n in PedGraph.nodes()}
    for i in range(PaddingCount):
        VarID = f'chr1:{100200+i}_G>C'
        q = random.randint(min_alt_freq, max_alt_freq)/100
        p = 1 - q
        if inherited_background:
            genotypes = gene_drop(parents_of, q)
            VarTable[VarID] = {node: genotypes[node] for node in SequencedNodes}
        else:
            VarTable[VarID] = {node : random.choices([0,1,2],[p**2, 2*p*q, q**2])[0] for node in SequencedNodes}

    return VarTable   