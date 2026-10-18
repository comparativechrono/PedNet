import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, auc
import numpy as np
from typing import Dict, Set, Tuple
from collections import Counter

#---------------------------------------
# PEDIGREE IMPORT
#---------------------------------------
def pedfile_readin(pedfile):
    '''
    Reads in pedigree data as a PED file format and converts to pandas data frame

    PARAMETERS:
    ------------
    pedfile (string): path to desired PED file for use in DAG analysis

    RETURN:
    -------
    df (pandas.DataFrame): data frame containing equivalent fields from PED file
        FamilyID
        IndividualID
        PaternalID
        MaternalID
        Sex
        Phenotype
    '''
    cols = ['FamilyID', 'IndividualID', 'PaternalID', 'MaternalID', 'Sex', 'Phenotype']
    df = pd.read_csv(pedfile, sep=r'\s+', header=None, names=cols)
    return df

#-------------------------
# PEDIGREE DAG CONVERSION
#-------------------------
def construct_pedigree_graph(df, rm_floaters= True):
    '''
    Constructs a directed acyclic graph representation of a given pedigree.

    PARAMETERS:
    -----------
    df (pandas.DataFrame): pedigree dataframe stroing PED file data (see PED file import above)
    rm_floaters (bool): in cases where individuals are included in the pedigree data despite having no relational data,
        this results in a node with no incoming or outgoing edges. This options decides if such nodes should be removed from the final pedigree graph
        default is True given floaters cause issues for many DAG analysis algorithms
    
    RETURN:
    -------
    G (networkX.Digraph): directed graph object depicted pedigree relational and phenotypic data
    '''
    G = nx.DiGraph()

    all_parents_set = set()
    founder_set = set()

    for _, row in df.iterrows():
        # Make sure IndividualID is treated as a string or int consistently if needed
        G.add_node(row['IndividualID'],
                  family=row['FamilyID'],
                  sex=row['Sex'],
                  phenotype=row['Phenotype'])

    for _, row in df.iterrows():
        # Ensure PaternalID and MaternalID are compared to string '0' if they are strings
        paternal_id = row['PaternalID']
        maternal_id = row['MaternalID']
        individual_id = row['IndividualID']

        if paternal_id != 0:
            G.add_edge(paternal_id, individual_id)
            all_parents_set.add(paternal_id)
        if maternal_id != 0:
            G.add_edge(maternal_id, individual_id)
            all_parents_set.add(maternal_id)
        if maternal_id == 0 and paternal_id == 0:
            founder_set.add(individual_id)

    #Removing founders with no children (i.e. floaters)
    if rm_floaters:
        floaters_set = founder_set - all_parents_set
        G.remove_nodes_from(floaters_set)


    return G

#----------------------------
# PEDIGREE DAG VISUALIZATION
#----------------------------
def plot_pedigree_tree(G, title="Pedigree (Tree Layout)"):
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')  # 'dot' gives top-down DAG style
    except ImportError:
        print("PyGraphviz not installed. Falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)

    node_colors = ['red' if G.nodes[n]['phenotype'] == 2 else 'lightblue' for n in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True)
    plt.title(title)
    plt.show()



#---------------------------------------
# SIMPLE DAG HELPER FUNCTIONS
#---------------------------------------
def parents(G, node):
    """Return a list of parent nodes for `node` (incoming edges)."""
    return list(G.predecessors(node))

def siblings(G, node):
    """Return siblings: nodes that share ≥ 1 parent with `node`."""
    sibs = set()
    for p in parents(G, node):
        sibs.update(G.successors(p))
    sibs.discard(node)
    return sibs

def children(G, node):
    """Return a list of child nodes for `node` (outgoing edges)."""
    return list(G.successors(node))

def generations(G):
    """Return a dictionary containing a key:value pair for each node (IndividualID=key) where the value is the biological generation that the node belonds to"""
    lvl={}
    Q=[(n,0) for n in G if G.in_degree(n)==0]
    while Q:
        n,d=Q.pop(0)
        #this check doesnt take into account children produced from one founder and one relative
        #leads all individuals to have the generation count to be minimum distance from most recent founder
        #if n in lvl: continue
        lvl[n]=d
        for c in G.successors(n): Q.append((c,d+1))
    return lvl

def longest_path_length(G):
    """Return the length of the longest path through the pedigree graph (i.e. the number of biological genetations depicted - 1)"""
    return nx.dag_longest_path_length(G)

def aff(G):
    """Return a list of nodes with an affected phenotype"""
    return [n for n in G.nodes if G.nodes[n]['phenotype']==2]

def unaff(G):
    """Return a list of nodes with an unaffected phenotype"""
    return [n for n in G.nodes if G.nodes[n]['phenotype']==1]

def pedigree_width(G: nx.DiGraph) -> int:
    """Returns the width of the pedigree (i.e. the size of the largest generation depicted)"""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG.")
    #transitive closure creates new graph including all origianl edges and adding edges between all nodes connected by a path
    #i.e. for AD pedigree adds 4 edges connecting both grandparents to both of their grandchildren
    P = nx.algorithms.dag.transitive_closure(G)
    left  = {f"{n}_L" for n in G}
    right = {f"{n}_R" for n in G}
    B = nx.DiGraph()
    B.add_nodes_from(left,  bipartite=0)
    B.add_nodes_from(right, bipartite=1)
    for u, v in P.edges:
        B.add_edge(f"{u}_L", f"{v}_R")
    match = nx.algorithms.bipartite.maximum_matching(B, top_nodes=left) #finds maximum number of node pairing each connected by eges that maximizes the number of nodes included in the set (no repeats)
    # this match includes both directions (one pairing left-right (normal) and one pairing right-left (reverse))
    matched = len(match) // 2
    width = G.number_of_nodes() - matched
    return width


#################### MODULAR PEDIGREE METRICS ####################
'''
Metrics: measures based on phenotype distribution across the pedigree for use in mode of inheritance classificaiton

Current List of Metrics:
-------------------------
1. Ratio Affected Parents
2. Generation Coverage
3. Affected Sibiling Pairing
4. Average Betweeness of Unaffected
5. Founder Influence
'''

# ---------------------------------------------------------------------
# 1. Ratio Affected Parent-Child Ratio
# ---------------------------------------------------------------------
def ratio_aff_parents(G):
    '''
    Calculates the ratio of parent-child pairings in which the child is affected and at least one of the parents is affected 
    compared to the total number of affected nodes
    '''
    aff_nodes = aff(G)
    aff_aff_partent = 0
    for n in aff_nodes:
        if any(G.nodes[p]['phenotype']==2 for p in parents(G,n)):
            aff_aff_partent +=1
    return aff_aff_partent/len(aff_nodes) if aff_nodes else 0


# ---------------------------------------------------------------------
# 2. Generation Coverage
# ---------------------------------------------------------------------
def gen_cov(G):
    '''
    Calculates the fraction of generations in which at least one affected individual is found 
    compared to the total number of generations depicted in the pedigree
    '''
    gen = generations(G)
    gens_aff = {gen[n] for n in aff(G)}
    return len(gens_aff)/(max(gen.values())+1) if gen else 0


# ---------------------------------------------------------------------
# 3. Affected Sibling Pairing Ratio
# ---------------------------------------------------------------------
def sibling_aff_ratio(G):
    '''
    Calculates the fraction of sibling pairs in which both of the siblings are affected 
    compared to the total number of sibling pairs depicted in the pedigree
    '''
    sib_pairs=0; aa_pairs=0
    for n in aff(G):
        for sib in siblings(G,n):
            if sib in aff(G):
                aa_pairs+=1
            sib_pairs+=1
    return aa_pairs/sib_pairs if sib_pairs else 0

# ---------------------------------------------------------------------
# 4. Average Betweeness of Unaffected
# ---------------------------------------------------------------------
def avg_bet_unaff(G):
    '''
    Calculates the average centrality betweenness of all unaffected nodes when subgraphed alone with all affected nodes
    '''
    unaff_nodes = unaff(G)
    aff_nodes = aff(G)
    unaff_bets = []
    for node in unaff_nodes:
        aff_node_SG = G.subgraph(nodes=(aff_nodes + [node]))
        bet = nx.betweenness_centrality(aff_node_SG, normalized=False)
        node_bet = bet[node]
        unaff_bets.append(node_bet)
    #trying to normalize it to the size of the affected subgraph
    return sum(unaff_bets)/len(unaff_bets) if unaff_bets else 0


# ---------------------------------------------------------------------
# PEDIGREE FEATURES WRAPPER
# ---------------------------------------------------------------------
def calc_pedigree_metrics(G):
    '''
    Wrapper function to calculate all graph metrics relevant to MOI classification 

    PARAMETERS:
    -----------
    G (networkX.DiGraph): DAG representation of the pedigree in quesiton


    RETURN:
    -------
    Metric Dictionary (dict): ditionary containing all of the MOI relevant metrics 'metric_name':metric_value(float)
    '''
    return {
        'ratio_aff_parent': ratio_aff_parents(G),
        'sibling_aff_ratio': sibling_aff_ratio(G),
        'gen_cov': gen_cov(G),
        'avg_bet_unaff': avg_bet_unaff(G)
    }



######################## CONSANGUINITY ANALYSIS ###############################
'''
Detects instances of consanguinity within a given pedigree.

PARAMETERS:
-----------
G (networkx.DiGraph): DAG representation of a given pedigree

RETURN:
-------
ancestry_dict (dict): dictionary depicting the biological ancestors of each one 
    one item per individual node = IndividualID:[ancestor IndividualIDs]
consanguinous_nodes (dict): dictionary depicting all nodes that are the result of consanguinous relationships
    one item per node that results from consanguinous relationship = Consanguinous Child IndividualID: {[Parent IDs], Common Ancestor ID, degree of separation}
'''
def consanguinity_analysis(G):
    revG = G.reverse()
    ancestry_dict = {}
    consanguinous_nodes = {}
    for n in revG.nodes():
        ancestors = nx.descendants(revG, n)
        ancestry_dict[n] = ancestors
        ancestry_subgraph = nx.subgraph(revG, [n]+ancestors)
        for ancestor in ancestors:
            if ancestry_subgraph.in_degree(ancestor) > 1:
                parents = list(revG.successors(n))
                degree_separation = nx.shortest_path(revG, parents[0], ancestor) + nx.shortest_path(revG, parents[1], ancestor)
                consanguinous_nodes[n] = {
                    'parents': parents,
                    'common_ancestor':ancestor,
                    'degree_separation':degree_separation
                }
    return ancestry_dict, consanguinous_nodes



#################### RULE-BASED MODE OF INHERITENCE FLAGS ####################
'''
Additional Boolean flags to be potentially implemented in MOI classification
in tandeom with phenotype distribution metrics listed above
'''

#--------------------------------------------
# AFFECTED CHILD WITH TWO UNAFFECTED PARENTS
#--------------------------------------------
def aff_child_with_unaff_parents(G):
    '''
    Determines if the pedigree contains at least one instance of an affected child where both parents are unaffected.
    This is a mendelian impossibility for a autosomal dominant phenotype.
    '''
    aff_nodes = aff(G)
    for node in aff_nodes:
        prnts = parents(G, node)
        if len(prnts) == 2 and not any(G.nodes[p]['phenotype'] == 2 for p in prnts):
            return True
    return False

def aff_parents_with_unaffected_child(G):
    '''
    Determines if the pedigree constains at least one instance of an unaffected child where both parents are affected.
    This is a mendelian impossibility for an autsomal recessive phenotype.
    '''
    unaff_nodes = unaff(G)
    for node in unaff_nodes:
        sblngs = siblings(G, node)
        prnts = parents(G, node)
        if len(prnts) == 2 and not any(G.nodes[p]['phenotype'] == 1 for p in prnts) and not any(G.nodes[s]['phenotype'] == 2 for s in sblngs):
            return True
    return False


#################### ADDITIONAL MODULAR GRAPH METRICS ####################
'''
Additional metrics: potentially useful pedigree graph metrics that were designed to make use of the graphical representation of pedigrees
but do not fit into current MOI classification scheme / require additional adjustments prior to implementation

Current List of Features:
-------------------------
1. Number of Nodes
2. Number of Edges
3. Number of Connected Components
4. Average Clustering Coefficient
5. Diameter
6. Average Shortest Path Length
7. Average Degree Centrality
8. Average Betweenness Centrality
9. Average Closeness Centrality
10. Transitive Reduction Size Ratio
11. Minimal Founder Coverage Size
12. Affected Generational Clustering
13. Founder Influence
'''

# ---------------------------------------------------------------------
# 1-6. Basic Graph Metrics
# ---------------------------------------------------------------------
def basic_graph_features(G):
    G_u = G.to_undirected()
    return {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_components': nx.number_connected_components(G_u),
        'avg_clustering': nx.average_clustering(G_u),
        'diameter': nx.diameter(G_u),
        'avg_path_len': nx.average_shortest_path_length(G_u)
    }

# ---------------------------------------------------------------------
# 7-9. Centralities
# ---------------------------------------------------------------------
def centralities(G):
    G_u = G.to_undirected()
    deg_cent = list(nx.degree_centrality(G_u).values())
    bet_cent = list(nx.betweenness_centrality(G_u).values())
    clos_cent = list(nx.closeness_centrality(G_u).values())

    return {'avg_degree_centrality': float(np.mean(deg_cent)),
            'avg_betweenness': float(np.mean(bet_cent)),
            'avg_closeness': float(np.mean(clos_cent))
    }

# ---------------------------------------------------------------------
# 10. Transitive Reduction Size
# ---------------------------------------------------------------------
# How does transitive reduction work with our pedigrees?
# nx.transitive_reduction only returns a list of duples for edges in transitive reduction
# would only cull child-parent relationships in cases of consanguinity between partent and other child
def transitive_reduction_ratio(G):
    red = nx.transitive_reduction(G)
    return red.number_of_edges()/G.number_of_edges()

# ---------------------------------------------------------------------
# 11. Minimal Founder Coverage
# ---------------------------------------------------------------------
def minimal_founder_cover_set(G: nx.DiGraph) -> set:
    """
    Return one minimal founder cover (greedy) as a Python set.
    """
    #different founder condition than used in score enhancement (no stipulation on genotype)
    founders = [n for n in G if G.in_degree(n) == 0]
    cover, uncovered = set(), set(G.nodes)
    while uncovered:
        best = max(founders, key=lambda f: len(nx.descendants(G, f) & uncovered) + (f in uncovered))
        cover.add(best)
        uncovered -= nx.descendants(G, best)
        uncovered.discard(best)
    return cover

def minimal_founder_coverage_size(G: nx.DiGraph) -> float:
    """
    Return the size of the minimal founder coverage set.
    """
    return len(minimal_founder_cover_set(G))


# ---------------------------------------------------------------------
# 12. Affected Generational Clustering
# ---------------------------------------------------------------------
def gen_aff_clustering(G):
    #reversing generational presentaiton of nodes
    aff_nodes = aff(G)
    gen = generations(G)
    aff_gens = {gen[n] for n in aff(G)}
    aff_gen_clusters = {g:0 for g in aff_gens}
    for node in aff_nodes:
        aff_gen_clusters[gen[node]] += 1
    avg_aff_gen_cluster = sum(aff_gen_clusters.values())/ len(aff_gen_clusters)
    #normalize to pedigree width
    return avg_aff_gen_cluster / pedigree_width(G)

# ---------------------------------------------------------------------
# 13. Founder Influence
# ---------------------------------------------------------------------
def founder_influence(G) -> Dict[str, float]:
    phen = nx.get_node_attributes(G, "phenotype")
    affected = {n for n, p in phen.items() if p == 2}
    memo_all, memo_aff = {}, {}
    def paths(u, memo, target=None):
        key = (u, id(target))
        if key in memo: return memo[key]
        total = 1 if target is None or u in target else 0
        for v in G.successors(u):
            total += paths(v, memo, target)
        memo[key] = total
        return total
    infl = {}
    for f in (n for n in G if G.in_degree(n)==0):
        all_p = paths(f, memo_all, None)
        aff_p = paths(f, memo_aff, affected)
        infl[f] = aff_p / all_p if all_p else 0
    return infl




# ---------------------------------------------------------------------
# ADDTIONAL GRAPH METRICS WRAPPER
# ---------------------------------------------------------------------
def additional_graph_metrics(G):
    add_metrics = {**basic_graph_features(G), **centralities(G)}
    add_metrics['transitive_reduction_ratio'] = transitive_reduction_ratio(G)
    add_metrics['width'] = pedigree_width(G)
    add_metrics['longest_path'] = longest_path_length(G)
    add_metrics['founder_cover_size'] = minimal_founder_coverage_size(G)
    add_metrics['aff_gen_clustering'] = gen_aff_clustering(G)
    add_metrics['founder_influence'] = founder_influence(G)
    
    return add_metrics

