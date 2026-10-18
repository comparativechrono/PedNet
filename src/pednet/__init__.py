"""PedNet: the mode of inheritance of a family, and how well each candidate variant co-segregates with its condition,
from exact pedigree likelihoods."""
from .genotypes import read_genotype_table, read_vcf
from .inheritance import InheritanceResult, ModeEvidence, infer_mode, infer_shared_mode
from .models import MODE_NAMES, MODES, Model
from .pedigree import Pedigree, PedigreeError, Person, pedigree_from_rows, read_ped, write_ped
from .segregation import (CombinedScore, Variant, VariantScore, evidence_strength, rank_variants,
                          rank_variants_across)

__version__ = '0.1.0'

__all__ = [
    'infer_mode', 'infer_shared_mode', 'rank_variants', 'rank_variants_across', 'evidence_strength', 'read_ped',
    'write_ped', 'pedigree_from_rows', 'read_vcf', 'read_genotype_table', 'Pedigree', 'Person', 'PedigreeError',
    'Variant', 'VariantScore', 'CombinedScore', 'InheritanceResult', 'ModeEvidence', 'Model', 'MODES', 'MODE_NAMES',
    '__version__',
]
