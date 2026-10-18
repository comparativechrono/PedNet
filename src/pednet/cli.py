"""The pednet command:

    pednet mode FAMILY.ped                 the mode of inheritance
    pednet rank FAMILY.ped VARIANTS.vcf    candidate variants ranked by co-segregation
    pednet check FAMILY.ped                what PedNet makes of a PED file
"""
import argparse
import csv
import sys
from typing import List, Optional

from . import __version__
from .genotypes import read_genotype_table, read_vcf, vcf_samples
from .inheritance import DEFAULT_PENETRANCE, DEFAULT_PHENOCOPY, infer_mode, infer_shared_mode
from .models import MODES
from .pedigree import PedigreeError, read_ped
from .segregation import DEFAULT_ERROR_RATE, MIN_VARIANT_FREQUENCY, rank_variants, rank_variants_across


def _floats(text: str) -> List[float]:
    try:
        return [float(x) for x in text.replace(',', ' ').split()]
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected numbers separated by commas, not {text!r}')


def _families(args):
    families = read_ped(args.ped, args.family)
    if not families:
        raise PedigreeError(f'There are no people in {args.ped}')
    if getattr(args, 'proband', None) and len(families) > 1:
        raise PedigreeError('--proband needs --family, since the PED file has more than one family')
    return families


def _common(parser):
    parser.add_argument('ped', help='the PED file')
    parser.add_argument('--family', help='only this family (by default every family in the file)')
    parser.add_argument('--proband', help='the person through whom the family was found (by default each affected '
                                          'person in turn)')
    parser.add_argument('--modes', default=','.join(MODES),
                        help='the modes of inheritance to weigh up, separated by commas (default: all four)')
    parser.add_argument('--penetrance', type=_floats, default=list(DEFAULT_PENETRANCE),
                        help='the penetrance, or several to average over (default: 0.6,0.8,0.95,0.99)')
    parser.add_argument('--phenocopy', type=float, default=DEFAULT_PHENOCOPY,
                        help=f'the chance of being affected without the disease genotype (default: {DEFAULT_PHENOCOPY})')


def _write_tsv(rows, handle):
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    writer = csv.DictWriter(handle, fieldnames=fields or ['candidate'], delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)


def _mode(args) -> int:
    families = _families(args)
    if args.together:
        results = [infer_shared_mode(families, args.modes, args.penetrance, phenocopy=args.phenocopy,
                                     threshold=args.threshold)]
    else:
        results = [infer_mode(pedigree, args.modes, args.penetrance, phenocopy=args.phenocopy, proband=args.proband,
                              threshold=args.threshold) for pedigree in families.values()]
    if args.tsv:
        _write_tsv([row for r in results for row in r.table()], sys.stdout)
        return 0
    for result in results:
        print(result)
        print()
    return 0


def _format(value, digits=2):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _print_table(header, table):
    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]
    for row in [header] + table:
        print('  '.join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip())


def _note_strangers(names, people):
    """Say which genotyped samples are in no family, and stop if none of them is in one"""
    missing = sorted(set(names) - set(people))
    if missing and len(missing) == len(set(names)):
        raise PedigreeError('None of the genotyped samples is in the PED file: their names must be the IDs of the '
                            f'PED file ({", ".join(missing[:3])}{"..." if len(missing) > 3 else ""})')
    if missing:
        print(f'note: {len(missing)} genotyped sample(s) are in no family of the PED file and are left out '
              f'({", ".join(missing[:5])}{"..." if len(missing) > 5 else ""})', file=sys.stderr)


def _read_variants(args, people):
    path = str(args.variants).lower()
    if path.endswith(('.vcf', '.vcf.gz', '.vcf.bgz')):
        _note_strangers(vcf_samples(args.variants), people)
        variants = read_vcf(args.variants, samples=people, frequency_key=args.frequency_key,
                            passed_only=args.passed_only, gene_key=args.gene_key)
    else:
        variants = read_genotype_table(args.variants)
        _note_strangers({pid for v in variants for pid in v.genotypes}, people)
    if args.max_frequency is not None:
        kept = [v for v in variants if v.allele_frequency is None or v.allele_frequency <= args.max_frequency]
        if len(kept) < len(variants):
            print(f'note: {len(variants) - len(kept)} variant(s) with an allele frequency above {args.max_frequency} '
                  'are left out', file=sys.stderr)
        variants = kept
    return variants


def _rank(args) -> int:
    families = _families(args)
    people = {pid for pedigree in families.values() for pid in pedigree.ids}
    variants = _read_variants(args, people)
    options = dict(modes=args.modes, penetrance=args.penetrance, phenocopy=args.phenocopy,
                   min_frequency=args.min_frequency, error_rate=args.error_rate, compound=not args.no_compound)
    if len(families) > 1:
        return _rank_across(args, families, variants, options)
    pedigree = next(iter(families.values()))
    scores = rank_variants(pedigree, variants, proband=args.proband, **options)
    if args.out:
        with open(args.out, 'w', newline='') as handle:
            _write_tsv([s.row() for s in scores], handle)
    shown = scores[:args.top] if args.top else scores
    pairs = sum(s.compound for s in scores)
    print(f'Family {pedigree.family}: {len(scores) - pairs} variants'
          + (f' and {pairs} pairs of variants in one gene' if pairs else '') + ', the most likely cause first')
    header = ['rank', 'candidate', 'log10 odds', 'log10 BF', 'evidence', 'best mode', 'affected carriers',
              'unaffected carriers', 'note']
    table = [[_format(s.rank, 1), s.id, _format(s.log10_odds), _format(s.log10_bf), s.evidence or '-',
              (s.best_mode + ' (compound)' if s.compound and s.best_mode else s.best_mode) or '-',
              f'{s.affected_carriers}/{s.affected_carriers + s.affected_noncarriers}',
              f'{s.unaffected_carriers}/{s.unaffected_carriers + s.unaffected_noncarriers}', s.notes]
             for s in shown]
    _print_table(header, table)
    if args.top and len(scores) > args.top:
        print(f'... and {len(scores) - args.top} more' + (f' (all in {args.out})' if args.out else ''))
    return 0


def _rank_across(args, families, variants, options) -> int:
    combined = rank_variants_across(families, variants, **options)
    if args.out:
        with open(args.out, 'w', newline='') as handle:
            _write_tsv([c.row() for c in combined], handle)
    print(f'Families {", ".join(families)}: {len(combined)} candidates carried by an affected person in at least '
          'one family, the most likely cause over the families first')
    shown = combined[:args.top] if args.top else combined
    header = ['rank', 'candidate', 'log10 odds', 'log10 BF', 'evidence', 'families (log10 BF)']
    table = [[_format(c.rank, 1), c.id, _format(c.log10_odds), _format(c.log10_bf), c.evidence or '-',
              ', '.join(f'{name} {s.log10_bf:.2f}' for name, s in c.families.items())] for c in shown]
    _print_table(header, table)
    if args.top and len(combined) > args.top:
        print(f'... and {len(combined) - args.top} more' + (f' (all in {args.out})' if args.out else ''))
    return 0


def _check(args) -> int:
    for name, pedigree in _families(args).items():
        generations = pedigree.generations()
        print(f'Family {name}: {len([p for p in pedigree.people if not p.placeholder])} people, '
              f'{len(pedigree.affected())} affected, {max(generations.values()) + 1} generations')
        related = pedigree.consanguineous()
        if related:
            print(f'  children of related parents: {", ".join(sorted(related))}')
        for note in pedigree.notes + pedigree.problems():
            print(f'  note: {note}')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='pednet', description='Mode of inheritance and variant co-segregation '
                                     'from pedigree likelihoods.')
    parser.add_argument('--version', action='version', version=f'pednet {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    mode = sub.add_parser('mode', help='the probability of each mode of inheritance')
    _common(mode)
    mode.add_argument('--threshold', type=float, default=0.8,
                      help='the posterior probability needed to call a mode (default: 0.8)')
    mode.add_argument('--together', action='store_true',
                      help='take the families as having the same condition, and give one answer for them all')
    mode.add_argument('--tsv', action='store_true', help='print a table separated by tabs')
    mode.set_defaults(run=_mode)

    rank = sub.add_parser('rank', help='candidate variants ranked by co-segregation with the condition',
                          description='Ranks candidate variants by how well they co-segregate with the condition. '
                          'With several families in the PED file (and no --family), the families are taken to have '
                          'the same condition, and each candidate is ranked by its evidence summed over the families '
                          'in which an affected person carries it.')
    _common(rank)
    rank.add_argument('variants', help='a VCF file (.vcf or .vcf.gz), or a table with a column per person')
    rank.add_argument('--frequency-key', default='AF', help='the INFO field with the allele frequency (default: AF)')
    rank.add_argument('--gene-key', help='the INFO field naming the gene (default: the first of CSQ, ANN, BCSQ, GENE, '
                                         'SYMBOL, Gene.refGene, Gene.ensGene and GENEINFO present)')
    rank.add_argument('--passed-only', action='store_true', help='keep only VCF sites that passed their filters')
    rank.add_argument('--max-frequency', type=float,
                      help='leave out variants with an allele frequency above this (by default none are left out)')
    rank.add_argument('--min-frequency', type=float, default=MIN_VARIANT_FREQUENCY,
                      help=f'treat rarer variants as having this frequency (default: {MIN_VARIANT_FREQUENCY})')
    rank.add_argument('--error-rate', type=float, default=DEFAULT_ERROR_RATE,
                      help=f'the chance that a genotype call is one copy out (default: {DEFAULT_ERROR_RATE})')
    rank.add_argument('--no-compound', action='store_true',
                      help='don\'t score pairs of variants in one gene as compound heterozygotes')
    rank.add_argument('--top', type=int, default=20, help='how many to show (default: 20; 0 for all)')
    rank.add_argument('--out', help='write every candidate\'s scores to this file, separated by tabs')
    rank.set_defaults(run=_rank)

    check = sub.add_parser('check', help='what PedNet makes of a PED file')
    check.add_argument('ped', help='the PED file')
    check.add_argument('--family', help='only this family')
    check.set_defaults(run=_check)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (PedigreeError, ValueError, OSError) as error:
        print(f'pednet: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
