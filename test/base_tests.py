import unittest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from davion.base import MetaFrame, GeneFrame
from davion.utils import *
from davion.normalisation import *


class TestGeneBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gf = GeneFrame()
        cls.gf.load_expression('data/tpm.csv')
        cls.gf.load_expression('data/counts.csv', exprs='counts')
        cls.gf.load_metadata('data/metadata.csv')
        cls.gf.metadata['quality'] = ['good'] * 10 + \
                                     ['bad'] * (cls.gf.metadata.shape[0] - 10)
        cls.gf.metadata['enjoyment'] = ['fun'] * 20 + \
                                       ['boring'] * (cls.gf.metadata.shape[0] - 20)
        cls.gf.metadata['numerical'] = range(cls.gf.metadata.shape[0])
        cls.gf.total_features()


class TestBase(unittest.TestCase):

    def setUp(self):
        self.mf = MetaFrame()
        self.mf.load_expression('data/tpm.csv')
        self.mf.load_expression('data/counts.csv', exprs='counts')
        self.mf.load_metadata('data/metadata.csv')
        self.mf.metadata['quality'] = ['good'] * 10 + ['bad'] * (self.mf.metadata.shape[0] - 10)

    def test_load(self):
        assert self.mf.expression.shape == (100, 389)
        assert (self.mf.expression.columns == self.mf.metadata.index).all()

    def test_query(self):
        good = self.mf.query('quality == "good"')
        assert good.expression.shape[1] == 10
        assert good.metadata.shape[0] == 10

    def test_slice(self):
        sliced = self.mf[0:5, -5:]
        assert sliced.metadata.shape[0] == 5
        assert sliced.expression.shape[1] == 5


class TestGeneFrame(TestGeneBase):

    def _add_mt_genes(self):
        mt_genes = pd.read_csv('../annotations/mouse_mt_genes.tsv', sep='\t')
        self.gf.add_feature_sets(mt=mt_genes['Gene ID'])

    def test_rename_genes(self):
        self.gf.map_genes('../annotations/mouse_annotation_map.tsv',
                          'Gene ID', 'Associated Gene Name', agg=np.sum,
                          sep='\t')
        assert not all(self.gf.expression.index.str.contains('ENSMUSG'))
        print(self.gf.expression.duplicated())
        assert self.gf.expression.duplicated().sum() == 0

    def test_mt_calc(self):
        self._add_mt_genes()
        assert 'mt_pct' in self.gf.metadata.columns

    def test_total_features(self):
        total_features = self.gf.total_features()

    def test_plot_qc(self):
        self.gf.total_features()
        self._add_mt_genes()
        plt.figure()
        self.gf.plot_qc()
        plt.savefig('data/qc_plot.pdf')

    def test_run_pca(self):
        self.gf._run_pca(n_components=3)
        assert self.gf.pca_scores.shape[1] == 3
        plt.figure()
        self.gf.plot_pca()
        self.gf.plot_pca(hue='quality')
        self.gf.plot_pca(shape='enjoyment')
        self.gf.plot_pca(hue='numerical', shape='enjoyment')
        plt.figure()
        self.gf.plot_pca(hue='quality', shape='enjoyment')
        plt.savefig('data/pca_plot.pdf')

    def test_pc_correlation(self):
        plt.figure()
        self.gf.feature_pca_correlation('total_features')
        plt.savefig('data/pca_corr.pdf')


class TestUtils(TestGeneBase):

    def xtest_get_efflen(self):
        lengths = get_effective_length(species='mouse')
        lengths.to_csv('../annotations/mouse_gene_lengths.csv')

    def test_convert_counts_tpm(self):
        lengths = pd.read_csv('../annotations/mouse_gene_lengths.csv',
                              index_col=0)
        tpm = count_to_tpm(self.gf.get_expression('counts'), lengths)
        assert int(tpm.sum(axis=1).mean()) == 1e6


if __name__ == '__main__':
    unittest.main()