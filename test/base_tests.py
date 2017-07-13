import unittest
import numpy as np
from davion.base import MetaFrame, GeneFrame
from matplotlib import pyplot as plt


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

    def tearDown(self):
        pass


class TestGeneFrame(unittest.TestCase):

    def setUp(self):
        self.gf = GeneFrame()
        self.gf.load_expression('data/tpm.csv')
        self.gf.load_expression('data/counts.csv', exprs='counts')
        self.gf.load_metadata('data/metadata.csv')
        self.gf.metadata['quality'] = ['good'] * 10 + \
                                      ['bad'] * (self.gf.metadata.shape[0] - 10)
        self.gf.metadata['enjoyment'] = ['fun'] * 20 + \
                                        ['boring'] * (self.gf.metadata.shape[0] - 20)
        self.gf.total_features()

    def test_rename_genes(self):
        self.gf.map_genes('../annotations/mouse_annotation_map.tsv',
                          'Gene ID', 'Associated Gene Name', agg=np.sum,
                          sep='\t')
        assert not all(self.gf.expression.index.str.contains('ENMUSG'))

    def test_mt_calc(self):
        mt_pct = self.gf.mt_percent(
            '../annotations/mouse_mt_genes.tsv', sep='\t')

    def test_total_features(self):
        total_features = self.gf.total_features()

    def test_plot_qc(self):
        self.gf.total_features()
        self.gf.mt_percent('../annotations/mouse_mt_genes.tsv', sep='\t')
        plt.figure()
        self.gf.plot_qc()
        plt.savefig('data/qc_plot.pdf')

    def test_run_pca(self):
        self.gf.run_pca(n_components=3)
        assert self.gf.pca_scores.shape[1] == 3
        plt.figure()
        self.gf.plot_pca()
        self.gf.plot_pca(hue='quality')
        self.gf.plot_pca(shape='enjoyment')
        self.gf.plot_pca(hue='quality', shape='enjoyment')
        plt.savefig('data/pca_plot.pdf')

    def test_pc_correlation(self):
        plt.figure()
        self.gf.feature_pca_correlation('total_features')
        plt.savefig('data/pca_corr.pdf')


if __name__ == '__main__':
    unittest.main()