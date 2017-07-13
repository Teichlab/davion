import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import itertools


class MetaFrame(object):

    default_exprs = 'tpm'

    def __init__(self, metadata=None, **kwargs):
        self.express_collection = {}
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = pd.DataFrame([])
        for exprs, table in kwargs.items():
            self.set_expression(table, exprs=exprs)

    def load_expression(self, filename, exprs=default_exprs):
        table = pd.read_csv(filename, index_col=0)
        self.set_expression(table, exprs=exprs)

    def load_metadata(self, filename):
        metadata = pd.read_csv(filename, index_col=0)
        self.set_metadata(metadata)

    def get_expression(self, exprs=default_exprs):
        return self.express_collection[exprs]

    def set_expression(self, new_expression, exprs=default_exprs):
        if not self.metadata.empty:
            assert ~bool(set(new_expression.columns)
                         .difference(set(self.metadata.index))), \
                'Indices do not match with expression %s' % exprs
            new_expression = new_expression.loc[:, self.metadata.index]
        self.express_collection[exprs] = new_expression

    def set_metadata(self, metadata):
        if self.express_collection:
            for exprs in self.express_collection:
                table = self.get_expression(exprs)
                assert ~bool(set(table.columns)
                             .difference(set(metadata.index))), \
                    'Indices do not match with expression %s' % exprs
                # Force the indices to match
                self.set_expression(table.loc[:, metadata.index], exprs)
        self.metadata = metadata

    def query(self, statement):
        """ Query by metadata """

        index = self.metadata.query(statement).index
        return MetaFrame(
            **{exprs: self.get_expression(exprs=exprs).loc[:, index]
               for exprs in self.express_collection.keys()},
            metadata=self.metadata.loc[index, :])

    def __getitem__(self, key):
        x, y = key
        return MetaFrame(
            **{exprs: self.get_expression(exprs=exprs).iloc[:, x]
               for exprs in self.express_collection.keys()},
            metadata=self.metadata.iloc[x, y])

    @property
    def expression(self):
        return self.get_expression()


class GeneFrame(MetaFrame):

    def __init__(self, *args, **kwargs):
        super(GeneFrame, self).__init__(*args, **kwargs)

        self.pca = None
        self.pca_scores = None

    def map_genes(self, annotations, original, replacement,
                  agg=np.sum, **kwargs):
        if isinstance(annotations, str):
            annotations = pd.read_csv(annotations, **kwargs)
        mapping = dict(zip(annotations[original], annotations[replacement]))
        for exprs, table in self.express_collection.items():
            table = table.rename(index=mapping)
            if agg is not None:
                table.groupby(table.index).agg(agg)

    def mt_percent(self, mt_genes, annotation='Associated Gene Name', **kwargs):
        # TODO: Generalise for ERCC's
        if isinstance(mt_genes, str):
            mt_genes = pd.read_csv(mt_genes, **kwargs)
        if isinstance(mt_genes, pd.DataFrame):
            mt_genes = list(mt_genes[annotation])
        overlapping = list(set(mt_genes).intersection(self.expression.index))
        pct_mt = self.expression.loc[overlapping, :].sum() / self.expression.sum()
        self.metadata['pct_mitochondrial'] = pct_mt
        return pct_mt

    def total_features(self, minimum=1):
        total_features = (self.expression > minimum).sum(axis=0)
        self.metadata['total_features'] = total_features
        return total_features

    def plot_qc(self, vars=['total_features', 'pct_mitochondrial'], **kwargs):
        sns.pairplot(data=self.metadata, vars=vars, **kwargs)

    def run_pca(self, exprs='tpm', n_components=50, **kwargs):
        self.pca = PCA(n_components=n_components, **kwargs)
        pca_scores = self.pca.fit_transform(self.get_expression(exprs).T)
        self.pca_scores = pd.DataFrame(pca_scores, index=self.metadata.index)

    def _plot_hue_shape(self, x, y, hue=None, shape=None, xlabel='',
                        ylabel='', title=''):
        y = pd.Series(y)
        x = pd.Series(x)
        if hue is not None:
            hue = pd.Series(hue)
            hue_set = list(set(hue))
            hues = itertools.cycle(plt.cm.rainbow(np.linspace(0, 1, len(hue_set))))
        else:
            hue = pd.Series([''] * len(x))
            hue_set = ['']
            hues = itertools.cycle(['b'])
        if shape is not None:
            shape = pd.Series(shape)
            shape_set = list(set(shape))
            shapes = itertools.cycle(['o', ',', '.', '*'])
        else:
            shape = pd.Series([''] * len(x))
            shape_set = ['']
            shapes = itertools.cycle(['o'])

        for hue_var in hue_set:
            color = next(hues)
            color_query = hue == hue_var
            for shape_var in shape_set:
                shape_query = shape == shape_var
                final_query = shape_query.values & color_query.values
                marker = next(shapes)
                label = hue_var + ' ' + shape_var
                plt.scatter(x[final_query], y.iloc[final_query],
                            color=color, marker=marker, label=label,
                            alpha=0.5)

        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    def plot_pca(self, pc1=0, pc2=1, hue=None, shape=None):
        if not len(self.pca_scores):
            self.run_pca()
        if isinstance(hue, str):
            hue = self.metadata[hue]
        if isinstance(shape, str):
            shape = self.metadata[shape]
        self._plot_hue_shape(
            self.pca_scores.loc[:, pc1], self.pca_scores.loc[:, pc2],
            hue=hue, shape=shape, xlabel='PC%d' % pc1, ylabel='PC%d' % pc2)

    def feature_pca_correlation(self, feature):
        if self.pca_scores is None:
            self.run_pca()
        feature_data = self.metadata[feature]
        correlations = []
        for pc in self.pca_scores:
            corr = np.corrcoef(x=feature_data, y=self.pca_scores[pc])[0, 1]
            correlations.append(corr)
        correlated = np.argsort(correlations)[::-1][:4]
        correlated_pcs = self.pca_scores.iloc[:, correlated]
        correlated_pcs[feature] = feature_data
        melted = correlated_pcs.melt(id_vars=feature, var_name='pc',
                                     value_name='score')
        sns.lmplot(x=feature, y='score', data=melted, col='pc', col_wrap=2)
