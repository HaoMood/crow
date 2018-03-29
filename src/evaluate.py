#!/usr/bin/env python
"""Gram pooling and evaluate the result.

This program is modified from CroW:
    https://github.com/yahoo/crow/

The compute_ap executable file provied by the Oxford5k dataset:
    http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
To compute the average precision for a ranked list of query christ_church_1,
run:
    ./compute_ap christ_church_1 ranked_list.txt
"""


from __future__ import division, print_function


__all__ = ['EvaluateManager']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-03-29'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-03-29'
__version__ = '1.0'


import itertools
import os
import sys
if sys.version[0] == '2':
    filter = itertools.ifilter
    input = raw_input
    map = itertools.imap
    range = xrange
    zip = itertools.izip
import tempfile

import numpy as np
import sklearn.decomposition
import sklearn.neighbors
import sklearn.preprocessing


class EvaluateManager(object):
    """Manager class to compute mAP.

    Attributes:
        _paths, dict of (str, str): Feature paths.
        _pca_model: sklearn.decomposition.PCA model for final PCA whitening.
    """
    def __init__(self, paths):
        print('Initiate.')
        self._paths = paths
        self._pca_model = None

    def fitPca(self):
        """Compute PCA whitening paramters from the via dataset.

        Args:
            dim, int: Dimension of the final retrieval descriptor.
        """
        print('Load via dataset features.')
        # Load all features.
        via_all_features, _ = self._loadFeature('via_all_conv')

        print('Fit PCA whitening paramters from via dataset.')
        # l2-normalize.
        via_all_features = np.vstack(via_all_features)
        sklearn.preprocessing.normalize(via_all_features, copy=False)
        self._pca_model = sklearn.decomposition.PCA(n_components=512,
                                                    whiten=True)
        self._pca_model.fit(via_all_features)

    def evaluate(self):
        """Evaluate the retrieval results."""
        print('Load test dataset features.')
        # Load all features.
        test_all_features, test_all_names = self._loadFeature('test_all_conv')
        test_crop_features, test_crop_names = self._loadFeature(
            'test_crop_conv')

        # l2-normalize, PCA whitening, and l2-normalize again.
        test_all_features, test_crop_features = self._normalization(
            test_all_features, test_crop_features)
        assert test_all_features.shape[1] == test_crop_features.shape[1]

        # Iterate queries, process them, rank results, and evaluate mAP.
        print('Evaluate the results.')
        all_ap = []
        for i in xrange(len(test_crop_names)):
            knn_model = sklearn.neighbors.NearestNeighbors(
                n_neighbors=len(test_all_features))
            knn_model.fit(test_all_features)
            _, ind = knn_model.kneighbors(
                test_crop_features[i].reshape(1, -1))
            ap = self._getAP(ind[0], test_crop_names[i], test_all_names)
            all_ap.append(ap)
        m_ap = np.mean(np.array(all_ap))
        print('mAP is', m_ap)

    def _loadFeature(self, conv_path):
        """Load and process conv features into weighted-pooled descriptors.

        This is a helper function of fitPca() and evaluate().

        Args:
            conv_path, str: Path of .npy conv features.

        Return:
            feature_list, list of np.ndarray: List of features.
            name_list, list of str: List of image names without extensions.
        """
        name_list = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(self._paths[conv_path])
            if os.path.isfile(os.path.join(self._paths[conv_path], f))])
        m = len(name_list)
        feature_list = []
        for i, name_i in enumerate(name_list):
            if i % 200 == 0:
                print('Processing %d/%d' % (i, m))
            # Load conv feature.
            conv_i = np.load('%s.npy' %
                             os.path.join(self._paths[conv_path], name_i))
            D, H, W = conv_i.shape
            assert D == 512
            conv_i = np.reshape(conv_i, (D, H * W))

            # CroW spatial weighting.
            alpha = np.sum(conv_i, axis=0, keepdims=True)
            assert alpha.shape == (1, H * W)
            sklearn.preprocessing.normalize(alpha, copy=False)

            # CroW channel weighting.
            Q = np.sum(conv_i > 0, axis=1, keepdims=True) / (H * W)
            beta = np.zeros((D, 1))
            beta[Q != 0] = np.log(np.sum(Q) / Q[Q != 0])

            y = np.sum((conv_i * alpha) * beta, axis=1)[np.newaxis, :]
            assert y.shape == (1, D)
            feature_list.append(y)
        return feature_list, name_list

    def _normalization(self, test_all_descriptors, test_crop_descriptors):
        """l2 normalize, PCA whitening, and l2 normalize again for test all and
        cropped query descriptors.

        This is a helper function of evaluate().

        Args:
            test_all_descriptors, np.ndarray: Before normalize.
            test_crop_descriptors, np.ndarray: Before normalize.

        Return
            test_all_descriptors, np.ndarray: After normalize.
            test_crop_descriptors, np.ndarray: After normalize.
        """
        test_all_descriptors = np.vstack(test_all_descriptors)
        test_crop_descriptors = np.vstack(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)

        test_all_descriptors = self._pca_model.transform(test_all_descriptors)
        test_crop_descriptors = self._pca_model.transform(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)
        assert test_all_descriptors.shape[1] == 512
        assert test_crop_descriptors.shape[1] == 512

        return test_all_descriptors, test_crop_descriptors

    def _getAP(self, ind, query_name, all_names):
        """Given a query, compute average precision for the results by calling
        to the compute_ap.

        This is a helper function of evaluate().
        """
        # Generate a temporary file.
        f = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = f.name
        f.writelines([all_names[i] + '\n' for i in ind])
        f.close()

        cmd = '%s %s %s' % (
            self._paths['compute_ap'],
            os.path.join(self._paths['groundtruth'], query_name), temp_filename)
        ap = os.popen(cmd).read()

        # Delete temporary file.
        os.remove(temp_filename)
        return float(ap.strip())


def main():
    """Main function of this program."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', type=str, required=True,
                        help='Dataset to evaluate.')
    parser.add_argument('--via', dest='via', type=str, required=True,
                        help='Dataset to assistant PCA whitening.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model to extract features.')
    args = parser.parse_args()
    if args.test not in ['oxford5k', 'paris6k']:
        raise AttributeError('--test parameter must be oxford5k/paris6k.')
    if args.via not in ['oxford5k', 'paris6k']:
        raise AttributeError('--via parameter must be Oxford5k/paris6k.')
    if args.model not in ['vgg16', 'vgg19']:
        raise AttributeError('--model parameter must be vgg16/vgg19.')

    project_root = os.popen('pwd').read().strip()
    test_data_root = os.path.join(project_root, 'data', args.test)
    via_data_root = os.path.join(project_root, 'data', args.via)
    paths = {
        'groundtruth': os.path.join(test_data_root, 'groundtruth/'),
        'test_all_conv': os.path.join(
            test_data_root, 'conv', args.model, 'all/'),
        'test_crop_conv': os.path.join(
            test_data_root, 'conv', args.model, 'crop/'),
        'via_all_conv': os.path.join(
            via_data_root, 'conv', args.model, 'all/'),
        'compute_ap': os.path.join(project_root, 'lib/compute_ap'),
    }
    for k in paths:
        if k != 'compute_ap':
            assert os.path.isdir(paths[k])
        else:
            assert os.path.isfile(paths[k])

    evaluate_manager = EvaluateManager(paths)
    evaluate_manager.fitPca()
    evaluate_manager.evaluate()


if __name__ == '__main__':
    main()
