"""Check independent centroid scoring and the declared objective-quality gate."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_centroid_capture import parse_captures
from objective_comparison import compare_captures, mean_objective, quality_cost, check_quality, validate_levels


def record(trainer, cost):
    return dict(dataset='known', metric='l2', roots=0, trainer=trainer, quality_cost=cost)


class ObjectiveTests(unittest.TestCase):
    def check_quietly(self, rows):
        with contextlib.redirect_stdout(io.StringIO()):
            return check_quality(rows)

    def test_multiline_capture_records(self):
        text = 'ignored output\nVXVECTOR_CENTROIDS {"trainer":"vx",\n"levels":[[1],\n[2]]}\n'
        text += 'NOTICE:  VXVECTOR_CENTROIDS {"trainer":"internal","levels":[[3]]}\n'
        self.assertEqual(parse_captures(text), [dict(trainer='vx', levels=[[1], [2]]),
                                               dict(trainer='internal', levels=[[3]])])
        legacy = text.replace('VXVECTOR_CENTROIDS', 'VEXTOR_CENTROIDS')
        self.assertEqual(parse_captures(legacy), parse_captures(text))
        with self.assertRaises(ValueError):
            parse_captures('VXVECTOR_CENTROIDS {broken}')

    def test_l2_known_answer_and_batching(self):
        points = np.array([[0.], [2.], [10.], [12.]])
        centers = np.array([[1.], [11.]])
        for batch in (1, 3, 100):
            self.assertEqual(mean_objective(points, centers, 'l2', batch), 1.0)

    def test_angular_metrics_and_quality_conversion(self):
        points = np.array([[1., 0.], [0., 1.]])
        centers = np.array([[1., 0.]])
        self.assertEqual(mean_objective(points, centers, 'cos'), 0.5)
        self.assertEqual(mean_objective(points, centers, 'ip'), -0.5)
        self.assertAlmostEqual(quality_cost(-0.9, 'ip'), 0.1)

    def test_cosine_zero_vector_semantics(self):
        self.assertEqual(mean_objective([[0., 0.], [1., 0.]], [[0., 0.]], 'cos'), 0.5)

    def test_invalid_objective_inputs(self):
        for points, centers in [([[float('inf')]], [[1.]]), ([[1.]], [[float('nan')]]),
                                ([[1., 2.]], [[1.]]), ([], [[1.]]), ([[]], [[]])]:
            with self.subTest(points=points, centers=centers), self.assertRaises(AssertionError):
                mean_objective(points, centers, 'l2')

    def test_quality_gate_rejects_worse_clustering(self):
        internal = mean_objective([[0.], [2.]], [[1.]], 'l2')
        vx = mean_objective([[0.], [2.]], [[1.4]], 'l2')
        with self.assertRaisesRegex(AssertionError, 'predefined quality limit'):
            self.check_quietly([record('internal', internal), record('vx', vx)])

    def test_quality_gate_compares_means_across_repeats(self):
        result = self.check_quietly([record('internal', 1), record('internal', 1),
                                    record('vx', 1.3), record('vx', 0.8)])
        self.assertAlmostEqual(result[0]['vx_mean_cost'], 1.05)
        self.assertEqual(result[0]['repeats'], 2)

    def test_quality_gate_rejects_nonfinite_costs(self):
        with self.assertRaisesRegex(AssertionError, 'finite'):
            quality_cost(float('nan'), 'ip')
        with self.assertRaisesRegex(AssertionError, 'finite'):
            self.check_quietly([record('vx', float('nan')), record('internal', 1)])

    def test_capture_pairing_and_known_objective(self):
        cases = {'known': dict(dataset='known', metric='l2', roots=0, seed=0,
                               points=np.array([[0.], [2.], [10.], [12.]], dtype='float32'))}
        captures = [dict(trainer=name, case='known', levels=[[[1.], [11.]], [[root]]])
                    for name, root in [('vx', 0.), ('internal', 6.)]]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_captures(captures, cases, leaves=2)['result'], 'passed')
        with self.assertRaisesRegex(AssertionError, 'Missing centroid'):
            compare_captures(captures[:1], cases, leaves=2)
        with self.assertRaisesRegex(AssertionError, 'duplicate centroid'):
            compare_captures(captures + captures[:1], cases, leaves=2)

    def test_capture_dimensions_and_normalization(self):
        with self.assertRaisesRegex(AssertionError, 'dimensions'):
            validate_levels([[[1.]], [[0.]]], 2, 0, 'vx', leaves=1)
        with self.assertRaisesRegex(AssertionError, 'unit norms'):
            validate_levels([[[2., 0.]], [[0., 0.]]], 2, 0, 'vx', leaves=1, spherical=True)


if __name__ == '__main__':
    unittest.main()
