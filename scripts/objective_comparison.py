"""Score captured centroids independently on a common float32 source dataset.

Raw objectives use the selected metric in float64. For the quality gate, inner
product on unit vectors uses 1 + mean negative dot product. This makes its
nonnegative directional cost comparable to cosine. L2 uses mean squared error.
"""

import hashlib
import json

import numpy as np


MAXIMUM_RELATIVE_INCREASE = 0.10
ABSOLUTE_TOLERANCE = 1e-6


def mean_objective(points, centroids, metric, batch_size=256):
    points = np.asarray(points, dtype='float64')
    centroids = np.asarray(centroids, dtype='float64')
    if (points.ndim != 2 or centroids.ndim != 2 or len(points) == 0 or
            len(centroids) == 0 or points.shape[1] == 0 or points.shape[1] != centroids.shape[1]):
        raise AssertionError('Objective inputs need nonempty matrices with matching dimensions')
    if not np.isfinite(points).all() or not np.isfinite(centroids).all():
        raise AssertionError('Objective inputs must be finite')
    if metric not in ('l2', 'ip', 'cos') or batch_size <= 0:
        raise AssertionError('Invalid objective metric or batch size')
    total = 0.0
    center_norms = np.sqrt(np.sum(centroids * centroids, axis=1))
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        if metric == 'l2':
            delta = batch[:, None, :] - centroids[None, :, :]
            costs = np.sum(delta * delta, axis=2)
        else:
            # Broadcasting keeps this oracle independent of native kernels and BLAS threads.
            dot = np.sum(batch[:, None, :] * centroids[None, :, :], axis=2)
            if metric == 'ip':
                costs = -dot
            else:
                norms = np.sqrt(np.sum(batch * batch, axis=1))
                denominator = norms[:, None] * center_norms[None, :]
                costs = 1 - np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0)
                costs[(norms[:, None] == 0) & (center_norms[None, :] == 0)] = 0
        total += float(costs.min(axis=1).sum())
    result = total / len(points)
    if not np.isfinite(result):
        raise AssertionError('Objective must be finite')
    return result


def quality_cost(raw_mean, metric):
    if not np.isfinite(raw_mean):
        raise AssertionError('Raw objective must be finite')
    return max(0.0, 1 + raw_mean) if metric == 'ip' else max(0.0, raw_mean)


def validate_levels(levels, dims, roots, trainer, leaves=32, spherical=False):
    if len(levels) != (3 if roots else 2):
        raise AssertionError('Centroid capture has the wrong tree depth')
    result = [np.asarray(level, dtype='float32') for level in levels]
    for level in result:
        if level.ndim != 2 or level.shape[1] != dims or not np.isfinite(level).all():
            raise AssertionError('Centroid capture has invalid dimensions or nonfinite coordinates')
    if len(result[0]) != leaves or len(result[-1]) != 1:
        raise AssertionError('Centroid capture has the wrong leaf or singleton-root count')
    if roots and not (len(result[1]) == roots if trainer == 'vx' else 1 <= len(result[1]) <= roots):
        raise AssertionError('Centroid capture has an invalid coarse centroid count')
    if spherical:
        for level in result[:-1]:
            norms = np.sqrt(np.sum(level.astype('float64') ** 2, axis=1))
            if not np.allclose(norms, 1, atol=3e-4, rtol=3e-4):
                raise AssertionError('Spherical centroids must have unit norms')
    return result


def check_quality(records):
    if not records:
        raise AssertionError('Objective comparison needs at least one result')
    groups = {}
    for row in records:
        if not np.isfinite(row['quality_cost']) or row['quality_cost'] < 0:
            raise AssertionError('Objective quality costs must be finite and nonnegative')
        key = (row['dataset'], row['metric'], row['roots'])
        groups.setdefault(key, {}).setdefault(row['trainer'], []).append(row['quality_cost'])
    summaries = []
    for key, trainers in groups.items():
        if set(trainers) != {'vx', 'internal'} or len(trainers['vx']) != len(trainers['internal']):
            raise AssertionError('Objective comparison needs paired trainer runs')
        vx, internal = (float(np.mean(trainers[name])) for name in ('vx', 'internal'))
        limit = internal * (1 + MAXIMUM_RELATIVE_INCREASE) + ABSOLUTE_TOLERANCE
        row = dict(dataset=key[0], metric=key[1], roots=key[2], vx_mean_cost=vx,
                   internal_mean_cost=internal, maximum_vx_cost=limit,
                   repeats=len(trainers['vx']))
        summaries.append(row)
    failures = [row for row in summaries if row['vx_mean_cost'] > row['maximum_vx_cost']]
    for row in summaries:
        print(json.dumps(dict(objective_summary=row)), flush=True)
    if failures:
        raise AssertionError(f'Vx objective exceeds the predefined quality limit: {failures}')
    return summaries


def compare_captures(captures, cases, leaves=32):
    if not cases:
        raise AssertionError('Objective comparison needs at least one case')
    expected = {(trainer, case) for trainer in ('vx', 'internal') for case in cases}
    found = {}
    for capture in captures:
        key = (capture['trainer'], capture['case'])
        if key not in expected or key in found:
            raise AssertionError(f'Unexpected or duplicate centroid capture: {key}')
        found[key] = capture
    if set(found) != expected:
        raise AssertionError(f'Missing centroid captures: {expected - set(found)}')
    records = []
    for (trainer, case), capture in found.items():
        spec = cases[case]
        points, metric, roots = spec['points'], spec['metric'], spec['roots']
        if metric != 'l2' and not np.allclose(np.sqrt(np.sum(points.astype('float64') ** 2, axis=1)), 1, atol=3e-4, rtol=3e-4):
            raise AssertionError('Angular objective comparisons require unit source vectors')
        levels = validate_levels(capture['levels'], points.shape[1], roots, trainer,
                                 leaves=leaves, spherical=metric != 'l2')
        raw = mean_objective(points, levels[0], metric)
        row = dict(case=case, trainer=trainer, dataset=spec['dataset'], metric=metric,
                   roots=roots, vx_seed=spec['seed'] if trainer == 'vx' else None,
                   paired_vx_seed=spec['seed'], rows=len(points), raw_mean_objective=raw,
                   quality_cost=quality_cost(raw, metric),
                   leaf_sha256=hashlib.sha256(levels[0].tobytes()).hexdigest())
        if roots:
            # Each trainer has its own leaves. This is diagnostic, not a matched root-training problem.
            row['own_leaf_root_mean_objective'] = mean_objective(levels[0], levels[1], metric)
            row['actual_coarse_count'] = len(levels[1])
        records.append(row)
        print(json.dumps(dict(objective_result=row)), flush=True)
    summaries = check_quality(records)
    return dict(result='passed', centroid_captures=len(captures), objective_comparisons=len(summaries),
                maximum_relative_increase=MAXIMUM_RELATIVE_INCREASE,
                absolute_tolerance=ABSOLUTE_TOLERANCE)
