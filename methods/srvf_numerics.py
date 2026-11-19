"""Numerical corrections shared by the two isolated SRVF notebook copies.

No datasets, model fits, global configuration, or output files are accessed.
Sphere statistics remain an ambient approximation for closed curves.
"""
import numpy as np

STEPS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1))


def segment_costs(q_ref, q, lam):
    """Trapezoidal L2 segment costs plus the integrated slope penalty.

    Both inputs include the endpoint at t=1. A DP segment has constant
    slope dj/di. Its di source-grid intervals each have width 1/(n-1).
    The returned costs integrate over time, rather than counting steps.
    """
    n = len(q_ref)
    h = 1.0 / (n - 1)
    output = {}
    for di, dj in STEPS:
        if di >= n or dj >= n:
            continue
        slope = dj / di
        cost = np.zeros((n - di, n - dj))
        for offset in range(di + 1):
            weight = .5 if offset in (0, di) else 1.0
            positions = np.arange(n - dj) + offset * slope
            warped = np.column_stack([
                np.interp(positions, np.arange(n), q[:, k])
                for k in range(q.shape[1])
            ]) * np.sqrt(slope)
            reference = q_ref[offset:n - di + offset]
            cost += weight * np.sum(
                (reference[:, None, :] - warped[None, :, :]) ** 2, axis=2
            )
        output[(di, dj)] = h * (cost + di * lam * (slope - 1.0) ** 2)
    return output


def dp_path(q_ref, q, lam=0.0):
    """Return the optimum for the stated finite DP/quadrature objective.

    This is a discrete approximation, not a global optimizer over all
    continuous warpings, rotations, cyclic shifts, and sphere means.
    """
    q_ref, q = np.asarray(q_ref, dtype=float), np.asarray(q, dtype=float)
    if q_ref.ndim != 2 or q_ref.shape != q.shape or len(q) < 2:
        raise ValueError('SRVFs must have the same (n >= 2, dimension) shape.')
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be finite and nonnegative.')
    if not np.all(np.isfinite(q_ref)) or not np.all(np.isfinite(q)):
        raise ValueError('SRVF values must be finite.')
    costs = segment_costs(q_ref, q, lam)
    n = len(q)
    distance = np.full((n, n), np.inf)
    predecessor = np.full((n, n, 2), -1, dtype=int)
    distance[0, 0] = 0.0
    for i in range(n):
        for j in range(n):
            if not np.isfinite(distance[i, j]):
                continue
            for (di, dj), local in costs.items():
                ni, nj = i + di, j + dj
                if ni >= n or nj >= n:
                    continue
                value = distance[i, j] + local[i, j]
                if value < distance[ni, nj]:
                    distance[ni, nj] = value
                    predecessor[ni, nj] = i, j
    path = [(n - 1, n - 1)]
    while path[-1] != (0, 0):
        previous = tuple(predecessor[path[-1]])
        if previous[0] < 0:
            raise RuntimeError('No admissible DP path reaches the endpoint.')
        path.append(previous)
    return np.asarray(path[::-1]), float(distance[-1, -1])


def penalized_dp(q_ref, q, lam=0.0, endpoint_duplicated=False):
    """Preserve the notebooks' length-N, endpoint-inclusive gamma contract."""
    count = len(q_ref)
    if not endpoint_duplicated:
        q_ref = np.vstack([q_ref, q_ref[0]])
        q = np.vstack([q, q[0]])
    path, _ = dp_path(q_ref, q, lam)
    scale = len(q_ref) - 1
    grid = np.linspace(0.0, 1.0, count)
    gamma = np.interp(grid, path[:, 0] / scale, path[:, 1] / scale)
    # Keep the original tiny strict-monotonicity regularization.
    gamma += 1e-7 * grid
    gamma = (gamma - gamma[0]) / (gamma[-1] - gamma[0])
    return gamma


def integral(values, endpoint_duplicated=False):
    if endpoint_duplicated:
        return np.trapezoid(values, dx=1.0 / (len(values) - 1), axis=0)
    return np.mean(values, axis=0)


def closure_diagnostics(q, endpoint_duplicated=False):
    """Measure closure on the sphere representative before any correction."""
    q = np.asarray(q)
    velocity = q * np.linalg.norm(q, axis=1, keepdims=True)
    residual = integral(velocity, endpoint_duplicated)
    norm2 = float(integral(np.sum(q*q, axis=1), endpoint_duplicated))
    return {'closure_vector': residual, 'closure_norm': float(np.linalg.norm(residual)),
            'srvf_norm': float(np.sqrt(max(norm2, 0.0)))}


def inverse_closed_display(q, endpoint_duplicated=False):
    """Return an inverse and an explicitly corrected display approximation.

    Subtracting t times the closure vector changes the inverse curve. It
    does not project q onto the closed-curve SRVF submanifold.
    """
    n = len(q)
    h = 1.0 / (n - 1 if endpoint_duplicated else n)
    grid = np.arange(n) * h
    velocity = q * np.linalg.norm(q, axis=1, keepdims=True)
    raw = np.zeros_like(q)
    raw[1:] = np.cumsum(.5 * (velocity[:-1] + velocity[1:]) * h, axis=0)
    diagnostics = closure_diagnostics(q, endpoint_duplicated)
    correction = grid[:, None] * diagnostics['closure_vector']
    closed = raw - correction
    diagnostics['display_correction_max'] = float(np.max(np.linalg.norm(correction, axis=1)))
    return raw, closed, diagnostics


def cyclic_shift(values, shift, endpoint_duplicated=False):
    """Shift unique periodic samples; never shift a duplicated endpoint."""
    if endpoint_duplicated:
        unique = np.roll(values[:-1], shift, axis=0)
        return np.concatenate([unique, unique[:1]], axis=0)
    return np.roll(values, shift, axis=0)


def tangent_fpca(q_aligned, mu, logarithm, n_modes=3, endpoint_duplicated=False):
    """SVD in the same discrete L2 metric used by the sphere operations."""
    n, dimension = mu.shape
    weights = np.full(n, 1.0/(n-1 if endpoint_duplicated else n))
    if endpoint_duplicated:
        weights[[0, -1]] *= .5
    root_weights = np.sqrt(np.repeat(weights, dimension))
    vectors = np.vstack([logarithm(q, mu).reshape(-1) for q in q_aligned])
    U, S, Vt = np.linalg.svd(vectors*root_weights, full_matrices=False)
    count = min(n_modes, len(S))
    return {'mu': mu, 'tangent_vectors': vectors,
            'pcs': Vt[:count]/root_weights, 'scores': U[:, :count]*S[:count],
            'singular_values': S[:count], 'n_modes': count,
            'all_eigenvalues': S**2/max(len(vectors)-1, 1)}


def joint_mode_curves(mu, q_aligned, reference, fpca, exponential,
                      alpha, n_modes=3, endpoint_duplicated=False):
    """Reconstruct both coordinates from the same tangent direction.

    A single isotropic scale and translation, learned from reference
    training curves, is shared by the mean and every mode. The raw
    sphere mean and modes remain available alongside display curves.
    """
    payload = fpca(q_aligned, mu, n_modes=n_modes)
    _, mean_closed, mean_diagnostics = inverse_closed_display(mu, endpoint_duplicated)
    if endpoint_duplicated:
        length = lambda x: np.linalg.norm(np.diff(x, axis=0), axis=1).sum()
    else:
        length = lambda x: np.linalg.norm(np.roll(x, -1, axis=0)-x, axis=1).sum()
    target_length = np.mean([length(x) for x in reference])
    scale = float(target_length / max(length(mean_closed), 1e-12))
    # Consistent quadrature also avoids double-counting the synthetic endpoint.
    center = integral(mean_closed, endpoint_duplicated)
    target = np.mean([integral(x, endpoint_duplicated) for x in reference], axis=0)
    translation = target - scale * center
    display = lambda x: scale * x + translation
    modes, diagnostics = [], [{'kind': 'mean', **mean_diagnostics}]
    for k in range(payload['n_modes']):
        # Same ddof=0 amplitude convention as the original plotting code.
        tau = 2.0 * np.std(payload['scores'][:, k])
        pair = []
        for sign in (-1.0, 1.0):
            tangent = (sign * tau * payload['pcs'][k]).reshape(mu.shape)
            q_mode = exponential(tangent, mu)
            _, closed, diagnostic = inverse_closed_display(q_mode, endpoint_duplicated)
            pair.append(display(closed))
            diagnostics.append({'kind': f'mode_{k+1}_{sign:+g}', **diagnostic})
        modes.append(pair)
    return {'mean': display(mean_closed), 'modes': modes, 'diagnostics': diagnostics,
            'scale': scale, 'translation': translation, 'fpca': payload,
            'alpha': alpha}
