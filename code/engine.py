"""Instrument the frozen notebooks at named, checked sites; keep their fitting code."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / 'methods'
SOURCES = {name: METHODS / f'{name}.ipynb' for name in ('synthetic', 'camus')}
SPLIT_SIGNATURE = ROOT / 'camus_split_signature.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=False).encode()


def key(obj):
    return hashlib.sha256(pack(obj)).hexdigest()


def atomic(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(pack(obj))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Frozen notebook hook changed: ' + old[:80])
    return source.replace(old, new)


def cells(branch):
    return [''.join(c['source']) for c in json.loads(SOURCES[branch].read_text())['cells']
            if c['cell_type'] == 'code']


def record_predictions(directory, task, outcome, model, ids, truth, predicted, scores=None):
    frame = pd.DataFrame(dict(patient_id=list(ids), observed=truth, predicted=predicted))
    if scores is not None:
        frame['probability'] = scores
    if frame.patient_id.duplicated().any() or not np.isfinite(frame.iloc[:, 1:]).all().all():
        raise ValueError('Invalid test predictions')
    frame.to_csv(directory / f'{task}_{outcome}_{model}.csv', index=False, float_format='%.17g')


def variance_table(eigenvalues, count):
    v = np.asarray(eigenvalues)
    if not np.isfinite(v).all() or v.sum() <= 0 or len(v) < count:
        raise ValueError('Invalid FPCA spectrum')
    pct = 100 * v[:count] / v.sum()
    return pd.DataFrame(dict(mode=np.arange(1, count+1), eigenvalue=v[:count],
                             explained_variance_percent=pct, cumulative_variance_percent=pct.cumsum()))


def parallel_alignment(source):
    """Batch the DP call within each fixed template, preserving curve order."""
    for name, indent in (('run_alignment', '        '),
                         ('align_to_fixed_template', '    ')):
        start = source.index('def ' + name + '(')
        end = source.index('\ndef ', start + 1)
        block = source[start:end]
        loop = indent + 'for i in range(K):\n'
        prep_start = indent + '    beta_i = center_beta(beta_raw[i].copy())'
        dp = indent + '    gamma_i = dp_align_srvf(q_template, q_i, lam=lambda_warp)'
        begin = block.index(loop)
        prep = block.index(prep_start, begin)
        stop = block.index(dp, prep)
        log = block[begin+len(loop):prep]
        preparation = block[prep:stop]
        parallel = (
            f'{indent}def prepare_curve(i):\n'
            f'{preparation}'
            f'{indent}    return beta_i, r_i, q_i\n\n'
            f'{indent}prepared = [prepare_curve(i) for i in range(K)]\n'
            f'{indent}gammas_in_order = dp_batch(q_template, [q for _, _, q in prepared], lambda_warp)\n'
            f'{indent}for i, (beta_i, r_i, q_i) in enumerate(prepared):\n'
            f'{log}'
            f'{indent}    gamma_i = gammas_in_order[i]'
        )
        block = replace_once(block, block[begin:stop+len(dp)], parallel)
        source = source[:start] + block + source[end:]
    return source


def load(branch, target, pilot=False, dp_batch=None):
    path = SOURCES[branch]
    source, plot = cells(branch)[:2]
    legacy = f'Path("runs/{branch}")'
    source = replace_once(source, legacy, 'destination')
    plot = replace_once(plot, legacy, 'destination')
    # One content signature per selector invocation, with the ordered raw arrays.
    selector = 'select_lambda_5fold_cv' + ('_real' if branch == 'camus' else '')
    start = source.index('def ' + selector + '(')
    end = source.index('\ndef ', start + 1)
    block = source[start:end]
    block = replace_once(block, '    for lam in LAMBDA_GRID:', '''    context = array_context(beta_raw, r_raw, X_raw)
    for grid_index, lam in enumerate(LAMBDA_GRID):''')
    rep = 'sim_id' if branch == 'synthetic' else 'repeat_id'
    block = replace_once(block, '            beta_train =', f'''            unit = [context, {rep}, grid_index, float(lam), fold_id]
            cached = checkpoint.read(unit)
            if cached is not None:
                rows.append(cached)
                continue
            beta_train =''')
    block = replace_once(block, '    cv_table =', '            checkpoint.write(unit, rows[-1])\n\n    cv_table =')
    source = source[:start] + block + source[end:]
    if dp_batch is not None:
        if branch != 'camus':
            raise ValueError('Parallel DP is implemented for the CAMUS grid.')
        source = parallel_alignment(source)
    convergence = []
    marker = '        mu = mu_new\n\n    return mu'
    source = replace_once(source, marker, '        mu = mu_new\n\n    convergence.append((float(diff), float(tol), it+1))\n    return mu')
    if branch == 'synthetic':
        marker = '    srvf_fpca_error_table.insert(0, "sim", sim_id)'
        source = replace_once(source, marker, '''    srvf_fpca_error_table['explained_variance_percent'] = variance_table(
        srvf_fpca_payload['all_eigenvalues'], 3)['explained_variance_percent'].values
''' + marker)
    else:
        source = replace_once(source, '    # Report the eigenvalues', '''    variance_table(payload['all_eigenvalues'], 10).to_csv(
        TAB_DIR / 'srvf_tangent_fpca_variance_explained_first10.csv', index=False)
    # Report the eigenvalues''')
        marker = '                metrics = evaluate_regression(pred_test, ys)'
        source = replace_once(source, marker, '''                record_predictions(pred_dir, 'regression', outcome, model_name,
                                   meta_test.loc[valid_test, 'patient_id'], ys, pred_test)
''' + marker)
        marker = '                            classification_predictions_for_fig.append({'
        source = replace_once(source, marker, '''                            record_predictions(pred_dir, 'classification', 'lowEF', model_name,
                                               meta_test.loc[valid_test, 'patient_id'], y_test,
                                               best_model.predict(Xs), y_prob)
''' + marker)
        marker = '        cluster_train = cluster_summary(meta_train, labels_train)'
        source = replace_once(source, marker, '''        pd.DataFrame(dict(patient_id=meta_test.patient_id, EF=meta_test.EF,
                          cluster=labels_test, model=cluster_name, k=best_k)).to_csv(
                          DATA_DIR / 'test_cluster_assignments.csv', index=False)
''' + marker)
    def array_context(*arrays):
        h = hashlib.sha256()
        for arr in arrays:
            a = np.ascontiguousarray(arr)
            h.update(pack([a.shape, a.dtype.str]))
            h.update(a.data)
        return h.hexdigest()
    env = dict(__name__='frozen_notebook', destination=target,
               convergence=convergence, array_context=array_context, record_predictions=record_predictions,
               pred_dir=target/'predictions', variance_table=variance_table,
               dp_batch=dp_batch)
    exec(compile(source, str(path), 'exec'), env)
    env['pred_dir'].mkdir()
    if pilot:
        # Only for the bounded execution test. Production uses notebook constants.
        env.update(N=24, N_FOLDS=2, N_ITER_ALIGN=1, LAMBDA_GRID=[0., .2])
        if branch=='synthetic':
            env.update(K=12, N_SIM=1)
        else:
            env['MAX_PATIENTS']=12
        env['s'] = np.linspace(0., 1., 24, endpoint=branch=='synthetic')
        env['dt'] = 1/(23 if branch=='synthetic' else 24)
    return env, compile(plot, str(path)+'#lambda-figure', 'exec')
