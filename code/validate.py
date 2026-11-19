"""Independent checks on scientific products before a run can be sealed."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from engine import SPLIT_SIGNATURE, key

FIGURES = {'synthetic':['fig_lambda_curve_repeated','fig_synthetic_pipeline_example','fig_thickness_fpca_comparison','fig_shape_fpca_comparison'],
           'camus':['fig_lambda_curve_real_data','fig01_real_data_pipeline_clean','fig_srvf_tangent_fpca_shape_thickness_modes',
                    'fig_EF_prediction_and_low_EF_ROC_combined']}
REG = {'linear','ridge','lasso','elasticnet','random_forest','svr','xgboost','lightgbm'}
CLASS = {'logistic','elasticnet_logistic','random_forest','svm','xgboost','lightgbm'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_split(metadata, train, test):
    expected = json.loads(SPLIT_SIGNATURE.read_text())
    observed = {
        'patient_order_sha256': key(metadata.patient_id.tolist()),
        'ef_values_sha256': key([float(value).hex() for value in metadata.EF]),
        'train_indices_sha256': key(np.asarray(train).tolist()),
        'test_indices_sha256': key(np.asarray(test).tolist()),
    }
    require(len(metadata)==500 and len(train)==350 and len(test)==150,
            'Unexpected cohort or split size')
    for field, value in observed.items():
        require(value==expected[field], f'Frozen CAMUS input mismatch: {field}')


def verify_predictions(output, metadata, ids):
    meta = metadata.set_index('patient_id')
    reg = pd.read_csv(output/'tables/regression_repeated_results.csv').query("outcome=='EF'").set_index('model')
    cls = pd.read_csv(output/'tables/classification_repeated_results.csv').set_index('model')
    for task, table, models in [('regression',reg,REG),('classification',cls,CLASS)]:
        require(set(table.index)==models and table.index.is_unique, f'Missing or duplicate {task} models')
        for name in sorted(models):
            outcome = 'EF' if task=='regression' else 'lowEF'
            df = pd.read_csv(output/'predictions'/f'{task}_{outcome}_{name}.csv',float_precision='round_trip')
            require(df.patient_id.tolist()==ids, f'{name} patient identity/order changed')
            y = meta.loc[ids,'EF'].to_numpy()
            pred = df.predicted.to_numpy()
            if task=='regression':
                metrics = dict(r2=1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2),
                               rmse=np.sqrt(np.mean((pred-y)**2)),mae=np.mean(abs(pred-y)))
            else:
                y = (y<50).astype(int)
                require(np.isin(pred,[0,1]).all() and df.probability.between(0,1).all(), 'Invalid classes/scores')
                metrics = dict(accuracy=accuracy_score(y,pred),f1=f1_score(y,pred,zero_division=0),auc=roc_auc_score(y,df.probability))
            np.testing.assert_array_equal(df.observed,y)
            for k,v in metrics.items():
                np.testing.assert_allclose(table.loc[name,k],v,rtol=1e-11,atol=1e-12)


def products(branch, output, config):
    tables = output/'tables'
    def table(name):
        return pd.read_csv(tables/(name+'.csv'))
    figures = FIGURES[branch].copy()
    review = []
    if branch=='synthetic':
        n = config['N_SIM']
        a = table('synthetic_alignment_repeated_results')
        f = table('synthetic_srvf_tangent_fpca_error_results')
        require(len(a)==6*n and a.groupby('sim').Metric.nunique().eq(6).all(), 'Missing alignment metrics')
        require(np.isfinite(a[['Before alignment','After alignment','Change']]).all().all(), 'Invalid distances')
        require(len(f)==3*n and f.groupby('sim').n_modes.apply(lambda s:set(s)=={1,2,3}).all(), 'Missing FPCA rows')
        require(np.isfinite(f.select_dtypes('number')).all().all(), 'Nonfinite FPCA results')
        cv = table('synthetic_lambda_repeated_results')
        require(set(cv.sim)==set(range(n)), 'Missing simulation IDs')
        groups = cv.groupby('sim')
        for i in range(n):
            j=json.loads((output/'data'/f'closure_diagnostics_sim_{i:03}.json').read_text())
            require(len(j['aligned_closure_norms'])==config['K'] and np.isfinite(j['aligned_closure_norms']).all(), 'Invalid closure diagnostics')
    else:
        cv = table('lambda_repeated_results')
        groups = cv.groupby('repeat')
        require(set(cv['repeat'])=={0}, 'Outer repeat count changed')
        a=table('alignment_repeated_results')
        require(set(a.split)=={'train','test'} and np.isfinite(a.select_dtypes('number')).all().all(), 'Invalid split distances')
        f=table('srvf_tangent_fpca_variance_explained_first10')
        require(list(f['mode'])==list(range(1,11)) and np.isfinite(f).all().all()
                and np.all(np.diff(f.eigenvalue)<=1e-12) and f.cumulative_variance_percent.iloc[-1]<=100.000001,'Invalid FPCA spectrum')
        audit=json.loads((output/'data/inner_cv_audit_repeat_000.json').read_text())
        metadata = pd.read_csv(output/'data/metadata.csv', float_precision='round_trip')
        train_order = audit['external_train_row_ids']
        test_order = audit['external_test_row_ids']
        check_split(metadata, train_order, test_order)
        train = set(train_order)
        test_ids = metadata.iloc[test_order].patient_id.tolist()
        verify_predictions(output, metadata, test_ids)
        require(bool(audit['groups']), 'Missing fold audit')
        for folds in audit['groups']:
            require(len(folds)==config['N_FOLDS'], 'Missing inner folds')
            seen=[]
            for fold in folds:
                tr,va=set(fold['train_row_ids']),set(fold['validation_row_ids'])
                require(tr.isdisjoint(va) and tr|va==train and fold['representation']['n_lambda_training_rows']==len(tr), 'Inner fold isolation')
                seen.extend(va)
            require(sorted(seen)==sorted(train),'Inner validation coverage')
        cl=pd.read_csv(output/'data/test_cluster_assignments.csv')
        require(cl.patient_id.tolist()==test_ids, 'Invalid cluster assignments')
        normal=cl[cl.EF>=50]
        require(len(normal)==52,'Normal EF subgroup changed')
        sizes=normal.groupby('cluster').size().sort_index()
        if len(sizes)>=2:
            figures.append('fig_normal_EF_cluster_summary')
        if len(sizes)>=2 and sizes.iloc[:2].min()>=2:
            subgroup=table('normal_EF_cluster_comparison_simplified')
            require(set(subgroup.metric)=={'EF','area','perimeter','compactness','mean_thickness','sd_thickness',
                        'shape_distance_to_template','srvf_distance_to_template'},'Incomplete normal-EF comparison')
            if len(sizes)>2:
                review.append('Table 8 compares the two smallest cluster labels; report all subgroup counts.')
        else:
            review.append('The realized subgroup lacks two eligible clusters for the Table 8 comparison.')
        j=json.loads((output/'data/closure_diagnostics_repeat_000.json').read_text())
        for s,k in [('train',350),('test',150)]:
            v=j[s+'_closure_norms']
            require(len(v)==k and np.isfinite(v).all(), 'Invalid CAMUS closure diagnostics')
    for _,g in groups:
        expected = Counter((v,i) for v in config['LAMBDA_GRID'] for i in range(config['N_FOLDS']))
        require(Counter(zip(g['lambda'],g.fold))==expected, 'Incomplete lambda x fold grid')
        require(np.isfinite(g.srvf_geodesic_validation_mse).all(), 'Invalid CV loss')
    for name in figures:
        path=output/'figures'/(name+'.pdf')
        with path.open('rb') as stream:
            require(stream.read(5)==b'%PDF-',f'Missing figure {name}')
    return review


def pilot_camus(output):
    cv=pd.read_csv(output/'tables/pilot_lambda.csv')
    require(Counter(zip(cv['lambda'],cv.fold))==Counter([(0.,0),(0.,1),(.2,0),(.2,1)])
            and np.isfinite(cv.select_dtypes('number')).all().all(), 'Invalid CAMUS pilot products')
