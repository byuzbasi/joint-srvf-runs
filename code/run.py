"""Local entry point: bounded pilot, production, signed CV resume and verification."""
import argparse
import contextlib
import fcntl
import importlib.metadata as md
import json
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]='1'
os.environ['MPLBACKEND']='Agg'
from engine import ROOT, SOURCES, atomic, digest, key, load
from parallel_dp import ParallelDP
from state import State
from validate import check_split, products, require, REG, CLASS, pilot_camus
import numpy as np
import pandas as pd

FIELDS='SEED K N N_SIM N_FOLDS LAMBDA_GRID N_ITER_ALIGN alpha ALPHA_EMBED RANDOM_STATE TRAIN_SIZE TEST_SIZE N_REPEATS N_PCA_COMPONENTS EF_CLASSIFICATION_THRESHOLD MAX_PATIENTS'.split()


def inventory(folder):
    return {str(p.relative_to(folder)): [p.stat().st_size,digest(p)] for p in sorted(folder.rglob('*')) if p.is_file()}


def signature(branch, mode):
    sources=json.loads((ROOT/'frozen_sources.json').read_text())
    for name, checksum in sources.items():
        require(digest(ROOT/name)==checksum,f'Frozen input changed: {name}')
    modules='numpy pandas scipy sklearn matplotlib'.split()
    if branch=='camus':
        modules+=['cv2','nibabel','xgboost','lightgbm']
    import importlib
    versions={m: str(importlib.import_module(m).__version__) for m in modules}
    for name in ['fdasrsf','jinja2','threadpoolctl','joblib','contourpy','Pillow','fonttools']:
        try:
            versions[name]=md.version(name)
        except md.PackageNotFoundError:
            versions[name]=None
    return dict(branch=branch,mode=mode,frozen=sources,code={p.name:digest(p) for p in sorted((ROOT/'code').glob('*.py'))},
                python=sys.version,machine=platform.machine(),system=platform.platform(),packages=versions)


def raw_inputs(env):
    directories=sorted((ROOT/'database_nifti').glob('patient*'))
    require(len(directories)==500,'Expected 500 CAMUS directories')
    files={}
    rows=[]
    for d in directories:
        mask=Path(env['choose_mask_file'](d))
        im=Path(env['choose_image_file'](d))
        require(mask.name==d.name+'_2CH_ED_gt.nii.gz' and im.name==d.name+'_2CH_ED.nii.gz','Unexpected image selection')
        for p in [mask,im,*(p for p in sorted(d.rglob('*')) if p.suffix in {'.cfg','.txt','.csv'})]:
            files[str(p.relative_to(ROOT))]=[p.stat().st_size,digest(p)]
        rows.append(dict(patient_id=d.name,**env['parse_patient_metadata'](d)))
    meta=pd.DataFrame(rows)
    train,test=env['make_split_indices'](meta,0)
    check_split(meta,train,test)
    return files,meta


def timing(env,branch):
    s=np.linspace(0,1,250,endpoint=branch=='synthetic')
    q=np.stack([np.cos(2*np.pi*s),np.sin(2*np.pi*s),.2*np.cos(4*np.pi*s)],axis=1)
    other=np.roll(q,4,axis=0)
    times=[]
    for lam in [0.,.2,1.]:
        began=time.monotonic()
        warp=env['dp_align_srvf'](q,other,lam)
        times.append(time.monotonic()-began)
        require(len(warp)==250 and np.all(np.diff(warp)>0) and np.allclose(warp[[0,-1]],[0,1]),'DP pilot failed')
    return times


def execute(action,branch,run_id,workers):
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix='srvf-mpl-') as cache, contextlib.ExitStack() as resources:
        os.environ['MPLCONFIGDIR']=cache
        runs=ROOT/'runs'
        folder=runs/run_id
        if action=='status':
            print((folder/'progress.json').read_text() if (folder/'progress.json').exists() else 'Preflight or no run record yet.')
            return
        if action not in ('resume','verify'):
            folder.mkdir(parents=True,exist_ok=False)
        with (folder/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if action=='verify':
                seal=json.loads((folder/'manifest.json').read_text())
                record=json.loads((folder/'signature.json').read_text())
                require(seal['signature']==key(record) and seal['branch']==branch,'Run identity mismatch')
                require(seal['outputs']==inventory(folder/seal['attempt']/'output'),'Output manifest mismatch')
                require(seal['checkpoint_files']==inventory(folder/'checkpoints'),'Checkpoint inventory mismatch')
                require(seal['diagnostics_sha256']==digest(folder/seal['attempt']/'diagnostics.json'),'Diagnostic manifest mismatch')
                if record['mode']=='run' or branch=='synthetic':
                    products(branch,folder/seal['attempt']/'output',seal['config'])
                else:
                    pilot_camus(folder/seal['attempt']/'output')
                for name,expected in record.get('raw_files',{}).items():
                    p=ROOT/name
                    require([p.stat().st_size,digest(p)]==expected,'Raw CAMUS input changed: '+name)
                print('Verified',run_id,seal['decision'])
                return
            require(not (folder/'manifest.json').exists(),'Sealed run already exists')
            mode=json.loads((folder/'signature.json').read_text())['mode'] if action=='resume' else action
            pilot=mode=='pilot'
            stamp=signature(branch,mode)
            attempt=folder/f'attempt-{1+len(list(folder.glob("attempt-*"))):03}'
            attempt.mkdir()
            batch=resources.enter_context(ParallelDP(workers)) if workers>1 else None
            with (attempt/'run.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log):
                target=attempt/'output'
                # State is attached after the raw files and settings are signed.
                env,plot=load(branch,target,pilot,dp_batch=batch)
                cfg={k:env[k] for k in FIELDS if k in env}
                cfg['DP_WORKERS']=workers
                meta=None
                if branch=='camus':
                    stamp['raw_files'],meta=raw_inputs(env)
                    require(set(env['regression_model_specs']())==REG and set(env['classification_model_specs']())==CLASS,'Model dependency unavailable')
                stamp['config']=cfg
                sf=folder/'signature.json'
                if action=='resume':
                    require(json.loads(sf.read_text())==stamp,'Resume signature mismatch')
                else:
                    atomic(sf,stamp)
                if branch=='synthetic':
                    datasets=cfg['N_SIM']
                elif pilot:
                    datasets=1
                else:
                    datasets=1+cfg['N_FOLDS']
                    # All eligible outcome subsets on this frozen cohort use the same rows.
                    for col in ['EDV','ESV']:
                        require(meta[col].isna().all() or meta[col].notna().all(),'Additional eligibility groups need explicit accounting')
                total=datasets*cfg['N_FOLDS']*len(cfg['LAMBDA_GRID'])
                state=State(folder,key(stamp),total)
                env.update(checkpoint=state,convergence=state.convergence)
                state.report()
                from sklearn.exceptions import ConvergenceWarning
                model_warnings=Counter()
                warning_context=warnings.catch_warnings()
                warning_context.__enter__()
                fallback=warnings.showwarning
                def log_warning(message, category, filename, lineno, file=None, line=None):
                    if issubclass(category,ConvergenceWarning):
                        model_warnings[str(message)]+=1
                    else:
                        fallback(message,category,filename,lineno,file,line)
                warnings.showwarning=log_warning
                warnings.simplefilter('always',ConvergenceWarning)
                try:
                    start=time.monotonic()
                    dp_seconds=timing(env,branch) if pilot else []
                    if branch=='camus' and pilot:
                        # Use only a small subset of the frozen outer training split.
                        train, _ = env['make_split_indices'](meta, 0)
                        patients = sorted(meta.iloc[train].patient_id)[:env['MAX_PATIENTS']]
                        training_data = Path(resources.enter_context(
                            tempfile.TemporaryDirectory(prefix='srvf-camus-pilot-')))
                        for patient_id in patients:
                            (training_data/patient_id).symlink_to(
                                ROOT/'database_nifti'/patient_id, target_is_directory=True)
                        env['ROOT_DIR'] = training_data
                        beta,_,r,x,small=env['build_real_dataset']()
                        require(small.patient_id.tolist()==patients, 'Unexpected pilot training rows')
                        require(len(small)==12,'Small CAMUS sample changed')
                        _,cv,_=env['select_lambda_5fold_cv_real'](beta,r,x)
                        cv.to_csv(target/'tables/pilot_lambda.csv',index=False)
                        pilot_camus(target)
                        review=['12-patient, reduced-grid CV execution test only']
                    else:
                        if branch=='camus':
                            old=env['make_split_indices']
                            def split(data,repeat):
                                tr,te=old(data,repeat)
                                check_split(data,tr,te)
                                return tr,te
                            env['make_split_indices']=split
                        env['main']()
                        exec(plot,env)
                        review=products(branch,target,cfg)
                    require(state.count==total,'Incomplete CV checkpoint count')
                    conv=np.asarray(state.convergence)
                    require(conv.ndim==2 and np.isfinite(conv).all(),'Missing or invalid sphere-mean diagnostics')
                    capped=int(np.sum(conv[:,0]>=conv[:,1]))
                    if capped:
                        review.append(f'{capped} sphere mean(s) ended above tolerance; review before interpretation.')
                    if model_warnings:
                        review.append('Review estimator convergence warnings before interpreting model comparisons.')
                    diag=dict(dp_workers=workers, dp_processes_observed=len(batch.worker_pids) if batch else 1,
                              model_convergence_warnings=dict(model_warnings),elapsed_seconds=time.monotonic()-start,dp_seconds_250=dp_seconds,mean_calls=len(conv),
                              above_tolerance=capped,largest_final_step=float(conv[:,0].max()),
                              highest_iteration_count=int(conv[:,2].max()),review=review)
                    atomic(attempt/'diagnostics.json',diag)
                    if branch=='camus':
                        raw_end,_=raw_inputs(env)
                        require(raw_end==stamp['raw_files'],'Raw inputs changed during execution')
                    state.close('validated_pilot' if pilot else ('scientific_review' if review else 'validated_outputs'))
                    atomic(folder/'manifest.json',dict(branch=branch,attempt=attempt.name,config=cfg,signature=key(stamp),
                           decision=state.phase,review=review,outputs=inventory(target),checkpoint_files=inventory(state.units),diagnostics_sha256=digest(attempt/'diagnostics.json')))
                except BaseException:
                    state.close('failed')
                    raise
                finally:
                    warning_context.__exit__(None,None,None)
            print(folder, state.phase)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['pilot','run','resume','status','verify'])
    p.add_argument('branch',choices=SOURCES)
    p.add_argument('id')
    p.add_argument('--workers',type=int,choices=[1,4],help='defaults to 1 for synthetic and 4 for CAMUS')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.id):
        p.error('Use a short alphanumeric run ID, optionally with hyphens or underscores.')
    workers = args.workers or (4 if args.branch == 'camus' else 1)
    if args.branch == 'synthetic' and workers != 1:
        p.error('The synthetic workflow uses one worker.')
    execute(args.action,args.branch,args.id,workers)
