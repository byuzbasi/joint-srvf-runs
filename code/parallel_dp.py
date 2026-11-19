"""Run the frozen CAMUS DP kernel on independent curves using spawned workers."""
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def initialize_worker():
    global penalized_dp
    methods = Path(__file__).resolve().parents[1] / 'methods'
    sys.path.insert(0, str(methods))
    from srvf_numerics import penalized_dp


def solve_chunk(task):
    reference, curves, lam = task
    gammas = [penalized_dp(reference, curve, lam, endpoint_duplicated=False)
              for curve in curves]
    return os.getpid(), gammas


class ParallelDP:
    def __init__(self, workers):
        self.workers = workers
        self.worker_pids = set()
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,
        )

    def __call__(self, reference, curves, lam):
        # Several chunks per worker keep the final batches balanced. Map
        # returns in input order even when workers finish out of order.
        chunk_size = max(1, (len(curves) + 4*self.workers - 1)//(4*self.workers))
        tasks = ((reference, curves[i:i+chunk_size], lam)
                 for i in range(0, len(curves), chunk_size))
        gammas = []
        for pid, chunk in self.pool.map(solve_chunk, tasks):
            self.worker_pids.add(pid)
            gammas.extend(chunk)
        return gammas

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.pool.shutdown(wait=True, cancel_futures=True)
