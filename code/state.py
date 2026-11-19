"""Verified scalar CV checkpoints and single-writer progress reporting."""
import json
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

import numpy as np

from engine import atomic, key


def now():
    return datetime.now(timezone.utc)


class State:
    def __init__(self, folder, signature, total):
        self.folder, self.signature, self.total = folder, signature, total
        self.units = folder / 'checkpoints'
        self.units.mkdir(exist_ok=True)
        self.seen = set()
        self.convergence = []
        self.initial = self.count = 0
        self.last = self.active = None
        self.failed = 0
        self.phase = 'preflight'
        self.events = threading.Event()
        self.lock = threading.RLock()
        self.start = time.monotonic()
        self.cv_seconds = 0.
        for file in self.units.glob('*.json'):
            self.unpack(file)
            self.seen.add(file.stem)
        self.count = self.initial = len(self.seen)
        self.worker = threading.Thread(target=self.heartbeat, daemon=True)
        self.worker.start()

    def unpack(self, file):
        doc = json.loads(file.read_text())
        body = doc['body']
        if (doc['sha256'] != key(body) or body['signature'] != self.signature
                or file.stem != key(body['unit'])):
            raise ValueError(f'Checkpoint signature mismatch: {file}')
        row = body['row']
        if (not np.isfinite(list(row.values())).all()
                or row['fold'] != body['unit'][-1] or row['lambda'] != body['unit'][-2]
                or row['srvf_geodesic_validation_mse'] < 0):
            raise ValueError(f'Invalid fold metrics: {file}')
        return body

    def read(self, unit):
        name = key(unit)
        with self.lock:
            if name in self.seen:
                body = self.unpack(self.units/(name+'.json'))
                self.convergence.extend(body['convergence'])
                return {column:body['row'][column] for column in body['columns']}
            self.phase = 'lambda_CV'
            self.active = unit
            self.begin = time.monotonic()
            self.offset = len(self.convergence)
        return None

    def write(self, unit, row):
        with self.lock:
            name = key(unit)
            path = self.units/(name+'.json')
            body = dict(signature=self.signature, unit=unit, row=row, columns=list(row), convergence=self.convergence[self.offset:])
            atomic(path, dict(body=body, sha256=key(body)))
            self.unpack(path)
            self.seen.add(name)
            self.count = len(self.seen)
            self.cv_seconds += time.monotonic()-self.begin
            self.last, self.active = unit, None
            self.phase = 'fit_or_summarize'
            self.report()

    def report(self):
        with self.lock:
            t = now()
            elapsed = time.monotonic()-self.start
            fresh = self.count - self.initial
            rate = fresh/self.cv_seconds if self.cv_seconds > 0 else None
            eta = (self.total-self.count)/rate if rate and self.count < self.total else None
            fields = dict(run_id=self.folder.name, phase=self.phase, unit='one lambda entry x fold x training dataset',
                          total=self.total, validated=self.count, committed=self.count, running=int(self.active is not None),
                          failed=self.failed, pending=max(0,self.total-self.count-int(self.active is not None)-self.failed),
                          percent_cv=round(100*self.count/self.total,2), elapsed_attempt_seconds=round(elapsed,2),
                          throughput_cv_units_per_second=rate, eta_cv_seconds=eta,
                          eta_cv_utc=(t+timedelta(seconds=eta)).isoformat() if eta else None,
                          whole_run_eta='unknown', heartbeat_utc=t.isoformat(), last_unit=self.last, active_unit=self.active)
            atomic(self.folder/'progress.json', fields)
            hist = self.folder/'progress.tsv'
            with hist.open('a') as stream:
                if stream.tell()==0:
                    stream.write('\t'.join(fields)+'\n')
                stream.write('\t'.join(str(v) for v in fields.values())+'\n')
            message = f"{fields['phase']} | CV {self.count}/{self.total} | attempt {elapsed:.0f}s"
            print(message, flush=True)
            if sys.stdout is not sys.__stdout__:
                print(message, file=sys.__stdout__, flush=True)

    def heartbeat(self):
        while not self.events.wait(45):
            self.report()

    def close(self, phase):
        self.events.set()
        self.worker.join()
        if phase=='failed' and self.active is not None:
            self.active, self.failed = None, 1
        self.phase = phase
        self.report()
