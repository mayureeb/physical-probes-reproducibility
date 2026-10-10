"""Check release integrity, selected waveforms, expected count arithmetic, and pipeline coverage."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
manifest=json.loads((ROOT/'source_provenance.json').read_text())
for record in manifest['files']:
 path=ROOT/record['file']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==record['released_sha256'], record['file']
m=module('candidates',ROOT/'experiments/E030_decision_directed_probe/probe_candidates.py')
waveforms=m.library()
assert len(waveforms)==36
with np.load(ROOT/'fixtures/selected_waveforms.npz',allow_pickle=False) as frozen:
 for a,b in [('decision_directed','piecewise_14'),('d_optimal','multisine_03'),('fixed_positive_first','fixed_positive_first'),('fixed_negative_first','fixed_negative_first')]:
  assert np.array_equal(frozen[a],waveforms[b]),a
for study in ['E027_confirmatory_replication','E028_model_based_planning','E029_temporal_state_estimation','E030_decision_directed_probe']:
 d=json.loads((ROOT/'reference_results'/study/'results.json').read_text())
 for key,values in d['descriptive'].items():
  rates=values.get('completion_by_tolerance',{'primary':values.get('completion')})
  for rate in rates.values():
   assert 0<=rate<=1 and abs(rate*d['worlds']-round(rate*d['worlds']))<1e-6,(study,key,rate)
subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--dry-run'],check=True,stdout=subprocess.DEVNULL)
print('Package checks passed: source/reference hashes, waveform identity, count arithmetic, and all pipeline entrypoints.')
