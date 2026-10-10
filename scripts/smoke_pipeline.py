"""Run a reduced E030 generation-to-evaluation pipeline in a temporary checkout.

Two worlds per split validate wiring; these outputs are not the paper results.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ml-python',required=True)
args=parser.parse_args()
ml_python=str(Path(args.ml_python).resolve())
with tempfile.TemporaryDirectory(prefix='physical-probes-smoke-') as temp:
 temp=Path(temp)
 shutil.copytree(ROOT/'experiments',temp/'experiments',ignore=shutil.ignore_patterns('__pycache__','data','checkpoints','figures','*.npz','*.log'))
 folder=temp/'experiments/E030_decision_directed_probe'
 for name in ['data','audit','checkpoints']:(folder/name).mkdir(exist_ok=True)
 shutil.copy2(ROOT/'fixtures/selected_waveforms.npz',folder/'audit/selected_waveforms.npz')
 shutil.copy2(ROOT/'reference_results/E030_decision_directed_probe/selection_lock.json',folder/'audit/selection_lock.json')
 spec=importlib.util.spec_from_file_location('smoke_generate',folder/'generate_confirmatory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.SPLITS={'train':(30130,2),'validation':(30230,2),'test':(30330,2)};m.CHUNK_SIZE=2
 m.main()
 # Evaluator fixes the original cohort size at 2048; change only the temporary smoke copy.
 evaluator=folder/'evaluate.py'
 evaluator.write_text(evaluator.read_text().replace('build.make_worlds(30330, 2048)','build.make_worlds(30330, 2)').replace('\"worlds\": 2048','\"worlds\": 2'))
 env=os.environ.copy();env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
 for py,filename in [(ml_python,'fit_and_freeze.py'),(sys.executable,'plan_and_freeze.py'),(sys.executable,'evaluate.py')]:
  subprocess.run([py,str(folder/filename)],cwd=folder,env=env,check=True)
 result=json.loads((folder/'results.json').read_text())
 assert result['worlds']==2
 assert len(result['descriptive'])==12
 assert all(0<=v['completion']<=1 for v in result['descriptive'].values())
 print('Reduced E030 end-to-end smoke passed: fresh generation, model selection/refit, action freezing, physical evaluation, and paired comparisons. Two worlds per split; not the paper results.')
