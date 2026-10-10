"""Compare numerical outputs of reruns against immutable reference results."""
from pathlib import Path
import argparse
import json
import math
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--atol',type=float,default=1e-6)
parser.add_argument('--rtol',type=float,default=1e-4)
args=parser.parse_args()
failures=[];checked=0;missing=[]
SKIP={'elapsed_seconds','timing','environment','physical_evaluation_seconds','generation_seconds','seconds'}
def compare(expected,actual,path):
 global checked
 if isinstance(expected,dict):
  for k,v in expected.items():
   if k in SKIP or 'sha256' in k or k.endswith('_seconds'):continue
   if k not in actual:failures.append(path+'/'+k+': missing');continue
   compare(v,actual[k],path+'/'+k)
 elif isinstance(expected,list):
  if len(expected)!=len(actual):failures.append(path+': different lengths');return
  for i,v in enumerate(expected):compare(v,actual[i],path+'/'+str(i))
 elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
  checked+=1
  if not math.isclose(expected,actual,abs_tol=args.atol,rel_tol=args.rtol):failures.append(f'{path}: expected {expected}, got {actual}')
for file in sorted((ROOT/'reference_results').glob('*/results.json')):
 actual=ROOT/'experiments'/file.parent.name/('audit/results.json' if file.parent.name.startswith('E023') else 'results.json')
 if not actual.exists():missing.append(file.parent.name);continue
 compare(json.loads(file.read_text()),json.loads(actual.read_text()),file.parent.name)
print(f'Compared {checked} numerical fields; {len(failures)} differences beyond atol={args.atol}, rtol={args.rtol}.')
for line in failures[:100]:print(line)
if missing:print('Not rerun:',', '.join(missing))
if failures:raise SystemExit(1)
if not checked:raise SystemExit('No generated outputs found; first run scripts/reproduce.py. Reference summaries are not rerun results.')
