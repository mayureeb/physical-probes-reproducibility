"""Recompute completion rates and paired tests from released physical landing errors.

This verifies published analysis, without retraining or rerunning the simulator.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def equal(a,b,path=''):
 if isinstance(a,dict):
  for k,v in a.items():equal(v,b[k],path+'/'+k)
 elif isinstance(a,list):
  for i,v in enumerate(a):equal(v,b[i],path+'/'+str(i))
 elif isinstance(a,(float,int)):
  assert np.isclose(a,b,atol=1e-12,rtol=1e-10),(path,a,b)
 else:assert a==b,(path,a,b)

output=ROOT/'validation_outputs';output.mkdir(exist_ok=True)
for study,filename in [('E027_confirmatory_replication','evaluate_confirmatory.py'),('E028_model_based_planning','evaluate.py'),('E029_temporal_state_estimation','evaluate.py')]:
 sys.path.insert(0,str(ROOT/'experiments'/study))
 m=load_module(study,ROOT/'experiments'/study/filename)
 folder=ROOT/'reference_results'/study
 with np.load(folder/'physical_replay.npz',allow_pickle=False) as d:
  errors={k.removesuffix('__landing_error'):d[k] for k in d.files}
 reference=json.loads((folder/'results.json').read_text())
 comparisons={}
 if study.startswith('E027'):
  rng=np.random.default_rng(m.BOOTSTRAP_SEED);locked=[]
  for tolerance in (0.10,0.20):
   for probe,short in [('fixed_positive_first','positive'),('fixed_negative_first','negative')]:
    name=f'{short}_minus_multisine_at_{tolerance:.2f}m';r=m.paired(errors[f'{probe}__primary']<=tolerance,errors['multisine__primary']<=tolerance,rng);comparisons[name]=r;locked.append((name,r))
  for name,a,b in [('multisine_17_minus_9_samples','multisine__samples_17','multisine__samples_9'),('positive_9_minus_401_samples','fixed_positive_first__samples_9','fixed_positive_first__samples_401'),('true_parameters_minus_structured','half_then_full_forward__true_parameter_action','half_then_full_forward__structured')]:
   r=m.paired(errors[a]<=0.06,errors[b]<=0.06,rng);comparisons[name]=r;locked.append((name,r))
  m.holm(locked)
  descriptive={label:{'mean_landing_error_m':float(e.mean()),'completion_by_tolerance':{str(t):float(np.mean(e<=t)) for t in m.TOLERANCES}} for label,e in errors.items()}
 elif study.startswith('E028'):
  rng=np.random.default_rng(m.BOOTSTRAP_SEED)
  for probe in m.PROBES:
   comparisons[f'{probe}__estimated_planner_minus_learned_action']=m.paired(errors[f'{probe}__estimated_planner']<=0.06,errors[f'{probe}__learned_action']<=0.06,rng)
   comparisons[f'{probe}__true_minus_estimated_planner']=m.paired(errors[f'{probe}__true_planner']<=0.06,errors[f'{probe}__estimated_planner']<=0.06,rng)
  m.holm(list(comparisons.values()))
  descriptive={label:{'mean_landing_error_m':float(e.mean()),'completion':float(np.mean(e<=0.06))} for label,e in errors.items()}
 else:
  rng=np.random.default_rng(2922901)
  for name,a,b in [('budget_33_minus_9','budget_33','budget_9'),('budget_401_minus_9','budget_401','budget_9'),('true_state_minus_401','true_state','budget_401')]:comparisons[name]=m.paired(errors[a]<=0.06,errors[b]<=0.06,rng)
  m.holm(list(comparisons.values()))
  descriptive={label:{'mean_landing_error_m':float(e.mean()),'completion':float(np.mean(e<=0.06))} for label,e in errors.items()}
 equal(comparisons,reference['comparisons']);equal(descriptive,reference['descriptive'])
 (output/f'{study}_reanalyzed.json').write_text(json.dumps({'comparisons':comparisons,'descriptive':descriptive},indent=2)+'\n')
 print(study+': completion, landing errors, all 20,000-resample paired intervals, exact p-values, and Holm adjustments match.',flush=True)
