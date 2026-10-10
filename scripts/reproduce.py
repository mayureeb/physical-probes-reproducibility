"""Run the published study pipelines in dependency order in this checkout."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Each step is (environment, experiment, filename, arguments).
PIPELINES = {
 'E023': [('sim','E023_information_ceiling','build_data.py',[]),('ml','E023_information_ceiling','learn.py',[]),('sim','E023_information_ceiling','evaluate.py',[])],
 'E024': [('ml','E024_real_dynamics_history','build_data.py',[]),('ml','E024_real_dynamics_history','learn.py',[]),('ml','E024_real_dynamics_history','evaluate.py',[])],
 'E025': [('sim','E025_decision_ambiguity','build_data.py',[]),('ml','E025_decision_ambiguity','learn.py',[]),('sim','E025_decision_ambiguity','evaluate.py',[]),('ml','E025_decision_ambiguity','analyze_interval_tolerance.py',[]),('sim','E025_decision_ambiguity','analyze_noise_robustness.py',[])],
 'E026': [('sim','E026_estimator_bottleneck','build_fresh_data.py',[]),('ml','E026_estimator_bottleneck','run_analyses.py',[])],
 'E027': [('sim','E027_confirmatory_replication','generate_fresh_test.py',[]),('ml','E027_confirmatory_replication','fit_predict.py',[]),('sim','E027_confirmatory_replication','evaluate_confirmatory.py',[])],
 'E028': [('sim','E028_model_based_planning','generate_test.py',[]),('ml','E028_model_based_planning','fit_estimator.py',[]),('sim','E028_model_based_planning','plan_before_targets.py',[]),('sim','E028_model_based_planning','evaluate.py',[])],
 'E029': [('sim','E029_temporal_state_estimation','generate_data.py',[]),('ml','E029_temporal_state_estimation','fit_estimators.py',[]),('sim','E029_temporal_state_estimation','plan_before_targets.py',[]),('sim','E029_temporal_state_estimation','evaluate.py',[])],
 'E030': [('sim','E030_decision_directed_probe','search_candidates.py',[]),('sim','E030_decision_directed_probe','generate_confirmatory.py',[]),('ml','E030_decision_directed_probe','fit_and_freeze.py',[]),('sim','E030_decision_directed_probe','plan_and_freeze.py',[]),('sim','E030_decision_directed_probe','evaluate.py',[])],
 'E031': [('sim','E031_direct_simulator_fit','lock_protocol.py',[]),('sim','E031_direct_simulator_fit','generate_data.py',[]),('sim','E031_direct_simulator_fit','fit_and_freeze.py',[]),('sim','E031_direct_simulator_fit','plan_and_freeze.py',[]),('sim','E031_direct_simulator_fit','evaluate.py',[]),('ml','E031_direct_simulator_fit','analyze_parameter_recovery.py',[])],
 'diagnostics': [('sim','posthoc_decision_diagnostics','run_posthoc.py',[]),('ml','posthoc_decision_diagnostics','run_structured_decoder.py',[]),('sim','posthoc_decision_diagnostics','evaluate_structured_physical.py',[]),('ml','posthoc_decision_diagnostics','analyze_uncertainty.py',[]),('ml','posthoc_decision_equivalence','analyze.py',[]),('ml','posthoc_path_cap_sensitivity','analyze.py',[]),('ml','posthoc_pool_bayes','analyze.py',[]),('sim','posthoc_decision_sensitivity','analyze.py',['--worlds','512','--output','results_512.json'])],
}
DEPENDENCIES={'E027':['E025'],'E028':['E025'],'E031':['E029'],'diagnostics':['E025','E030','E031']}

def ordered(requested):
 result=[]
 def visit(name):
  for dep in DEPENDENCIES.get(name,[]):visit(dep)
  if name not in result:result.append(name)
 for name in requested:visit(name)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--study',choices=['all']+list(PIPELINES),default='all')
 parser.add_argument('--simulation-python',default=os.environ.get('MJX_PYTHON',sys.executable))
 parser.add_argument('--ml-python',default=os.environ.get('ML_PYTHON',sys.executable))
 parser.add_argument('--dry-run',action='store_true')
 parser.add_argument('--skip-dependencies',action='store_true',help='Only when prerequisite outputs already exist.')
 args=parser.parse_args()
 py={'sim':str(Path(args.simulation_python).resolve()),'ml':str(Path(args.ml_python).resolve())}
 requested=list(PIPELINES) if args.study=='all' else [args.study]
 studies=requested if args.skip_dependencies else ordered(requested)
 env=os.environ.copy();env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');env.setdefault('MPLBACKEND','Agg')
 logs=ROOT/'run_logs'
 for study in studies:
  for environment,name,script,extras in PIPELINES[study]:
   folder=ROOT/'experiments'/name
   command=[py[environment],str(folder/script),*extras]
   print(study,environment,name,script,*extras,flush=True)
   if not (folder/script).exists():raise FileNotFoundError(folder/script)
   if args.dry_run:continue
   for directory in ['audit','data','checkpoints']: (folder/directory).mkdir(exist_ok=True)
   logs.mkdir(exist_ok=True)
   with (logs/f'{name}_{Path(script).stem}.log').open('w') as log:
    subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   if study=='E024' and script=='build_data.py':
    subprocess.run([py['ml'],str(ROOT/'scripts/verify_real_data.py')],env=env,check=True)
 if not args.dry_run:print('Study pipelines finished. Compare generated outputs with reference_results; see docs/RESULTS.md.')

if __name__=='__main__':main()
