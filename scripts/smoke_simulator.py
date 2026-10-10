"""Replay one frozen synthetic world through the original probe and release code."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/E025_decision_ambiguity'))
import build_data as build

fixture=np.load(ROOT/'fixtures/simulation_world.npz',allow_pickle=False)
theta=fixture['parameters'];goal=float(fixture['goal']);speed=float(fixture['release_speed'])
world=build.e020.e014.World(family=4,**dict(zip(build.PARAMETERS,map(float,theta))),goal=goal,seed=0)
_,models,initial=build.e020.make_models([world])
params=build.e020.parameters([world])
final,trace=build.batch_probe_with_cost(models,initial,params,build.jp.asarray(build.probes.library()['fixed_positive_first']))
data,z1,z2,*_=final
outcome=build.e021.batch_release(models,data,params,z1,z2,build.jp.asarray([speed],dtype=build.jp.float32))
landing=float(np.asarray(outcome[0])[0])
assert np.isfinite(np.asarray(trace)).all()
assert abs(landing-goal)<=0.06, (landing,goal)
print(f'Physical replay passed: goal={goal:.6f} m, landing={landing:.6f} m, error={abs(landing-goal):.6f} m')
