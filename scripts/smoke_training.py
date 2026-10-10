"""Train the published matched-action GRU on a tiny synthetic example."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/E025_decision_ambiguity'))
import learn

torch.set_num_threads(1)
rng=np.random.default_rng(42)
x=rng.normal(size=(8,9,3)).astype(np.float32)
y=np.linspace(0.3,1.0,8,dtype=np.float32)
vx=x[:4];target=y[:4];probe='fixed_positive_first'
validation={f'{probe}__oracle_speed':target,f'{probe}__interval_low':target-0.05,f'{probe}__interval_high':target+0.05}
model,mean,scale,record=learn.fit_gru(x,y,vx,validation,probe,25,'cpu')
pred=learn.predict_gru(model,vx,mean,scale,'cpu')
assert pred.shape==(4,) and np.isfinite(pred).all()
assert 1<=record['epoch']<=80
print('Training smoke passed: original 80-epoch loop, validation checkpoint selection, and finite decoded predictions on a tiny synthetic dataset. This is not a reproduction of the paper results.')
