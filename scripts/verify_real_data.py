"""Verify newly downloaded TartanDrive bytes against the recorded download hashes."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
recorded=json.loads((ROOT/'reference_results/E024_real_dynamics_history/download_manifest.json').read_text())
current=json.loads((ROOT/'experiments/E024_real_dynamics_history/data/download_manifest.json').read_text())
expected={(x['run'],x['path']):(x['sha256'],x['bytes']) for x in recorded['files']}
actual={(x['run'],x['path']):(x['sha256'],x['bytes']) for x in current['files']}
if expected!=actual:raise RuntimeError('Downloaded raw files differ from the recorded experiment. Do not describe this as an exact data reproduction.')
print(f'All {len(expected)} raw file sizes/hashes match the recorded download manifest.')
