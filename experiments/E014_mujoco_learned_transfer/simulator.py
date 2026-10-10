"""Hybrid MuJoCo contact simulator for E014 rendered transfer experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco

DT=.002; ACCEL=8.; TOLERANCE=.06
FAMILIES=('static','memory_linear','memory_squared','memory_saturating','dual_target')
XML='''<mujoco><option timestep=".002" gravity="0 0 -9.81"/><worldbody><geom type="plane" size="4 1 .1" friction="{mu} .005 .0001"/><site name="target" type="box" pos="{goal} 0 .002" size=".08 .08 .002" rgba="0 1 0 .35"/><body name="object" pos="0 0 .055"><joint name="slide" type="slide" axis="1 0 0" damping="{damping}"/><geom type="box" size=".05 .05 .05" mass="{mass}" friction="{mu} .005 .0001" rgba=".1 .3 .9 1"/></body><camera name="scene" pos="1.4 -2 1.25" xyaxes=".82 .57 0 -.3 .43 .85"/></worldbody><actuator><motor name="fixture" joint="slide" gear="1" ctrllimited="true" ctrlrange="-50 50"/></actuator></mujoco>'''
@dataclass(frozen=True)
class World:
 family:int; mass:float; mu:float; damping:float; goal:float; gain:float; strength:float; tau1:float; tau2:float; mix:float; seed:int
def make_worlds(seed,count,families):
 rng=np.random.default_rng(seed); out=[]
 for _ in range(count):
  common=dict(mass=rng.uniform(.6,1.5),mu=rng.uniform(.06,.22),damping=rng.uniform(.01,.12),goal=rng.uniform(.3,1.),gain=rng.uniform(14,30),strength=rng.uniform(.2,1.),tau1=rng.uniform(.05,.25),tau2=rng.uniform(.7,1.8),mix=rng.uniform(.25,.7),seed=int(rng.integers(2**31)))
  out.extend(World(int(f),**common) for f in families)
 return out
def model(world): return mujoco.MjModel.from_xml_string(XML.format(mu=world.mu,damping=world.damping,mass=world.mass,goal=world.goal))
def extra_force(world,speed,z1,z2):
 if world.family==0:return 0.
 if world.family==1:return world.strength*z1
 if world.family==2:return world.strength*z1*z1
 if world.family==3:return world.strength*np.tanh(z1/.6)
 return world.strength*((1-world.mix)*z1+world.mix*z2)
def update(z,target,h,tau): return target+(z-target)*np.exp(-h/tau)
def simulate(world,release_speed,peak=1.2,render=False):
 m=model(world); d=mujoco.MjData(m); renderer=mujoco.Renderer(m,96,96) if render else None; rng=np.random.default_rng(world.seed); z1=z2=0.; trace=[]
 leg=np.arange(0,peak+1e-9,ACCEL*DT); targets=np.concatenate((leg,leg[-2::-1],-leg[1:],-leg[-2::-1],np.zeros(100)))
 for target in targets:
  speed=abs(float(d.qvel[0])); z1=update(z1,speed,DT,world.tau1); z2=update(z2,speed,DT,world.tau2); d.qfrc_applied[0]=-np.sign(d.qvel[0])*extra_force(world,speed,z1,z2); d.ctrl[0]=np.clip(world.gain*(target-d.qvel[0]),-50,50); mujoco.mj_step(m,d); trace.append((d.qvel[0],d.ctrl[0]+d.qfrc_constraint[0]+rng.normal(0,.04)))
 scene=None
 if renderer:
  renderer.update_scene(d,camera='scene'); scene=renderer.render().copy(); renderer.close()
 for _ in range(2500):
  if d.qvel[0]>=release_speed:break
  speed=abs(float(d.qvel[0])); z1=update(z1,speed,DT,world.tau1); z2=update(z2,speed,DT,world.tau2); d.qfrc_applied[0]=-np.sign(d.qvel[0])*extra_force(world,speed,z1,z2); d.ctrl[0]=np.clip(world.gain*(release_speed-d.qvel[0]),0,50); mujoco.mj_step(m,d)
 release=float(d.qpos[0]); d.ctrl[0]=0
 for _ in range(15000):
  speed=abs(float(d.qvel[0])); z1=update(z1,speed,DT,world.tau1); z2=update(z2,speed,DT,world.tau2); d.qfrc_applied[0]=-np.sign(d.qvel[0])*extra_force(world,speed,z1,z2); mujoco.mj_step(m,d)
  if abs(d.qvel[0])<1e-4:break
 return {'distance':float(d.qpos[0]-release),'trace':np.asarray(trace,dtype=np.float32),'scene':scene,'probe_z':(z1,z2)}
