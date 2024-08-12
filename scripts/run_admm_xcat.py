#!/usr/bin/env python3

import pickle
import os
import sys
import cupy as cp
from pathlib import Path
import sigpy
import pprint
import inspect

# insert path above "scripts" folder:
file_path = Path(__file__).parent.resolve()
if not file_path.parent in sys.path:
    sys.path.insert(0,str(file_path.parent))

from admm.admm import admm_mr
from motion.motion_demons import motion_fun_demons
from motion.motion_inversion import invert_mvf

num_iter = 100
dataset_path    = Path(file_path / '../data/xcat_sims')
output_base_path= Path(file_path / '../output/xcat_test')
datasets=['spokes_400_001.p']
ds_idx = list(range(len(datasets)))


img_shape = (80,256,256)
beta=0.001
rho=0.1
target_gate_index=2
device=0 # CUDA device to use


parms = {}
parms['demons']='diffeomorphic'
parms['scaling']=[[4,4,4],[2,2,2]]
parms['scaling_sigmas']=[8,4]

parms['intensitythreshold']=1e-3
parms['smoothing']=3 #3,5,7,9

parms['spacing']=(2.0,2.0,2.0)
parms['normalization']=[]

for dataset in datasets:

    ex1,ds,Fs,ex2,ex3 = pickle.load(open(os.path.join(dataset_path,dataset),'rb'))

    output_path = os.path.join(output_base_path,dataset[:-2],'iters_' + str(num_iter), 'smooth' + str(parms['smoothing']))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    admm_mr(ds,Fs,img_shape,
            motion_fun_demons,
            invert_mvf,
            motion_parms=parms,
            rho=rho, beta=beta,target_gate_index=target_gate_index,
            output_dir=output_path,
            device=device,
            do_pre_initialization=True,
            num_iter=num_iter,
            motion_base='zu_lam') # est. motion by aligning z+u and lambda

