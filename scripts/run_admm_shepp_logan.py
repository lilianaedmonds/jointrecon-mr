import pickle
import os
import sigpy
import cupy as cp
import sys
import pprint
import inspect

# insert path above "scripts" folder:
sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])



dataset_path='data/shepp_logan/6_gate/100s/sinusoidal'
output_base_path = 'output/shepp_logan_dim_0/6_gate/TEST'
datasets=['shepp_logan_dim_0_op_kspace.p']


from admm.admm import admm_mr
from motion.motion_demons import motion_fun_demons, get_demons_parms
from motion.motion_inversion import invert_mvf


img_shape = (64,128,128)
beta=0.001
rho=0.1
target_gate_index=1
device=0



ds_idx = list(range(len(datasets)))

parms = {}
parms['demons']='diffeomorphic'
parms['scaling']=[[4,4,1],[2,2,1]]
parms['scaling_sigmas']=[8,4]

parms['intensitythreshold']=1e-1
parms['smoothing']=3

parms['spacing']=(1.0,1.0,1.0)
parms['normalization']=[]

for ds_index in ds_idx:

    dataset = datasets[ds_index]
    with cp.cuda.Device(device):
        Fs, ds = pickle.load(open(os.path.join(dataset_path,dataset),'rb'))

    print("Shapes and types of Fs and ds:")
    print([f.ishape for f in Fs], type(Fs))
    print(Fs)
    print(len(ds), type(ds))

    output_path = os.path.join(output_base_path,dataset[:-2],'rho_' + str(rho) )
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
            num_iter=50,
            motion_base='zu_lam')

