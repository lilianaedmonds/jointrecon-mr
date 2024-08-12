import os
import numpy as np
import sigpy as sp
import sys
import pickle as p
sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])
project_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

from admm.utils_moco import golden_angle_coords_3d
output_dir = os.path.join(project_directory,'data','shepp_logan', '6_gate', '100s', 'sinusoidal')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img_shape = (64,128,128)
num_gates= 6
motion_extent = 2
num_spokes= 100 # per gate
# base cases:
# generate 3 phantom datasets - in each one, motion occurs
# along one of the axes:

coords = golden_angle_coords_3d(img_shape,num_gates*num_spokes,128)
norm_op = sp.linop.NUFFT(img_shape,coords)
max_eig = sp.app.MaxEig(norm_op.H * norm_op, dtype=np.complex64, max_iter=30, device=0).run()

Fs=[]


count=0
for gate in range(num_gates):
    Fs.append((1/np.sqrt(max_eig))*sp.linop.NUFFT(img_shape, coord=coords[:,count:count+num_spokes,...]))
    count+=num_spokes

# test:
print('Op norms:')
for gate in range(num_gates):
    max_eig = sp.app.MaxEig(Fs[gate].H * Fs[gate], dtype=np.complex64, max_iter=30, device=0).run()
    print(max_eig)

# Uncomment for Sinusoidal pattern
gate_indices = np.arange(num_gates)
freqs = 2 * np.pi / num_gates
shifts = motion_extent * np.sin(freqs * gate_indices)

# Uncomment for Triangular wave pattern
# half_gates = num_gates // 2
# shifts = np.concatenate([
#     np.linspace(0, motion_extent, half_gates),
#     np.linspace(motion_extent, 0, num_gates - half_gates)
# ])

for dim in range(3):
     
    ds=[]

    # get phantom
    ph = np.zeros((num_gates,*img_shape),dtype=np.complex64)
    ph[1, ...] = sp.shepp_logan(img_shape, dtype=np.complex64)

    for gate in range(num_gates):
        if gate != 1:  # Skip the reference gate
            ph[gate, ...] = np.roll(ph[1, ...], int(shifts[gate]), dim)

    # Save the absolute values of the phantom
    with open(os.path.join(output_dir, 'shepp_logan_dim_' + str(dim) + '.v'), 'wb') as f:
        f.write(np.reshape(np.abs(np.moveaxis(ph, 0, -1)), -1, order='F').astype(np.float32))

    # Create and save motion vectors
    vecs = np.zeros((num_gates, *img_shape, 3), dtype=np.float32)

    for gate in range(num_gates):
        if gate != 1:  # Skip the reference gate
            vecs[gate, ..., dim] = shifts[gate]

    vecs_inv = -vecs

    with open(os.path.join(output_dir,'shepp_logan_dim_'+str(dim)+'_vecs.mvf'),'wb') as f:
        f.write(np.reshape(np.moveaxis(vecs,0,-1),-1,order='F').astype(np.float32))
    with open(os.path.join(output_dir,'shepp_logan_dim_'+str(dim)+'_vecs_inv.mvf'),'wb') as f:
        f.write(np.reshape(np.moveaxis(vecs_inv,0,-1),-1,order='F').astype(np.float32))
    
    ph -= ph.min()
    ph /= ph.max()

    # project pahtnom data
    for gate in range(num_gates):
        ds.append(Fs[gate](ph[gate,...]))

    # save: 
    with open(os.path.join(output_dir,'shepp_logan_dim_'+str(dim)+ '_op_kspace.p'),'wb') as f:
        p.dump([Fs,ds],f)