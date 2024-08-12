# JointRecon MR



## Getting started

Use Python >= 3.10

Create virtual environment and activate:
```
python -m venv ./venv
source ./venv/bin/activate
```
Install required libraries (numpy, sigpy, scipy, SimpleITK):
```
pip install -r requirements.txt
```

Create Shepp Logan example:
```
python phantoms/create_shepp_logan_phantoms.py
```
This will create three 3d phantom datasets (with three motion phases, and motion in one dimension), consisting of kspace and vector (motion) data.

Run ADMM on Shepp Logan phantom example:
```
python scripts/run_admm_shepp_logan.py
```
Output will be written to the `output``directory.

Download MR simulations from XCAT data:
```
cd data
./download_xcat_sims.py
```
Output data will be written to data/xcat_sims directory
