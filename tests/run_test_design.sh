#!/bin/bash
# Reference SLURM wrapper for the design integration tests.
# Edit the paths below and submit with: sbatch tests/run_test_design.sh
#
#SBATCH -J foldmapper_test
#SBATCH -p rome_veryshort
#SBATCH --qos=veryshort
#SBATCH -c 4
#SBATCH -o /home/hlkim/apps/foldmapper/logs/%x_%j.out
#SBATCH -e /home/hlkim/apps/foldmapper/logs/%x_%j.err

export FOLDMAPPER_PYTHON="/home/hlkim/anaconda3/envs/delta_if_env/bin/python"
export FOLDMAPPER_REPO="/home/hlkim/apps/foldmapper"
export FOLDMAPPER_CKPT="/home/hlkim/project/delta_if/training/260214/3/outputs/model_weights/epoch_last.pt"
export FOLDMAPPER_TEST_MONOMER="/home/hlkim/project/delta_if/full_data_filtering/260116/pdb_for_foldseek/AF-A0A842SDA8-F1-model_v4_TED02.pdb"
export FOLDMAPPER_TEST_BINDER="/home/hlkim/project/PPIBindBench/data/backbone_clean/9mxd_B_A.pdb"
export FOLDMAPPER_TEST_TRIMER="/home/hlkim/project/delta_if/rfdiffusion_foldability_t01/data/PUBLIC_RELEASE/insilico_tested/C3_Covid_binders/af2/C3_covid_binder_AF2_1.pdb"
export FOLDMAPPER_BINDER_NATIVE_PREFIX="RRKWQ"
export FOLDMAPPER_TRIMER_LEN="205"

echo "Node: $(hostname), started: $(date)"
bash "$FOLDMAPPER_REPO/tests/test_design.sh"
echo "Finished: $(date)"
