#!/bin/bash
# Reference SLURM wrapper for the design integration tests.
# Edit the paths below, then submit with: sbatch tests/run_test_design.sh
# See tests/README.md for what each input file must contain.
#
#SBATCH -J foldmapper_test
#SBATCH -c 4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
## Set the partition / QoS for your site, e.g.
##SBATCH -p <partition>
##SBATCH --qos=<qos>

# export FOLDMAPPER_PYTHON="/path/to/env/bin/python"   # optional; defaults to `python`
export FOLDMAPPER_REPO="/path/to/foldmapper"
export FOLDMAPPER_CKPT="/path/to/foldmapper_d512_blosum.pt"

export FOLDMAPPER_TEST_MONOMER="/path/to/monomer.pdb"        # single chain
export FOLDMAPPER_TEST_BINDER="/path/to/binder_complex.pdb"  # chain B = binder, A = target
export FOLDMAPPER_TEST_TRIMER="/path/to/trimer.pdb"          # C3 homo-trimer, chains A/B/C

# These two describe the files above - update them together (tests/README.md).
export FOLDMAPPER_BINDER_NATIVE_PREFIX="RRKWQ"
export FOLDMAPPER_TRIMER_LEN="205"

for v in FOLDMAPPER_REPO FOLDMAPPER_CKPT FOLDMAPPER_TEST_MONOMER \
         FOLDMAPPER_TEST_BINDER FOLDMAPPER_TEST_TRIMER; do
    case "${!v}" in
        /path/to/*) echo "ERROR: edit $v at the top of this script first." >&2; exit 1 ;;
    esac
done

echo "Node: $(hostname), started: $(date)"
bash "$FOLDMAPPER_REPO/tests/test_design.sh"
echo "Finished: $(date)"
