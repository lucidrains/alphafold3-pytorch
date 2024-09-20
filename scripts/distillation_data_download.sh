#!/bin/bash

######################### Batch Headers #########################
#SBATCH --partition=copy                                      # use partition `gpu` for GPU nodes
#SBATCH --account=pawsey1018                                  # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=1                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time 0-48:00:00                                     # time limit for the job (up to 48 hours: `0-48:00:00`)
#SBATCH --job-name=distillation_data_download                 # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#################################################################

# Load required modules
module load pawseyenv/2024.05
module load singularity/4.1.0-slurm

# Ensure AWS CLI is installed
awsv2 --install

# Define paths
AFDB_URL="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_cif_v4.tar"
OUTPUT_DIR="./data/afdb_data"

MMCIF_OUTPUT_DIR="$OUTPUT_DIR/unfiltered_train_mmcifs"
MSA_OUTPUT_DIR="$OUTPUT_DIR/data_caches/train"

mkdir -p "$MMCIF_OUTPUT_DIR"
mkdir -p "$MSA_OUTPUT_DIR"

# Run download commands
bash -c "
    wget -O $OUTPUT_DIR/afdb_swissprot_cif_v4.tar $AFDB_URL \
    && tar -xvf $OUTPUT_DIR/afdb_swissprot_cif_v4.tar -C $MMCIF_OUTPUT_DIR \
    && awsv2 s3 cp s3://openfold/pdb/ $MSA_OUTPUT_DIR --recursive --no-sign-request
"

# Inform user of task completion
echo "Task completed for job: $SLURM_JOB_NAME"
