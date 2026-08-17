#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -A m5216
#SBATCH --job-name=matsim-agents
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Example job submission script for matsim-agents on Perlmutter
#
# BEFORE SUBMITTING:
# 1. Replace <your_allocation> with your actual project allocation
# 2. Adjust -N (number of nodes), -t (time), and -q (queue) as needed
# 3. Modify the python command to run your specific matsim-agents job
#
# QUEUE OPTIONS:
#   -q regular     - Standard GPU queue (A100 GPUs)
#   -q premium     - Premium GPU queue (faster turnaround)
#   -q flex        - Flex queue (lower priority, more availability)
#
# NODE OPTIONS:
#   -C gpu         - GPU nodes (A100)
#   -C cpu         - CPU-only nodes
#
# SUBMISSION:
#   sbatch job_perlmutter.sh

set -euo pipefail

# Setup environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/setup_matsim_perlmutter.sh" --gpu

# Print environment info
echo "================================"
echo "Job Information"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU Devices:"
nvidia-smi -L || echo "No GPU info available"
echo ""
echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version | head -n 1)"
echo ""

# Change to work directory (optional)
# cd /path/to/work/directory

# Example: Run matsim-agents with a configuration
# Adjust command as needed for your use case
echo "Starting matsim-agents job..."
python -m matsim_agents.main --config example_config.yaml

echo "Job completed successfully!"
