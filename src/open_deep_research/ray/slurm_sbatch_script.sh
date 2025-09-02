#!/bin/bash
#SBATCH --job-name=ray-vllm-langgraph
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ray_job_%j.out
#SBATCH --error=ray_job_%j.err

# Configuration
export MODEL_NAME="microsoft/DialoGPT-medium"  # Change to your desired model
export RAY_HEAD_PORT=10001
export RAY_TMPDIR="${SCRATCH}/ray_tmp_${SLURM_JOB_ID}"

# Create temporary directory for Ray
mkdir -p $RAY_TMPDIR

# Load required modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate your virtual environment
source /path/to/your/venv/bin/activate  # Adjust path

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Time limit: $SLURM_TIME_LIMIT"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Temporary Ray directory: $RAY_TMPDIR"

# Get node information
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

echo "Head node: $HEAD_NODE"
echo "All nodes: ${ALL_NODES[*]}"

# Function to setup Ray head node
setup_head_node() {
    echo "=== Setting up Ray head node on $HEAD_NODE ==="
    
    # Export Ray environment variables
    export RAY_ADDRESS=""
    export RAY_TMPDIR="$RAY_TMPDIR"
    
    # Check GPU availability
    nvidia-smi
    
    # Start Ray head node
    ray start --head \
        --node-ip-address=$HEAD_NODE \
        --port=$RAY_HEAD_PORT \
        --redis-port=$(shuf -i 6379-6399 -n 1) \
        --temp-dir=$RAY_TMPDIR \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$(nvidia-smi -L | wc -l) \
        --object-store-memory=$(( SLURM_MEM_PER_NODE * 1024 * 1024 / 2 )) \
        --verbose
    
    sleep 10
    
    echo "Ray head node started successfully"
    
    # Store head node address for workers
    echo "${HEAD_NODE}:${RAY_HEAD_PORT}" > $RAY_TMPDIR/ray_head_address
    
    echo "Ray cluster status:"
    ray status
}

# Function to setup Ray worker nodes
setup_worker_nodes() {
    echo "=== Setting up Ray worker nodes ==="
    
    # Read head node address
    RAY_HEAD_ADDRESS=$(cat $RAY_TMPDIR/ray_head_address)
    
    # Start worker nodes on remaining nodes
    for node in "${ALL_NODES[@]:1}"; do
        echo "Starting worker on $node connecting to $RAY_HEAD_ADDRESS"
        
        srun --nodes=1 --nodelist=$node --exclusive bash -c "
            export RAY_TMPDIR='$RAY_TMPDIR'
            nvidia-smi
            ray start --address='$RAY_HEAD_ADDRESS' \
                --temp-dir='$RAY_TMPDIR' \
                --num-cpus='$SLURM_CPUS_PER_TASK' \
                --num-gpus=\$(nvidia-smi -L | wc -l) \
                --object-store-memory=\$(( $SLURM_MEM_PER_NODE * 1024 * 1024 / 2 )) \
                --verbose
            sleep infinity  # Keep worker alive
        " &
        
        sleep 5
    done
    
    # Wait for all workers to connect
    sleep 30
    
    echo "All worker nodes started"
    ray status
}

# Function to run the application
run_application() {
    echo "=== Running VLLM + LangGraph Application ==="
    
    # Set Ray address for application
    export RAY_ADDRESS="ray://${HEAD_NODE}:${RAY_HEAD_PORT}"
    
    # Run the main application
    python main_application.py \
        --model-name "$MODEL_NAME" \
        --num-tasks $(( SLURM_JOB_NUM_NODES * 2 )) \
        --cluster-mode
}

# Function to cleanup
cleanup() {
    echo "=== Cleaning up ==="
    
    # Stop Ray on all nodes
    ray stop --force || true
    
    # Kill any remaining processes
    pkill -f ray || true
    
    # Clean up temporary directory
    rm -rf $RAY_TMPDIR || true
    
    echo "Cleanup completed"
}

# Trap cleanup function
trap cleanup EXIT INT TERM

# Main execution
main() {
    cd $SLURM_SUBMIT_DIR
    
    echo "=== Starting Ray cluster setup ==="
    
    # Setup head node
    if [ "$SLURM_PROCID" == "0" ]; then
        setup_head_node
        
        # Setup worker nodes if we have multiple nodes
        if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
            setup_worker_nodes
        fi
        
        # Wait a bit more for cluster to stabilize
        sleep 10
        
        # Display final cluster status
        echo "=== Final Ray cluster status ==="
        python -c "
import ray
ray.init('ray://${HEAD_NODE}:${RAY_HEAD_PORT}')
print('Cluster resources:', ray.cluster_resources())
print('Available resources:', ray.available_resources())
nodes = ray.nodes()
print(f'Number of nodes: {len(nodes)}')
for i, node in enumerate(nodes):
    print(f'Node {i}: {node[\"NodeManagerAddress\"]} - Alive: {node[\"Alive\"]}')
ray.shutdown()
        "
        
        # Run the application
        run_application
        
        # Keep job alive to see results
        echo "=== Application completed. Keeping cluster alive for inspection ==="
        sleep 300  # Keep alive for 5 minutes
        
    else
        # Non-head processes just wait
        sleep infinity
    fi
}

# Execute main function
main

echo "=== Ray job completed ===" 