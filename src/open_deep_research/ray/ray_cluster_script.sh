#!/bin/bash

# setup_ray_cluster.sh
# Script to set up Ray cluster with VLLM actors

set -e

# Configuration
export RAY_CLUSTER_NAME="vllm-cluster"
export RAY_HEAD_PORT=10001
export RAY_REDIS_PORT=6379
export MODEL_NAME="microsoft/DialoGPT-medium"  # Change to your model
export WORKFLOW_CONFIG_PATH="workflow_config.json"

# Function to setup Ray head node
setup_ray_head() {
    echo "Setting up Ray head node..."
    
    # Kill any existing Ray processes
    ray stop --force || true
    
    # Start Ray head node
    ray start --head \
        --port=$RAY_HEAD_PORT \
        --redis-port=$RAY_REDIS_PORT \
        --num-cpus=0 \
        --num-gpus=$(nvidia-smi -L | wc -l) \
        --object-store-memory=2000000000 \
        --verbose
    
    echo "Ray head node started on $(hostname -I | awk '{print $1}'):$RAY_HEAD_PORT"
    
    # Wait for Ray to be ready
    sleep 10
    
    # Display cluster status
    python -c "
import ray
ray.init('ray://$(hostname -I | awk '{print $1}'):$RAY_HEAD_PORT')
print('Ray cluster resources:')
print(ray.cluster_resources())
ray.shutdown()
"
}

# Function to setup Ray worker node
setup_ray_worker() {
    if [ -z "$RAY_HEAD_ADDRESS" ]; then
        echo "Error: RAY_HEAD_ADDRESS environment variable not set"
        echo "Usage: RAY_HEAD_ADDRESS=head_node_ip:port $0 worker"
        exit 1
    fi
    
    echo "Setting up Ray worker node to connect to $RAY_HEAD_ADDRESS..."
    
    # Kill any existing Ray processes
    ray stop --force || true
    
    # Start Ray worker node
    ray start --address=$RAY_HEAD_ADDRESS \
        --num-cpus=0 \
        --num-gpus=$(nvidia-smi -L | wc -l) \
        --object-store-memory=2000000000 \
        --verbose
    
    echo "Ray worker node started and connected to $RAY_HEAD_ADDRESS"
}

# Function to run the main application
run_application() {
    echo "Starting VLLM + LangGraph application..."
    
    # Check if Ray is running
    if ! ray status > /dev/null 2>&1; then
        echo "Error: Ray is not running. Please start Ray cluster first."
        exit 1
    fi
    
    # Run the main application
    python main_application.py \
        --model-name "$MODEL_NAME" \
        --workflow-config "$WORKFLOW_CONFIG_PATH" \
        --num-tasks 2
}

# Function to check system requirements
check_requirements() {
    echo "Checking system requirements..."
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. GPU detection may fail."
    else
        echo "Available GPUs: $(nvidia-smi -L | wc -l)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    fi
    
    # Check Python dependencies
    python -c "
import sys
required_packages = ['ray', 'vllm', 'torch', 'transformers']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} (missing)')

if missing_packages:
    print(f'\\nMissing packages: {missing_packages}')
    print('Install with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('\\nAll required packages are available.')
"
}

# Function to stop Ray cluster
stop_ray() {
    echo "Stopping Ray cluster..."
    ray stop --force
    echo "Ray cluster stopped."
}

# Function to monitor cluster
monitor_cluster() {
    echo "Monitoring Ray cluster..."
    
    python -c "
import ray
import time

ray.init()

print('Ray Dashboard URL:', ray.get_dashboard_url())
print('Cluster Resources:', ray.cluster_resources())
print('Available Resources:', ray.available_resources())

# Monitor for 60 seconds
for i in range(12):
    time.sleep(5)
    print(f'[{i*5}s] Available resources:', ray.available_resources())

ray.shutdown()
"
}

# Main script logic
case "${1:-help}" in
    "head")
        check_requirements
        setup_ray_head
        ;;
    "worker")
        check_requirements
        setup_ray_worker
        ;;
    "run")
        run_application
        ;;
    "stop")
        stop_ray
        ;;
    "monitor")
        monitor_cluster
        ;;
    "check")
        check_requirements
        ;;
    "help"|*)
        echo "Usage: $0 {head|worker|run|stop|monitor|check}"
        echo ""
        echo "Commands:"
        echo "  head     - Start Ray head node"
        echo "  worker   - Start Ray worker node (requires RAY_HEAD_ADDRESS)"
        echo "  run      - Run the VLLM + LangGraph application"
        echo "  stop     - Stop Ray cluster"
        echo "  monitor  - Monitor cluster status"
        echo "  check    - Check system requirements"
        echo ""
        echo "Environment Variables:"
        echo "  RAY_HEAD_ADDRESS - Head node address for worker (e.g., 192.168.1.100:10001)"
        echo "  MODEL_NAME       - HuggingFace model name (default: microsoft/DialoGPT-medium)"
        echo ""
        echo "Examples:"
        echo "  $0 head                                    # Start head node"
        echo "  RAY_HEAD_ADDRESS=192.168.1.100:10001 $0 worker  # Start worker"
        echo "  $0 run                                     # Run application"
        ;;
esac