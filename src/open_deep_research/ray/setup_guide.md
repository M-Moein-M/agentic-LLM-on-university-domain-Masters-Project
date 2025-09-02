# Ray + VLLM + LangGraph Distributed Setup

This repository provides a complete setup for running LangGraph workflows with VLLM actors on a multi-node, multi-GPU Ray cluster.

## Architecture Overview

- **Ray Cluster**: Distributed computing framework managing resources across nodes
- **VLLM Actors**: GPU-accelerated language model inference engines (one per GPU)
- **LangGraph Workflow Tasks**: Your custom workflows wrapped as Ray tasks
- **Multi-Node Support**: Scales across cluster nodes with SLURM integration

## Quick Start

### 1. Installation

```bash
# Clone your repository and install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x setup_ray_cluster.sh
```

### 2. Local Development/Testing

```bash
# Check system requirements
./setup_ray_cluster.sh check

# Start local Ray cluster (single node)
./setup_ray_cluster.sh head

# In another terminal, run your application
python main_application.py \
    --model-name "microsoft/DialoGPT-medium" \
    --num-tasks 2 \
    --interactive
```

### 3. Multi-Node Cluster (SLURM)

```bash
# Submit job to SLURM
sbatch submit_ray_job.sbatch

# Check job status
squeue -u $USER

# View output
tail -f ray_job_<jobid>.out
```

## File Structure

```
your_project/
├── vllm_ray_actor.py          # VLLM actor definitions
├── langgraph_ray_task.py      # LangGraph workflow wrapper
├── main_application.py        # Main orchestration script
├── setup_ray_cluster.sh       # Ray cluster setup script
├── submit_ray_job.sbatch      # SLURM batch script
├── workflow_config.json       # Configuration file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Integrating Your LangGraph Workflow

### Option 1: Modify the Template

In `langgraph_ray_task.py`, replace the `MockLangGraphWorkflow` class with your actual LangGraph workflow:

```python
def _create_langgraph_workflow(self):
    from your_langgraph_module import YourWorkflowGraph
    
    # Initialize your workflow with VLLM pool
    workflow = YourWorkflowGraph(llm_pool=self.vllm_pool)
    return workflow
```

### Option 2: Extend the Base Class

Create a new file with your specific workflow:

```python
from langgraph_ray_task import LangGraphWorkflowTask
from your_langgraph_module import YourWorkflowGraph

@ray.remote
class MyCustomWorkflowTask(LangGraphWorkflowTask):
    def _create_langgraph_workflow(self):
        return YourWorkflowGraph(llm_pool=self.vllm_pool)
    
    def execute_workflow(self, input_data):
        # Custom execution logic
        return self.workflow.invoke(input_data)
```

## Configuration

### Model Selection

Update the model in your scripts:

```bash
export MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"  # Example
# or
python main_application.py --model-name "meta-llama/Llama-2-7b-chat-hf"
```

### Workflow Configuration

Edit `workflow_config.json`:

```json
{
  "vllm_settings": {
    "gpu_memory_utilization": 0.8,
    "max_model_len": 4096
  },
  "sampling_params": {
    "temperature": 0.7,
    "max_tokens": 512
  }
}
```

### SLURM Parameters

Edit `submit_ray_job.sbatch`:

```bash
#SBATCH --nodes=4           # Number of nodes
#SBATCH --gres=gpu:8        # GPUs per node
#SBATCH --time=04:00:00     # Time limit
#SBATCH --mem=64G           # Memory per node
```

## Usage Examples

### 1. Interactive Testing

```bash
python main_application.py \
    --model-name "microsoft/DialoGPT-medium" \
    --interactive
```

Commands in interactive mode:
- Type prompts directly for single inference
- `batch` - Run predefined batch test
- `status` - Check system status
- `quit` - Exit

### 2. Batch Processing

```bash
python main_application.py \
    --model-name "microsoft/DialoGPT-medium" \
    --num-tasks 4 \
    --workflow-config custom_config.json
```

### 3. Connect to Existing Cluster

```bash
# If Ray cluster is already running
python main_application.py \
    --ray-address "ray://head_node_ip:10001" \
    --model-name "your-model" \
    --num-tasks 8
```

## Monitoring

### Ray Dashboard

```bash
# Local cluster
http://localhost:8265

# Remote cluster (port forward)
ssh -L 8265:head_node:8265 your_cluster
```

### Cluster Status

```bash
./setup_ray_cluster.sh monitor
```

### SLURM Job Monitoring

```bash
# Check job status
squeue -u $USER

# View job output
tail -f ray_job_<jobid>.out

# Cancel job if needed
scancel <jobid>
```

## Performance Tuning

### GPU Memory Optimization

```python
# In workflow_config.json
"vllm_settings": {
    "gpu_memory_utilization": 0.8,  # Adjust based on GPU memory
    "max_model_len": 4096,          # Reduce for smaller GPUs
}
```

### Batch Size Tuning

```python
# In your workflow
def generate_batch(self, prompts, batch_size=8):
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Process batch...
```

### Multi-GPU Models

For models requiring multiple GPUs:

```python
@ray.remote(num_gpus=2)  # Use 2 GPUs per actor
class VLLMActorMultiGPU:
    def __init__(self, model_name):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=2,  # Use 2 GPUs
            gpu_memory_utilization=0.9
        )
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `gpu_memory_utilization` in config
   - Decrease `max_model_len` or batch sizes
   - Use smaller models or fewer actors per GPU

2. **Ray Connection Issues**
   - Check firewall settings for Ray ports
   - Ensure consistent Ray versions across nodes
   - Verify node connectivity

3. **SLURM Job Failures**
   - Check resource availability: `sinfo`
   - Verify module loading in sbatch script
   - Check disk space in `$SCRATCH` directory

### Debugging Commands

```bash
# Check Ray cluster status
ray status

# View Ray logs
ray logs

# Check GPU utilization
nvidia-smi

# Monitor Ray resources
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

### Log Analysis

```bash
# View SLURM job logs
less ray_job_<jobid>.out
less ray_job_<jobid>.err

# Ray specific logs
tail -f /tmp/ray/session_*/logs/raylet.out
```

## Extending the System

### Adding Custom Metrics

```python
import ray
from ray import serve

@ray.remote
class MetricsCollector:
    def collect_metrics(self):
        # Custom metrics collection
        pass
```

### Adding Load Balancing

```python
class SmartWorkflowManager(WorkflowManager):
    def select_least_loaded_task(self):
        # Implement load balancing logic
        pass
```

### Integration with External Systems

```python
# Add callbacks or webhooks
def on_workflow_complete(result):
    # Send results to external system
    pass
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Ray documentation: https://docs.ray.io/
3. Check VLLM documentation: https://vllm.readthedocs.io/
4. Review LangGraph documentation: https://langchain-ai.github.io/langgraph/

## License

[Your License Here]