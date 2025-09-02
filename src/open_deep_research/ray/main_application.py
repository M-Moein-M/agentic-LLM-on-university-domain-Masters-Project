#!/usr/bin/env python3
"""
Main application that orchestrates VLLM actors with LangGraph workflows using Ray
"""

import argparse
import json
import logging
import ray
import time
from typing import Dict, Any, List
from langgraph_ray_task import WorkflowManager, LangGraphWorkflowTask

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_workflow_config(config_path: str) -> Dict[str, Any]:
    """Load workflow configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using default config")
        return {
            "max_iterations": 5,
            "temperature": 0.7,
            "max_tokens": 512
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file {config_path}: {e}")
        raise


def create_sample_workload() -> List[Dict[str, Any]]:
    """Create sample workload for testing"""
    return [
        {
            "prompts": [
                "Explain the concept of machine learning in simple terms.",
                "What are the main benefits of using distributed computing?",
                "How does natural language processing work?"
            ],
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": 200,
                "top_p": 0.9
            }
        },
        {
            "prompts": [
                "Write a short story about a robot learning to paint.",
                "Describe the future of artificial intelligence.",
                "What makes a good programming language?"
            ],
            "sampling_params": {
                "temperature": 0.8,
                "max_tokens": 300,
                "top_p": 0.95
            }
        }
    ]


def run_distributed_workflow(
    model_name: str,
    num_tasks: int,
    workflow_config: Dict[str, Any],
    cluster_mode: bool = False
):
    """
    Run distributed workflow with VLLM actors
    
    Args:
        model_name: HuggingFace model name
        num_tasks: Number of workflow tasks to create
        workflow_config: Configuration for workflows
        cluster_mode: Whether running on multi-node cluster
    """
    logger.info(f"Starting distributed workflow with {num_tasks} tasks")
    logger.info(f"Model: {model_name}")
    logger.info(f"Cluster mode: {cluster_mode}")
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        if cluster_mode:
            # In cluster mode, connect to existing cluster
            ray.init()
        else:
            # Local mode
            ray.init(ignore_reinit_error=True)
    
    # Display cluster information
    logger.info("Ray cluster resources:")
    logger.info(ray.cluster_resources())
    logger.info("Available resources:")
    logger.info(ray.available_resources())
    
    # Create workflow manager
    workflow_manager = WorkflowManager()
    
    try:
        # Create workflow tasks
        logger.info(f"Creating {num_tasks} workflow tasks...")
        for i in range(num_tasks):
            task = workflow_manager.create_task(model_name, workflow_config)
            logger.info(f"Created task {i+1}/{num_tasks}")
        
        # Wait for tasks to initialize
        logger.info("Waiting for tasks to initialize...")
        time.sleep(10)
        
        # Get system information from all tasks
        logger.info("Gathering system information from tasks...")
        system_info = workflow_manager.get_all_system_info()
        
        for i, info in enumerate(system_info):
            if info.get("status") == "healthy":
                logger.info(f"Task {i+1}: {len(info.get('vllm_actors', []))} VLLM actors ready")
            else:
                logger.warning(f"Task {i+1}: {info.get('status', 'unknown status')}")
        
        # Create and run sample workload
        logger.info("Creating sample workload...")
        workloads = create_sample_workload()
        
        # Execute workloads
        for i, workload in enumerate(workloads):
            logger.info(f"Executing workload {i+1}/{len(workloads)}")
            
            start_time = time.time()
            results = workflow_manager.execute_on_all_tasks(workload)
            execution_time = time.time() - start_time
            
            logger.info(f"Workload {i+1} completed in {execution_time:.2f}s")
            
            # Process results
            successful_results = [r for r in results if r.get("status") == "success"]
            logger.info(f"Successful executions: {len(successful_results)}/{len(results)}")
            
            # Display sample results
            for j, result in enumerate(successful_results[:2]):  # Show first 2 results
                processed_count = result.get("processed_count", 0)
                sample_output = result.get("results", [])[:1]  # First result only
                logger.info(f"Task {j+1} processed {processed_count} prompts")
                if sample_output:
                    logger.info(f"Sample output: {sample_output[0][:100]}...")
        
        # Performance monitoring
        logger.info("=== Performance Summary ===")
        final_system_info = workflow_manager.get_all_system_info()
        
        total_actors = sum(len(info.get("vllm_actors", [])) for info in final_system_info)
        healthy_tasks = sum(1 for info in final_system_info if info.get("status") == "healthy")
        
        logger.info(f"Total VLLM actors: {total_actors}")
        logger.info(f"Healthy tasks: {healthy_tasks}/{len(final_system_info)}")
        logger.info(f"Average actors per task: {total_actors/len(final_system_info):.1f}")
        
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Shutting down workflow tasks...")
        workflow_manager.shutdown_all_tasks()
        
        if not cluster_mode:
            # Only shutdown Ray if we're not in cluster mode
            ray.shutdown()
        
        logger.info("Workflow execution completed")


def run_interactive_mode(model_name: str, workflow_config: Dict[str, Any]):
    """Run in interactive mode for testing and debugging"""
    logger.info("Starting interactive mode...")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Create single workflow task for interactive testing
    task = LangGraphWorkflowTask.remote(model_name, workflow_config)
    
    try:
        # Get system info
        system_info = ray.get(task.get_system_info.remote())
        logger.info(f"Task system info: {system_info}")
        
        print("\n=== Interactive VLLM + LangGraph Testing ===")
        print("Enter prompts (one per line). Type 'quit' to exit.")
        print("Type 'status' to check system status.")
        print("Type 'batch' to run predefined batch test.")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'status':
                    info = ray.get(task.get_system_info.remote())
                    print(f"Status: {info}")
                elif user_input.lower() == 'batch':
                    # Run batch test
                    test_data = {
                        "prompts": [
                            "Hello, how are you?",
                            "What is the weather like today?",
                            "Tell me a joke."
                        ],
                        "sampling_params": {"temperature": 0.8, "max_tokens": 100}
                    }
                    
                    print("Running batch test...")
                    start_time = time.time()
                    result = ray.get(task.execute_workflow.remote(test_data))
                    execution_time = time.time() - start_time
                    
                    print(f"Execution time: {execution_time:.2f}s")
                    print(f"Status: {result.get('status')}")
                    
                    if result.get('results'):
                        for i, output in enumerate(result['results']):
                            print(f"Output {i+1}: {output}")
                
                elif user_input:
                    # Single prompt processing
                    test_data = {
                        "prompts": [user_input],
                        "sampling_params": {"temperature": 0.7, "max_tokens": 150}
                    }
                    
                    start_time = time.time()
                    result = ray.get(task.execute_workflow.remote(test_data))
                    execution_time = time.time() - start_time
                    
                    print(f"[{execution_time:.2f}s] {result.get('results', ['No result'])[0]}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        # Cleanup
        ray.get(task.shutdown.remote())
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="VLLM + LangGraph Distributed Application")
    parser.add_argument("--model-name", type=str, required=True,
                      help="HuggingFace model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--workflow-config", type=str, default="workflow_config.json",
                      help="Path to workflow configuration file")
    parser.add_argument("--num-tasks", type=int, default=2,
                      help="Number of workflow tasks to create")
    parser.add_argument("--cluster-mode", action="store_true",
                      help="Run in cluster mode (connect to existing Ray cluster)")
    parser.add_argument("--interactive", action="store_true",
                      help="Run in interactive mode for testing")
    parser.add_argument("--ray-address", type=str, default=None,
                      help="Ray cluster address (e.g., ray://head_node:10001)")
    
    args = parser.parse_args()
    
    # Load workflow configuration
    workflow_config = load_workflow_config(args.workflow_config)
    logger.info(f"Loaded workflow config: {workflow_config}")
    
    # Connect to Ray cluster if address provided
    if args.ray_address:
        logger.info(f"Connecting to Ray cluster at {args.ray_address}")
        ray.init(args.ray_address)
        args.cluster_mode = True
    
    try:
        if args.interactive:
            run_interactive_mode(args.model_name, workflow_config)
        else:
            run_distributed_workflow(
                args.model_name, 
                args.num_tasks, 
                workflow_config,
                args.cluster_mode
            )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()