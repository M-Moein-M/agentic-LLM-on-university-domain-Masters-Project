import ray
from typing import Dict, Any, List
import logging
from vllm_ray_actor import VLLMActorPool

@ray.remote
class LangGraphWorkflowTask:
    """
    Ray task wrapper for LangGraph workflow that uses VLLM actors
    """
    
    def __init__(self, model_name: str, workflow_config: Dict[str, Any] = None):
        """
        Initialize the LangGraph workflow task
        
        Args:
            model_name: Model name for VLLM actors
            workflow_config: Configuration for your LangGraph workflow
        """
        self.model_name = model_name
        self.workflow_config = workflow_config or {}
        
        # Initialize VLLM actor pool
        self.vllm_pool = None
        self._initialize_vllm_pool()
        
        # Initialize your LangGraph workflow here
        self.workflow = self._create_langgraph_workflow()
        
        logging.info("LangGraph workflow task initialized")
    
    def _initialize_vllm_pool(self):
        """Initialize the VLLM actor pool"""
        try:
            self.vllm_pool = VLLMActorPool.remote(
                model_name=self.model_name,
                gpu_memory_utilization=0.8,  # Leave some memory for other processes
                max_model_len=4096
            )
            logging.info("VLLM actor pool initialized")
        except Exception as e:
            logging.error(f"Failed to initialize VLLM pool: {e}")
            raise
    
    def _create_langgraph_workflow(self):
        """
        Create and configure your LangGraph workflow
        This is where you'd integrate your existing LangGraph setup
        """
        # Import your LangGraph components here
        # from your_langgraph_module import YourWorkflowGraph
        
        # Example placeholder - replace with your actual LangGraph workflow
        class MockLangGraphWorkflow:
            def __init__(self, llm_pool):
                self.llm_pool = llm_pool
            
            def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                """
                Process input through your LangGraph workflow
                Replace this with your actual workflow logic
                """
                # Example: Extract prompts from input
                prompts = input_data.get("prompts", [])
                
                if prompts and self.llm_pool:
                    # Generate using VLLM actors
                    sampling_params = input_data.get("sampling_params", {})
                    results = ray.get(
                        self.llm_pool.generate_batch.remote(prompts, sampling_params)
                    )
                    
                    return {
                        "status": "success",
                        "results": results,
                        "processed_count": len(results)
                    }
                
                return {"status": "no_prompts", "results": []}
        
        return MockLangGraphWorkflow(self.vllm_pool)
    
    def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Workflow execution results
        """
        try:
            logging.info(f"Executing workflow with input keys: {list(input_data.keys())}")
            
            # Execute your LangGraph workflow
            result = self.workflow.invoke(input_data)
            
            logging.info("Workflow execution completed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Workflow execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information including VLLM actors status"""
        try:
            vllm_info = ray.get(self.vllm_pool.get_actor_info.remote()) if self.vllm_pool else []
            
            return {
                "model_name": self.model_name,
                "workflow_config": self.workflow_config,
                "vllm_actors": vllm_info,
                "ray_node_id": ray.get_runtime_context().get_node_id(),
                "status": "healthy"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def shutdown(self):
        """Cleanup resources"""
        if self.vllm_pool:
            ray.get(self.vllm_pool.shutdown.remote())
        logging.info("LangGraph workflow task shutdown completed")


@ray.remote
def create_langgraph_workflow_task(model_name: str, workflow_config: Dict[str, Any] = None):
    """
    Factory function to create LangGraph workflow tasks
    
    Args:
        model_name: Model name for VLLM
        workflow_config: Workflow configuration
        
    Returns:
        Initialized LangGraphWorkflowTask
    """
    return LangGraphWorkflowTask(model_name, workflow_config)


# Utility functions for managing multiple workflow tasks
class WorkflowManager:
    """Manager for multiple LangGraph workflow tasks"""
    
    def __init__(self):
        self.tasks = []
        self.task_counter = 0
    
    def create_task(self, model_name: str, workflow_config: Dict[str, Any] = None) -> ray.ObjectRef:
        """Create a new workflow task"""
        task = LangGraphWorkflowTask.remote(model_name, workflow_config)
        self.tasks.append(task)
        self.task_counter += 1
        logging.info(f"Created workflow task #{self.task_counter}")
        return task
    
    def execute_on_all_tasks(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute workflow on all tasks"""
        if not self.tasks:
            return []
        
        # Execute on all tasks in parallel
        execution_refs = [
            task.execute_workflow.remote(input_data) 
            for task in self.tasks
        ]
        
        # Collect results
        results = ray.get(execution_refs)
        return results
    
    def get_all_system_info(self) -> List[Dict[str, Any]]:
        """Get system info from all tasks"""
        if not self.tasks:
            return []
        
        info_refs = [task.get_system_info.remote() for task in self.tasks]
        return ray.get(info_refs)
    
    def shutdown_all_tasks(self):
        """Shutdown all workflow tasks"""
        if self.tasks:
            shutdown_refs = [task.shutdown.remote() for task in self.tasks]
            ray.get(shutdown_refs)
            self.tasks = []
            logging.info("All workflow tasks shutdown")