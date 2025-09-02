import ray
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import logging

@ray.remote(num_gpus=1)
class VLLMActorGPU:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 4096):
        """
        Initialize VLLM actor with a specific GPU
        
        Args:
            model_name: HuggingFace model name or path
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum sequence length
        """
        self.model_name = model_name
        self.gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else 0
        
        logging.info(f"Initializing VLLM on GPU {self.gpu_id} with model {model_name}")
        
        # Initialize VLLM engine
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=1,  # Single GPU per actor
            trust_remote_code=True
        )
        
        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )
        
        logging.info(f"VLLM Actor initialized successfully on GPU {self.gpu_id}")
    
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        """
        Generate completions for given prompts
        
        Args:
            prompts: List of input prompts
            sampling_params: Optional sampling parameters override
            
        Returns:
            List of generated completions
        """
        if sampling_params:
            params = SamplingParams(**sampling_params)
        else:
            params = self.default_sampling_params
            
        outputs = self.llm.generate(prompts, params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "gpu_id": self.gpu_id,
            "device": f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        }
    
    def health_check(self) -> bool:
        """Simple health check"""
        try:
            test_output = self.generate(["Hello"], {"max_tokens": 5})
            return len(test_output) > 0
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False


@ray.remote
class VLLMActorPool:
    """Pool manager for VLLM actors"""
    
    def __init__(self, model_name: str, num_actors: int = None, **vllm_kwargs):
        """
        Initialize a pool of VLLM actors
        
        Args:
            model_name: Model name to load
            num_actors: Number of actors (defaults to number of GPUs available)
            **vllm_kwargs: Additional arguments for VLLM initialization
        """
        self.model_name = model_name
        self.vllm_kwargs = vllm_kwargs
        
        # Determine number of actors based on available GPUs
        if num_actors is None:
            cluster_resources = ray.cluster_resources()
            num_actors = int(cluster_resources.get("GPU", 1))
        
        self.num_actors = num_actors
        self.actors = []
        
        # Create actors
        self._initialize_actors()
        
    def _initialize_actors(self):
        """Initialize VLLM actors"""
        logging.info(f"Initializing {self.num_actors} VLLM actors")
        
        for i in range(self.num_actors):
            actor = VLLMActorGPU.remote(
                model_name=self.model_name,
                **self.vllm_kwargs
            )
            self.actors.append(actor)
        
        # Wait for all actors to initialize
        health_checks = [actor.health_check.remote() for actor in self.actors]
        results = ray.get(health_checks)
        
        successful_actors = sum(results)
        logging.info(f"Successfully initialized {successful_actors}/{self.num_actors} actors")
        
    def generate_batch(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        """
        Generate completions using available actors in round-robin fashion
        
        Args:
            prompts: List of prompts to process
            sampling_params: Sampling parameters
            
        Returns:
            List of generated completions
        """
        if not self.actors:
            raise RuntimeError("No actors available")
        
        # Distribute prompts across actors
        actor_tasks = []
        for i, prompt in enumerate(prompts):
            actor_idx = i % len(self.actors)
            task = self.actors[actor_idx].generate.remote([prompt], sampling_params)
            actor_tasks.append(task)
        
        # Collect results
        results = ray.get(actor_tasks)
        # Flatten results since each actor returns a list
        flattened_results = [result[0] for result in results]
        
        return flattened_results
    
    def get_actor_info(self) -> List[Dict[str, Any]]:
        """Get information about all actors"""
        info_tasks = [actor.get_model_info.remote() for actor in self.actors]
        return ray.get(info_tasks)
    
    def shutdown(self):
        """Shutdown all actors"""
        for actor in self.actors:
            ray.kill(actor)
        self.actors = []