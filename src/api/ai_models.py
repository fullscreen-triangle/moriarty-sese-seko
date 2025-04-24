import json
import requests
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Union
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_models")

class MoriartyLLMClient:
    """
    Client for accessing specialized AI models via Hugging Face Inference API.
    Provides analysis capabilities for different biomechanical data types.
    """
    
    def __init__(self, api_token=None):
        """
        Initialize the LLM client with Hugging Face API token.
        
        Args:
            api_token: Hugging Face API token (defaults to HF_API_TOKEN environment variable)
        """
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            logger.warning("No Hugging Face API token provided. Set HF_API_TOKEN environment variable.")
            
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Model mapping for different analysis tasks
        self.models = {
            "biomechanical_analysis": "anthropic/claude-3-opus-20240229",
            "movement_patterns": "meta-llama/llama-3-70b-instruct",
            "technical_reporting": "anthropic/claude-3-haiku-20240307",
            "sprint_specialist": "your-username/sprint-biomechanics-expert",
            "performance_comparison": "your-username/performance-comparison-model",
            "coaching_insights": "anthropic/claude-3-sonnet-20240229",
            "quick_analysis": "mistralai/Mixtral-8x7B-Instruct-v0.1"
        }
        
        # Cache for storing responses
        self.cache = {}
        
    async def analyze(self, analysis_type: str, data: Dict, **kwargs) -> Dict:
        """
        Route analysis to appropriate model via Hugging Face API.
        
        Args:
            analysis_type: Type of analysis to perform
            data: Data to analyze
            **kwargs: Additional parameters for specific analysis types
            
        Returns:
            Analysis results from the AI model
        """
        model_id = self.models.get(analysis_type)
        if not model_id:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        # Generate cache key based on analysis type and data
        cache_key = f"{analysis_type}:{hash(json.dumps(data, sort_keys=True))}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Prepare prompt based on analysis type
        prompt = self._generate_prompt(analysis_type, data, **kwargs)
            
        # Call Hugging Face API
        result = await self._call_api(model_id, prompt)
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def analyze_sync(self, analysis_type: str, data: Dict, **kwargs) -> Dict:
        """
        Synchronous version of analyze method.
        
        Args:
            analysis_type: Type of analysis to perform
            data: Data to analyze
            **kwargs: Additional parameters for specific analysis types
            
        Returns:
            Analysis results from the AI model
        """
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.analyze(analysis_type, data, **kwargs))
        finally:
            loop.close()
    
    def _generate_prompt(self, analysis_type: str, data: Dict, **kwargs) -> str:
        """
        Generate appropriate prompt based on analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            data: Data to analyze
            **kwargs: Additional parameters for specific analysis types
            
        Returns:
            Prompt string for the AI model
        """
        if analysis_type == "biomechanical_analysis":
            return self._biomechanical_analysis_prompt(data, **kwargs)
        elif analysis_type == "movement_patterns":
            return self._movement_patterns_prompt(data, **kwargs)
        elif analysis_type == "technical_reporting":
            return self._technical_reporting_prompt(data, **kwargs)
        elif analysis_type == "sprint_specialist":
            return self._sprint_specialist_prompt(data, **kwargs)
        elif analysis_type == "performance_comparison":
            return self._performance_comparison_prompt(data, **kwargs)
        elif analysis_type == "coaching_insights":
            return self._coaching_insights_prompt(data, **kwargs)
        elif analysis_type == "quick_analysis":
            return self._quick_analysis_prompt(data, **kwargs)
        else:
            raise ValueError(f"No prompt template for analysis type: {analysis_type}")
    
    async def _call_api(self, model_id: str, prompt: str) -> Dict:
        """
        Call Hugging Face Inference API.
        
        Args:
            model_id: Hugging Face model ID
            prompt: Prompt for the model
            
        Returns:
            Model response
        """
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        
        payload = {"inputs": prompt}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API error ({response.status}): {error_text}")
    
    # Prompt templates for different analysis types
    def _biomechanical_analysis_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for biomechanical analysis.
        
        Args:
            data: Biomechanical data
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        athlete_info = kwargs.get("athlete_info", {})
        return f"""
        Analyze the following biomechanical data for a {athlete_info.get('sport', 'sprint')} athlete:
        
        {json.dumps(data, indent=2)}
        
        Provide a comprehensive analysis covering:
        1. Stride mechanics (length, cadence, contact time)
        2. Joint kinematics (knee angles, hip extension)
        3. Force production patterns
        4. Technical efficiency assessment
        5. Specific recommendations for improvement
        
        Include specific numerical insights and coaching recommendations.
        """
    
    def _movement_patterns_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for movement pattern analysis.
        
        Args:
            data: Pose sequence data
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        reference_patterns = kwargs.get("reference_patterns", {})
        return f"""
        Compare the athlete's movement patterns with reference patterns for elite performance.
        
        Athlete's movement data:
        {json.dumps(data, indent=2)}
        
        Reference patterns:
        {json.dumps(reference_patterns, indent=2)}
        
        Identify:
        1. Key deviations from optimal technique
        2. Temporal sequence issues
        3. Coordination patterns that need improvement
        4. Specific technical adjustments recommended
        """
    
    def _technical_reporting_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for technical reporting.
        
        Args:
            data: Analysis results
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        athlete_profile = kwargs.get("athlete_profile", {})
        return f"""
        Create a comprehensive technical report based on the following biomechanics analysis:
        
        {json.dumps(data, indent=2)}
        
        Athlete profile:
        {json.dumps(athlete_profile, indent=2)}
        
        Format the report with these sections:
        1. Executive Summary
        2. Stride Analysis
        3. Joint Kinematics
        4. Force Production
        5. Technical Efficiency
        6. Recommendations
        7. Training Implications
        
        Use professional language appropriate for coaches and sports scientists.
        """
    
    def _sprint_specialist_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for sprint technique analysis.
        
        Args:
            data: Sprint-specific metrics
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        return f"""
        Analyze these sprint-specific metrics and provide elite-level coaching insights:
        
        {json.dumps(data, indent=2)}
        
        Focus on:
        1. Block clearance mechanics
        2. Acceleration phase technique
        3. Maximum velocity mechanics
        4. Speed endurance factors
        """
    
    def _performance_comparison_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for performance comparison.
        
        Args:
            data: Current performance data
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        previous_performances = kwargs.get("previous_performances", {})
        return f"""
        Compare the athlete's current performance with previous performances:
        
        Current:
        {json.dumps(data, indent=2)}
        
        Previous:
        {json.dumps(previous_performances, indent=2)}
        
        Identify:
        1. Key improvements
        2. Remaining technical issues
        3. Progress metrics
        4. Next development priorities
        """
    
    def _coaching_insights_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for coaching insights.
        
        Args:
            data: Analysis data
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        coach_context = kwargs.get("coach_context", {})
        return f"""
        Provide elite-level coaching insights based on this biomechanical analysis:
        
        {json.dumps(data, indent=2)}
        
        Coach context:
        {json.dumps(coach_context, indent=2)}
        
        Include:
        1. Key technical observations
        2. Specific drills to address issues
        3. Progression timeline
        4. Expected performance improvements
        """
    
    def _quick_analysis_prompt(self, data: Dict, **kwargs) -> str:
        """
        Generate prompt for quick analysis.
        
        Args:
            data: Basic analysis data
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt
        """
        return f"""
        Provide a quick analysis of these biomechanical metrics:
        
        {json.dumps(data, indent=2)}
        
        Focus on 2-3 most critical technical issues and immediate recommendations.
        """


# Standalone functions for specialized analysis

def analyze_biomechanics(biomechanical_data: Dict, athlete_info: Dict) -> Dict:
    """
    Analyze biomechanical data using Claude 3 Opus via Hugging Face.
    
    Args:
        biomechanical_data: Biomechanical data to analyze
        athlete_info: Information about the athlete
        
    Returns:
        Analysis results
    """
    API_URL = "https://api-inference.huggingface.co/models/anthropic/claude-3-opus-20240229"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    
    prompt = f"""
    Analyze the following biomechanical data for a {athlete_info.get('sport', 'sprint')} athlete:
    
    {json.dumps(biomechanical_data, indent=2)}
    
    Provide a comprehensive analysis covering:
    1. Stride mechanics (length, cadence, contact time)
    2. Joint kinematics (knee angles, hip extension)
    3. Force production patterns
    4. Technical efficiency assessment
    5. Specific recommendations for improvement
    
    Include specific numerical insights and coaching recommendations.
    """
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def analyze_movement_patterns(pose_sequence_data: Dict, reference_patterns: Dict) -> Dict:
    """
    Compare athlete's movement patterns with reference patterns using LLaMA 3.
    
    Args:
        pose_sequence_data: Sequence of pose data
        reference_patterns: Reference patterns for comparison
        
    Returns:
        Analysis results
    """
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/llama-3-70b-instruct"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    
    prompt = f"""
    Compare the athlete's movement patterns with reference patterns for elite performance.
    
    Athlete's movement data:
    {json.dumps(pose_sequence_data, indent=2)}
    
    Reference patterns:
    {json.dumps(reference_patterns, indent=2)}
    
    Identify:
    1. Key deviations from optimal technique
    2. Temporal sequence issues
    3. Coordination patterns that need improvement
    4. Specific technical adjustments recommended
    """
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def generate_technical_report(analysis_results: Dict, athlete_profile: Dict) -> Dict:
    """
    Generate a comprehensive technical report using Claude 3 Haiku.
    
    Args:
        analysis_results: Results of biomechanical analysis
        athlete_profile: Profile information about the athlete
        
    Returns:
        Technical report
    """
    API_URL = "https://api-inference.huggingface.co/models/anthropic/claude-3-haiku-20240307"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    
    prompt = f"""
    Create a comprehensive technical report based on the following sprint biomechanics analysis:
    
    {json.dumps(analysis_results, indent=2)}
    
    Athlete profile:
    {json.dumps(athlete_profile, indent=2)}
    
    Format the report with these sections:
    1. Executive Summary
    2. Stride Analysis
    3. Joint Kinematics
    4. Force Production
    5. Technical Efficiency
    6. Recommendations
    7. Training Implications
    
    Use professional language appropriate for coaches and sports scientists.
    """
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def analyze_sprint_technique(sprint_specific_metrics: Dict) -> Dict:
    """
    Analyze sprint technique using a custom fine-tuned model (placeholder).
    
    Args:
        sprint_specific_metrics: Sprint-specific metrics to analyze
        
    Returns:
        Sprint technique analysis
    """
    # Replace with your actual fine-tuned model
    API_URL = "https://api-inference.huggingface.co/models/your-username/sprint-biomechanics-expert"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    
    prompt = f"""
    Analyze these sprint-specific metrics and provide elite-level coaching insights:
    
    {json.dumps(sprint_specific_metrics, indent=2)}
    
    Focus on:
    1. Block clearance mechanics
    2. Acceleration phase technique
    3. Maximum velocity mechanics
    4. Speed endurance factors
    """
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def compare_performances(current_performance: Dict, previous_performances: Dict) -> Dict:
    """
    Compare current performance with previous performances.
    
    Args:
        current_performance: Current performance data
        previous_performances: Previous performance data
        
    Returns:
        Performance comparison analysis
    """
    # Replace with your actual fine-tuned model
    API_URL = "https://api-inference.huggingface.co/models/your-username/performance-comparison-model"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    
    prompt = f"""
    Compare the athlete's current performance with previous performances:
    
    Current:
    {json.dumps(current_performance, indent=2)}
    
    Previous:
    {json.dumps(previous_performances, indent=2)}
    
    Identify:
    1. Key improvements
    2. Remaining technical issues
    3. Progress metrics
    4. Next development priorities
    """
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json() 