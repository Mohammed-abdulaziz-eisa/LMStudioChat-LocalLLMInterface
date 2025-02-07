import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"llm_inference_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalLLMInference:
    def __init__(
        self, 
        base_url: str = "http://localhost:1234/v1", 
        api_key: Optional[str] = None,
        model: str = 'llama-3.2-3b-instruct'
    ):
        load_dotenv()  
        
        self.client = OpenAI(
            base_url=base_url, 
            api_key=os.getenv('LM_STUDIO_API_KEY', api_key or 'lm-studio')
        )
        self.model = model
        
        logger.info(f"Initialized LLM Inference with model: {self.model}")

    def generate_response(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=temperature
            )
            
            response = completion.choices[0].message.content
            logger.info(f"Generated response for prompt: {prompt[:]}...")
            return response
        
        except Exception as e:
            logger.error(f"LLM Inference Error: {e}")
            raise

def main():
    """main execution for local LLM inference."""
    try:
        inference_engine = LocalLLMInference()
        meal_plan_prompt = "Give me a meal plan for today as Friday meal."
        
        response = inference_engine.generate_response(
            prompt=meal_plan_prompt,
            system_message="You are a nutritional assistant specializing in daily meal planning."
        )
        
        print("Generated Meal Plan:")
        print(response)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()