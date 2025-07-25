"""
Stage 1: HTML Pruning using zstanjj/HTML-Pruner-Phi-3.8B model
"""

import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLPruner:
    """HTML Pruning using the zstanjj/HTML-Pruner-Phi-3.8B model."""
    
    def __init__(self, model_name: str = "zstanjj/HTML-Pruner-Phi-3.8B"):
        """
        Initialize the HTML Pruner.
        
        Args:
            model_name: The name of the HTML pruning model to use
        """
        self.model_name = model_name
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """Load the HTML pruning model and tokenizer."""
        try:
            logger.info(f"Loading HTML Pruner model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            logger.info("HTML Pruner model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading HTML Pruner model: {e}")
            raise
    
    def prune_html(self, raw_html: str, max_length: int = 4096) -> str:
        """
        Prune HTML content to remove noise and keep main content.
        
        Args:
            raw_html: Raw HTML content with all tags, styles, scripts
            max_length: Maximum length of input text to process
            
        Returns:
            Cleaned HTML with main content (headings, paragraphs, lists, tables)
        """
        if not self.model or not self.tokenizer:
            self.load_model()
        
        try:
            # Truncate input if too long
            if len(raw_html) > max_length:
                raw_html = raw_html[:max_length]
                logger.warning(f"Input HTML truncated to {max_length} characters")
            
            # Create prompt for the model
            prompt = f"Clean the following HTML content, keeping only the main content (headings, paragraphs, lists, tables) and removing scripts, styles, and other noise:\n\n{raw_html}\n\nCleaned HTML:"
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate cleaned HTML
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the cleaned HTML (after the prompt)
            if "Cleaned HTML:" in generated_text:
                cleaned_html = generated_text.split("Cleaned HTML:")[-1].strip()
            else:
                cleaned_html = generated_text[len(prompt):].strip()
            
            logger.info(f"HTML pruning completed. Input length: {len(raw_html)}, Output length: {len(cleaned_html)}")
            return cleaned_html
            
        except Exception as e:
            logger.error(f"Error during HTML pruning: {e}")
            # Fallback: return original HTML if pruning fails
            logger.warning("Falling back to original HTML content")
            return raw_html
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("HTML Pruner resources cleaned up")