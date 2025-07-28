"""
Topic Analysis using Ollama llama3.2:3b for Ukrainian political text analysis
"""

import json
import requests
from typing import Dict, Any, List, Optional
from ..utils.logging import PipelineLogger


class TopicAnalyzer:
    """Topic Analyzer using Ollama for extracting topics and sentiment from Ukrainian political text."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize the Topic Analyzer.
        
        Args:
            ollama_host: Ollama server URL
            model: Model name to use for analysis
        """
        self.ollama_host = ollama_host
        self.model = model
        self.logger = PipelineLogger(__name__)
        
        # Ukrainian prompt for topic extraction
        self.system_prompt = """Ти - експерт з аналізу українських політичних текстів. 
Твоє завдання - виділити 2-3 основні теми з тексту та визначити ставлення до кожної теми.

Для кожної теми визнач позицію автора:
- positive (позитивне ставлення)
- negative (негативне ставлення)  
- neutral (нейтральне ставлення)

Відповідь надай у форматі JSON:
{
    "topics": ["тема1", "тема2", "тема3"],
    "sentiment_by_topic": {
        "тема1": "positive/negative/neutral",
        "тема2": "positive/negative/neutral", 
        "тема3": "positive/negative/neutral"
    },
    "confidence": 0.85
}

Якщо текст не є політичним або недостатньо інформативним, поверни порожній список тем."""

        self.user_prompt_template = """Проаналізуй наступний український політичний текст і виділи основні теми:

Текст: "{text}"

Надай відповідь у JSON форматі."""
    
    def analyze_chunk(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a text chunk for topics and sentiment.
        
        Args:
            text: Text content to analyze
            metadata: Additional metadata about the text chunk
            
        Returns:
            Dictionary with topics, sentiment analysis, and confidence score
        """
        try:
            self.logger.debug(f"Analyzing chunk with {len(text)} characters")
            
            # Check if Ollama is available
            if not self._check_ollama_availability():
                self.logger.warning("Ollama service unavailable, returning empty analysis")
                return self._empty_analysis()
            
            # Prepare the prompt
            user_prompt = self.user_prompt_template.format(text=text[:1000])  # Limit text length
            
            # Make request to Ollama
            response = self._call_ollama(user_prompt)
            
            if not response:
                self.logger.warning("No response from Ollama, returning empty analysis")
                return self._empty_analysis()
            
            # Parse the response
            analysis = self._parse_response(response)
            
            # Add metadata
            analysis['source_metadata'] = {
                'text_length': len(text),
                'url': metadata.get('url', ''),
                'element_type': metadata.get('element_type', ''),
                'position': metadata.get('position', 0)
            }
            
            self.logger.debug(f"Analysis completed, found {len(analysis.get('topics', []))} topics")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error during topic analysis: {str(e)}")
            return self._empty_analysis()
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {str(e)}")
            return False
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Make a request to Ollama API.
        
        Args:
            prompt: The user prompt to send
            
        Returns:
            Response text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request to Ollama failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error calling Ollama: {str(e)}")
            return None
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from Ollama.
        
        Args:
            response: Raw response text from Ollama
            
        Returns:
            Parsed analysis dictionary
        """
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Look for JSON block markers
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            
            # Try to parse JSON
            analysis = json.loads(response)
            
            # Validate structure
            if not isinstance(analysis, dict):
                raise ValueError("Response is not a dictionary")
            
            # Ensure required fields
            topics = analysis.get("topics", [])
            sentiment_by_topic = analysis.get("sentiment_by_topic", {})
            confidence = analysis.get("confidence", 0.5)
            
            # Validate data types
            if not isinstance(topics, list):
                topics = []
            if not isinstance(sentiment_by_topic, dict):
                sentiment_by_topic = {}
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                confidence = 0.5
            
            # Clean topics (limit to 3)
            topics = [str(topic).strip() for topic in topics[:3] if topic]
            
            # Validate sentiment values
            valid_sentiments = {"positive", "negative", "neutral"}
            cleaned_sentiment = {}
            for topic in topics:
                sentiment = sentiment_by_topic.get(topic, "neutral")
                if sentiment not in valid_sentiments:
                    sentiment = "neutral"
                cleaned_sentiment[topic] = sentiment
            
            return {
                "topics": topics,
                "sentiment_by_topic": cleaned_sentiment,
                "confidence": float(confidence)
            }
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {str(e)}")
            return self._empty_analysis()
        except Exception as e:
            self.logger.warning(f"Error parsing response: {str(e)}")
            return self._empty_analysis()
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "topics": [],
            "sentiment_by_topic": {},
            "confidence": 0.0
        }
    
    def analyze_batch(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple text blocks for topics and sentiment.
        
        Args:
            text_blocks: List of text block dictionaries
            
        Returns:
            List of text blocks with added topic analysis
        """
        try:
            self.logger.info(f"Starting batch topic analysis for {len(text_blocks)} blocks")
            
            analyzed_blocks = []
            for i, block in enumerate(text_blocks):
                text = block.get('text', '')
                if len(text.strip()) < 10:  # Skip very short texts
                    analysis = self._empty_analysis()
                else:
                    analysis = self.analyze_chunk(text, block)
                
                # Add analysis to block
                analyzed_block = block.copy()
                analyzed_block['topic_analysis'] = analysis
                analyzed_blocks.append(analyzed_block)
                
                # Log progress for long batches
                if (i + 1) % 10 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(text_blocks)} blocks")
            
            self.logger.info(f"Batch topic analysis completed for {len(analyzed_blocks)} blocks")
            return analyzed_blocks
            
        except Exception as e:
            self.logger.error(f"Error during batch analysis: {str(e)}")
            # Return original blocks without analysis on error
            return [block.copy() for block in text_blocks]
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the topic analysis service.
        
        Returns:
            Dictionary with service information
        """
        return {
            "ollama_host": self.ollama_host,
            "model": self.model,
            "service_available": self._check_ollama_availability(),
            "supported_language": "Ukrainian",
            "analysis_type": "Political topic extraction with sentiment"
        }