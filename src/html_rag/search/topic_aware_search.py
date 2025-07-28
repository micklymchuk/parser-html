"""
Topic-Aware Search system using Llama for intelligent query analysis and document retrieval.
"""

import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

from ..utils.logging import PipelineLogger


class TopicAwareSearcher:
    """Topic-aware search system that uses Llama to understand user queries and find relevant documents."""
    
    def __init__(self, vector_store, text_embedder, ollama_host: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize the Topic-Aware Searcher.
        
        Args:
            vector_store: Vector store instance for document retrieval
            text_embedder: Text embedder for semantic search
            ollama_host: Ollama server URL
            model: Model name to use for query analysis
        """
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.ollama_host = ollama_host
        self.model = model
        self.logger = PipelineLogger(__name__)
        
        # Ukrainian prompt for query analysis
        self.system_prompt = """Ти - експерт з аналізу пошукових запитів для системи пошуку українських політичних документів.
Твоє завдання - проаналізувати запит користувача і визначити намір, теми, персону та стратегію пошуку.

Типи намірів:
- find_contradictions: пошук суперечностей або зміни позиції
- search_topic: пошук за конкретною темою 
- analyze_positions: аналіз позицій персони
- general: загальний пошук

Стратегії пошуку:
- contradiction_analysis: пошук документів з однаковими темами але різними настроями
- topic_focused: фільтрація за конкретними темами
- general: широкий пошук з контекстом тем

Відповідь надай у форматі JSON:
{
    "intent": "find_contradictions|search_topic|analyze_positions|general",
    "topics": ["тема1", "тема2"],
    "person": "ім'я_персони",
    "time_period": "період_часу",
    "search_strategy": "contradiction_analysis|topic_focused|general",
    "keywords": ["ключове_слово1", "ключове_слово2"]
}

Якщо інформація не вказана, залиш поле порожнім або null."""

        self.user_prompt_template = """Проаналізуй наступний пошуковий запит українською мовою:

Запит: "{query}"

Визнач намір, теми, персону (якщо є), часовий період (якщо є) та оптимальну стратегію пошуку.
Надай відповідь у JSON форматі."""
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to extract intent, topics, person, and search strategy.
        
        Args:
            query: User search query in Ukrainian
            
        Returns:
            Dictionary with query analysis results
        """
        try:
            self.logger.debug(f"Analyzing query: {query}")
            
            # Check if Ollama is available
            if not self._check_ollama_availability():
                self.logger.warning("Ollama service unavailable, using fallback analysis")
                return self._fallback_query_analysis(query)
            
            # Prepare the prompt
            user_prompt = self.user_prompt_template.format(query=query)
            
            # Make request to Ollama
            response = self._call_ollama(user_prompt)
            
            if not response:
                self.logger.warning("No response from Ollama, using fallback analysis")
                return self._fallback_query_analysis(query)
            
            # Parse the response
            analysis = self._parse_query_response(response)
            
            self.logger.debug(f"Query analysis completed: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error during query analysis: {str(e)}")
            return self._fallback_query_analysis(query)
    
    def topic_aware_search(self, query: str, n_results: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Main topic-aware search method that analyzes query and applies appropriate search strategy.
        
        Args:
            query: User search query
            n_results: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents with enhanced metadata
        """
        try:
            self.logger.info(f"Starting topic-aware search for: '{query}'")
            
            # Analyze the query
            query_analysis = self.analyze_query(query)
            
            # Apply search strategy based on analysis
            search_strategy = query_analysis.get('search_strategy', 'general')
            
            if search_strategy == 'contradiction_analysis':
                results = self._contradiction_search(query, query_analysis, n_results)
            elif search_strategy == 'topic_focused':
                results = self._topic_focused_search(query, query_analysis, n_results)
            else:
                results = self._general_search(query, query_analysis, n_results)
            
            # Enhance results with query context
            enhanced_results = self._enhance_results_with_context(results, query_analysis)
            
            self.logger.info(f"Topic-aware search completed: found {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error during topic-aware search: {str(e)}")
            # Fallback to basic search
            return self._fallback_search(query, n_results)
    
    def _contradiction_search(self, query: str, analysis: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        """
        Search for contradictions - documents with same topics but different sentiments.
        
        Args:
            query: Original query
            analysis: Query analysis results
            n_results: Number of results to return
            
        Returns:
            List of documents showing contradictions
        """
        try:
            self.logger.debug("Executing contradiction search strategy")
            
            topics = analysis.get('topics', [])
            person = analysis.get('person')
            
            if not topics:
                # If no topics identified, do broad search
                return self._general_search(query, analysis, n_results)
            
            # Search for documents with the identified topics
            all_results = []
            
            for topic in topics:
                # Search for documents containing this topic
                topic_filter = {'topics': topic}
                if person:
                    # Add person filtering if available - could be in URL or content
                    topic_filter['person'] = person
                
                # Get documents with topic analysis
                topic_docs = self.vector_store.filter_by_metadata(topic_filter, limit=n_results * 2)
                all_results.extend(topic_docs)
            
            # Group by topics and find contradictions
            contradictions = self._find_contradictions(all_results, topics)
            
            # If no contradictions found, return regular topic search
            if not contradictions:
                return self._topic_focused_search(query, analysis, n_results)
            
            return contradictions[:n_results]
            
        except Exception as e:
            self.logger.error(f"Error in contradiction search: {str(e)}")
            return self._general_search(query, analysis, n_results)
    
    def _topic_focused_search(self, query: str, analysis: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        """
        Search focused on specific topics with filtering.
        
        Args:
            query: Original query
            analysis: Query analysis results
            n_results: Number of results to return
            
        Returns:
            List of topic-filtered documents
        """
        try:
            self.logger.debug("Executing topic-focused search strategy")
            
            topics = analysis.get('topics', [])
            person = analysis.get('person')
            time_period = analysis.get('time_period')
            
            # Build metadata filter
            metadata_filter = {}
            
            # Filter by topics if available
            if topics:
                # For multiple topics, we'll search for documents containing any of them
                topic_results = []
                for topic in topics:
                    filter_copy = metadata_filter.copy()
                    # Search in topic_analysis metadata
                    docs = self._search_by_topic_metadata(topic, person, time_period, n_results)
                    topic_results.extend(docs)
                
                # Remove duplicates and sort by relevance
                seen_ids = set()
                unique_results = []
                for doc in topic_results:
                    doc_id = doc.get('id', doc.get('text', '')[:50])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_results.append(doc)
                
                return unique_results[:n_results]
            
            # If no topics, filter by person or time
            if person:
                metadata_filter['person'] = person
            if time_period:
                metadata_filter['time_period'] = time_period
            
            if metadata_filter:
                results = self.vector_store.filter_by_metadata(metadata_filter, limit=n_results)
                return results
            
            # Fallback to general search
            return self._general_search(query, analysis, n_results)
            
        except Exception as e:
            self.logger.error(f"Error in topic-focused search: {str(e)}")
            return self._general_search(query, analysis, n_results)
    
    def _general_search(self, query: str, analysis: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        """
        General search with topic context enhancement.
        
        Args:
            query: Original query
            analysis: Query analysis results
            n_results: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            self.logger.debug("Executing general search strategy")
            
            # Use semantic search with query embedding
            query_embedding = self.text_embedder.embed_query(query)
            
            # Build metadata filter for context
            metadata_filter = {}
            person = analysis.get('person')
            time_period = analysis.get('time_period')
            
            if person:
                metadata_filter['person'] = person
            if time_period:
                metadata_filter['time_period'] = time_period
            
            # Perform semantic search
            results = self.vector_store.search_by_similarity(
                query_embedding=query_embedding,
                n_results=n_results,
                metadata_filter=metadata_filter if metadata_filter else None
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in general search: {str(e)}")
            return []
    
    def _search_by_topic_metadata(self, topic: str, person: Optional[str], time_period: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """
        Search for documents by topic in the topic_analysis metadata.
        
        Args:
            topic: Topic to search for
            person: Optional person filter
            time_period: Optional time period filter
            limit: Maximum number of results
            
        Returns:
            List of documents containing the topic
        """
        try:
            # Get all documents to filter client-side (ChromaDB metadata filtering limitations)
            all_docs = self.vector_store.filter_by_metadata({}, limit=limit * 10)
            
            matching_docs = []
            for doc in all_docs:
                metadata = doc.get('metadata', {})
                topic_analysis = metadata.get('topic_analysis', {})
                topics = topic_analysis.get('topics', [])
                
                # Check if topic matches
                topic_match = any(topic.lower() in doc_topic.lower() or doc_topic.lower() in topic.lower() 
                                for doc_topic in topics)
                
                if topic_match:
                    # Apply additional filters
                    if person and person.lower() not in doc.get('text', '').lower():
                        continue
                    if time_period and time_period not in metadata.get('url', ''):
                        continue
                    
                    matching_docs.append(doc)
                
                if len(matching_docs) >= limit:
                    break
            
            return matching_docs
            
        except Exception as e:
            self.logger.error(f"Error searching by topic metadata: {str(e)}")
            return []
    
    def _find_contradictions(self, documents: List[Dict[str, Any]], topics: List[str]) -> List[Dict[str, Any]]:
        """
        Find contradictions in documents - same topics with different sentiments.
        
        Args:
            documents: List of documents to analyze
            topics: Topics to look for contradictions in
            
        Returns:
            List of documents showing contradictions
        """
        try:
            # Group documents by topics and sentiments
            topic_sentiment_groups = defaultdict(lambda: defaultdict(list))
            
            for doc in documents:
                metadata = doc.get('metadata', {})
                topic_analysis = metadata.get('topic_analysis', {})
                doc_topics = topic_analysis.get('topics', [])
                sentiment_by_topic = topic_analysis.get('sentiment_by_topic', {})
                
                for topic in topics:
                    # Find matching topics
                    matching_topics = [t for t in doc_topics if topic.lower() in t.lower() or t.lower() in topic.lower()]
                    
                    for match_topic in matching_topics:
                        sentiment = sentiment_by_topic.get(match_topic, 'neutral')
                        topic_sentiment_groups[topic][sentiment].append(doc)
            
            # Find contradictions - same topic with different sentiments
            contradictions = []
            for topic, sentiment_groups in topic_sentiment_groups.items():
                sentiments = list(sentiment_groups.keys())
                
                # Look for opposing sentiments
                if 'positive' in sentiments and 'negative' in sentiments:
                    # Add documents with opposing sentiments
                    contradictions.extend(sentiment_groups['positive'])
                    contradictions.extend(sentiment_groups['negative'])
                elif len(sentiments) > 1:
                    # Add documents with different sentiments
                    for sentiment in sentiments:
                        contradictions.extend(sentiment_groups[sentiment])
            
            # Remove duplicates
            seen_ids = set()
            unique_contradictions = []
            for doc in contradictions:
                doc_id = doc.get('id', doc.get('text', '')[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_contradictions.append(doc)
            
            return unique_contradictions
            
        except Exception as e:
            self.logger.error(f"Error finding contradictions: {str(e)}")
            return documents
    
    def _enhance_results_with_context(self, results: List[Dict[str, Any]], query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhance search results with query context and relevance scoring.
        
        Args:
            results: Raw search results
            query_analysis: Query analysis results
            
        Returns:
            Enhanced results with additional context
        """
        try:
            enhanced_results = []
            
            for result in results:
                enhanced_result = result.copy()
                
                # Add query context
                enhanced_result['query_context'] = {
                    'matched_intent': query_analysis.get('intent'),
                    'matched_topics': self._find_matching_topics(result, query_analysis.get('topics', [])),
                    'search_strategy': query_analysis.get('search_strategy'),
                    'relevance_score': self._calculate_relevance_score(result, query_analysis)
                }
                
                enhanced_results.append(enhanced_result)
            
            # Sort by relevance score
            enhanced_results.sort(key=lambda x: x['query_context']['relevance_score'], reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error enhancing results: {str(e)}")
            return results
    
    def _find_matching_topics(self, document: Dict[str, Any], query_topics: List[str]) -> List[str]:
        """Find topics in document that match query topics."""
        try:
            metadata = document.get('metadata', {})
            topic_analysis = metadata.get('topic_analysis', {})
            doc_topics = topic_analysis.get('topics', [])
            
            matching_topics = []
            for query_topic in query_topics:
                for doc_topic in doc_topics:
                    if query_topic.lower() in doc_topic.lower() or doc_topic.lower() in query_topic.lower():
                        matching_topics.append(doc_topic)
            
            return matching_topics
            
        except Exception as e:
            self.logger.debug(f"Error finding matching topics: {str(e)}")
            return []
    
    def _calculate_relevance_score(self, document: Dict[str, Any], query_analysis: Dict[str, Any]) -> float:
        """Calculate relevance score based on query analysis."""
        try:
            score = 0.0
            
            # Base similarity score
            if 'similarity_score' in document:
                score += document['similarity_score'] * 0.4
            
            # Topic matching bonus
            query_topics = query_analysis.get('topics', [])
            matched_topics = self._find_matching_topics(document, query_topics)
            if query_topics:
                topic_match_ratio = len(matched_topics) / len(query_topics)
                score += topic_match_ratio * 0.3
            
            # Person matching bonus
            person = query_analysis.get('person')
            if person and person.lower() in document.get('text', '').lower():
                score += 0.2
            
            # Intent-specific bonuses
            intent = query_analysis.get('intent')
            if intent == 'find_contradictions':
                # Check for sentiment analysis
                metadata = document.get('metadata', {})
                topic_analysis = metadata.get('topic_analysis', {})
                if topic_analysis.get('sentiment_by_topic'):
                    score += 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.debug(f"Error calculating relevance score: {str(e)}")
            return 0.5
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {str(e)}")
            return False
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Make a request to Ollama API."""
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
                    "max_tokens": 400
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
    
    def _parse_query_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from Ollama."""
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
            
            # Validate and clean structure
            cleaned_analysis = {
                'intent': analysis.get('intent', 'general'),
                'topics': [str(t).strip() for t in analysis.get('topics', []) if t],
                'person': str(analysis.get('person', '')).strip() if analysis.get('person') else None,
                'time_period': str(analysis.get('time_period', '')).strip() if analysis.get('time_period') else None,
                'search_strategy': analysis.get('search_strategy', 'general'),
                'keywords': [str(k).strip() for k in analysis.get('keywords', []) if k]
            }
            
            # Clean empty values
            cleaned_analysis = {k: v for k, v in cleaned_analysis.items() if v}
            
            return cleaned_analysis
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {str(e)}")
            return self._fallback_query_analysis("")
        except Exception as e:
            self.logger.warning(f"Error parsing response: {str(e)}")
            return self._fallback_query_analysis("")
    
    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis when Ollama is unavailable."""
        analysis = {
            'intent': 'general',
            'topics': [],
            'person': None,
            'time_period': None,
            'search_strategy': 'general',
            'keywords': []
        }
        
        # Simple keyword detection
        query_lower = query.lower()
        
        # Detect contradiction intent
        contradiction_keywords = ['суперечність', 'протиріччя', 'зміна позиції', 'інакше думав', 'по-різному']
        if any(keyword in query_lower for keyword in contradiction_keywords):
            analysis['intent'] = 'find_contradictions'
            analysis['search_strategy'] = 'contradiction_analysis'
        
        # Detect person names (simple pattern)
        person_indicators = ['тимошенко', 'янукович', 'зеленський', 'порошенко']
        for person in person_indicators:
            if person in query_lower:
                analysis['person'] = person.title()
                break
        
        # Extract basic topics
        topic_keywords = ['освіта', 'приватизація', 'економіка', 'політика', 'реформи']
        found_topics = [topic for topic in topic_keywords if topic in query_lower]
        if found_topics:
            analysis['topics'] = found_topics
            analysis['search_strategy'] = 'topic_focused'
        
        return analysis
    
    def _fallback_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Fallback search method when topic-aware search fails."""
        try:
            query_embedding = self.text_embedder.embed_query(query)
            results = self.vector_store.search_by_similarity(
                query_embedding=query_embedding,
                n_results=n_results
            )
            return results
        except Exception as e:
            self.logger.error(f"Fallback search failed: {str(e)}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search system."""
        return {
            'ollama_host': self.ollama_host,
            'model': self.model,
            'service_available': self._check_ollama_availability(),
            'supported_intents': ['find_contradictions', 'search_topic', 'analyze_positions', 'general'],
            'supported_strategies': ['contradiction_analysis', 'topic_focused', 'general'],
            'language': 'Ukrainian'
        }