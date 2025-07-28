"""
Simplified Topic-Aware Search system using Llama for semantic and topic-based search.
Contradictions analysis moved to post-processing.
"""

import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

from ..utils.logging import PipelineLogger


class SimplifiedTopicSearcher:
    """Simplified topic-aware search system focused on semantic and topic-based retrieval."""

    def __init__(self, vector_store, text_embedder, ollama_host: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize the Simplified Topic-Aware Searcher.

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

        # Simplified Ukrainian prompt for query analysis
        self.system_prompt = """Ти - експерт з аналізу пошукових запитів для системи пошуку українських політичних документів.
Твоє завдання - проаналізувати запит користувача і визначити теми та персону для пошуку.

Відповідь надай у форматі JSON:
{
    "topics": ["тема1", "тема2"],
    "person": "ім'я_персони або null",
    "search_type": "topic_focused|semantic",
    "keywords": ["ключове_слово1", "ключове_слово2"]
}

search_type:
- topic_focused: якщо запит про конкретні теми (приватизація, освіта, економіка)
- semantic: якщо запит більш загальний або складний

Якщо інформація не вказана, залиш поле порожнім або null."""

        self.user_prompt_template = """Проаналізуй наступний пошуковий запит українською мовою:

Запит: "{query}"

Визнач теми, персону (якщо є) та тип пошуку.
Надай відповідь у JSON форматі."""

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to extract topics, person, and search type.

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
            self.logger.info(f"Starting simplified topic-aware search for: '{query}'")

            # Analyze the query
            query_analysis = self.analyze_query(query)

            # Apply search strategy based on analysis
            search_type = query_analysis.get('search_type', 'semantic')

            if search_type == 'topic_focused':
                results = self._topic_focused_search(query, query_analysis, n_results)
            else:
                results = self._semantic_search(query, query_analysis, n_results)

            # Enhance results with query context
            enhanced_results = self._enhance_results_with_context(results, query_analysis)

            self.logger.info(f"Topic-aware search completed: found {len(enhanced_results)} results")
            return enhanced_results

        except Exception as e:
            self.logger.error(f"Error during topic-aware search: {str(e)}")
            # Fallback to basic search
            return self._fallback_search(query, n_results)

    def _topic_focused_search(self, query: str, analysis: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        """
        Search focused on specific topics using topic_analysis metadata.

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

            if not topics:
                # If no topics identified, fall back to semantic search
                return self._semantic_search(query, analysis, n_results)

            # Search for documents containing the identified topics
            topic_results = []

            for topic in topics:
                # Get documents with this topic
                docs = self._search_by_topic_metadata(topic, person, n_results * 2)
                topic_results.extend(docs)

            # Remove duplicates and sort by relevance
            unique_results = self._deduplicate_results(topic_results)

            # If not enough results, combine with semantic search
            if len(unique_results) < n_results:
                self.logger.debug(f"Only {len(unique_results)} topic results, adding semantic search")
                semantic_results = self._semantic_search(query, analysis, n_results - len(unique_results))

                # Merge results, avoiding duplicates
                for sem_result in semantic_results:
                    if not any(self._are_same_document(sem_result, ur) for ur in unique_results):
                        unique_results.append(sem_result)

            return unique_results[:n_results]

        except Exception as e:
            self.logger.error(f"Error in topic-focused search: {str(e)}")
            return self._semantic_search(query, analysis, n_results)

    def _semantic_search(self, query: str, analysis: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        """
        Semantic search with optional person filtering.

        Args:
            query: Original query
            analysis: Query analysis results
            n_results: Number of results to return

        Returns:
            List of semantically relevant documents
        """
        try:
            self.logger.debug("Executing semantic search strategy")

            # Use semantic search with query embedding
            query_embedding = self.text_embedder.embed_query(query)

            # Build metadata filter for person if specified
            metadata_filter = {}
            person = analysis.get('person')

            if person:
                # Try different metadata fields where person might be stored
                # We can't use OR filters in ChromaDB easily, so we'll search without filter
                # and filter results post-processing
                pass

            # Perform semantic search
            results = self.vector_store.search_by_similarity(
                query_embedding=query_embedding,
                n_results=n_results * 2,  # Get more to allow for person filtering
                metadata_filter=None  # We'll filter post-search
            )

            # Apply person filtering if specified
            if person:
                filtered_results = []
                person_lower = person.lower()

                for result in results:
                    # Check in text content
                    if person_lower in result.get('text', '').lower():
                        filtered_results.append(result)
                    # Check in URL
                    elif person_lower in result.get('metadata', {}).get('url', '').lower():
                        filtered_results.append(result)
                    # Check in wayback metadata
                    elif person_lower in result.get('metadata', {}).get('wayback_original_url', '').lower():
                        filtered_results.append(result)

                    if len(filtered_results) >= n_results:
                        break

                results = filtered_results

            return results[:n_results]

        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []

    def _search_by_topic_metadata(self, topic: str, person: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """
        Search for documents by topic in the topic_analysis metadata.

        Args:
            topic: Topic to search for
            person: Optional person filter
            limit: Maximum number of results

        Returns:
            List of documents containing the topic
        """
        try:
            # Get all documents to filter client-side (ChromaDB metadata filtering limitations)
            all_docs = self.vector_store.filter_by_metadata({}, limit=limit * 5)

            matching_docs = []
            topic_lower = topic.lower()

            for doc in all_docs:
                metadata = doc.get('metadata', {})
                topic_analysis = metadata.get('topic_analysis', {})
                topics = topic_analysis.get('topics', [])

                # Check if topic matches
                topic_match = any(
                    topic_lower in doc_topic.lower() or doc_topic.lower() in topic_lower
                    for doc_topic in topics
                )

                if topic_match:
                    # Apply person filter if specified
                    if person and person.lower() not in doc.get('text', '').lower():
                        continue

                    matching_docs.append(doc)

                if len(matching_docs) >= limit:
                    break

            return matching_docs

        except Exception as e:
            self.logger.error(f"Error searching by topic metadata: {str(e)}")
            return []

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
                    'matched_topics': self._find_matching_topics(result, query_analysis.get('topics', [])),
                    'search_type': query_analysis.get('search_type'),
                    'person_match': self._check_person_match(result, query_analysis.get('person')),
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
                    if (query_topic.lower() in doc_topic.lower() or
                            doc_topic.lower() in query_topic.lower()):
                        matching_topics.append(doc_topic)

            return matching_topics

        except Exception as e:
            self.logger.debug(f"Error finding matching topics: {str(e)}")
            return []

    def _check_person_match(self, document: Dict[str, Any], person: Optional[str]) -> bool:
        """Check if document matches the specified person."""
        if not person:
            return False

        person_lower = person.lower()
        text = document.get('text', '').lower()
        metadata = document.get('metadata', {})
        url = metadata.get('url', '').lower()

        return person_lower in text or person_lower in url

    def _calculate_relevance_score(self, document: Dict[str, Any], query_analysis: Dict[str, Any]) -> float:
        """Calculate relevance score based on query analysis."""
        try:
            score = 0.0

            # Base similarity score
            if 'similarity_score' in document:
                score += document['similarity_score'] * 0.6

            # Topic matching bonus
            query_topics = query_analysis.get('topics', [])
            matched_topics = self._find_matching_topics(document, query_topics)
            if query_topics:
                topic_match_ratio = len(matched_topics) / len(query_topics)
                score += topic_match_ratio * 0.3

            # Person matching bonus
            if self._check_person_match(document, query_analysis.get('person')):
                score += 0.1

            return min(score, 1.0)  # Cap at 1.0

        except Exception as e:
            self.logger.debug(f"Error calculating relevance score: {str(e)}")
            return 0.5

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents from results."""
        seen_texts = set()
        unique_results = []

        for result in results:
            text_key = result.get('text', '')[:100]  # Use first 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)

        return unique_results

    def _are_same_document(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> bool:
        """Check if two documents are the same."""
        text1 = doc1.get('text', '')[:100]
        text2 = doc2.get('text', '')[:100]
        return text1 == text2

    # --- Ollama Integration Methods (same as before) ---

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
                    "max_tokens": 300
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
                self.logger.error(f"Ollama API error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Request to Ollama failed: {str(e)}")
            return None

    def _parse_query_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from Ollama."""
        try:
            # Clean response
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

            # Parse JSON
            analysis = json.loads(response)

            # Validate and clean structure
            cleaned_analysis = {
                'topics': [str(t).strip() for t in analysis.get('topics', []) if t],
                'person': str(analysis.get('person', '')).strip() if analysis.get('person') else None,
                'search_type': analysis.get('search_type', 'semantic'),
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
            'topics': [],
            'person': None,
            'search_type': 'semantic',
            'keywords': []
        }

        # Simple keyword detection
        query_lower = query.lower()

        # Detect person names
        person_indicators = ['тимошенко', 'янукович', 'зеленський', 'порошенко']
        for person in person_indicators:
            if person in query_lower:
                analysis['person'] = person.title()
                break

        # Extract basic topics
        topic_keywords = ['освіта', 'приватизація', 'економіка', 'політика', 'реформи', 'енергетика']
        found_topics = [topic for topic in topic_keywords if topic in query_lower]
        if found_topics:
            analysis['topics'] = found_topics
            analysis['search_type'] = 'topic_focused'

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
            'supported_search_types': ['topic_focused', 'semantic'],
            'language': 'Ukrainian'
        }


# --- POST-PROCESSING: Contradiction Analysis ---

class ContradictionAnalyzer:
    """Post-processing analyzer for finding contradictions in search results."""

    def __init__(self):
        self.logger = PipelineLogger("ContradictionAnalyzer")

    def find_contradictions_in_results(
            self,
            results: List[Dict[str, Any]],
            topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find contradictions in search results based on topic analysis.

        Args:
            results: Search results from topic-aware search
            topics: Optional list of topics to focus on

        Returns:
            Dictionary with contradiction analysis
        """
        try:
            self.logger.info(f"Analyzing {len(results)} results for contradictions")

            # Group results by topics and sentiments
            topic_sentiment_groups = self._group_by_topic_sentiment(results, topics)

            # Find contradictions
            contradictions = self._identify_contradictions(topic_sentiment_groups)

            # Create contradiction pairs
            contradiction_pairs = self._create_contradiction_pairs(contradictions)

            analysis = {
                'total_results': len(results),
                'topics_analyzed': list(topic_sentiment_groups.keys()),
                'contradictions_found': len(contradiction_pairs),
                'contradiction_pairs': contradiction_pairs,
                'summary': self._create_contradiction_summary(contradiction_pairs)
            }

            self.logger.info(f"Found {len(contradiction_pairs)} contradiction pairs")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in contradiction analysis: {str(e)}")
            return {'error': str(e)}

    def _group_by_topic_sentiment(
            self,
            results: List[Dict[str, Any]],
            focus_topics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group results by topic and sentiment."""
        groups = defaultdict(lambda: defaultdict(list))

        for result in results:
            metadata = result.get('metadata', {})
            topic_analysis = metadata.get('topic_analysis', {})
            topics = topic_analysis.get('topics', [])
            sentiment_by_topic = topic_analysis.get('sentiment_by_topic', {})

            # Process each topic in the document
            for topic in topics:
                # Filter by focus topics if specified
                if focus_topics:
                    if not any(ft.lower() in topic.lower() or topic.lower() in ft.lower()
                               for ft in focus_topics):
                        continue

                sentiment = sentiment_by_topic.get(topic, 'neutral')
                groups[topic][sentiment].append(result)

        return dict(groups)

    def _identify_contradictions(
            self,
            topic_groups: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, List[Tuple[str, List[Dict[str, Any]]]]]:
        """Identify contradictions within topic groups."""
        contradictions = {}

        for topic, sentiment_groups in topic_groups.items():
            sentiments = list(sentiment_groups.keys())

            # Look for opposing sentiments
            contradictory_pairs = []

            if 'positive' in sentiments and 'negative' in sentiments:
                contradictory_pairs.append(('positive vs negative',
                                            sentiment_groups['positive'] + sentiment_groups['negative']))

            # Also check for neutral vs strong opinions
            if 'neutral' in sentiments:
                if 'positive' in sentiments and len(sentiment_groups['positive']) > 0:
                    contradictory_pairs.append(('neutral vs positive',
                                                sentiment_groups['neutral'] + sentiment_groups['positive']))
                if 'negative' in sentiments and len(sentiment_groups['negative']) > 0:
                    contradictory_pairs.append(('neutral vs negative',
                                                sentiment_groups['neutral'] + sentiment_groups['negative']))

            if contradictory_pairs:
                contradictions[topic] = contradictory_pairs

        return contradictions

    def _create_contradiction_pairs(
            self,
            contradictions: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]]
    ) -> List[Dict[str, Any]]:
        """Create structured contradiction pairs."""
        pairs = []

        for topic, contradiction_list in contradictions.items():
            for contradiction_type, documents in contradiction_list:
                if len(documents) >= 2:
                    pair = {
                        'topic': topic,
                        'contradiction_type': contradiction_type,
                        'documents': documents,
                        'document_count': len(documents),
                        'sentiments': self._extract_sentiments_from_docs(documents, topic)
                    }
                    pairs.append(pair)

        return pairs

    def _extract_sentiments_from_docs(
            self,
            documents: List[Dict[str, Any]],
            topic: str
    ) -> List[str]:
        """Extract sentiments for a specific topic from documents."""
        sentiments = []

        for doc in documents:
            metadata = doc.get('metadata', {})
            topic_analysis = metadata.get('topic_analysis', {})
            sentiment_by_topic = topic_analysis.get('sentiment_by_topic', {})

            sentiment = sentiment_by_topic.get(topic, 'unknown')
            sentiments.append(sentiment)

        return sentiments

    def _create_contradiction_summary(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of found contradictions."""
        if not pairs:
            return {'message': 'No contradictions found'}

        topics_with_contradictions = list(set(pair['topic'] for pair in pairs))
        contradiction_types = Counter(pair['contradiction_type'] for pair in pairs)

        return {
            'topics_with_contradictions': topics_with_contradictions,
            'contradiction_types': dict(contradiction_types),
            'total_contradictory_documents': sum(pair['document_count'] for pair in pairs)
        }