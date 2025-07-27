"""
Trend Analysis Component for Content Analytics.

This module provides trend analysis functionality including:
- Time-series analysis of content metrics
- Sentiment trend detection
- Topic evolution tracking
- Controversy pattern analysis
- Comparative analysis across time periods
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

try:
    import numpy as np
    from scipy import stats
    import pandas as pd
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Trend analysis dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

from ..analytics.models import (
    TrendAnalysis, ContentAnalytics, SentimentLabel, 
    ControversyLevel, AnalyticsResult
)
from ..core.config import ContentAnalyticsConfig
from ..exceptions.pipeline_exceptions import PipelineError, handle_pipeline_error
from ..utils.logging import PipelineLogger

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyzes trends and patterns in content analytics data over time.
    
    Provides comprehensive trend analysis including sentiment evolution,
    topic changes, controversy patterns, and statistical trend detection.
    """
    
    def __init__(self, config: Optional[ContentAnalyticsConfig] = None):
        """
        Initialize the trend analyzer.
        
        Args:
            config: Content analytics configuration
        """
        self.config = config or ContentAnalyticsConfig()
        self.logger = PipelineLogger("TrendAnalyzer")
        self._trend_cache = {}
    
    @handle_pipeline_error
    def analyze_trends(self, analytics_data: List[ContentAnalytics], 
                      time_window: str = 'daily') -> TrendAnalysis:
        """
        Analyze trends in a collection of content analytics.
        
        Args:
            analytics_data: List of content analytics results
            time_window: Time window for aggregation ('hourly', 'daily', 'weekly', 'monthly')
            
        Returns:
            Comprehensive trend analysis
        """
        try:
            if not analytics_data:
                return self._empty_trend_analysis()
            
            # Sort data by timestamp
            sorted_data = sorted(analytics_data, key=lambda x: x.timestamp)
            
            # Aggregate data by time window
            aggregated_data = self._aggregate_by_time_window(sorted_data, time_window)
            
            # Analyze different trend components
            sentiment_trends = self._analyze_sentiment_trends(aggregated_data)
            controversy_trends = self._analyze_controversy_trends(aggregated_data)
            topic_trends = self._analyze_topic_trends(aggregated_data)
            volume_trends = self._analyze_volume_trends(aggregated_data)
            entity_trends = self._analyze_entity_trends(aggregated_data)
            
            # Calculate overall trend direction
            overall_trend = self._calculate_overall_trend(aggregated_data)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(aggregated_data)
            
            # Generate predictions if enough data
            predictions = self._generate_predictions(aggregated_data) if len(aggregated_data) >= 5 else {}
            
            # Calculate trend strength and confidence
            trend_strength = self._calculate_trend_strength(aggregated_data)
            confidence = self._calculate_confidence(aggregated_data, trend_strength)
            
            return TrendAnalysis(
                time_period=f"{sorted_data[0].timestamp} to {sorted_data[-1].timestamp}",
                trend_direction=overall_trend,
                trend_strength=trend_strength,
                confidence=confidence,
                sentiment_trends=sentiment_trends,
                controversy_trends=controversy_trends,
                topic_trends=topic_trends,
                volume_trends=volume_trends,
                entity_trends=entity_trends,
                anomalies=anomalies,
                predictions=predictions,
                data_points=len(analytics_data),
                time_window=time_window
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return self._empty_trend_analysis()
    
    def _aggregate_by_time_window(self, data: List[ContentAnalytics], 
                                time_window: str) -> List[Dict[str, Any]]:
        """Aggregate analytics data by time window."""
        try:
            # Group data by time window
            time_groups = defaultdict(list)
            
            for item in data:
                # Create time key based on window
                if time_window == 'hourly':
                    time_key = item.timestamp.strftime('%Y-%m-%d %H:00:00')
                elif time_window == 'daily':
                    time_key = item.timestamp.strftime('%Y-%m-%d')
                elif time_window == 'weekly':
                    # Get Monday of the week
                    monday = item.timestamp - timedelta(days=item.timestamp.weekday())
                    time_key = monday.strftime('%Y-%m-%d')
                elif time_window == 'monthly':
                    time_key = item.timestamp.strftime('%Y-%m')
                else:
                    time_key = item.timestamp.strftime('%Y-%m-%d')  # Default to daily
                
                time_groups[time_key].append(item)
            
            # Aggregate each time group
            aggregated = []
            for time_key, group_data in sorted(time_groups.items()):
                agg_point = self._aggregate_group(group_data, time_key)
                aggregated.append(agg_point)
            
            return aggregated
            
        except Exception as e:
            self.logger.warning(f"Error aggregating by time window: {e}")
            return []
    
    def _aggregate_group(self, group_data: List[ContentAnalytics], time_key: str) -> Dict[str, Any]:
        """Aggregate a group of analytics data for a time period."""
        try:
            group_size = len(group_data)
            
            # Aggregate sentiment
            sentiment_scores = [item.sentiment.score for item in group_data]
            avg_sentiment = sum(sentiment_scores) / group_size
            sentiment_distribution = Counter(item.sentiment.label.value for item in group_data)
            
            # Aggregate controversy
            controversy_scores = [item.controversy.score for item in group_data]
            avg_controversy = sum(controversy_scores) / group_size
            controversy_distribution = Counter(item.controversy.level.value for item in group_data)
            
            # Aggregate topics
            topic_counts = Counter()
            for item in group_data:
                topic_counts[item.topics.primary_topic] += 1
            
            # Aggregate entities
            entity_counts = Counter()
            total_entities = 0
            for item in group_data:
                total_entities += len(item.entities)
                for entity in item.entities:
                    entity_counts[entity.normalized_form or entity.text] += 1
            
            # Quality and processing metrics
            quality_scores = [item.quality_score for item in group_data if item.quality_score]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            processing_times = [item.processing_time for item in group_data]
            avg_processing_time = sum(processing_times) / group_size
            
            return {
                'time_key': time_key,
                'timestamp': datetime.fromisoformat(time_key) if '-' in time_key else datetime.now(),
                'count': group_size,
                'avg_sentiment': avg_sentiment,
                'sentiment_distribution': dict(sentiment_distribution),
                'avg_controversy': avg_controversy,
                'controversy_distribution': dict(controversy_distribution),
                'topic_distribution': dict(topic_counts.most_common(10)),
                'entity_distribution': dict(entity_counts.most_common(20)),
                'total_entities': total_entities,
                'avg_quality': avg_quality,
                'avg_processing_time': avg_processing_time,
                'raw_data': group_data
            }
            
        except Exception as e:
            self.logger.warning(f"Error aggregating group: {e}")
            return {'time_key': time_key, 'count': 0}
    
    def _analyze_sentiment_trends(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        try:
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            
            if not sentiment_scores:
                return {}
            
            # Calculate trend
            trend_direction = self._calculate_linear_trend(sentiment_scores)
            
            # Calculate volatility
            volatility = self._calculate_volatility(sentiment_scores)
            
            # Find extremes
            max_sentiment = max(sentiment_scores)
            min_sentiment = min(sentiment_scores)
            max_index = sentiment_scores.index(max_sentiment)
            min_index = sentiment_scores.index(min_sentiment)
            
            # Detect sentiment shifts
            shifts = self._detect_sentiment_shifts(aggregated_data)
            
            return {
                'trend_direction': trend_direction,
                'volatility': volatility,
                'max_sentiment': max_sentiment,
                'min_sentiment': min_sentiment,
                'max_date': aggregated_data[max_index].get('time_key'),
                'min_date': aggregated_data[min_index].get('time_key'),
                'sentiment_shifts': shifts,
                'average': sum(sentiment_scores) / len(sentiment_scores)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment trends: {e}")
            return {}
    
    def _analyze_controversy_trends(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze controversy trends over time."""
        try:
            controversy_scores = [point.get('avg_controversy', 0) for point in aggregated_data]
            
            if not controversy_scores:
                return {}
            
            # Calculate trend
            trend_direction = self._calculate_linear_trend(controversy_scores)
            
            # Calculate volatility
            volatility = self._calculate_volatility(controversy_scores)
            
            # Analyze controversy level distribution over time
            level_evolution = []
            for point in aggregated_data:
                level_dist = point.get('controversy_distribution', {})
                level_evolution.append(level_dist)
            
            # Detect controversy spikes
            spikes = self._detect_controversy_spikes(aggregated_data)
            
            return {
                'trend_direction': trend_direction,
                'volatility': volatility,
                'average': sum(controversy_scores) / len(controversy_scores),
                'level_evolution': level_evolution,
                'controversy_spikes': spikes,
                'peak_controversy': max(controversy_scores) if controversy_scores else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing controversy trends: {e}")
            return {}
    
    def _analyze_topic_trends(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topic trends and evolution over time."""
        try:
            # Track topic evolution
            topic_evolution = []
            all_topics = set()
            
            for point in aggregated_data:
                topics = point.get('topic_distribution', {})
                topic_evolution.append(topics)
                all_topics.update(topics.keys())
            
            # Calculate topic stability
            stability_scores = {}
            for topic in all_topics:
                appearances = sum(1 for point in topic_evolution if topic in point)
                stability_scores[topic] = appearances / len(topic_evolution)
            
            # Find emerging and declining topics
            emerging_topics = self._find_emerging_topics(topic_evolution)
            declining_topics = self._find_declining_topics(topic_evolution)
            
            # Calculate topic diversity over time
            diversity_scores = []
            for topics in topic_evolution:
                # Shannon diversity index
                total = sum(topics.values())
                if total > 0:
                    diversity = -sum((count/total) * math.log(count/total) 
                                   for count in topics.values() if count > 0)
                    diversity_scores.append(diversity)
                else:
                    diversity_scores.append(0)
            
            return {
                'topic_evolution': topic_evolution,
                'stability_scores': stability_scores,
                'emerging_topics': emerging_topics,
                'declining_topics': declining_topics,
                'diversity_trend': self._calculate_linear_trend(diversity_scores),
                'average_diversity': sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0,
                'total_unique_topics': len(all_topics)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing topic trends: {e}")
            return {}
    
    def _analyze_volume_trends(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content volume trends over time."""
        try:
            volume_data = [point.get('count', 0) for point in aggregated_data]
            
            if not volume_data:
                return {}
            
            # Calculate trend
            trend_direction = self._calculate_linear_trend(volume_data)
            
            # Calculate growth rate
            growth_rates = []
            for i in range(1, len(volume_data)):
                if volume_data[i-1] > 0:
                    growth_rate = (volume_data[i] - volume_data[i-1]) / volume_data[i-1]
                    growth_rates.append(growth_rate)
            
            avg_growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 0
            
            # Find peaks and valleys
            peaks = []
            valleys = []
            for i in range(1, len(volume_data) - 1):
                if volume_data[i] > volume_data[i-1] and volume_data[i] > volume_data[i+1]:
                    peaks.append({'index': i, 'value': volume_data[i], 
                                'date': aggregated_data[i].get('time_key')})
                elif volume_data[i] < volume_data[i-1] and volume_data[i] < volume_data[i+1]:
                    valleys.append({'index': i, 'value': volume_data[i], 
                                  'date': aggregated_data[i].get('time_key')})
            
            return {
                'trend_direction': trend_direction,
                'average_volume': sum(volume_data) / len(volume_data),
                'max_volume': max(volume_data),
                'min_volume': min(volume_data),
                'growth_rate': avg_growth_rate,
                'peaks': peaks[-5:],  # Last 5 peaks
                'valleys': valleys[-5:],  # Last 5 valleys
                'volatility': self._calculate_volatility(volume_data)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing volume trends: {e}")
            return {}
    
    def _analyze_entity_trends(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entity mention trends over time."""
        try:
            # Track entity evolution
            entity_evolution = []
            all_entities = set()
            
            for point in aggregated_data:
                entities = point.get('entity_distribution', {})
                entity_evolution.append(entities)
                all_entities.update(entities.keys())
            
            # Find trending entities
            trending_up = self._find_trending_entities(entity_evolution, direction='up')
            trending_down = self._find_trending_entities(entity_evolution, direction='down')
            
            # Calculate entity consistency
            consistency_scores = {}
            for entity in all_entities:
                appearances = sum(1 for point in entity_evolution if entity in point)
                consistency_scores[entity] = appearances / len(entity_evolution)
            
            # Most mentioned entities overall
            entity_totals = Counter()
            for entities in entity_evolution:
                for entity, count in entities.items():
                    entity_totals[entity] += count
            
            return {
                'trending_up': trending_up,
                'trending_down': trending_down,
                'consistency_scores': dict(sorted(consistency_scores.items(), 
                                                key=lambda x: x[1], reverse=True)[:20]),
                'most_mentioned': dict(entity_totals.most_common(20)),
                'total_unique_entities': len(all_entities)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing entity trends: {e}")
            return {}
    
    def _calculate_linear_trend(self, values: List[float]) -> str:
        """Calculate linear trend direction for a series of values."""
        try:
            if len(values) < 2:
                return 'stable'
            
            if DEPENDENCIES_AVAILABLE:
                # Use scipy for robust trend calculation
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Consider statistical significance
                if p_value < 0.05:  # Statistically significant
                    if slope > 0.01:
                        return 'increasing'
                    elif slope < -0.01:
                        return 'decreasing'
                    else:
                        return 'stable'
                else:
                    return 'stable'
            else:
                # Simple trend calculation
                first_half = sum(values[:len(values)//2]) / (len(values)//2)
                second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                
                diff = second_half - first_half
                threshold = 0.1  # Adjust based on scale
                
                if diff > threshold:
                    return 'increasing'
                elif diff < -threshold:
                    return 'decreasing'
                else:
                    return 'stable'
                    
        except Exception:
            return 'stable'
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of values."""
        try:
            if len(values) < 2:
                return 0.0
            
            if DEPENDENCIES_AVAILABLE:
                return float(np.std(values))
            else:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                return math.sqrt(variance)
                
        except Exception:
            return 0.0
    
    def _detect_sentiment_shifts(self, aggregated_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect significant sentiment shifts."""
        shifts = []
        
        try:
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            
            for i in range(1, len(sentiment_scores)):
                prev_sentiment = sentiment_scores[i-1]
                curr_sentiment = sentiment_scores[i]
                
                # Detect significant shifts (threshold of 0.3)
                if abs(curr_sentiment - prev_sentiment) > 0.3:
                    shift_type = 'positive' if curr_sentiment > prev_sentiment else 'negative'
                    shifts.append({
                        'date': aggregated_data[i].get('time_key'),
                        'type': shift_type,
                        'magnitude': abs(curr_sentiment - prev_sentiment),
                        'from_score': prev_sentiment,
                        'to_score': curr_sentiment
                    })
            
        except Exception as e:
            self.logger.warning(f"Error detecting sentiment shifts: {e}")
        
        return shifts[-10:]  # Return last 10 shifts
    
    def _detect_controversy_spikes(self, aggregated_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect controversy spikes."""
        spikes = []
        
        try:
            controversy_scores = [point.get('avg_controversy', 0) for point in aggregated_data]
            
            if len(controversy_scores) < 3:
                return spikes
            
            # Calculate rolling average for baseline
            window_size = min(3, len(controversy_scores))
            for i in range(window_size, len(controversy_scores)):
                baseline = sum(controversy_scores[i-window_size:i]) / window_size
                current = controversy_scores[i]
                
                # Spike if current is significantly higher than baseline
                if current > baseline * 1.5 and current > 0.5:
                    spikes.append({
                        'date': aggregated_data[i].get('time_key'),
                        'score': current,
                        'baseline': baseline,
                        'magnitude': current / baseline
                    })
            
        except Exception as e:
            self.logger.warning(f"Error detecting controversy spikes: {e}")
        
        return spikes[-5:]  # Return last 5 spikes
    
    def _find_emerging_topics(self, topic_evolution: List[Dict[str, int]]) -> List[str]:
        """Find topics that are becoming more prominent."""
        try:
            if len(topic_evolution) < 3:
                return []
            
            emerging = []
            mid_point = len(topic_evolution) // 2
            
            # Compare first half vs second half
            first_half_topics = Counter()
            second_half_topics = Counter()
            
            for topics in topic_evolution[:mid_point]:
                for topic, count in topics.items():
                    first_half_topics[topic] += count
            
            for topics in topic_evolution[mid_point:]:
                for topic, count in topics.items():
                    second_half_topics[topic] += count
            
            # Find topics that increased significantly
            for topic in second_half_topics:
                first_count = first_half_topics.get(topic, 0)
                second_count = second_half_topics[topic]
                
                # Emerging if appeared more in second half or is new
                if second_count > first_count * 2 or (first_count == 0 and second_count > 1):
                    emerging.append(topic)
            
            return emerging[:10]  # Top 10 emerging topics
            
        except Exception:
            return []
    
    def _find_declining_topics(self, topic_evolution: List[Dict[str, int]]) -> List[str]:
        """Find topics that are becoming less prominent."""
        try:
            if len(topic_evolution) < 3:
                return []
            
            declining = []
            mid_point = len(topic_evolution) // 2
            
            # Compare first half vs second half
            first_half_topics = Counter()
            second_half_topics = Counter()
            
            for topics in topic_evolution[:mid_point]:
                for topic, count in topics.items():
                    first_half_topics[topic] += count
            
            for topics in topic_evolution[mid_point:]:
                for topic, count in topics.items():
                    second_half_topics[topic] += count
            
            # Find topics that decreased significantly
            for topic in first_half_topics:
                first_count = first_half_topics[topic]
                second_count = second_half_topics.get(topic, 0)
                
                # Declining if appeared much less in second half
                if first_count > second_count * 2 and first_count > 2:
                    declining.append(topic)
            
            return declining[:10]  # Top 10 declining topics
            
        except Exception:
            return []
    
    def _find_trending_entities(self, entity_evolution: List[Dict[str, int]], 
                              direction: str = 'up') -> List[str]:
        """Find entities trending up or down."""
        try:
            if len(entity_evolution) < 3:
                return []
            
            trending = []
            mid_point = len(entity_evolution) // 2
            
            # Compare first half vs second half
            first_half_entities = Counter()
            second_half_entities = Counter()
            
            for entities in entity_evolution[:mid_point]:
                for entity, count in entities.items():
                    first_half_entities[entity] += count
            
            for entities in entity_evolution[mid_point:]:
                for entity, count in entities.items():
                    second_half_entities[entity] += count
            
            if direction == 'up':
                # Find entities mentioned more in second half
                for entity in second_half_entities:
                    first_count = first_half_entities.get(entity, 0)
                    second_count = second_half_entities[entity]
                    
                    if second_count > first_count * 1.5 and second_count > 2:
                        trending.append(entity)
            
            else:  # direction == 'down'
                # Find entities mentioned less in second half
                for entity in first_half_entities:
                    first_count = first_half_entities[entity]
                    second_count = second_half_entities.get(entity, 0)
                    
                    if first_count > second_count * 1.5 and first_count > 2:
                        trending.append(entity)
            
            return trending[:10]  # Top 10 trending entities
            
        except Exception:
            return []
    
    def _calculate_overall_trend(self, aggregated_data: List[Dict[str, Any]]) -> str:
        """Calculate overall trend across all metrics."""
        try:
            # Consider multiple factors
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            controversy_scores = [point.get('avg_controversy', 0) for point in aggregated_data]
            volume_scores = [point.get('count', 0) for point in aggregated_data]
            
            sentiment_trend = self._calculate_linear_trend(sentiment_scores)
            controversy_trend = self._calculate_linear_trend(controversy_scores)
            volume_trend = self._calculate_linear_trend(volume_scores)
            
            # Combine trends with weights
            trend_weights = {'increasing': 1, 'stable': 0, 'decreasing': -1}
            
            weighted_score = (
                trend_weights[sentiment_trend] * 0.4 +
                trend_weights[volume_trend] * 0.4 +
                trend_weights[controversy_trend] * 0.2  # Controversy less weight for overall
            )
            
            if weighted_score > 0.2:
                return 'improving'
            elif weighted_score < -0.2:
                return 'declining'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _calculate_trend_strength(self, aggregated_data: List[Dict[str, Any]]) -> float:
        """Calculate strength of the trend (0-1)."""
        try:
            if len(aggregated_data) < 2:
                return 0.0
            
            # Calculate based on consistency of trend direction
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            
            if DEPENDENCIES_AVAILABLE and len(sentiment_scores) > 2:
                x = np.arange(len(sentiment_scores))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, sentiment_scores)
                # R-squared as strength indicator
                return abs(r_value) ** 2
            else:
                # Simple consistency measure
                changes = []
                for i in range(1, len(sentiment_scores)):
                    change = sentiment_scores[i] - sentiment_scores[i-1]
                    changes.append(1 if change > 0 else -1 if change < 0 else 0)
                
                if not changes:
                    return 0.0
                
                # Count consistent direction changes
                positive_changes = sum(1 for c in changes if c > 0)
                negative_changes = sum(1 for c in changes if c < 0)
                
                max_consistent = max(positive_changes, negative_changes)
                return max_consistent / len(changes)
                
        except Exception:
            return 0.0
    
    def _calculate_confidence(self, aggregated_data: List[Dict[str, Any]], 
                            trend_strength: float) -> float:
        """Calculate confidence in trend analysis."""
        try:
            # Base confidence on data amount and trend strength
            data_points = len(aggregated_data)
            
            # More data points = higher confidence
            data_confidence = min(1.0, data_points / 10)
            
            # Higher trend strength = higher confidence
            strength_confidence = trend_strength
            
            # Lower volatility = higher confidence
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            volatility = self._calculate_volatility(sentiment_scores)
            volatility_confidence = max(0, 1 - volatility)
            
            # Combine factors
            confidence = (data_confidence * 0.4 + strength_confidence * 0.4 + volatility_confidence * 0.2)
            
            return max(0.1, min(0.95, confidence))
            
        except Exception:
            return 0.5
    
    def _detect_anomalies(self, aggregated_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalous data points."""
        anomalies = []
        
        try:
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            controversy_scores = [point.get('avg_controversy', 0) for point in aggregated_data]
            
            if len(sentiment_scores) < 3:
                return anomalies
            
            # Calculate thresholds for anomaly detection
            sentiment_mean = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_std = self._calculate_volatility(sentiment_scores)
            
            controversy_mean = sum(controversy_scores) / len(controversy_scores)
            controversy_std = self._calculate_volatility(controversy_scores)
            
            # Detect outliers (> 2 standard deviations)
            for i, point in enumerate(aggregated_data):
                sentiment = sentiment_scores[i]
                controversy = controversy_scores[i]
                
                sentiment_z = abs(sentiment - sentiment_mean) / max(sentiment_std, 0.1)
                controversy_z = abs(controversy - controversy_mean) / max(controversy_std, 0.1)
                
                if sentiment_z > 2 or controversy_z > 2:
                    anomalies.append({
                        'date': point.get('time_key'),
                        'type': 'outlier',
                        'sentiment_score': sentiment,
                        'controversy_score': controversy,
                        'sentiment_z_score': sentiment_z,
                        'controversy_z_score': controversy_z
                    })
            
        except Exception as e:
            self.logger.warning(f"Error detecting anomalies: {e}")
        
        return anomalies[-10:]  # Return last 10 anomalies
    
    def _generate_predictions(self, aggregated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate simple predictions based on trends."""
        predictions = {}
        
        try:
            if len(aggregated_data) < 3:
                return predictions
            
            sentiment_scores = [point.get('avg_sentiment', 0) for point in aggregated_data]
            controversy_scores = [point.get('avg_controversy', 0) for point in aggregated_data]
            
            # Simple linear extrapolation
            if DEPENDENCIES_AVAILABLE:
                # Predict next sentiment value
                x = np.arange(len(sentiment_scores))
                sentiment_slope, sentiment_intercept = np.polyfit(x, sentiment_scores, 1)
                next_sentiment = sentiment_slope * len(sentiment_scores) + sentiment_intercept
                
                # Predict next controversy value
                controversy_slope, controversy_intercept = np.polyfit(x, controversy_scores, 1)
                next_controversy = controversy_slope * len(controversy_scores) + controversy_intercept
                
                predictions['next_sentiment'] = float(np.clip(next_sentiment, -1, 1))
                predictions['next_controversy'] = float(np.clip(next_controversy, 0, 1))
            else:
                # Simple average of recent trend
                recent_sentiment = sentiment_scores[-3:]
                recent_controversy = controversy_scores[-3:]
                
                predictions['next_sentiment'] = sum(recent_sentiment) / len(recent_sentiment)
                predictions['next_controversy'] = sum(recent_controversy) / len(recent_controversy)
            
            # Confidence in predictions
            predictions['confidence'] = min(0.7, len(aggregated_data) / 10)
            
        except Exception as e:
            self.logger.warning(f"Error generating predictions: {e}")
        
        return predictions
    
    def _empty_trend_analysis(self) -> TrendAnalysis:
        """Return empty trend analysis for error cases."""
        return TrendAnalysis(
            time_period="",
            trend_direction="stable",
            trend_strength=0.0,
            confidence=0.0,
            sentiment_trends={},
            controversy_trends={},
            topic_trends={},
            volume_trends={},
            entity_trends={},
            anomalies=[],
            predictions={},
            data_points=0,
            time_window="daily"
        )