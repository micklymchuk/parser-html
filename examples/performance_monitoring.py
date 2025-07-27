#!/usr/bin/env python3
"""
Performance monitoring example for HTML RAG Pipeline.

This example demonstrates:
- Processing time metrics and analysis
- Memory usage tracking and optimization
- Database performance statistics
- Resource utilization monitoring
- Performance profiling and bottleneck identification
- Optimization recommendations
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import psutil
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from html_rag.core.pipeline import create_pipeline
from html_rag.core.config import PipelineConfig
from html_rag.utils.logging import setup_logging, PipelineLogger
from html_rag.utils.metrics import (
    MetricsCollector, PerformanceProfiler, track_processing,
    benchmark_function, profiler
)
from html_rag.exceptions.pipeline_exceptions import PipelineError

# Setup logging
setup_logging(level="INFO", log_file="logs/performance_monitoring.log")
logger = PipelineLogger("PerformanceMonitoring")


class SystemMonitor:
    """Advanced system monitoring for pipeline performance."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.samples = []
        self.process = psutil.Process()
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"🔧 System monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("🔧 System monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        while self.monitoring:
            try:
                # System-wide metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                # Process-specific metrics
                process_memory = self.process.memory_info()
                process_cpu = self.process.cpu_percent()
                
                sample = {
                    'timestamp': time.time(),
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_available_gb': memory.available / (1024**3),
                        'memory_percent': memory.percent,
                        'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                        'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
                    },
                    'process': {
                        'memory_rss_mb': process_memory.rss / (1024**2),
                        'memory_vms_mb': process_memory.vms / (1024**2),
                        'cpu_percent': process_cpu,
                        'num_threads': self.process.num_threads()
                    }
                }
                
                self.samples.append(sample)
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.samples:
            return {}
        
        # Calculate statistics
        system_cpu = [s['system']['cpu_percent'] for s in self.samples]
        system_memory = [s['system']['memory_percent'] for s in self.samples]
        process_memory = [s['process']['memory_rss_mb'] for s in self.samples]
        process_cpu = [s['process']['cpu_percent'] for s in self.samples]
        
        return {
            'duration_seconds': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'],
            'sample_count': len(self.samples),
            'system': {
                'cpu_avg': sum(system_cpu) / len(system_cpu),
                'cpu_max': max(system_cpu),
                'memory_avg': sum(system_memory) / len(system_memory),
                'memory_max': max(system_memory)
            },
            'process': {
                'memory_peak_mb': max(process_memory),
                'memory_avg_mb': sum(process_memory) / len(process_memory),
                'cpu_avg': sum(process_cpu) / len(process_cpu),
                'cpu_max': max(process_cpu)
            },
            'raw_samples': self.samples
        }


def create_test_documents(num_docs: int = 20) -> List[Dict[str, Any]]:
    """Create test documents of varying sizes for performance testing."""
    
    documents = []
    
    # Templates of different sizes
    templates = {
        'small': '''<!DOCTYPE html>
        <html><head><title>Small Doc {i}</title></head>
        <body>
            <h1>Small Document {i}</h1>
            <p>This is a small test document with minimal content.</p>
        </body></html>''',
        
        'medium': '''<!DOCTYPE html>
        <html><head><title>Medium Doc {i}</title></head>
        <body>
            <h1>Medium Document {i}</h1>
            <p>This is a medium-sized test document with more substantial content for performance testing.</p>
            <h2>Section 1</h2>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            <h2>Section 2</h2>
            <ul>
                <li>Item 1 with some description</li>
                <li>Item 2 with more details</li>
                <li>Item 3 with comprehensive information</li>
            </ul>
            <h3>Subsection</h3>
            <p>Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.</p>
        </body></html>''',
        
        'large': '''<!DOCTYPE html>
        <html><head><title>Large Doc {i}</title></head>
        <body>
            <h1>Large Document {i}</h1>
            <p>This is a large test document designed to test performance with substantial content.</p>
            ''' + '\n'.join([f'''
            <h2>Section {j}</h2>
            <p>This is section {j} with detailed content for performance testing. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
            <h3>Subsection {j}.1</h3>
            <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
            <ul>
                <li>List item {j}.1 with comprehensive details</li>
                <li>List item {j}.2 with extensive information</li>
                <li>List item {j}.3 with thorough explanations</li>
            </ul>
            ''' for j in range(1, 6)]) + '''
        </body></html>'''
    }
    
    # Create documents with different sizes
    for i in range(num_docs):
        if i < num_docs // 3:
            template_type = 'small'
        elif i < 2 * num_docs // 3:
            template_type = 'medium'
        else:
            template_type = 'large'
        
        html_content = templates[template_type].format(i=i+1)
        
        documents.append({
            'html': html_content,
            'url': f'https://test.com/doc_{i+1:03d}',
            'size_category': template_type,
            'estimated_size': len(html_content)
        })
    
    return documents


def run_performance_benchmark(
    pipeline,
    documents: List[Dict[str, Any]],
    monitor: SystemMonitor
) -> Dict[str, Any]:
    """Run comprehensive performance benchmark."""
    
    logger.info(f"🚀 Starting performance benchmark with {len(documents)} documents...")
    
    # Start monitoring
    monitor.start_monitoring(interval=0.5)
    
    # Track overall processing
    with track_processing(enable_resource_monitoring=True) as metrics:
        start_time = time.time()
        
        # Process documents one by one to track individual performance
        individual_results = []
        
        for i, doc in enumerate(documents):
            logger.info(f"📄 Processing document {i+1}/{len(documents)} ({doc['size_category']})")
            
            # Benchmark individual document processing
            doc_start = time.time()
            
            try:
                result = pipeline.process_html(
                    raw_html=doc['html'],
                    url=doc['url']
                )
                
                doc_duration = time.time() - doc_start
                
                individual_results.append({
                    'document_index': i,
                    'size_category': doc['size_category'],
                    'estimated_size': doc['estimated_size'],
                    'processing_time': doc_duration,
                    'success': result.get('success', False),
                    'text_blocks': result.get('text_blocks_count', 0),
                    'embedded_blocks': result.get('embedded_blocks_count', 0),
                    'original_html_length': result.get('original_html_length', 0),
                    'cleaned_html_length': result.get('cleaned_html_length', 0)
                })
                
                # Log progress
                if result['success']:
                    throughput = doc['estimated_size'] / doc_duration if doc_duration > 0 else 0
                    logger.info(f"   ✅ Success: {doc_duration:.2f}s, {throughput:.0f} chars/sec")
                else:
                    logger.warning(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                doc_duration = time.time() - doc_start
                individual_results.append({
                    'document_index': i,
                    'size_category': doc['size_category'],
                    'estimated_size': doc['estimated_size'],
                    'processing_time': doc_duration,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"   ❌ Exception: {str(e)}")
        
        total_time = time.time() - start_time
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Compile benchmark results
    benchmark_results = {
        'total_documents': len(documents),
        'total_time': total_time,
        'individual_results': individual_results,
        'metrics': metrics.get_metrics_dict() if metrics else {},
        'system_monitoring': monitor.get_summary()
    }
    
    return benchmark_results


def analyze_performance_results(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance benchmark results."""
    
    individual_results = benchmark_results['individual_results']
    successful_results = [r for r in individual_results if r.get('success', False)]
    failed_results = [r for r in individual_results if not r.get('success', False)]
    
    # Overall statistics
    total_docs = len(individual_results)
    success_count = len(successful_results)
    failure_count = len(failed_results)
    
    # Performance by size category
    size_performance = {}
    for category in ['small', 'medium', 'large']:
        category_results = [r for r in successful_results if r.get('size_category') == category]
        
        if category_results:
            processing_times = [r['processing_time'] for r in category_results]
            sizes = [r['estimated_size'] for r in category_results]
            
            size_performance[category] = {
                'count': len(category_results),
                'avg_time': sum(processing_times) / len(processing_times),
                'min_time': min(processing_times),
                'max_time': max(processing_times),
                'avg_size': sum(sizes) / len(sizes),
                'throughput_chars_per_sec': sum(sizes) / sum(processing_times)
            }
    
    # Processing efficiency
    if successful_results:
        total_chars_processed = sum(r['estimated_size'] for r in successful_results)
        total_processing_time = sum(r['processing_time'] for r in successful_results)
        total_text_blocks = sum(r['text_blocks'] for r in successful_results)
        total_embedded_blocks = sum(r['embedded_blocks'] for r in successful_results)
        
        efficiency_metrics = {
            'overall_throughput_chars_per_sec': total_chars_processed / total_processing_time,
            'avg_processing_time_per_doc': total_processing_time / len(successful_results),
            'avg_text_blocks_per_doc': total_text_blocks / len(successful_results),
            'avg_embedded_blocks_per_doc': total_embedded_blocks / len(successful_results),
            'text_extraction_efficiency': total_text_blocks / total_docs if total_docs > 0 else 0
        }
    else:
        efficiency_metrics = {}
    
    # System resource analysis
    system_summary = benchmark_results.get('system_monitoring', {})
    
    analysis = {
        'summary': {
            'total_documents': total_docs,
            'successful_processing': success_count,
            'failed_processing': failure_count,
            'success_rate': success_count / total_docs if total_docs > 0 else 0,
            'total_time': benchmark_results['total_time']
        },
        'performance_by_size': size_performance,
        'efficiency_metrics': efficiency_metrics,
        'system_resources': system_summary,
        'error_analysis': {
            'failure_count': failure_count,
            'common_errors': {}  # Could be expanded to categorize errors
        }
    }
    
    return analysis


def generate_optimization_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on performance analysis."""
    
    recommendations = []
    
    # Success rate recommendations
    success_rate = analysis['summary']['success_rate']
    if success_rate < 0.95:
        recommendations.append(
            f"⚠️  Success rate is {success_rate:.1%}. Consider improving error handling and input validation."
        )
    
    # Performance recommendations
    efficiency = analysis.get('efficiency_metrics', {})
    
    # Throughput recommendations
    throughput = efficiency.get('overall_throughput_chars_per_sec', 0)
    if throughput < 1000:
        recommendations.append(
            "🐌 Low throughput detected. Consider enabling batch processing or increasing worker count."
        )
    elif throughput > 10000:
        recommendations.append(
            "🚀 Excellent throughput! Current configuration is well-optimized."
        )
    
    # Processing time recommendations
    avg_time = efficiency.get('avg_processing_time_per_doc', 0)
    if avg_time > 5.0:
        recommendations.append(
            "⏰ High average processing time. Consider using basic HTML cleaning for better performance."
        )
    
    # Size-based recommendations
    size_performance = analysis.get('performance_by_size', {})
    
    if 'large' in size_performance:
        large_avg_time = size_performance['large'].get('avg_time', 0)
        if large_avg_time > 10.0:
            recommendations.append(
                "📄 Large documents are slow to process. Consider implementing content chunking or size limits."
            )
    
    # Resource recommendations
    system_resources = analysis.get('system_resources', {})
    process_info = system_resources.get('process', {})
    
    peak_memory = process_info.get('memory_peak_mb', 0)
    if peak_memory > 1000:  # > 1GB
        recommendations.append(
            f"💾 High memory usage detected ({peak_memory:.0f}MB). Consider processing smaller batches."
        )
    
    avg_cpu = process_info.get('cpu_avg', 0)
    if avg_cpu > 80:
        recommendations.append(
            "🖥️  High CPU usage. Consider reducing parallel processing or optimizing model inference."
        )
    elif avg_cpu < 20:
        recommendations.append(
            "🖥️  Low CPU usage. Consider increasing parallelism to utilize available resources."
        )
    
    # Model-specific recommendations
    text_extraction_efficiency = efficiency.get('text_extraction_efficiency', 0)
    if text_extraction_efficiency < 0.5:
        recommendations.append(
            "📝 Low text extraction efficiency. Review HTML cleaning and parsing configuration."
        )
    
    return recommendations


def main():
    """Performance monitoring demonstration."""
    
    logger.info("="*60)
    logger.info("HTML RAG Pipeline - Performance Monitoring Example")
    logger.info("="*60)
    
    try:
        # Example 1: Setup performance monitoring
        logger.info("\n1. Setting up performance monitoring...")
        
        # Create system monitor
        monitor = SystemMonitor()
        
        # Create performance profiler
        perf_profiler = PerformanceProfiler()
        
        logger.info("✅ Performance monitoring tools initialized")
        
        # Example 2: Configure pipeline for performance testing
        logger.info("\n2. Configuring pipeline for performance testing...")
        
        # Test different configurations
        configs = {
            'default': PipelineConfig(
                collection_name="performance_test_default",
                persist_directory="./perf_test_default_db",
                enable_metrics=True
            ),
            'optimized': PipelineConfig(
                collection_name="performance_test_optimized",
                persist_directory="./perf_test_optimized_db",
                enable_metrics=True,
                prefer_basic_cleaning=True,  # Faster processing
                batch_size=64,
                max_chunk_size=256
            )
        }
        
        logger.info("✅ Pipeline configurations prepared")
        
        # Example 3: Create test documents
        logger.info("\n3. Creating test documents...")
        
        test_documents = create_test_documents(num_docs=15)
        
        # Analyze document distribution
        size_counts = {}
        total_size = 0
        for doc in test_documents:
            category = doc['size_category']
            size_counts[category] = size_counts.get(category, 0) + 1
            total_size += doc['estimated_size']
        
        logger.info(f"📄 Created {len(test_documents)} test documents:")
        for category, count in size_counts.items():
            logger.info(f"   {category}: {count} documents")
        logger.info(f"   Total estimated size: {total_size:,} characters")
        
        # Example 4: Run performance benchmarks
        logger.info("\n4. Running performance benchmarks...")
        
        benchmark_results = {}
        
        for config_name, config in configs.items():
            logger.info(f"\n📊 Testing {config_name} configuration...")
            
            # Create pipeline with current configuration
            pipeline = create_pipeline(config=config)
            
            # Run benchmark
            results = run_performance_benchmark(pipeline, test_documents, monitor)
            benchmark_results[config_name] = results
            
            # Quick summary
            total_time = results['total_time']
            success_count = sum(1 for r in results['individual_results'] if r.get('success', False))
            throughput = len(test_documents) / total_time
            
            logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
            logger.info(f"   ✅ Success rate: {success_count}/{len(test_documents)} ({success_count/len(test_documents):.1%})")
            logger.info(f"   📈 Throughput: {throughput:.2f} docs/sec")
            
            # Cleanup pipeline
            pipeline.cleanup()
        
        # Example 5: Analyze performance results
        logger.info("\n5. Analyzing performance results...")
        
        analyses = {}
        for config_name, results in benchmark_results.items():
            analysis = analyze_performance_results(results)
            analyses[config_name] = analysis
            
            logger.info(f"\n📊 {config_name.capitalize()} Configuration Analysis:")
            logger.info(f"   Success rate: {analysis['summary']['success_rate']:.1%}")
            logger.info(f"   Avg processing time: {analysis['efficiency_metrics'].get('avg_processing_time_per_doc', 0):.2f}s/doc")
            logger.info(f"   Throughput: {analysis['efficiency_metrics'].get('overall_throughput_chars_per_sec', 0):.0f} chars/sec")
            
            # System resources
            if 'system_resources' in analysis and analysis['system_resources']:
                process_info = analysis['system_resources'].get('process', {})
                logger.info(f"   Peak memory: {process_info.get('memory_peak_mb', 0):.0f}MB")
                logger.info(f"   Avg CPU: {process_info.get('cpu_avg', 0):.1f}%")
        
        # Example 6: Performance comparison
        logger.info("\n6. Comparing configuration performance...")
        
        if len(analyses) >= 2:
            config_names = list(analyses.keys())
            config1, config2 = config_names[0], config_names[1]
            
            analysis1 = analyses[config1]
            analysis2 = analyses[config2]
            
            # Compare key metrics
            metrics_comparison = {
                'success_rate': {
                    config1: analysis1['summary']['success_rate'],
                    config2: analysis2['summary']['success_rate']
                },
                'avg_processing_time': {
                    config1: analysis1['efficiency_metrics'].get('avg_processing_time_per_doc', 0),
                    config2: analysis2['efficiency_metrics'].get('avg_processing_time_per_doc', 0)
                },
                'throughput': {
                    config1: analysis1['efficiency_metrics'].get('overall_throughput_chars_per_sec', 0),
                    config2: analysis2['efficiency_metrics'].get('overall_throughput_chars_per_sec', 0)
                }
            }
            
            logger.info(f"📊 Performance Comparison: {config1} vs {config2}")
            for metric, values in metrics_comparison.items():
                val1, val2 = values[config1], values[config2]
                if val1 > 0 and val2 > 0:
                    improvement = ((val2 - val1) / val1) * 100
                    better_config = config2 if val2 > val1 else config1
                    logger.info(f"   {metric}: {better_config} is {abs(improvement):.1f}% better")
        
        # Example 7: Generate optimization recommendations
        logger.info("\n7. Generating optimization recommendations...")
        
        for config_name, analysis in analyses.items():
            recommendations = generate_optimization_recommendations(analysis)
            
            logger.info(f"\n💡 Recommendations for {config_name} configuration:")
            if recommendations:
                for rec in recommendations:
                    logger.info(f"   {rec}")
            else:
                logger.info("   ✅ Configuration appears to be well-optimized!")
        
        # Example 8: Export performance data
        logger.info("\n8. Exporting performance monitoring data...")
        
        # Compile comprehensive performance report
        performance_report = {
            'test_configuration': {
                'document_count': len(test_documents),
                'document_distribution': size_counts,
                'total_test_size': total_size
            },
            'pipeline_configurations': {name: config.dict() for name, config in configs.items()},
            'benchmark_results': benchmark_results,
            'performance_analyses': analyses,
            'optimization_recommendations': {
                config_name: generate_optimization_recommendations(analysis)
                for config_name, analysis in analyses.items()
            },
            'metadata': {
                'test_timestamp': time.time(),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'python_version': sys.version
                }
            }
        }
        
        # Export to file
        export_file = Path("performance_monitoring_report.json")
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Performance report exported to {export_file}")
        
        # Example 9: Performance summary
        logger.info("\n9. Performance monitoring summary...")
        
        # Find best performing configuration
        best_config = None
        best_throughput = 0
        
        for config_name, analysis in analyses.items():
            throughput = analysis['efficiency_metrics'].get('overall_throughput_chars_per_sec', 0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config_name
        
        logger.info(f"🏆 Performance Summary:")
        logger.info(f"   Best configuration: {best_config}")
        logger.info(f"   Best throughput: {best_throughput:.0f} chars/sec")
        logger.info(f"   Configurations tested: {len(analyses)}")
        logger.info(f"   Documents processed: {len(test_documents) * len(configs)}")
        
        # Overall recommendations
        overall_recommendations = []
        for recs in performance_report['optimization_recommendations'].values():
            overall_recommendations.extend(recs)
        
        unique_recommendations = list(set(overall_recommendations))
        
        if unique_recommendations:
            logger.info(f"\n💡 Key Optimization Opportunities:")
            for rec in unique_recommendations[:5]:  # Top 5 recommendations
                logger.info(f"   {rec}")
        
        logger.info("\n" + "="*60)
        logger.info("Performance monitoring example completed successfully! 🎉")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)