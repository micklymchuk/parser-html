"""
Stage 0: Wayback Snapshot Processor for processing Wayback Machine snapshots
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaybackProcessor:
    """Processor for handling Wayback Machine snapshot directories."""
    
    def __init__(self):
        """Initialize the Wayback Processor."""
        self.processed_count = 0
        self.error_count = 0
    
    def process_snapshots_directory(self, directory_path: str, require_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Process an entire directory of Wayback Machine snapshots.
        
        Args:
            directory_path: Path to the directory containing snapshots
            require_metadata: If True, only process HTML files with corresponding meta.json files
                            If False, process all HTML files (with or without metadata)
            
        Returns:
            List of dictionaries containing HTML content and enhanced metadata
        """
        try:
            logger.info(f"Processing snapshots directory: {directory_path}")
            logger.info(f"Require metadata: {require_metadata}")
            
            # Convert to Path object for easier handling
            directory = Path(directory_path)
            
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not directory.is_dir():
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            if require_metadata:
                # Original behavior: only process HTML files with meta.json
                html_files_to_process = self.find_html_meta_pairs(directory)
                if not html_files_to_process:
                    logger.warning(f"No HTML-metadata pairs found in {directory_path}")
                    return []
                logger.info(f"Found {len(html_files_to_process)} HTML-metadata pairs")
            else:
                # New behavior: process all HTML files (with or without metadata)
                html_files_to_process = self.find_all_html_files(directory)
                if not html_files_to_process:
                    logger.warning(f"No HTML files found in {directory_path}")
                    return []
                logger.info(f"Found {len(html_files_to_process)} HTML files to process")
            
            # Process each file
            processed_snapshots = []
            self.processed_count = 0
            self.error_count = 0
            
            for item in tqdm(html_files_to_process, desc="Processing snapshots"):
                try:
                    if require_metadata:
                        # item is (html_path, meta_path) tuple
                        html_path, meta_path = item
                        snapshot_data = self.read_snapshot_data(html_path, meta_path)
                    else:
                        # item is html_path only
                        html_path = item
                        snapshot_data = self.read_html_file(html_path)
                    
                    if snapshot_data:
                        processed_snapshots.append(snapshot_data)
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing snapshot {html_path}: {e}")
                    self.error_count += 1
                    continue
            
            logger.info(f"Processing completed: {self.processed_count} successful, {self.error_count} errors")
            return processed_snapshots
            
        except Exception as e:
            logger.error(f"Error processing snapshots directory: {e}")
            raise
    
    def find_html_meta_pairs(self, directory: Path) -> List[Tuple[Path, Path]]:
        """
        Find matching HTML and meta.json file pairs in the directory.
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of tuples containing (html_path, meta_path) pairs
        """
        try:
            logger.info(f"Searching for HTML-metadata pairs in {directory}")
            
            html_meta_pairs = []
            
            # Recursively search for HTML files
            html_files = []
            for pattern in ['**/*.html', '**/*.htm']:
                html_files.extend(directory.glob(pattern))
            
            logger.info(f"Found {len(html_files)} HTML files")
            
            # For each HTML file, look for corresponding meta.json
            for html_file in html_files:
                # Construct expected meta.json filename
                meta_file = html_file.with_suffix(html_file.suffix + '.meta.json')
                
                if meta_file.exists():
                    html_meta_pairs.append((html_file, meta_file))
                    logger.debug(f"Found pair: {html_file.name} <-> {meta_file.name}")
                else:
                    logger.warning(f"No metadata file found for {html_file.name}")
            
            logger.info(f"Found {len(html_meta_pairs)} complete HTML-metadata pairs")
            return html_meta_pairs
            
        except Exception as e:
            logger.error(f"Error finding HTML-metadata pairs: {e}")
            return []
    
    def find_all_html_files(self, directory: Path) -> List[Path]:
        """
        Find all HTML files in the directory (with or without metadata).
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of HTML file paths
        """
        try:
            logger.info(f"Searching for all HTML files in {directory}")
            
            # Recursively search for HTML files
            html_files = []
            for pattern in ['**/*.html', '**/*.htm']:
                html_files.extend(directory.glob(pattern))
            
            logger.info(f"Found {len(html_files)} HTML files total")
            
            # Log which files have metadata and which don't
            with_metadata = 0
            without_metadata = 0
            
            for html_file in html_files:
                meta_file = html_file.with_suffix(html_file.suffix + '.meta.json')
                if meta_file.exists():
                    with_metadata += 1
                    logger.debug(f"HTML with metadata: {html_file.name}")
                else:
                    without_metadata += 1
                    logger.debug(f"HTML without metadata: {html_file.name}")
            
            logger.info(f"HTML files breakdown: {with_metadata} with metadata, {without_metadata} without metadata")
            return html_files
            
        except Exception as e:
            logger.error(f"Error finding HTML files: {e}")
            return []
    
    def read_snapshot_data(self, html_path: Path, meta_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read HTML content and metadata from snapshot files.
        
        Args:
            html_path: Path to HTML file
            meta_path: Path to metadata JSON file
            
        Returns:
            Dictionary containing HTML content and metadata, or None if error
        """
        try:
            logger.debug(f"Reading snapshot data: {html_path.name}")
            
            # Read HTML content
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings if UTF-8 fails
                logger.warning(f"UTF-8 decode failed for {html_path.name}, trying other encodings")
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(html_path, 'r', encoding=encoding) as f:
                            html_content = f.read()
                        logger.info(f"Successfully read {html_path.name} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.error(f"Failed to decode {html_path.name} with any encoding")
                    return None
            
            if not html_content.strip():
                logger.warning(f"Empty HTML content in {html_path.name}")
                return None
            
            # Read metadata
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {meta_path.name}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error reading metadata {meta_path.name}: {e}")
                return None
            
            # Validate metadata structure
            required_fields = ['archive_url', 'original_url', 'timestamp']
            missing_fields = [field for field in required_fields if field not in metadata]
            
            if missing_fields:
                logger.warning(f"Missing required metadata fields in {meta_path.name}: {missing_fields}")
                # Add default values for missing fields
                for field in missing_fields:
                    if field == 'archive_url':
                        metadata[field] = f"unknown_archive_url_{html_path.stem}"
                    elif field == 'original_url':
                        metadata[field] = f"unknown_original_url_{html_path.stem}"
                    elif field == 'timestamp':
                        metadata[field] = "unknown_timestamp"
            
            # Enhance metadata with file information
            metadata.update({
                'html_file_path': str(html_path),
                'meta_file_path': str(meta_path),
                'html_file_size': html_path.stat().st_size,
                'html_content_length': len(html_content)
            })
            
            # Ensure domain is extracted if not present
            if 'domain' not in metadata and metadata.get('original_url'):
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(metadata['original_url'])
                    metadata['domain'] = parsed_url.netloc
                except Exception:
                    metadata['domain'] = 'unknown_domain'
            
            # Create snapshot data structure
            snapshot_data = {
                'html': html_content,
                'url': metadata['original_url'],
                'wayback_metadata': metadata
            }
            
            logger.debug(f"Successfully processed snapshot: {html_path.name}")
            return snapshot_data
            
        except Exception as e:
            logger.error(f"Error reading snapshot data from {html_path.name}: {e}")
            return None
    
    def read_html_file(self, html_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read HTML file without metadata (creates synthetic metadata).
        
        Args:
            html_path: Path to HTML file
            
        Returns:
            Dictionary containing HTML content and synthetic metadata, or None if error
        """
        try:
            logger.debug(f"Reading HTML file without metadata: {html_path.name}")
            
            # Read HTML content with encoding handling
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings if UTF-8 fails
                logger.warning(f"UTF-8 decode failed for {html_path.name}, trying other encodings")
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(html_path, 'r', encoding=encoding) as f:
                            html_content = f.read()
                        logger.info(f"Successfully read {html_path.name} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.error(f"Failed to decode {html_path.name} with any encoding")
                    return None
            
            if not html_content.strip():
                logger.warning(f"Empty HTML content in {html_path.name}")
                return None
            
            # Create synthetic metadata from filename and file info
            file_stat = html_path.stat()
            
            # Try to extract domain and URL info from filename
            filename = html_path.stem  # filename without extension
            
            # Common wayback filename patterns: domain_.html, domain_page_.html
            if '_' in filename:
                domain_part = filename.split('_')[0]
                # Clean up common wayback filename artifacts
                domain = domain_part.replace('-', '.').replace('_', '.')
            else:
                domain = 'unknown_domain'
            
            # Try to extract timestamp from parent directory name
            timestamp = 'unknown_timestamp'
            for parent in html_path.parents:
                parent_name = parent.name
                # Check if it looks like a wayback timestamp (14 digits)
                if parent_name.isdigit() and len(parent_name) == 14:
                    timestamp = parent_name
                    break
            
            # Construct URLs
            if domain != 'unknown_domain':
                original_url = f"https://{domain}/"
                if timestamp != 'unknown_timestamp':
                    archive_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
                else:
                    archive_url = f"https://web.archive.org/web/*/{original_url}"
            else:
                original_url = f"unknown_url_{filename}"
                archive_url = f"unknown_archive_url_{filename}"
            
            # Extract title from HTML content if possible
            title = 'Unknown Title'
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                title_tag = soup.find('title')
                if title_tag and title_tag.string:
                    title = title_tag.string.strip()
            except Exception:
                # If BeautifulSoup fails, try simple regex
                import re
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
            
            # Create synthetic metadata
            metadata = {
                'archive_url': archive_url,
                'original_url': original_url,
                'timestamp': timestamp,
                'title': title,
                'content_length': len(html_content),
                'domain': domain,
                'file_path': str(html_path),
                'html_file_path': str(html_path),
                'html_file_size': file_stat.st_size,
                'html_content_length': len(html_content),
                'metadata_source': 'synthetic',  # Flag to indicate this is synthetic metadata
                'file_modified_time': file_stat.st_mtime
            }
            
            # Create snapshot data structure
            snapshot_data = {
                'html': html_content,
                'url': original_url,
                'wayback_metadata': metadata
            }
            
            logger.debug(f"Successfully processed HTML file with synthetic metadata: {html_path.name}")
            return snapshot_data
            
        except Exception as e:
            logger.error(f"Error reading HTML file {html_path.name}: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        Get statistics about the last processing operation.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'total_attempted': self.processed_count + self.error_count,
            'success_rate': (self.processed_count / (self.processed_count + self.error_count) * 100) 
                          if (self.processed_count + self.error_count) > 0 else 0
        }
    
    def validate_snapshot_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Validate a snapshot directory structure and provide diagnostics.
        
        Args:
            directory_path: Path to the directory to validate
            
        Returns:
            Dictionary with validation results and diagnostics
        """
        try:
            logger.info(f"Validating snapshot directory: {directory_path}")
            
            directory = Path(directory_path)
            validation_results = {
                'is_valid': False,
                'directory_exists': directory.exists(),
                'is_directory': directory.is_dir() if directory.exists() else False,
                'html_files_count': 0,
                'meta_files_count': 0,
                'paired_files_count': 0,
                'orphaned_html_files': [],
                'orphaned_meta_files': [],
                'errors': []
            }
            
            if not validation_results['directory_exists']:
                validation_results['errors'].append(f"Directory does not exist: {directory_path}")
                return validation_results
            
            if not validation_results['is_directory']:
                validation_results['errors'].append(f"Path is not a directory: {directory_path}")
                return validation_results
            
            # Count HTML files
            html_files = list(directory.glob('**/*.html')) + list(directory.glob('**/*.htm'))
            validation_results['html_files_count'] = len(html_files)
            
            # Count meta files
            meta_files = list(directory.glob('**/*.meta.json'))
            validation_results['meta_files_count'] = len(meta_files)
            
            # Find pairs and orphans
            html_with_meta = []
            html_without_meta = []
            
            for html_file in html_files:
                meta_file = html_file.with_suffix(html_file.suffix + '.meta.json')
                if meta_file.exists():
                    html_with_meta.append(html_file)
                else:
                    html_without_meta.append(html_file)
            
            # Find orphaned meta files
            meta_without_html = []
            for meta_file in meta_files:
                # Remove .meta.json to get original filename
                original_name = meta_file.name.replace('.meta.json', '')
                html_file = meta_file.parent / original_name
                if not html_file.exists():
                    meta_without_html.append(meta_file)
            
            validation_results.update({
                'paired_files_count': len(html_with_meta),
                'orphaned_html_files': [str(f) for f in html_without_meta],
                'orphaned_meta_files': [str(f) for f in meta_without_html]
            })
            
            # Determine if directory is valid
            validation_results['is_valid'] = (
                validation_results['directory_exists'] and
                validation_results['is_directory'] and
                validation_results['paired_files_count'] > 0
            )
            
            if validation_results['paired_files_count'] == 0:
                validation_results['errors'].append("No complete HTML-metadata pairs found")
            
            logger.info(f"Validation completed: {validation_results['paired_files_count']} valid pairs found")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating directory: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
    
    def extract_timestamp_info(self, timestamp: str) -> Dict[str, Any]:
        """
        Extract detailed information from Wayback Machine timestamp.
        
        Args:
            timestamp: Wayback timestamp string (format: YYYYMMDDHHMMSS)
            
        Returns:
            Dictionary with parsed timestamp information
        """
        try:
            if not timestamp or len(timestamp) != 14:
                return {'error': 'Invalid timestamp format'}
            
            from datetime import datetime
            
            # Parse timestamp
            dt = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            
            return {
                'timestamp': timestamp,
                'datetime': dt.isoformat(),
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'minute': dt.minute,
                'second': dt.second,
                'weekday': dt.strftime('%A'),
                'human_readable': dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
        except Exception as e:
            logger.error(f"Error parsing timestamp {timestamp}: {e}")
            return {'error': f'Failed to parse timestamp: {str(e)}'}
    
    def filter_snapshots_by_criteria(
        self, 
        snapshots: List[Dict[str, Any]], 
        domain_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        min_content_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter snapshots based on various criteria.
        
        Args:
            snapshots: List of snapshot data dictionaries
            domain_filter: Filter by domain (exact match)
            year_filter: Filter by year from timestamp
            min_content_length: Minimum content length threshold
            
        Returns:
            Filtered list of snapshots
        """
        try:
            logger.info(f"Filtering {len(snapshots)} snapshots with criteria")
            
            filtered_snapshots = snapshots.copy()
            
            # Filter by domain
            if domain_filter:
                filtered_snapshots = [
                    s for s in filtered_snapshots 
                    if s.get('wayback_metadata', {}).get('domain') == domain_filter
                ]
                logger.info(f"After domain filter '{domain_filter}': {len(filtered_snapshots)} snapshots")
            
            # Filter by year
            if year_filter:
                year_filtered = []
                for snapshot in filtered_snapshots:
                    timestamp = snapshot.get('wayback_metadata', {}).get('timestamp', '')
                    if timestamp and len(timestamp) >= 4:
                        try:
                            snapshot_year = int(timestamp[:4])
                            if snapshot_year == year_filter:
                                year_filtered.append(snapshot)
                        except ValueError:
                            continue
                filtered_snapshots = year_filtered
                logger.info(f"After year filter '{year_filter}': {len(filtered_snapshots)} snapshots")
            
            # Filter by content length
            if min_content_length:
                filtered_snapshots = [
                    s for s in filtered_snapshots 
                    if s.get('wayback_metadata', {}).get('content_length', 0) >= min_content_length
                ]
                logger.info(f"After content length filter (>={min_content_length}): {len(filtered_snapshots)} snapshots")
            
            logger.info(f"Filtering completed: {len(filtered_snapshots)} snapshots remaining")
            return filtered_snapshots
            
        except Exception as e:
            logger.error(f"Error filtering snapshots: {e}")
            return snapshots