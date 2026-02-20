"""
CSV Reporter untuk ESG Reports
Generate reports dalam format CSV untuk analisis lebih lanjut
"""

import csv
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVReporter:
    """
    Generate CSV reports untuk traffic analysis
    
    Reports:
    - Vehicle detection log
    - Speed violations
    - Helmet violations
    - Hourly/ Daily summary
    - ESG compliance metrics
    """
    
    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize reporter
        
        Args:
            output_dir: Directory for report files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory log
        self.vehicle_log: List[Dict] = []
        self.violation_log: List[Dict] = []
        self.speed_log: List[Dict] = []
        
        # Summary counters
        self.hourly_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total': 0, 'cars': 0, 'motorcycles': 0, 'buses': 0, 'trucks': 0,
            'violations': 0, 'speeding': 0, 'no_helmet': 0,
            'avg_speed': 0.0, 'speed_sum': 0.0, 'speed_count': 0
        })
    
    def log_vehicle(self, result: Dict):
        """
        Log vehicle detection
        
        Args:
            result: AnalysisResult as dictionary
        """
        timestamp = datetime.now()
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        
        entry = {
            'timestamp': timestamp.isoformat(),
            'track_id': result.get('track_id', 0),
            'class_name': result.get('class_name', 'unknown'),
            'class_id': result.get('class_id', -1),
            'speed_kmh': result.get('speed_kmh', 0),
            'direction': result.get('direction', 'UNKNOWN'),
            'helmet_status': result.get('helmet_status', 'N/A'),
            'is_violation': result.get('is_violation', False),
            'bbox_x1': result.get('bbox', [0,0,0,0])[0],
            'bbox_y1': result.get('bbox', [0,0,0,0])[1],
            'bbox_x2': result.get('bbox', [0,0,0,0])[2],
            'bbox_y2': result.get('bbox', [0,0,0,0])[3],
            'confidence': result.get('confidence', 0)
        }
        
        self.vehicle_log.append(entry)
        
        # Update hourly stats
        stats = self.hourly_stats[hour_key]
        stats['total'] += 1
        
        class_name = entry['class_name']
        if class_name == 'car':
            stats['cars'] += 1
        elif class_name == 'motorcycle':
            stats['motorcycles'] += 1
        elif class_name == 'bus':
            stats['buses'] += 1
        elif class_name == 'truck':
            stats['trucks'] += 1
        
        speed = entry['speed_kmh']
        if speed > 0:
            stats['speed_sum'] += speed
            stats['speed_count'] += 1
            stats['avg_speed'] = stats['speed_sum'] / stats['speed_count']
    
    def log_violation(self, result: Dict):
        """
        Log traffic violation
        
        Args:
            result: AnalysisResult as dictionary
        """
        timestamp = datetime.now()
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        
        entry = {
            'timestamp': timestamp.isoformat(),
            'track_id': result.get('track_id', 0),
            'class_name': result.get('class_name', 'unknown'),
            'speed_kmh': result.get('speed_kmh', 0),
            'helmet_status': result.get('helmet_status', 'N/A'),
            'violation_types': [],
            'direction': result.get('direction', 'UNKNOWN')
        }
        
        # Determine violation types
        if result.get('is_speeding', False):
            entry['violation_types'].append('SPEEDING')
        if result.get('helmet_status') == 'NO_HELMET':
            entry['violation_types'].append('NO_HELMET')
        
        entry['violation_types'] = ','.join(entry['violation_types'])
        
        self.violation_log.append(entry)
        
        # Update hourly stats
        stats = self.hourly_stats[hour_key]
        stats['violations'] += 1
        
        if result.get('is_speeding', False):
            stats['speeding'] += 1
        if result.get('helmet_status') == 'NO_HELMET':
            stats['no_helmet'] += 1
    
    def log_speed(self, result: Dict):
        """
        Log speed measurement
        
        Args:
            result: AnalysisResult as dictionary
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'track_id': result.get('track_id', 0),
            'class_name': result.get('class_name', 'unknown'),
            'speed_kmh': result.get('speed_kmh', 0),
            'is_speeding': result.get('is_speeding', False)
        }
        
        self.speed_log.append(entry)
    
    def generate_vehicle_report(self, 
                                filename: Optional[str] = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> str:
        """
        Generate vehicle detection CSV report
        
        Args:
            filename: Output filename
            start_time: Filter start time
            end_time: Filter end time
        
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"vehicle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        
        # Filter by time if provided
        data = self.vehicle_log
        if start_time or end_time:
            data = [d for d in data if self._in_time_range(d['timestamp'], start_time, end_time)]
        
        # Write CSV
        if data:
            fieldnames = list(data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Vehicle report generated: {filepath}")
        else:
            logger.warning("No data to write to vehicle report")
            # Write empty file with headers
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("timestamp,track_id,class_name,speed_kmh,direction,helmet_status,is_violation\n")
        
        return str(filepath)
    
    def generate_violation_report(self, 
                                   filename: Optional[str] = None,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> str:
        """
        Generate violation CSV report
        
        Args:
            filename: Output filename
            start_time: Filter start time
            end_time: Filter end time
        
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        
        # Filter by time
        data = self.violation_log
        if start_time or end_time:
            data = [d for d in data if self._in_time_range(d['timestamp'], start_time, end_time)]
        
        # Write CSV
        if data:
            fieldnames = list(data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Violation report generated: {filepath}")
        else:
            logger.warning("No data to write to violation report")
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("timestamp,track_id,class_name,speed_kmh,helmet_status,violation_types\n")
        
        return str(filepath)
    
    def generate_summary_report(self, 
                                 filename: Optional[str] = None) -> str:
        """
        Generate summary report (hourly/daily statistics)
        
        Args:
            filename: Output filename
        
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert hourly stats to list
        data = []
        for hour_key, stats in sorted(self.hourly_stats.items()):
            data.append({
                'hour': hour_key,
                **stats
            })
        
        # Write CSV
        if data:
            fieldnames = ['hour', 'total', 'cars', 'motorcycles', 'buses', 'trucks',
                         'violations', 'speeding', 'no_helmet', 'avg_speed']
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Summary report generated: {filepath}")
        else:
            logger.warning("No data to write to summary report")
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("hour,total,cars,motorcycles,buses,trucks,violations,speeding,no_helmet,avg_speed\n")
        
        return str(filepath)
    
    def generate_esg_report(self,
                            filename: Optional[str] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> str:
        """
        Generate ESG (Environmental, Social, Governance) compliance report
        
        Args:
            filename: Output filename
            start_time: Filter start time
            end_time: Filter end time
        
        Returns:
            Path to generated file
        """
        if filename is None:
            filename = f"esg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        
        # Calculate ESG metrics
        total_vehicles = len(self.vehicle_log)
        total_violations = len(self.violation_log)
        
        # Safety compliance rate
        safety_rate = ((total_vehicles - total_violations) / total_vehicles * 100) if total_vehicles > 0 else 100
        
        # Helmet compliance (motorcycles only)
        motorcycles = [v for v in self.vehicle_log if v['class_name'] == 'motorcycle']
        helmet_violations = len([v for v in self.violation_log if 'NO_HELMET' in v.get('violation_types', '')])
        helmet_compliance = ((len(motorcycles) - helmet_violations) / len(motorcycles) * 100) if motorcycles else 100
        
        # Speed compliance
        speeding_count = len([v for v in self.violation_log if 'SPEEDING' in v.get('violation_types', '')])
        speed_compliance = ((total_vehicles - speeding_count) / total_vehicles * 100) if total_vehicles > 0 else 100
        
        # Vehicle type distribution
        vehicle_types = defaultdict(int)
        for v in self.vehicle_log:
            vehicle_types[v['class_name']] += 1
        
        # Average speed
        speeds = [v['speed_kmh'] for v in self.vehicle_log if v['speed_kmh'] > 0]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Create report
        report = {
            'report_time': datetime.now().isoformat(),
            'period_start': start_time.isoformat() if start_time else 'N/A',
            'period_end': end_time.isoformat() if end_time else 'N/A',
            'total_vehicles_detected': total_vehicles,
            'total_violations': total_violations,
            'safety_compliance_rate': f"{safety_rate:.2f}%",
            'helmet_compliance_rate': f"{helmet_compliance:.2f}%",
            'speed_compliance_rate': f"{speed_compliance:.2f}%",
            'average_speed_kmh': f"{avg_speed:.2f}",
            'cars': vehicle_types.get('car', 0),
            'motorcycles': vehicle_types.get('motorcycle', 0),
            'buses': vehicle_types.get('bus', 0),
            'trucks': vehicle_types.get('truck', 0),
            'speeding_violations': speeding_count,
            'helmet_violations': helmet_violations
        }
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(report.keys()))
            writer.writeheader()
            writer.writerow(report)
        
        logger.info(f"ESG report generated: {filepath}")
        
        return str(filepath)
    
    def _in_time_range(self, timestamp_str: str, 
                       start: Optional[datetime], 
                       end: Optional[datetime]) -> bool:
        """Check if timestamp is in time range"""
        try:
            ts = datetime.fromisoformat(timestamp_str)
            if start and ts < start:
                return False
            if end and ts > end:
                return False
            return True
        except:
            return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'total_vehicles_logged': len(self.vehicle_log),
            'total_violations_logged': len(self.violation_log),
            'hours_tracked': len(self.hourly_stats),
            'latest_hour': max(self.hourly_stats.keys()) if self.hourly_stats else None
        }
    
    def clear_logs(self):
        """Clear all in-memory logs"""
        self.vehicle_log.clear()
        self.violation_log.clear()
        self.speed_log.clear()
        self.hourly_stats.clear()
        logger.info("All logs cleared")
    
    def save_state(self, filename: str = "reporter_state.json"):
        """Save current state to JSON file"""
        filepath = self.output_dir / filename
        
        state = {
            'vehicle_log': self.vehicle_log,
            'violation_log': self.violation_log,
            'hourly_stats': dict(self.hourly_stats)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filename: str = "reporter_state.json"):
        """Load state from JSON file"""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            logger.warning(f"State file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.vehicle_log = state.get('vehicle_log', [])
        self.violation_log = state.get('violation_log', [])
        self.hourly_stats = defaultdict(lambda: {
            'total': 0, 'cars': 0, 'motorcycles': 0, 'buses': 0, 'trucks': 0,
            'violations': 0, 'speeding': 0, 'no_helmet': 0,
            'avg_speed': 0.0, 'speed_sum': 0.0, 'speed_count': 0
        }, state.get('hourly_stats', {}))
        
        logger.info(f"State loaded from {filepath}")


# Test standalone
if __name__ == "__main__":
    print("Testing CSV Reporter...")
    
    reporter = CSVReporter()
    
    # Add some test data
    for i in range(10):
        result = {
            'track_id': i,
            'class_name': ['car', 'motorcycle', 'bus', 'truck'][i % 4],
            'class_id': [2, 3, 5, 7][i % 4],
            'speed_kmh': 30 + i * 5,
            'direction': ['UP', 'DOWN', 'LEFT', 'RIGHT'][i % 4],
            'helmet_status': ['HELMET', 'NO_HELMET', 'N/A', 'N/A'][i % 4],
            'is_violation': i % 3 == 0,
            'is_speeding': i % 5 == 0,
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9
        }
        
        reporter.log_vehicle(result)
        if result['is_violation']:
            reporter.log_violation(result)
    
    # Generate reports
    print(f"Vehicle report: {reporter.generate_vehicle_report()}")
    print(f"Violation report: {reporter.generate_violation_report()}")
    print(f"Summary report: {reporter.generate_summary_report()}")
    print(f"ESG report: {reporter.generate_esg_report()}")
    
    print(f"Statistics: {reporter.get_statistics()}")
