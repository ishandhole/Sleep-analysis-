
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTIONS - Helper functions used throughout the program
# =============================================================================

def format_time(minutes: float) -> str:
    """Convert minutes to readable format (e.g., 480 -> '8h 0m')."""
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours == 0:
        return f"{mins}m"
    elif mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"

def parse_time(time_str: str) -> float:
    """Parse time string (24h or 12h format) to decimal hours."""
    time_str = time_str.upper().strip()
    is_pm = 'PM' in time_str
    is_am = 'AM' in time_str
    
    # Remove AM/PM and extract numbers
    time_part = time_str.replace('PM', '').replace('AM', '').strip()
    hour, minute = map(int, time_part.split(':'))
    
    # Convert to 24-hour format
    if is_pm and hour != 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0
    
    return hour + minute/60

def calculate_sleep_cycles(hours: float) -> Dict[str, Any]:
    """Calculate sleep cycles (each cycle is 90 minutes = 1.5 hours)."""
    cycle_length = 1.5
    total_cycles = hours / cycle_length
    complete_cycles = int(total_cycles)
    partial_minutes = (total_cycles - complete_cycles) * 90
    
    return {
        'total_cycles': round(total_cycles, 1),
        'complete_cycles': complete_cycles,
        'partial_cycle_minutes': round(partial_minutes),
        'recommended_durations': [6.0, 7.5, 9.0]  # 4, 5, 6 cycles
    }

def get_max_efficiency_by_duration(hours: float) -> float:
    """
    Calculate maximum efficiency based on sleep duration.
    
    Rules:
    - 6-9 hours: max efficiency 90%
    - 3-6 hours: max efficiency 70%
    - Below 3 hours: max efficiency 50%
    """
    if 6 <= hours <= 9:
        return 90.0
    elif 3 <= hours < 6:
        return 70.0
    elif hours < 3:
        return 50.0
    else:
        # For hours > 9, allow higher efficiency but cap at 95%
        return 95.0

# =============================================================================
# DATA COLLECTION - Get user input and generate sample sleep data
# =============================================================================

def get_user_sleep_input() -> Dict[str, Any]:
    """Ask user for their sleep patterns and return as dictionary."""
    print("\n🛏️  Sleep Pattern Input")
    print("=" * 40)
    
    try:
        # Get bedtime and wake time
        bedtime_str = input("Bedtime (e.g., 22:30 or 10:30 PM): ").strip()
        waketime_str = input("Wake time (e.g., 07:00 or 7:00 AM): ").strip()
        
        bedtime_hour = parse_time(bedtime_str)
        wake_hour = parse_time(waketime_str)
        
        # Calculate sleep duration
        if wake_hour < bedtime_hour:  # Next day
            sleep_duration = (24 - bedtime_hour) + wake_hour
        else:
            sleep_duration = wake_hour - bedtime_hour
        
        print(f"\n📊 Calculated: {sleep_duration:.1f} hours")
        confirm = input("Correct? (y/n) or enter custom hours: ").strip().lower()
        
        if confirm in ['n', 'no']:
            sleep_duration = float(input("Enter sleep duration (hours): "))
        elif confirm not in ['y', 'yes', '']:
            sleep_duration = float(confirm)
        
        # Get additional details
        interruptions = int(input("Night awakenings (e.g., 2): ") or "1")
        latency = float(input("Minutes to fall asleep (e.g., 15): ") or "15")
        
        # Calculate efficiency (estimate based on interruptions and latency)
        efficiency = max(70, 95 - (interruptions * 5) - max(0, latency - 20) * 0.5)
        
        # Apply maximum efficiency constraint based on sleep duration
        max_efficiency = get_max_efficiency_by_duration(sleep_duration)
        efficiency = min(efficiency, max_efficiency)
        
        # Calculate cycles
        cycles = calculate_sleep_cycles(sleep_duration)
        
        # Display summary
        print(f"\n✅ Summary:")
        print(f"   Duration: {sleep_duration:.1f}h | Efficiency: {efficiency:.1f}%")
        print(f"   Cycles: {cycles['total_cycles']} | Latency: {latency}min")
        
        return {
            'bedtime_hour': bedtime_hour,
            'wake_hour': wake_hour,
            'sleep_duration': sleep_duration,
            'sleep_efficiency': efficiency,
            'sleep_interruptions': interruptions,
            'sleep_latency': latency,
            'cycle_info': cycles
        }
        
    except Exception as e:
        print(f"❌ Error: {e}. Using defaults (8.5h, 85% efficiency)")
        default_duration = 8.5
        default_efficiency = 85.0
        # Apply maximum efficiency constraint
        max_efficiency = get_max_efficiency_by_duration(default_duration)
        default_efficiency = min(default_efficiency, max_efficiency)
        
        return {
            'bedtime_hour': 22.5,
            'wake_hour': 7.0,
            'sleep_duration': default_duration,
            'sleep_efficiency': default_efficiency,
            'sleep_interruptions': 1,
            'sleep_latency': 15.0,
            'cycle_info': calculate_sleep_cycles(default_duration)
        }

class SleepDataCollector:
    """Generate sample sleep data based on user's typical patterns."""
    
    def generate_sample_data(self, days: int = 30, user_input: Optional[Dict] = None) -> pd.DataFrame:
        """Generate realistic sleep data with natural variation."""
        np.random.seed(42)  # For reproducible results
        
        # Use user input or defaults
        if user_input:
            base_duration = user_input['sleep_duration']
            base_efficiency = user_input['sleep_efficiency']
            base_interruptions = user_input['sleep_interruptions']
            base_latency = user_input['sleep_latency']
            base_bedtime = user_input['bedtime_hour']
        else:
            base_duration, base_efficiency = 8.0, 85.0
            base_interruptions, base_latency = 1, 15.0
            base_bedtime = 22.5
        
        # Generate data for each day
        start_date = datetime.now() - timedelta(days=days)
        data = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            
            # Add realistic variation (±10-15%)
            duration = np.random.normal(base_duration, 0.3)
            duration = max(4, min(12, duration))  # Clamp to 4-12 hours
            
            # Calculate efficiency with variation
            efficiency = np.random.normal(base_efficiency, 3)
            efficiency = max(60, efficiency)  # Minimum 60%
            
            # Apply maximum efficiency constraint based on sleep duration
            max_efficiency = get_max_efficiency_by_duration(duration)
            efficiency = min(efficiency, max_efficiency)
            
            latency = max(1, np.random.normal(base_latency, 5))
            awakenings = max(0, int(np.random.normal(base_interruptions, 0.5)))
            
            # Calculate sleep stages (REM, deep, light)
            total_minutes = duration * 60
            rem_percent = np.random.normal(22, 5) / 100
            deep_percent = np.random.normal(15, 5) / 100
            light_percent = 1 - rem_percent - deep_percent
            
            # Calculate times
            bedtime = date.replace(hour=int(base_bedtime), minute=int((base_bedtime % 1) * 60))
            wake_time = bedtime + timedelta(hours=duration)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'bedtime': bedtime.strftime('%Y-%m-%d %H:%M:%S'),
                'wake_time': wake_time.strftime('%Y-%m-%d %H:%M:%S'),
                'sleep_duration_hours': round(duration, 2),
                'sleep_duration_minutes': round(duration * 60),
                'sleep_efficiency': round(efficiency, 1),
                'sleep_latency_minutes': round(latency, 1),
                'awakenings_count': awakenings,
                'rem_duration_minutes': round(total_minutes * rem_percent),
                'deep_sleep_duration_minutes': round(total_minutes * deep_percent),
                'light_sleep_duration_minutes': round(total_minutes * light_percent),
                'heart_rate_avg': round(np.random.normal(60, 8)),
                'heart_rate_min': round(np.random.normal(45, 5)),
                'heart_rate_max': round(np.random.normal(75, 10)),
                'temperature_avg': round(np.random.normal(98.6, 0.8), 1),
                'movement_score': round(np.random.normal(15, 5)),
                'stress_level': round(np.random.uniform(1, 10), 1)
            })
        
        return pd.DataFrame(data)

# =============================================================================
# DATA PREPROCESSING - Clean and prepare data for analysis
# =============================================================================

class SleepDataPreprocessor:
    """Clean sleep data and create useful features."""
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all preprocessing steps in order."""
        df = df.copy()
        
        # Step 1: Fix data types
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        for col in ['bedtime', 'wake_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Step 2: Fill missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Step 3: Create new features
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.day_name()
            df['is_weekend'] = df['date'].dt.weekday >= 5
            df['month'] = df['date'].dt.month
        
        if 'bedtime' in df.columns:
            df['bedtime_hour'] = df['bedtime'].dt.hour + df['bedtime'].dt.minute/60
        
        # Step 4: Calculate sleep quality score (0-100)
        df['sleep_quality_score'] = self._calculate_quality_score(df)
        
        # Step 5: Calculate sleep debt (difference from 8 hours target)
        if 'sleep_duration_hours' in df.columns:
            df['sleep_debt'] = 8.0 - df['sleep_duration_hours']
            df['cumulative_sleep_debt'] = df['sleep_debt'].cumsum()
        
        return df
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite sleep quality score (0-100)."""
        score = pd.Series(index=df.index, data=50.0)  # Start at 50
        
        # Duration component (best: 7-9 hours)
        if 'sleep_duration_hours' in df.columns:
            duration_score = df['sleep_duration_hours'].apply(
                lambda x: 100 if 7 <= x <= 9 else max(0, 100 - abs(x - 8) * 10)
            )
            score = 0.4 * duration_score + 0.6 * score
        
        # Efficiency component (best: 85%+)
        if 'sleep_efficiency' in df.columns:
            efficiency_score = df['sleep_efficiency'].clip(0, 100)
            score = 0.4 * efficiency_score + 0.6 * score
        
        # Latency component (best: <20 minutes)
        if 'sleep_latency_minutes' in df.columns:
            latency_score = df['sleep_latency_minutes'].apply(
                lambda x: max(0, 100 - x * 2) if x >= 0 else 50
            )
            score = 0.1 * latency_score + 0.9 * score
        
        return score.clip(0, 100)

# =============================================================================
# METRICS CALCULATION - Calculate sleep statistics and averages
# =============================================================================

class SleepMetrics:
    """Calculate all sleep-related metrics and statistics."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.metrics = {}
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all available metrics."""
        metrics = {}
        
        # Duration metrics
        if 'sleep_duration_hours' in self.data.columns:
            duration = self.data['sleep_duration_hours'].dropna()
            metrics.update({
                'avg_sleep_duration_hours': duration.mean(),
                'median_sleep_duration_hours': duration.median(),
                'min_sleep_duration_hours': duration.min(),
                'max_sleep_duration_hours': duration.max(),
                'nights_adequate_sleep': ((duration >= 7) & (duration <= 9)).sum(),
                'percentage_adequate_sleep': ((duration >= 7) & (duration <= 9)).mean() * 100
            })
        
        # Efficiency metrics
        if 'sleep_efficiency' in self.data.columns:
            efficiency = self.data['sleep_efficiency'].dropna()
            metrics.update({
                'avg_sleep_efficiency': efficiency.mean(),
                'median_sleep_efficiency': efficiency.median(),
                'nights_good_efficiency': (efficiency >= 85).sum(),
                'percentage_good_efficiency': (efficiency >= 85).mean() * 100
            })
        
        # Latency metrics
        if 'sleep_latency_minutes' in self.data.columns:
            latency = self.data['sleep_latency_minutes'].dropna()
            metrics.update({
                'avg_sleep_latency_minutes': latency.mean(),
                'median_sleep_latency_minutes': latency.median(),
                'nights_normal_onset': (latency <= 30).sum()
            })
        
        # Awakening metrics
        if 'awakenings_count' in self.data.columns:
            awakenings = self.data['awakenings_count'].dropna()
            metrics.update({
                'avg_awakenings_count': awakenings.mean(),
                'nights_restful': (awakenings <= 1).sum(),
                'percentage_restful_nights': (awakenings <= 1).mean() * 100
            })
        
        # Sleep stage metrics
        for stage in ['rem', 'deep', 'light']:
            col = f'{stage}_sleep_duration_minutes'
            if col in self.data.columns:
                stage_data = self.data[col].dropna()
                metrics[f'avg_{stage}_duration_minutes'] = stage_data.mean()
                metrics[f'avg_{stage}_duration_formatted'] = format_time(stage_data.mean())
        
        # Timing metrics
        if 'bedtime' in self.data.columns:
            bedtimes = pd.to_datetime(self.data['bedtime'])
            bedtime_minutes = bedtimes.dt.hour * 60 + bedtimes.dt.minute
            avg_bedtime_min = bedtime_minutes.mean()
            metrics['avg_bedtime'] = f"{int(avg_bedtime_min//60):02d}:{int(avg_bedtime_min%60):02d}"
        
        if 'wake_time' in self.data.columns:
            wake_times = pd.to_datetime(self.data['wake_time'])
            wake_minutes = wake_times.dt.hour * 60 + wake_times.dt.minute
            avg_wake_min = wake_minutes.mean()
            metrics['avg_wake_time'] = f"{int(avg_wake_min//60):02d}:{int(avg_wake_min%60):02d}"
        
        # Weekend vs weekday
        if 'is_weekend' in self.data.columns and 'sleep_duration_hours' in self.data.columns:
            weekday_sleep = self.data[~self.data['is_weekend']]['sleep_duration_hours'].mean()
            weekend_sleep = self.data[self.data['is_weekend']]['sleep_duration_hours'].mean()
            metrics.update({
                'avg_weekday_sleep_hours': weekday_sleep,
                'avg_weekend_sleep_hours': weekend_sleep,
                'weekend_sleep_difference_hours': weekend_sleep - weekday_sleep
            })
        
        self.metrics = metrics
        return metrics
    
    def get_summary(self) -> str:
        """Generate a readable summary of key metrics."""
        if not self.metrics:
            self.calculate_all_metrics()
        
        lines = ["=== Sleep Metrics Summary ===\n"]
        
        if 'avg_sleep_duration_hours' in self.metrics:
            lines.append(f"Average Sleep Duration: {self.metrics['avg_sleep_duration_hours']:.1f} hours")
        if 'avg_sleep_efficiency' in self.metrics:
            lines.append(f"Average Sleep Efficiency: {self.metrics['avg_sleep_efficiency']:.1f}%")
        if 'avg_sleep_latency_minutes' in self.metrics:
            lines.append(f"Average Sleep Latency: {self.metrics['avg_sleep_latency_minutes']:.1f} minutes")
        if 'avg_awakenings_count' in self.metrics:
            lines.append(f"Average Awakenings: {self.metrics['avg_awakenings_count']:.1f}")
        if 'avg_bedtime' in self.metrics:
            lines.append(f"Average Bedtime: {self.metrics['avg_bedtime']}")
        if 'avg_wake_time' in self.metrics:
            lines.append(f"Average Wake Time: {self.metrics['avg_wake_time']}")
        
        return "\n".join(lines)

# =============================================================================
# PATTERN ANALYSIS - Find trends and patterns in sleep data
# =============================================================================

class SleepAnalyzer:
    """Analyze sleep patterns, trends, and generate insights."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
    
    def analyze_trends(self, window: int = 7) -> Dict[str, Any]:
        """Analyze if sleep metrics are improving, worsening, or stable."""
        trends = {}
        
        if 'date' not in self.data.columns:
            return trends
        
        metrics = ['sleep_duration_hours', 'sleep_efficiency', 'sleep_latency_minutes', 'awakenings_count']
        
        for metric in metrics:
            if metric in self.data.columns:
                values = self.data[metric].dropna()
                if len(values) > 1:
                    # Calculate trend slope (linear regression)
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    # Determine direction
                    if abs(slope) < 0.01:
                        direction = 'stable'
                    elif slope > 0:
                        # For duration/efficiency: positive is good
                        # For latency/awakenings: positive is bad
                        if metric in ['sleep_latency_minutes', 'awakenings_count']:
                            direction = 'worsening'
                        else:
                            direction = 'improving'
                    else:
                        if metric in ['sleep_latency_minutes', 'awakenings_count']:
                            direction = 'improving'
                        else:
                            direction = 'worsening'
                    
                    trends[metric] = {
                        'slope': slope,
                        'direction': direction,
                        'change': values.iloc[-1] - values.iloc[0]
                    }
        
        return trends
    
    def analyze_weekly_patterns(self) -> Dict[str, Any]:
        """Compare weekday vs weekend sleep patterns."""
        patterns = {}
        
        if 'date' not in self.data.columns:
            return patterns
        
        self.data['is_weekend'] = self.data['date'].dt.weekday >= 5
        metrics = ['sleep_duration_hours', 'sleep_efficiency']
        
        for metric in metrics:
            if metric in self.data.columns:
                weekday_avg = self.data[~self.data['is_weekend']][metric].mean()
                weekend_avg = self.data[self.data['is_weekend']][metric].mean()
                
                patterns[metric] = {
                    'weekday_average': weekday_avg,
                    'weekend_average': weekend_avg,
                    'difference': weekend_avg - weekday_avg
                }
        
        return patterns
    
    def generate_insights(self) -> List[str]:
        """Generate helpful insights about sleep patterns."""
        insights = []
        
        # Analyze trends
        trends = self.analyze_trends()
        for metric, trend_data in trends.items():
            if trend_data['direction'] == 'worsening':
                metric_name = metric.replace('_', ' ').title()
                insights.append(f"⚠️ {metric_name} is declining over time.")
            elif trend_data['direction'] == 'improving':
                metric_name = metric.replace('_', ' ').title()
                insights.append(f"✅ {metric_name} is improving - keep it up!")
        
        # Analyze weekly patterns
        weekly = self.analyze_weekly_patterns()
        if 'sleep_duration_hours' in weekly:
            diff = weekly['sleep_duration_hours']['difference']
            if diff > 1:
                insights.append(f"📊 You sleep {diff:.1f}h longer on weekends. Try to maintain consistent sleep.")
        
        if not insights:
            insights.append("✅ Your sleep patterns are stable with no major concerns.")
        
        return insights

# =============================================================================
# VISUALIZATION - Create charts and graphs
# =============================================================================

class SleepVisualizer:
    """Create visualizations of sleep data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.sns.set_style("whitegrid")
            self.available = True
        except ImportError:
            self.available = False
    
    def plot_overview_dashboard(self) -> None:
        """Create a dashboard with 4 key charts."""
        if not self.available:
            print("⚠️ Install matplotlib and seaborn: pip install matplotlib seaborn")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = self.plt.subplots(2, 2, figsize=(14, 12))
        
        # Chart 1: Sleep duration over time
        if all(col in self.data.columns for col in ['date', 'sleep_duration_hours']):
            dates = pd.to_datetime(self.data['date'])
            duration = self.data['sleep_duration_hours']
            
            ax1.plot(dates, duration, marker='o', linewidth=1, markersize=3, alpha=0.7, label='Daily')
            rolling_avg = duration.rolling(window=7, min_periods=1).mean()
            ax1.plot(dates, rolling_avg, color='red', linewidth=2, label='7-day average')
            ax1.axhline(y=8, color='green', linestyle='--', alpha=0.7, label='Target (8h)')
            ax1.set_title('Sleep Duration Trend')
            ax1.set_ylabel('Hours')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Chart 2: Sleep efficiency distribution
        if 'sleep_efficiency' in self.data.columns:
            efficiency = self.data['sleep_efficiency'].dropna()
            ax2.hist(efficiency, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            ax2.axvline(efficiency.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {efficiency.mean():.1f}%')
            ax2.set_title('Sleep Efficiency Distribution')
            ax2.set_xlabel('Efficiency (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Chart 3: Sleep latency vs duration
        if all(col in self.data.columns for col in ['sleep_latency_minutes', 'sleep_duration_hours']):
            ax3.scatter(self.data['sleep_latency_minutes'], self.data['sleep_duration_hours'], 
                       alpha=0.6, s=30)
            ax3.set_xlabel('Sleep Latency (minutes)')
            ax3.set_ylabel('Sleep Duration (hours)')
            ax3.set_title('Sleep Latency vs Duration')
            ax3.grid(True, alpha=0.3)
        
        # Chart 4: Average sleep by day of week
        if 'date' in self.data.columns and 'sleep_duration_hours' in self.data.columns:
            self.data['day_of_week'] = pd.to_datetime(self.data['date']).dt.day_name()
            daily_avg = self.data.groupby('day_of_week')['sleep_duration_hours'].mean()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_avg = daily_avg.reindex([day for day in day_order if day in daily_avg.index])
            
            ax4.bar(range(len(daily_avg)), daily_avg.values, color='lightcoral', alpha=0.7)
            ax4.set_xticks(range(len(daily_avg)))
            ax4.set_xticklabels([day[:3] for day in daily_avg.index], rotation=45)
            ax4.set_title('Average Sleep by Day of Week')
            ax4.set_ylabel('Hours')
            ax4.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        self.plt.show()

# =============================================================================
# QUALITY PREDICTION - Predict sleep quality scores
# =============================================================================

class SleepQualityPredictor:
    """Predict sleep quality based on multiple factors."""
    
    def predict_quality(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sleep quality score (0-100) for each night."""
        score = pd.Series(index=data.index, data=50.0)
        
        # Duration component (best: 7-9 hours = 100 points)
        if 'sleep_duration_hours' in data.columns:
            duration_score = data['sleep_duration_hours'].apply(
                lambda x: 100 if 7 <= x <= 9 else max(0, 100 - abs(x - 8) * 15)
            )
            score = 0.3 * duration_score + 0.7 * score
        
        # Efficiency component (best: 85%+ = 100 points)
        if 'sleep_efficiency' in data.columns:
            efficiency_score = data['sleep_efficiency'].clip(0, 100)
            score = 0.3 * efficiency_score + 0.7 * score
        
        # Latency component (best: <20 min = 100 points)
        if 'sleep_latency_minutes' in data.columns:
            latency_score = data['sleep_latency_minutes'].apply(
                lambda x: max(0, 100 - max(0, x - 20) * 3)
            )
            score = 0.2 * latency_score + 0.8 * score
        
        # Awakenings component (best: 0 = 100 points)
        if 'awakenings_count' in data.columns:
            awakenings_score = data['awakenings_count'].apply(
                lambda x: max(0, 100 - x * 20)
            )
            score = 0.2 * awakenings_score + 0.8 * score
        
        return score.clip(0, 100)

# =============================================================================
# MAIN APPLICATION - Orchestrate all components
# =============================================================================

class SleepPatternAnalysis:
    """Main class that runs the complete sleep analysis pipeline."""
    
    def __init__(self):
        self.data = None
        self.collector = SleepDataCollector()
        self.preprocessor = SleepDataPreprocessor()
        self.metrics_calculator = None
        self.analyzer = None
        self.visualizer = None
    
    def load_sample_data(self, days: int = 30) -> None:
        """Step 1: Get user input and generate sample data."""
        print(f"\n🔄 Generating {days} days of sample sleep data...")
        user_input = get_user_sleep_input()
        self.data = self.collector.generate_sample_data(days=days, user_input=user_input)
        print(f"✅ Generated {len(self.data)} records")
    
    def preprocess_data(self) -> None:
        """Step 2: Clean and prepare data."""
        if self.data is None:
            print("❌ No data loaded")
            return
        print("🔄 Preprocessing data...")
        self.data = self.preprocessor.preprocess_data(self.data)
        print("✅ Preprocessing complete")
    
    def calculate_metrics(self) -> None:
        """Step 3: Calculate sleep metrics."""
        if self.data is None:
            print("❌ No data available")
            return
        print("🔄 Calculating metrics...")
        self.metrics_calculator = SleepMetrics(self.data)
        self.metrics_calculator.calculate_all_metrics()
        print("✅ Metrics calculated")
    
    def analyze_patterns(self) -> None:
        """Step 4: Analyze trends and patterns."""
        if self.data is None:
            print("❌ No data available")
            return
        print("🔄 Analyzing patterns...")
        self.analyzer = SleepAnalyzer(self.data)
        insights = self.analyzer.generate_insights()
        print(f"✅ Analysis complete - {len(insights)} insights found")
    
    def predict_quality(self) -> None:
        """Step 5: Predict sleep quality."""
        if self.data is None:
            print("❌ No data available")
            return
        print("🔄 Predicting sleep quality...")
        predictor = SleepQualityPredictor()
        quality_scores = predictor.predict_quality(self.data)
        self.data['predicted_quality_score'] = quality_scores
        print(f"✅ Quality prediction complete - Average: {quality_scores.mean():.1f}/100")
    
    def generate_visualizations(self) -> None:
        """Step 6: Create visualizations."""
        if self.data is None:
            print("❌ No data available")
            return
        print("🔄 Creating visualizations...")
        self.visualizer = SleepVisualizer(self.data)
        if self.visualizer.available:
            self.visualizer.plot_overview_dashboard()
            print("✅ Visualizations displayed")
        else:
            print("⚠️ Install matplotlib and seaborn for visualizations")
    
    def generate_report(self) -> str:
        """Step 7: Generate final report."""
        if self.data is None:
            return "No data available for report."
        
        report = [
            "=" * 60,
            "SLEEP PATTERN ANALYSIS REPORT",
            "=" * 60,
            f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Period: {len(self.data)} nights\n"
        ]
        
        # Basic statistics
        if 'sleep_duration_hours' in self.data.columns:
            avg_duration = self.data['sleep_duration_hours'].mean()
            adequate = ((self.data['sleep_duration_hours'] >= 7) & 
                       (self.data['sleep_duration_hours'] <= 9)).sum()
            report.extend([
                "SLEEP DURATION:",
                f"• Average: {avg_duration:.1f} hours",
                f"• Nights with adequate sleep (7-9h): {adequate}\n"
            ])
        
        if 'sleep_efficiency' in self.data.columns:
            avg_efficiency = self.data['sleep_efficiency'].mean()
            good_efficiency = (self.data['sleep_efficiency'] >= 85).sum()
            report.extend([
                "SLEEP EFFICIENCY:",
                f"• Average: {avg_efficiency:.1f}%",
                f"• Nights with good efficiency (≥85%): {good_efficiency}\n"
            ])
        
        # Quality prediction
        if 'predicted_quality_score' in self.data.columns:
            avg_quality = self.data['predicted_quality_score'].mean()
            report.extend([
                "SLEEP QUALITY:",
                f"• Average predicted quality: {avg_quality:.1f}/100\n"
            ])
        
        # Insights
        if self.analyzer:
            insights = self.analyzer.generate_insights()
            if insights:
                report.extend(["KEY INSIGHTS:"])
                for insight in insights:
                    report.append(f"• {insight}")
                report.append("")
        
        # Recommendations
        report.extend([
            "RECOMMENDATIONS:",
            "• Maintain consistent sleep schedule (same bedtime/wake time)",
            "• Aim for 7-9 hours of sleep nightly",
            "• Keep sleep efficiency above 85%",
            "• Minimize sleep latency (fall asleep within 20 minutes)",
            "• Reduce night awakenings for better sleep quality",
            "\n" + "=" * 60
        ])
        
        return "\n".join(report)
    
    def run_complete_analysis(self) -> None:
        """Run all analysis steps in order."""
        print("\n🚀 SLEEP PATTERN ANALYSIS")
        print("=" * 50)
        
        # Run all steps
        self.load_sample_data(days=30)
        self.preprocess_data()
        self.calculate_metrics()
        self.analyze_patterns()
        self.predict_quality()
        self.generate_visualizations()
        
        # Generate and display report
        print("\n📋 FINAL REPORT")
        print("=" * 50)
        report = self.generate_report()
        print(report)
        
        # Display metrics summary
        if self.metrics_calculator:
            print("\n" + self.metrics_calculator.get_summary())
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 50)

def main():
    """Main function to run the sleep analysis tool."""
    print("""
    ╔══════════════════════════════════════╗
    ║   SLEEP PATTERN ANALYSIS TOOL        ║
    ╚══════════════════════════════════════╝
    """)
    
    # Create analysis object and run
    analysis = SleepPatternAnalysis()
    analysis.run_complete_analysis()
    
if __name__ == "__main__":
    main()
