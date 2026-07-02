"""
Cluster Analysis Visualization Script
====================================

Generates comprehensive plots for each of the 18 user clusters,
showing reading patterns, category preferences, and key statistics.

Author: AI Assistant
Date: 2025-01-27
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ClusterVisualizer:
    """Generate comprehensive visualizations for user clusters"""
    
    def __init__(self):
        self.output_dir = "res/cluster_plots"
        self.data_dir = "dataset/processed"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cluster metadata
        self.cluster_data = {
            0: {"name": "Music Enthusiasts", "icon": "🎵", "primary_category": "music", "primary_pct": 27.9, "users": 1513, "pct": 3.2, "weight": 14.007, "session_length": 7.48, "session_freq": 2.12, "morning": 29.3, "daytime": 49.4, "evening": 21.3},
            1: {"name": "News Junkies", "icon": "📰", "primary_category": "news", "primary_pct": 68.9, "users": 4911, "pct": 10.4, "weight": 5.175, "session_length": 16.26, "session_freq": 2.83, "morning": 23.7, "daytime": 59.0, "evening": 17.3},
            2: {"name": "Sports Fanatics", "icon": "⚽", "primary_category": "sports", "primary_pct": 57.3, "users": 3882, "pct": 8.2, "weight": 5.924, "session_length": 15.24, "session_freq": 2.71, "morning": 31.1, "daytime": 51.0, "evening": 17.9},
            3: {"name": "Finance Enthusiasts", "icon": "💰", "primary_category": "finance", "primary_pct": 38.8, "users": 1924, "pct": 4.1, "weight": 9.282, "session_length": 11.86, "session_freq": 2.48, "morning": 28.4, "daytime": 54.1, "evening": 17.5},
            4: {"name": "Lifestyle Enthusiasts", "icon": "🏠", "primary_category": "lifestyle", "primary_pct": 49.8, "users": 1915, "pct": 4.1, "weight": 9.856, "session_length": 11.30, "session_freq": 2.54, "morning": 30.4, "daytime": 51.8, "evening": 17.8},
            5: {"name": "Health Enthusiasts", "icon": "🏥", "primary_category": "health", "primary_pct": 38.0, "users": 1714, "pct": 3.6, "weight": 8.930, "session_length": 12.15, "session_freq": 2.39, "morning": 30.2, "daytime": 51.2, "evening": 18.6},
            6: {"name": "Auto Enthusiasts", "icon": "🚗", "primary_category": "autos", "primary_pct": 38.2, "users": 1623, "pct": 3.4, "weight": 8.171, "session_length": 12.74, "session_freq": 2.53, "morning": 32.0, "daytime": 47.6, "evening": 20.4},
            7: {"name": "Heavy Readers", "icon": "📚", "primary_category": "news", "primary_pct": 32.1, "users": 2838, "pct": 6.0, "weight": 73.551, "session_length": 84.20, "session_freq": 10.66, "morning": 32.9, "daytime": 45.8, "evening": 21.3},
            8: {"name": "Evening News Readers", "icon": "🌃", "primary_category": "news", "primary_pct": 33.5, "users": 4775, "pct": 10.1, "weight": 4.933, "session_length": 17.43, "session_freq": 2.45, "morning": 4.3, "daytime": 6.5, "evening": 89.2},
            9: {"name": "TV & Entertainment Fans", "icon": "📺", "primary_category": "tv", "primary_pct": 16.0, "users": 7423, "pct": 15.7, "weight": 3.066, "session_length": 20.10, "session_freq": 2.82, "morning": 11.7, "daytime": 83.1, "evening": 5.2},
            10: {"name": "Travel Enthusiasts", "icon": "✈️", "primary_category": "travel", "primary_pct": 27.3, "users": 1712, "pct": 3.6, "weight": 10.075, "session_length": 11.19, "session_freq": 2.34, "morning": 28.7, "daytime": 50.5, "evening": 20.8},
            11: {"name": "Video Enthusiasts", "icon": "📹", "primary_category": "video", "primary_pct": 27.4, "users": 1362, "pct": 2.9, "weight": 10.756, "session_length": 10.45, "session_freq": 2.31, "morning": 31.2, "daytime": 43.2, "evening": 25.6},
            12: {"name": "Super Heavy Users", "icon": "🔥", "primary_category": "news", "primary_pct": 28.7, "users": 5, "pct": 0.0, "weight": 76.903, "session_length": 87.60, "session_freq": 10.00, "morning": 5.4, "daytime": 54.3, "evening": 40.3},
            13: {"name": "Morning Readers", "icon": "🌅", "primary_category": "news", "primary_pct": 30.4, "users": 5805, "pct": 12.3, "weight": 3.393, "session_length": 19.03, "session_freq": 2.27, "morning": 88.8, "daytime": 8.3, "evening": 2.9},
            14: {"name": "Entertainment Enthusiasts", "icon": "🎭", "primary_category": "entertainment", "primary_pct": 33.0, "users": 1655, "pct": 3.5, "weight": 11.050, "session_length": 10.20, "session_freq": 2.30, "morning": 28.6, "daytime": 50.5, "evening": 20.9},
            15: {"name": "Movie Enthusiasts", "icon": "🎬", "primary_category": "movies", "primary_pct": 32.6, "users": 1448, "pct": 3.1, "weight": 13.892, "session_length": 7.51, "session_freq": 2.14, "morning": 29.0, "daytime": 51.9, "evening": 19.1},
            16: {"name": "Weather Enthusiasts", "icon": "🌤️", "primary_category": "weather", "primary_pct": 25.0, "users": 1121, "pct": 2.4, "weight": 13.296, "session_length": 8.08, "session_freq": 2.21, "morning": 27.3, "daytime": 47.2, "evening": 25.5},
            17: {"name": "Food & Drink Enthusiasts", "icon": "🍽️", "primary_category": "foodanddrink", "primary_pct": 38.6, "users": 1622, "pct": 3.4, "weight": 5.176, "session_length": 15.68, "session_freq": 2.65, "morning": 29.4, "daytime": 52.1, "evening": 18.5}
        }
        
        # Category preferences for each cluster (top 8 categories)
        self.category_preferences = {
            0: {"music": 27.9, "news": 19.0, "tv": 11.7, "sports": 9.6, "lifestyle": 9.0, "finance": 3.7, "entertainment": 5.9, "health": 3.5},
            1: {"news": 68.9, "sports": 7.3, "finance": 4.5, "lifestyle": 4.4, "tv": 3.9, "health": 1.6, "entertainment": 1.1, "movies": 1.5},
            2: {"sports": 57.3, "news": 18.4, "tv": 4.5, "lifestyle": 4.0, "finance": 3.6, "health": 1.9, "entertainment": 1.2, "movies": 1.9},
            3: {"finance": 38.8, "news": 25.2, "sports": 9.1, "lifestyle": 7.4, "tv": 4.3, "health": 2.1, "entertainment": 1.2, "movies": 1.8},
            4: {"lifestyle": 49.8, "news": 17.3, "tv": 7.7, "sports": 4.6, "finance": 3.6, "health": 3.6, "entertainment": 2.2, "movies": 2.2},
            5: {"health": 38.0, "news": 18.3, "lifestyle": 11.6, "tv": 6.4, "sports": 5.7, "finance": 5.6, "entertainment": 2.1, "movies": 1.8},
            6: {"autos": 38.2, "news": 20.2, "sports": 9.6, "finance": 6.8, "lifestyle": 6.6, "tv": 4.1, "health": 3.2, "entertainment": 1.4},
            7: {"news": 32.1, "sports": 14.0, "lifestyle": 10.5, "finance": 7.8, "tv": 6.9, "health": 5.9, "foodanddrink": 5.9, "entertainment": 3.5},
            8: {"news": 33.5, "sports": 14.6, "lifestyle": 10.1, "finance": 8.1, "tv": 8.0, "health": 5.1, "foodanddrink": 4.2, "entertainment": 3.2},
            9: {"news": 24.4, "tv": 16.0, "lifestyle": 11.9, "sports": 11.2, "finance": 6.3, "health": 7.3, "foodanddrink": 7.2, "entertainment": 5.5},
            10: {"travel": 27.3, "news": 22.8, "lifestyle": 9.8, "sports": 8.7, "finance": 6.9, "tv": 5.4, "health": 4.6, "entertainment": 2.1},
            11: {"video": 27.4, "news": 26.0, "lifestyle": 10.1, "sports": 8.4, "tv": 6.0, "finance": 4.4, "health": 2.9, "entertainment": 2.6},
            12: {"news": 28.7, "foodanddrink": 12.7, "lifestyle": 11.9, "finance": 7.9, "health": 6.8, "travel": 4.7, "video": 4.2, "sports": 4.4},
            13: {"news": 30.4, "sports": 13.8, "tv": 11.2, "lifestyle": 11.1, "finance": 6.9, "health": 5.9, "foodanddrink": 5.2, "entertainment": 3.7},
            14: {"entertainment": 33.0, "news": 16.0, "tv": 11.3, "lifestyle": 10.4, "sports": 7.2, "finance": 3.5, "health": 3.7, "foodanddrink": 3.3},
            15: {"movies": 32.6, "news": 20.6, "tv": 11.3, "sports": 10.7, "lifestyle": 7.7, "finance": 4.3, "health": 2.7, "entertainment": 2.1},
            16: {"news": 26.2, "weather": 25.0, "sports": 9.9, "lifestyle": 8.0, "finance": 6.1, "tv": 5.7, "health": 3.3, "foodanddrink": 3.1},
            17: {"foodanddrink": 38.6, "news": 17.2, "lifestyle": 10.9, "sports": 7.1, "tv": 6.2, "finance": 5.6, "health": 4.8, "entertainment": 1.9}
        }
        
        # Overall averages for comparison
        self.overall_avg = {
            "news": 29.2, "sports": 14.3, "lifestyle": 10.7, "tv": 8.6, "finance": 7.1,
            "health": 4.8, "foodanddrink": 4.8, "autos": 3.7, "entertainment": 3.7,
            "movies": 4.1, "travel": 3.0, "music": 2.8, "video": 2.0, "weather": 1.3
        }
    
    def create_cluster_plot(self, cluster_id):
        """Create comprehensive plot for a specific cluster"""
        data = self.cluster_data[cluster_id]
        prefs = self.category_preferences[cluster_id]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.5, 1], width_ratios=[1, 1, 1])
        
        # Color scheme
        primary_color = plt.cm.Set3(cluster_id % 12)
        accent_color = plt.cm.Pastel1(cluster_id % 9)
        
        # Title
        fig.suptitle(f'{data["icon"]} CLUSTER {cluster_id}: {data["name"].upper()}', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Cluster Overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        overview_text = f"""
CLUSTER OVERVIEW
Users: {data['users']:,} ({data['pct']:.1f}%)
Cluster Weight: {data['weight']:.1f}
Primary Focus: {data['primary_category'].title()} ({data['primary_pct']:.1f}%)

SESSION BEHAVIOR
Avg Length: {data['session_length']:.1f} articles
Session Freq: {data['session_freq']:.1f} sessions
        """
        ax1.text(0.05, 0.95, overview_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=accent_color, alpha=0.7))
        
        # 2. Reading Time Distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        times = ['Morning\n(4-9AM)', 'Daytime\n(9AM-4PM)', 'Evening\n(4PM-4AM)']
        values = [data['morning'], data['daytime'], data['evening']]
        colors = ['#FFB6C1', '#87CEEB', '#DDA0DD']
        
        bars = ax2.bar(times, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Reading Time Distribution', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Highlight dominant time
        max_idx = values.index(max(values))
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(3)
        
        # 3. Cluster Size Comparison (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Show this cluster vs others
        all_sizes = [self.cluster_data[i]['users'] for i in range(18)]
        all_names = [f"C{i}" for i in range(18)]
        
        colors_comp = ['red' if i == cluster_id else 'lightgray' for i in range(18)]
        bars = ax3.bar(range(18), all_sizes, color=colors_comp, alpha=0.7)
        bars[cluster_id].set_color(primary_color)
        bars[cluster_id].set_alpha(1.0)
        
        ax3.set_title('Cluster Size Comparison', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Number of Users')
        ax3.set_xlabel('Cluster ID')
        ax3.set_xticks(range(0, 18, 2))
        
        # Highlight this cluster
        ax3.annotate(f'This Cluster\n{data["users"]:,} users', 
                    xy=(cluster_id, data['users']), xytext=(cluster_id, data['users'] + 1000),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    ha='center', fontweight='bold', color='red')
        
        # 4. Category Preferences (middle, spans all columns)
        ax4 = fig.add_subplot(gs[1, :])
        
        categories = list(prefs.keys())
        cluster_values = list(prefs.values())
        overall_values = [self.overall_avg.get(cat, 0) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, cluster_values, width, label=f'Cluster {cluster_id}', 
                       color=primary_color, alpha=0.8)
        bars2 = ax4.bar(x + width/2, overall_values, width, label='Overall Average', 
                       color='gray', alpha=0.6)
        
        ax4.set_title('Category Preferences vs Overall Average', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Percentage (%)')
        ax4.set_xlabel('Categories')
        ax4.set_xticks(x)
        ax4.set_xticklabels([cat.title() for cat in categories], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, cluster_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Highlight primary category
        primary_idx = categories.index(data['primary_category'])
        bars1[primary_idx].set_edgecolor('red')
        bars1[primary_idx].set_linewidth(3)
        
        # 5. Key Statistics (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        # Calculate some derived stats
        engagement_score = (data['session_length'] * data['session_freq']) / 10
        specialization_score = data['primary_pct'] / 10
        
        stats_text = f"""
KEY STATISTICS
Engagement Score: {engagement_score:.1f}/10
Specialization Score: {specialization_score:.1f}/10
Cluster Weight: {data['weight']:.1f}

READING BEHAVIOR
Most Active Time: {times[values.index(max(values))].split('\\n')[0]}
Primary Interest: {data['primary_category'].title()}
        """
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=primary_color, alpha=0.3))
        
        # 6. Comparison Metrics (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Compare key metrics to overall average
        overall_session_length = 19.35
        overall_session_freq = 3.01
        overall_weight = 20.0  # approximate average
        
        metrics = ['Session\nLength', 'Session\nFrequency', 'Cluster\nWeight']
        cluster_vals = [data['session_length'], data['session_freq'], data['weight']]
        overall_vals = [overall_session_length, overall_session_freq, overall_weight]
        
        # Normalize for comparison
        normalized_cluster = []
        normalized_overall = []
        for i, (c_val, o_val) in enumerate(zip(cluster_vals, overall_vals)):
            if i == 2:  # Weight - use log scale
                normalized_cluster.append(np.log10(c_val + 1))
                normalized_overall.append(np.log10(o_val + 1))
            else:
                normalized_cluster.append(c_val)
                normalized_overall.append(o_val)
        
        x = np.arange(2)  # Only first 2 metrics
        bars1 = ax6.bar(x - 0.2, normalized_cluster[:2], 0.4, label=f'Cluster {cluster_id}', 
                       color=primary_color, alpha=0.8)
        bars2 = ax6.bar(x + 0.2, normalized_overall[:2], 0.4, label='Average', 
                       color='gray', alpha=0.6)
        
        # Special handling for weight (separate subplot would be better, but let's simplify)
        # Add weight as text annotation instead
        weight_text = f"Cluster Weight: {data['weight']:.1f}\n(Avg: {overall_weight:.1f})"
        
        ax6.set_title('Performance Metrics', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Value')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics[:2])
        ax6.legend(loc='upper left')
        
        # Add weight as text
        ax6.text(0.7, 0.8, weight_text, transform=ax6.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # 7. Distinctive Features (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Calculate distinctive features
        distinctive_cats = []
        for cat, pct in prefs.items():
            overall_pct = self.overall_avg.get(cat, 0)
            if pct > overall_pct + 5:  # 5% threshold
                distinctive_cats.append(f"↑ {cat.title()}: +{pct - overall_pct:.1f}%")
        
        if len(distinctive_cats) > 5:
            distinctive_cats = distinctive_cats[:5]
        
        # Add behavioral insights
        if data['session_length'] > 50:
            behavior = "Heavy Reader"
        elif data['session_length'] < 10:
            behavior = "Light Reader"
        else:
            behavior = "Moderate Reader"
        
        if max(values) > 70:
            time_pattern = "Strong Time Preference"
        else:
            time_pattern = "Flexible Timing"
        
        features_text = f"""
DISTINCTIVE FEATURES
{chr(10).join(distinctive_cats[:4])}

BEHAVIORAL PROFILE
Reading Style: {behavior}
Time Pattern: {time_pattern}
Specialization: {data['primary_pct']:.1f}% focused
        """
        ax7.text(0.05, 0.95, features_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=accent_color, alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{self.output_dir}/cluster_{cluster_id:02d}_{data['name'].lower().replace(' ', '_').replace('&', 'and')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Generated plot for Cluster {cluster_id}: {data['name']} -> {filename}")
    
    def generate_all_plots(self):
        """Generate plots for all 18 clusters"""
        print("🎨 Generating cluster visualization plots...")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 60)
        
        for cluster_id in range(18):
            self.create_cluster_plot(cluster_id)
        
        print("=" * 60)
        print(f"🎉 Successfully generated 18 cluster plots in {self.output_dir}/")
        print("\nPlots include:")
        print("  • Cluster overview and statistics")
        print("  • Reading time distribution")
        print("  • Category preferences vs overall average")
        print("  • Performance metrics comparison")
        print("  • Distinctive features analysis")
        print("  • Behavioral profiling")
    
    def create_summary_plot(self):
        """Create a summary plot showing all clusters"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cluster sizes
        cluster_ids = list(range(18))
        sizes = [self.cluster_data[i]['users'] for i in cluster_ids]
        names = [f"C{i}: {self.cluster_data[i]['name'][:15]}" for i in cluster_ids]
        
        colors = plt.cm.Set3(np.linspace(0, 1, 18))
        bars = ax1.bar(cluster_ids, sizes, color=colors, alpha=0.8)
        ax1.set_title('Cluster Sizes Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Users')
        ax1.set_xticks(cluster_ids)
        
        # 2. Cluster weights
        weights = [self.cluster_data[i]['weight'] for i in cluster_ids]
        bars = ax2.bar(cluster_ids, weights, color=colors, alpha=0.8)
        ax2.set_title('Cluster Weights Distribution', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Cluster Weight')
        ax2.set_xticks(cluster_ids)
        ax2.set_yscale('log')
        
        # 3. Session behavior scatter
        session_lengths = [self.cluster_data[i]['session_length'] for i in cluster_ids]
        session_freqs = [self.cluster_data[i]['session_freq'] for i in cluster_ids]
        
        scatter = ax3.scatter(session_lengths, session_freqs, c=cluster_ids, 
                             cmap='Set3', s=sizes, alpha=0.7, edgecolors='black')
        ax3.set_title('Session Behavior by Cluster', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Average Session Length')
        ax3.set_ylabel('Session Frequency')
        
        # Add cluster labels
        for i, (x, y) in enumerate(zip(session_lengths, session_freqs)):
            ax3.annotate(f'C{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Primary category distribution
        primary_cats = [self.cluster_data[i]['primary_category'] for i in cluster_ids]
        primary_pcts = [self.cluster_data[i]['primary_pct'] for i in cluster_ids]
        
        # Group by category
        cat_counts = Counter(primary_cats)
        unique_cats = list(cat_counts.keys())
        cat_colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_cats)))
        
        wedges, texts, autotexts = ax4.pie(cat_counts.values(), labels=unique_cats, 
                                          colors=cat_colors, autopct='%1.0f%%', startangle=90)
        ax4.set_title('Primary Category Distribution', fontweight='bold', fontsize=14)
        
        plt.suptitle('🎯 CLUSTER ANALYSIS SUMMARY - 18 USER CLUSTERS', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save summary plot
        summary_filename = f"{self.output_dir}/00_cluster_summary.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Generated summary plot -> {summary_filename}")


def main():
    """Main execution function"""
    visualizer = ClusterVisualizer()
    
    # Generate summary plot first
    visualizer.create_summary_plot()
    
    # Generate individual cluster plots
    visualizer.generate_all_plots()
    
    print("\n🎨 All cluster visualizations completed!")
    print(f"📁 Check the plots in: {visualizer.output_dir}/")


if __name__ == "__main__":
    main()
