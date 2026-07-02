"""
Evening News Readers (Cluster 8) - Beautiful Reading Time Distribution Plot
==========================================================================

Creates a stunning visualization for Cluster 8's reading time preferences.

Author: AI Assistant  
Date: 2025-01-27
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")

# Set font preferences with fallback order
font_list = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']
available_fonts = [f.name for f in fm.fontManager.ttflist]

# Find the first available font from our preference list
selected_font = 'DejaVu Sans'  # Default fallback
for font in font_list:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = selected_font

def create_evening_readers_plot():
    """Create a beautiful plot for Evening News Readers (Cluster 8)"""
    
    # Data for Cluster 8: Evening News Readers
    reading_times = ['Morning', 'Daytime', 'Evening']
    percentages = [4.3, 6.5, 89.2]
    
    # Create figure with high DPI for crisp quality - wider format
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Beautiful color scheme - evening theme
    colors = ['#FFB6C1', '#87CEEB', '#2E0854']  # Light pink, sky blue, deep purple
    edge_colors = ['#FF69B4', '#4682B4', '#8B008B']  # Darker edges
    
    # Create bars with gradient effect
    bars = ax.bar(reading_times, percentages, 
                  color=colors, 
                  edgecolor=edge_colors, 
                  linewidth=3,
                  alpha=0.8,
                  capsize=10)
    
    # Remove special highlighting - keep all bars consistent
    
    # Remove glow effect - keeping it simple
    
    # Customize the plot
    ax.set_title('EVENING NEWS READERS (CLUSTER 8)\nReading Time Distribution', 
                fontsize=20, fontweight='bold', pad=30, color='#2E0854')
    
    ax.set_ylabel('Reading Activity (%)', fontsize=16, fontweight='bold', color='#2E0854')
    ax.set_xlabel('Time of Day', fontsize=16, fontweight='bold', color='#2E0854')
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, 100)
    
    # Add percentage labels on top of bars - consistent styling
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{percentage}%', 
               ha='center', va='bottom', 
               fontsize=16, fontweight='bold', 
               color='#2E0854')
    
    # Remove grid for cleaner look
    
    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#2E0854')
    ax.tick_params(axis='x', which='major', labelsize=14, colors='#2E0854', pad=10)
    
    # Remove background color for clean look
    
    # Remove annotation - keeping it clean
    
    # Remove cluster information box for cleaner look
    
    # Add comparison with overall average
    overall_evening = 22.3  # Overall average evening reading
    ax.axhline(y=overall_evening, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(2.5, overall_evening + 2, f'Overall Avg: {overall_evening}%', 
            fontsize=10, color='red', fontweight='bold')
    
    # Remove decorative elements and highlights - clean design
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2E0854')
    ax.spines['bottom'].set_color('#2E0854')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = 'res/cluster_plots/cluster_08_evening_readers_minimal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✨ Beautiful Evening Readers plot saved to: {output_path}")
    
    # Also display the plot
    plt.show()
    plt.close()

def main():
    """Main execution function"""
    print("🎨 Creating beautiful Evening News Readers plot...")
    create_evening_readers_plot()
    print("🌃 Evening News Readers visualization completed!")

if __name__ == "__main__":
    main()
