#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze basketball shot detection system debug logs and generate visualization reports
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from datetime import datetime

def load_debug_log(log_file):
    """
    Load debug log file
    
    Args:
        log_file: Path to debug log file
    
    Returns:
        dict: Debug log data
    """
    with open(log_file, 'r') as f:
        data = json.load(f)
    return data

def analyze_shot_patterns(debug_data):
    """
    Analyze shot patterns
    
    Args:
        debug_data: Debug log data
    
    Returns:
        tuple: (successful shots list, failed shots list)
    """
    successful_shots = []
    failed_shots = []
    
    for shot in debug_data.get('detailed_shots', []):
        # Extract debug info
        debug_info = {}
        if 'debug_info' in shot:
            debug_info = shot['debug_info']
        
        # Add other shot data to debug_info
        for key, value in shot.items():
            if key != 'debug_info' and key not in debug_info:
                debug_info[key] = value
        
        # Determine if shot was successful
        if shot.get('is_successful', False):
            successful_shots.append(debug_info)
        else:
            failed_shots.append(debug_info)
    
    return successful_shots, failed_shots

def visualize_shot_distribution(debug_data, successful_shots, failed_shots, output_dir):
    """
    Generate shot distribution visualization
    
    Args:
        debug_data: Debug log data
        successful_shots: List of successful shots
        failed_shots: List of failed shots
        output_dir: Output directory
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Draw hoop
        if successful_shots or failed_shots:
            try:
                # Use hoop information from the first shot
                all_shots = successful_shots + failed_shots
                if all_shots and 'hoop_info' in all_shots[0]:
                    hoop_info = all_shots[0]['hoop_info']
                    if ('position' in hoop_info and 'x' in hoop_info['position'] and 
                        'y' in hoop_info['position'] and 'width' in hoop_info and 'height' in hoop_info):
                        hoop_x = hoop_info['position']['x']
                        hoop_y = hoop_info['position']['y']
                        hoop_width = hoop_info['width']
                        hoop_height = hoop_info['height']
                        
                        # Draw rim
                        rim = plt.Circle((hoop_x, hoop_y), hoop_width/2, fill=False, color='black', linewidth=2)
                        plt.gca().add_patch(rim)
                        
                        # Draw backboard
                        backboard_width = hoop_width * 2
                        backboard_height = hoop_height * 0.2
                        backboard_x = hoop_x
                        backboard_y = hoop_y - hoop_height
                        backboard = Rectangle((backboard_x - backboard_width/2, backboard_y - backboard_height/2),
                                            backboard_width, backboard_height, fill=False, color='gray', linewidth=2)
                        plt.gca().add_patch(backboard)
            except Exception as e:
                print(f"Error drawing hoop: {str(e)}")
        
        # Draw successful shot points
        successful_points = []
        for shot in successful_shots:
            try:
                if ('prediction' in shot and 'predicted_x_at_rim' in shot['prediction'] and 
                    'hoop_info' in shot and 'rim_height' in shot['hoop_info']):
                    x = shot['prediction']['predicted_x_at_rim']
                    y = shot['hoop_info']['rim_height']
                    plt.scatter(x, y, color='green', s=50, alpha=0.7, marker='o')
                    successful_points.append((x, y))
            except Exception as e:
                print(f"Error drawing successful shot point: {str(e)}")
        
        # Draw failed shot points
        failed_points = []
        for shot in failed_shots:
            try:
                if ('prediction' in shot and 'predicted_x_at_rim' in shot['prediction'] and 
                    'hoop_info' in shot and 'rim_height' in shot['hoop_info']):
                    x = shot['prediction']['predicted_x_at_rim']
                    y = shot['hoop_info']['rim_height']
                    plt.scatter(x, y, color='red', s=50, alpha=0.7, marker='x')
                    failed_points.append((x, y))
            except Exception as e:
                print(f"Error drawing failed shot point: {str(e)}")
        
        plt.title('Shot Distribution', fontsize=16)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add legend only when there is data
        legend_elements = []
        if successful_points or failed_points:
            if successful_points:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Successful Shots'))
            if failed_points:
                legend_elements.append(plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Failed Shots'))
            plt.legend(handles=legend_elements)
        
        # Save image
        output_file = os.path.join(output_dir, 'shot_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Shot distribution visualization generated: {output_file}")
    except Exception as e:
        print(f"Error generating shot distribution visualization: {str(e)}")

def visualize_distance_distribution(debug_data, successful_shots, failed_shots, output_dir):
    """
    Generate distance distribution visualization
    
    Args:
        debug_data: Debug log data
        successful_shots: List of successful shots
        failed_shots: List of failed shots
        output_dir: Output directory
    """
    plt.figure(figsize=(12, 8))
    
    # Collect distance data
    successful_distances = []
    failed_distances = []
    
    for shot in successful_shots:
        if 'shot_analysis' in shot:
            distance = shot['shot_analysis']['horizontal_distance_from_center']
            successful_distances.append(distance)
    
    for shot in failed_shots:
        if 'shot_analysis' in shot:
            distance = shot['shot_analysis']['horizontal_distance_from_center']
            failed_distances.append(distance)
    
    # Create histogram
    bins = np.linspace(-50, 50, 20)
    plt.hist(successful_distances, bins=bins, alpha=0.7, color='green', label='Successful Shots')
    plt.hist(failed_distances, bins=bins, alpha=0.7, color='red', label='Failed Shots')
    
    plt.title('Horizontal Distance Distribution from Hoop Center', fontsize=16)
    plt.xlabel('Horizontal Distance from Hoop Center', fontsize=12)
    plt.ylabel('Number of Shots', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save image
    plt.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_shot_summary(debug_data, successful_shots, failed_shots, output_dir):
    """
    Generate shot summary report
    
    Args:
        debug_data: Debug log data
        successful_shots: List of successful shots
        failed_shots: List of failed shots
        output_dir: Output directory
    """
    total_shots = len(successful_shots) + len(failed_shots)
    success_rate = len(successful_shots) / total_shots * 100 if total_shots > 0 else 0
    
    # Count direct hits and rebound hits
    direct_hits = 0
    rebound_hits = 0
    
    for shot in successful_shots:
        if 'shot_analysis' in shot:
            if shot['shot_analysis']['is_direct_hit']:
                direct_hits += 1
            elif shot['shot_analysis']['is_rebound_hit']:
                rebound_hits += 1
    
    # Count failure reasons
    left_misses = 0
    right_misses = 0
    other_misses = 0
    
    for shot in failed_shots:
        if 'failure_reason' in shot:
            failure_reason = shot['failure_reason'].lower()
            if 'left' in failure_reason or 'missed left' in failure_reason:
                left_misses += 1
            elif 'right' in failure_reason or 'missed right' in failure_reason:
                right_misses += 1
            else:
                other_misses += 1
    
    # Calculate percentages safely
    direct_hits_pct = direct_hits / len(successful_shots) * 100 if len(successful_shots) > 0 else 0
    rebound_hits_pct = rebound_hits / len(successful_shots) * 100 if len(successful_shots) > 0 else 0
    left_misses_pct = left_misses / len(failed_shots) * 100 if len(failed_shots) > 0 else 0
    right_misses_pct = right_misses / len(failed_shots) * 100 if len(failed_shots) > 0 else 0
    other_misses_pct = other_misses / len(failed_shots) * 100 if len(failed_shots) > 0 else 0
    
    # Generate report
    report = f"""Shot Analysis Summary
===================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Statistics:
------------------
Total Shots: {total_shots}
Successful Shots: {len(successful_shots)} ({success_rate:.1f}%)
Failed Shots: {len(failed_shots)} ({100 - success_rate:.1f}%)

Successful Shot Analysis:
-----------------------
Direct Hits: {direct_hits} ({direct_hits_pct:.1f}% of successful shots)
Rim Rebound Hits: {rebound_hits} ({rebound_hits_pct:.1f}% of successful shots)

Failed Shot Analysis:
------------------
Missed Left of Hoop: {left_misses} ({left_misses_pct:.1f}% of failed shots)
Missed Right of Hoop: {right_misses} ({right_misses_pct:.1f}% of failed shots)
Other Failure Reasons: {other_misses} ({other_misses_pct:.1f}% of failed shots)
"""
    
    # Save report
    with open(os.path.join(output_dir, 'shot_summary.txt'), 'w') as f:
        f.write(report)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Analyze basketball shot detection system debug logs')
    parser.add_argument('--log', type=str, required=True, help='Path to debug log file')
    parser.add_argument('--output', type=str, default='analysis_output', help='Output directory')
    args = parser.parse_args(argv)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load debug log
    debug_data = load_debug_log(args.log)
    
    # Analyze shot patterns
    successful_shots, failed_shots = analyze_shot_patterns(debug_data)
    
    # Generate visualization reports
    visualize_shot_distribution(debug_data, successful_shots, failed_shots, args.output)
    visualize_distance_distribution(debug_data, successful_shots, failed_shots, args.output)
    generate_shot_summary(debug_data, successful_shots, failed_shots, args.output)
    
    print(f"Analysis complete! Visualization reports saved to {args.output} directory")

if __name__ == "__main__":
    main()