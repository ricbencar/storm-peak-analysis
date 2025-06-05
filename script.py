"""
Storm Peaks Analysis Script

Purpose:
This script performs an analysis of storm peak events by combining storm wave data with tide level data. 
It identifies significant storm peaks based on wave height, filters for major storms, 
and generates plots visualizing these events over various time periods. It also outputs a CSV file 
listing the major storm events found over the entire common data period.

1. Using vectorized operations for faster datetime parsing in the tide data.
2. Calculating storm peaks only once on the full dataset and filtering them for each plot,
   instead of re-calculating peaks inside the plotting loop.

Input Files:
1. Storm Data CSV (e.g., 'input.csv'):
   - Expected format: CSV file with a 'datetime' column and a 'swh' (Significant Wave Height) column.
2. Tide Level CSV (e.g., 'tide-levels.csv'):
   - Expected format: CSV file with a header and two columns: 'datetime' (YYYY-MM-DD HH:MM) and 'tide' (float).

Output Files:
1. PNG Plot(s) (e.g., 'storm-peaks.png', 'storm-peaks-last-10years.png'):
   - Visualizations of significant wave height (swh) and tide levels over specified periods.
2. CSV Data File (e.g., 'storm-peaks.txt'):
   - A comma-separated values file listing details of "major storms".

User Configuration:
- File paths for input and output.
- Thresholds for peak detection.
- Periods for which plots should be generated.

Installation of Required Packages:
  pip install pandas numpy matplotlib scipy

Running the Script:
1. Place input CSV files ('input.csv', 'tide-levels.csv') in the same directory as the script.
2. Run from your terminal: python script.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime 
from pandas.tseries.offsets import DateOffset
from scipy.signal import find_peaks
import warnings
import os

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')


def generate_storm_plot(df_plot_data, picos_tempestades_all_plot, principais_tempestades_plot, 
                        base_output_plot_name, period_config, 
                        major_storm_threshold_swh, label_relevant_peaks_threshold_swh, 
                        plots_output_dir):
    """
    Generates and saves a storm peak plot for a given data period.
    (This function is unchanged from the original script)
    """
    title_suffix = period_config['title']
    file_suffix = period_config['suffix']
    
    # Ensure the plots directory exists, create it if not
    if not os.path.exists(plots_output_dir):
        try:
            os.makedirs(plots_output_dir)
            print(f"Created directory: {plots_output_dir}")
        except OSError as e:
            print(f"Error creating directory {plots_output_dir}: {e}. Plots will be saved in current directory.")
            plots_output_dir = "." # Fallback to current directory

    # Construct full file paths for the output PNG plot
    output_plot_png_file = os.path.join(plots_output_dir, f"{base_output_plot_name}{file_suffix}.png")
    
    print(f"Generating storm peaks plot for: {title_suffix} (saving to {output_plot_png_file})...")
    
    # If there's no data for the period, skip plotting
    if df_plot_data.empty:
        print(f"Warning: No data to plot for {title_suffix}.")
        return

    # Create the figure and primary y-axis (for SWH)
    fig, ax1 = plt.subplots(figsize=(15, 7.5)) 
    lines = [] # To store plot elements for the legend

    # Plot Significant Wave Height (SWH) timeseries
    line_swh, = ax1.plot(df_plot_data.index, df_plot_data['swh'], linestyle='-', color='black', label='Significant Wave Height (swh)', linewidth=0.9, zorder=2)
    lines.append(line_swh)

    # Configure primary y-axis (ax1)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Significant Wave Height, swh (m)', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a secondary y-axis (for Tide Level), sharing the same x-axis
    ax2 = ax1.twinx()
    if 'tide' in df_plot_data.columns and not df_plot_data['tide'].dropna().empty:
        # Plot Tide Level timeseries
        line_tide, = ax2.plot(df_plot_data.index, df_plot_data['tide'], linestyle='-', color='lightgray', label='Tide Level', linewidth=1.2, alpha=0.5, zorder=1) 
        lines.append(line_tide)
        
        # Adjust y-limits for the tide axis to fit the data snugly with some padding
        tide_min = df_plot_data['tide'].min()
        tide_max = df_plot_data['tide'].max()
        tide_range = tide_max - tide_min
        padding = tide_range * 0.1 if tide_range > 0.1 else 0.05 # 10% padding, or fixed if range is small
        ax2.set_ylim(tide_min - padding, tide_max + padding)

    # Configure secondary y-axis (ax2)
    ax2.set_ylabel('Tide Level (m)', fontsize=14, color='dimgray') 
    ax2.tick_params(axis='y', labelcolor='dimgray') 
    ax2.grid(False) # Turn off grid for the secondary axis to avoid visual clutter

    # --- Peak Annotation Logic ---
    labeled_major_storm = False
    labeled_significant_peak = False
    
    if not picos_tempestades_all_plot.empty:
        for idx, row in picos_tempestades_all_plot.iterrows():
            peak_swh = row['swh']
            if peak_swh > major_storm_threshold_swh:
                ax1.text(idx, peak_swh, f"{peak_swh:.2f}", size=9, color='blue', ha='center', va='bottom', zorder=4)
                if not labeled_major_storm:
                    line_major_marker, = ax1.plot([], [], 'bo', markersize=5, label=f'Major Storm Peaks (swh > {major_storm_threshold_swh:.1f}m)')
                    lines.append(line_major_marker)
                    labeled_major_storm = True
            elif peak_swh > label_relevant_peaks_threshold_swh:
                ax1.text(idx, peak_swh, f"{peak_swh:.2f}", size=8, color='black', ha='center', va='bottom', zorder=4)
                if not labeled_significant_peak:
                    line_sig_marker, = ax1.plot([], [], 'o', color='black', markersize=4, label=f'Relevant Peaks ({label_relevant_peaks_threshold_swh:.1f}m < swh <= {major_storm_threshold_swh:.1f}m)')
                    lines.append(line_sig_marker)
                    labeled_significant_peak = True
        
        if not labeled_major_storm and not labeled_significant_peak:
            line_minor_peaks, = ax1.plot([],[], 'co', markersize=3, label=f'{len(picos_tempestades_all_plot)} other peaks found (swh <= {label_relevant_peaks_threshold_swh:.1f}m)', zorder=3)
            lines.append(line_minor_peaks)
            
    else: # No peaks at all in this period
        line_no_peaks, = ax1.plot([],[], 'x', color='gray', markersize=5, label='No peaks found in period', zorder=3)
        lines.append(line_no_peaks)

    # Set x-axis limits to the data range for the current plot
    ax1.set_xlim([df_plot_data.index.min(), df_plot_data.index.max()])

    # Dynamically adjust x-axis date locators and formatters
    time_delta_days = (df_plot_data.index.max() - df_plot_data.index.min()).days
    if time_delta_days <= 0:
        major_locator = mdates.AutoDateLocator()
        major_formatter = mdates.ConciseDateFormatter(major_locator)
        minor_locator = mdates.AutoDateLocator()
    elif time_delta_days <= 30:
        major_locator = mdates.DayLocator(interval=5)
        major_formatter = mdates.DateFormatter('%Y-%m-%d')
        minor_locator = mdates.DayLocator()
    elif time_delta_days <= 366 :
        major_locator = mdates.MonthLocator(interval=1)
        major_formatter = mdates.DateFormatter('%Y-%m')
        minor_locator = mdates.WeekdayLocator(byweekday=mdates.MO)
    elif time_delta_days <= 366 * 5:
        major_locator = mdates.MonthLocator(interval=6)
        major_formatter = mdates.DateFormatter('%Y-%m')
        minor_locator = mdates.MonthLocator(interval=1)
    elif time_delta_days <= 366 * 15:
        major_locator = mdates.YearLocator(1)
        major_formatter = mdates.DateFormatter('%Y')
        minor_locator = mdates.MonthLocator(interval=3)
    else:
        major_locator = mdates.YearLocator(2)
        major_formatter = mdates.DateFormatter('%Y')
        minor_locator = mdates.YearLocator(1)
        
    ax1.xaxis.set_major_locator(major_locator)
    ax1.xaxis.set_major_formatter(major_formatter)
    ax1.xaxis.set_minor_locator(minor_locator)
    
    ax1.legend(handles=lines, loc='upper left')
    plt.suptitle(f'Storm Peaks Analysis: {title_suffix}', fontsize=20)
    ax1.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2) 

    fig.autofmt_xdate()
    fig.tight_layout()
    
    try:
        plt.savefig(output_plot_png_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_plot_png_file}")
    except Exception as e:
        print(f"Error saving plot {output_plot_png_file}: {e}")
    
    plt.close(fig)

def main():
    """
    Main function to orchestrate the storm peak analysis.
    """
    # --- User Configuration Section ---
    plot_periods_config = [
        {'suffix': '', 'years_ago': None, 'title': 'Full Period'},
        {'suffix': '-last-10years', 'years_ago': 10, 'title': 'Last 10 Years'},
        {'suffix': '-last-5years', 'years_ago': 5, 'title': 'Last 5 Years'},
        {'suffix': '-last-2years', 'years_ago': 2, 'title': 'Last 2 Years'},
        {'suffix': '-last-year', 'years_ago': 1, 'title': 'Last Year'}
    ]
    input_csv_path = 'input.csv'
    tide_csv_path = 'tide-levels.csv'
    base_output_plot_name = 'storm-peaks'
    output_storms_txt = 'storm-peaks.txt'
    plots_output_dir = 'plots'
    peak_height_threshold_swh = 2.0
    peak_distance_hours = 72
    major_storm_threshold_swh = 6.0
    label_relevant_peaks_threshold_swh = 4.0
    # --- End of User Configuration Section ---

    # --- Step 1: Load and Preprocess Storm Data ---
    print(f"Loading storm data from {input_csv_path}...")
    try:
        df_inputs = pd.read_csv(input_csv_path, parse_dates=['datetime'], index_col='datetime')
        if df_inputs.empty:
            print(f"Error: Storm data file {input_csv_path} is empty.")
            return
    except FileNotFoundError:
        print(f"Error: File not found - {input_csv_path}.")
        return
    except Exception as e:
        print(f"Error reading storm data file {input_csv_path}: {e}")
        return

    print("Preprocessing storm data (resampling and interpolating)...")
    df_inputs_resampled = df_inputs.resample('h').mean()
    df_inputs_interpolated = df_inputs_resampled.interpolate(method='spline', order=2, limit_direction='both')
    df_inputs_interpolated.dropna(subset=['swh'], inplace=True)
    if df_inputs_interpolated.empty:
        print("Error: Storm data is empty after preprocessing.")
        return

    # --- Step 2: Load and Preprocess Tide Data ---
    print(f"Loading tide data from {tide_csv_path}...")
    df_tides_interpolated = pd.DataFrame()
    try:
        # UPDATED: Load tide data assuming a header and 'datetime', 'tide' columns.
        # pandas automatically parses the 'datetime' column and sets it as the index.
        df_tides = pd.read_csv(
            tide_csv_path, 
            parse_dates=['datetime'], 
            index_col='datetime'
        )
        
        if not df_tides.empty:
            # Clean up any rows that might be missing tide values
            df_tides.dropna(subset=['tide'], inplace=True)
            
            if not df_tides.empty:
                # Check for and handle duplicate timestamps by averaging
                if df_tides.index.duplicated().any():
                    print(f"Warning: Duplicate timestamps found in {tide_csv_path}. Averaging 'tide' values.")
                    df_tides = df_tides.groupby(df_tides.index).agg({'tide': 'mean'})
                
                print("Preprocessing tide data (resampling and interpolating)...")
                df_tides_resampled = df_tides[['tide']].resample('h').mean()
                df_tides_interpolated = df_tides_resampled.interpolate(method='linear', limit_direction='both')
                df_tides_interpolated.dropna(inplace=True)
        
        if df_tides_interpolated.empty:
            print(f"Warning: Tide data is effectively empty after processing from {tide_csv_path}.")
    except FileNotFoundError:
        print(f"Error: Tide data file not found - {tide_csv_path}.")
    except Exception as e:
        print(f"Error processing tide data file {tide_csv_path}: {e}")

    if df_tides_interpolated.empty:
        print("Error: Processed tide data is empty. Cannot proceed.")
        return 

    # --- Step 3: Determine Common Timeframe and Merge Data ---
    print("Determining common timeframe and merging data...")
    common_start_time = max(df_inputs_interpolated.index.min(), df_tides_interpolated.index.min())
    common_end_time = min(df_inputs_interpolated.index.max(), df_tides_interpolated.index.max())

    if pd.isna(common_start_time) or pd.isna(common_end_time) or common_start_time >= common_end_time:
        print("Error: No overlapping timeframe found between storm and tide data.")
        return
    
    print(f"Overall common timeframe for analysis: {common_start_time} to {common_end_time}")
    
    df_merged_full_common_period = df_inputs_interpolated.join(df_tides_interpolated, how='inner')
    df_merged_full_common_period = df_merged_full_common_period[common_start_time:common_end_time]
    df_merged_full_common_period.dropna(subset=['swh', 'tide'], inplace=True) 
            
    if df_merged_full_common_period.empty:
        print("Error: Merged DataFrame is empty. No data to analyze.")
        return

    # --- Step 4: Identify All Storms ONCE and Save Major Storms CSV ---
    print(f"\nIdentifying all storm peaks for the full common period...")
    full_peaks_indices, _ = find_peaks(
        df_merged_full_common_period['swh'], 
        height=peak_height_threshold_swh, 
        distance=peak_distance_hours
    )
    
    if len(full_peaks_indices) > 0:
        picos_tempestades_all_full = df_merged_full_common_period.iloc[full_peaks_indices]
        principais_tempestades_full = picos_tempestades_all_full[picos_tempestades_all_full['swh'] > major_storm_threshold_swh].copy()

        if not principais_tempestades_full.empty:
            print(f"Found {len(principais_tempestades_full)} major storm peaks. Saving to {output_storms_txt}...")
            output_columns = ['swh', 'tide'] + [col for col in ['mwd', 'pp1d', 'wind', 'dwi'] if col in principais_tempestades_full.columns]
            if principais_tempestades_full.index.name is None:
                principais_tempestades_full.index.name = 'datetime'
            principais_tempestades_full[output_columns].to_csv(
                output_storms_txt, sep=',', index=True, index_label='datetime', header=True, 
                float_format='%.2f', na_rep='NaN'
            )
        else:
            print(f"No major storms (swh > {major_storm_threshold_swh}m) found to save to CSV.")
            with open(output_storms_txt, 'w') as f: f.write(f"No major storms found.\n")
    else:
        print(f"No storm peaks found in the full common period.")
        picos_tempestades_all_full = pd.DataFrame() # Ensure it's an empty DF if no peaks
        with open(output_storms_txt, 'w') as f: f.write("No storm peaks found.\n")
    
    # --- Step 5: Loop to Generate Plots for Each Period ---
    if df_merged_full_common_period.empty:
        print("Cannot generate plots as there is no merged data.")
        return

    latest_date_in_data = df_merged_full_common_period.index.max()

    for period_conf in plot_periods_config:
        period_start_date = df_merged_full_common_period.index.min()
        if period_conf['years_ago'] is not None:
            try:
                calculated_start = latest_date_in_data - DateOffset(years=period_conf['years_ago'])
                period_start_date = max(period_start_date, calculated_start)
            except Exception as e:
                print(f"Error calculating date range for '{period_conf['title']}': {e}. Skipping plot.")
                continue
        
        period_end_date = latest_date_in_data
        current_df_period = df_merged_full_common_period[period_start_date:period_end_date].copy()
        
        if current_df_period.empty:
            print(f"\nNo data available for the period '{period_conf['title']}'. Skipping plot.")
            continue
        
        print(f"\nProcessing plot for: {period_conf['title']} (from {current_df_period.index.min().date()} to {current_df_period.index.max().date()})")

        # OPTIMIZATION: Filter pre-calculated peaks for the current period instead of re-calculating
        picos_all_period_df = picos_tempestades_all_full.loc[period_start_date:period_end_date]
        principais_period_df = picos_all_period_df[picos_all_period_df['swh'] > major_storm_threshold_swh]
        
        generate_storm_plot(
            df_plot_data=current_df_period,
            picos_tempestades_all_plot=picos_all_period_df,
            principais_tempestades_plot=principais_period_df,
            base_output_plot_name=base_output_plot_name,
            period_config=period_conf,
            major_storm_threshold_swh=major_storm_threshold_swh,
            label_relevant_peaks_threshold_swh=label_relevant_peaks_threshold_swh,
            plots_output_dir=plots_output_dir 
        )

    print("\nScript finished successfully.")

if __name__ == '__main__':
    main()