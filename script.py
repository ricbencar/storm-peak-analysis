# Detailed Script Description:
"""
Storm Peaks Analysis Script

Purpose:
This script performs an analysis of storm peak events by combining storm wave data with tide level data. 
It identifies significant storm peaks based on wave height, filters for major storms, 
and generates plots visualizing these events over various time periods. It also outputs a CSV file 
listing the major storm events found over the entire common data period.

Input Files:
1. Storm Data CSV (e.g., 'input.csv'):
   - Expected format: CSV file with a 'datetime' column and a 'swh' (Significant Wave Height) column, 
     among others (like 'mwd', 'pp1d', 'wind', 'dwi').
   - The 'datetime' column should be parsable by pandas (e.g., 'YYYY-MM-DD HH:MM:SS').
   - Example:
     datetime,swh,mwd,pp1d,wind,dwi
     1940-01-01 00:00:00,4.318,176.48,9.43,14.81,128.85
     ...

2. Tide Level CSV (e.g., 'tide-levels.csv'):
   - Expected format: CSV file. The script expects data rows to represent date, time, and tide value.
   - It specifically skips the first row (assumed to be a header like "datetime,tide") and then expects 
     three comma-separated values per data line: Date (YYYY-MM-DD), Time (HH:MM), Tide Value (float).
   - Example (after skipping header):
     1980-01-01,12:00,3.061828
     1980-01-01,01:00,3.483248
     ...

Output Files:
1. PNG Plot(s) (e.g., 'storm-peaks.png', 'storm-peaks-last-10years.png'):
   - Visualizations of significant wave height (swh) and tide levels over specified periods.
   - Storm peaks above certain thresholds are annotated on the plots.
   - Plots are saved in a subdirectory specified by 'plots_output_dir' (default: 'plots').

2. CSV Data File (e.g., 'storm-peaks.txt'):
   - A comma-separated values file listing details of "major storms" identified over the 
     entire common period where both storm and tide data are available.
   - Includes datetime, swh, tide, and other available storm parameters.

User Configuration:
- File paths for input and output.
- Thresholds for peak detection (general peak height, distance between peaks, major storm height).
- Periods for which plots should be generated (e.g., full period, last 10 years, last 1 year).
- Threshold for labeling significant peaks (default is 5.0m).

Installation of Required Packages:
This script requires the following Python packages. You can install them using pip:
  pip install pandas numpy matplotlib scipy

Running the Script:
1. Ensure Python is installed on your system.
2. Install the required packages as listed above.
3. Place the input CSV files ('input.csv', 'tide-levels.csv') in the same directory as the script,
   or update the 'input_csv_path' and 'tide_csv_path' variables in the script.
4. Run the script from your terminal: python your_script_name.py (e.g., python storm_analysis_script.py)
5. Outputs will be generated in the script's directory and the 'plots' subdirectory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime 
from pandas.tseries.offsets import DateOffset # For date calculations
from scipy.signal import find_peaks
import warnings
import os # For directory creation

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy') # For find_peaks with empty data

def parse_tide_datetime(date_str, time_str):
    """
    Helper function to parse date and time strings and combine them into a pandas datetime object.

    Parameters:
    date_str (str): The date string (e.g., 'YYYY-MM-DD').
    time_str (str): The time string (e.g., 'HH:MM').

    Returns:
    pandas.Timestamp: A datetime object if parsing is successful, otherwise pandas.NaT (Not a Time).
    """
    try:
        # Combine date and time strings, strip any leading/trailing whitespace, and convert to datetime
        return pd.to_datetime(f"{str(date_str).strip()} {str(time_str).strip()}")
    except Exception as e:
        # Print an error message if parsing fails for a specific entry (optional)
        # print(f"Debug: Failed to parse date '{date_str}' and time '{time_str}': {e}")
        return pd.NaT # Return Not a Time if parsing fails

def generate_storm_plot(df_plot_data, picos_tempestades_all_plot, principais_tempestades_plot, 
                        base_output_plot_name, period_config, 
                        major_storm_threshold_swh, label_significant_peaks_threshold_swh, 
                        plots_output_dir):
    """
    Generates and saves a storm peak plot for a given data period.

    Parameters:
    df_plot_data (pd.DataFrame): DataFrame containing 'swh' (significant wave height) and 'tide' 
                                 for the specific period to be plotted. Index must be DatetimeIndex.
    picos_tempestades_all_plot (pd.DataFrame): DataFrame of all storm peaks identified in df_plot_data 
                                               (above 'peak_height_threshold_swh').
    principais_tempestades_plot (pd.DataFrame): DataFrame of "major" storm peaks identified in df_plot_data
                                                (subset of picos_tempestades_all_plot, above 
                                                 'major_storm_threshold_swh').
    base_output_plot_name (str): Base name for output plot files (e.g., 'storm-peaks').
    period_config (dict): Dictionary containing:
                          'suffix' (str): Suffix for filename (e.g., '', '-last-10years').
                          'title' (str): Suffix for plot title (e.g., 'Full Period', 'Last 10 Years').
    major_storm_threshold_swh (float): The SWH threshold used to classify "major storms".
    label_significant_peaks_threshold_swh (float): The SWH threshold above which all peaks (not just major) should be labeled.
    plots_output_dir (str): The directory where the plot images will be saved.
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
    ax1.set_ylabel('Wave Height, swh (m)', fontsize=14, color='black')
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
    # Flags to ensure legend entries are added only once if corresponding peaks exist
    labeled_major_storm = False
    labeled_significant_peak = False
    
    # Annotate storm peaks on ax1 (SWH axis)
    # Iterate through all peaks found for the current period
    if not picos_tempestades_all_plot.empty:
        for idx, row in picos_tempestades_all_plot.iterrows():
            peak_swh = row['swh']
            # Label Major Storms (e.g., swh > 6.0m)
            if peak_swh > major_storm_threshold_swh:
                ax1.text(idx, peak_swh, f"{peak_swh:.2f}", size=9, color='blue', ha='center', va='bottom', zorder=4)
                if not labeled_major_storm:
                    # Add legend entry for major storms (dummy plot for handle)
                    line_major_marker, = ax1.plot([], [], 'bo', markersize=5, label=f'Major Storm Peaks (swh > {major_storm_threshold_swh:.1f}m)')
                    lines.append(line_major_marker)
                    labeled_major_storm = True
            # Label Other Significant Peaks (e.g., 4.0m < swh <= 6.0m)
            elif peak_swh > label_significant_peaks_threshold_swh:
                ax1.text(idx, peak_swh, f"{peak_swh:.2f}", size=8, color='darkorange', ha='center', va='bottom', zorder=4)
                if not labeled_significant_peak:
                     # Add legend entry for other significant peaks (dummy plot for handle)
                    line_sig_marker, = ax1.plot([], [], 'o', color='darkorange', markersize=4, label=f'Significant Peaks ({label_significant_peaks_threshold_swh:.1f}m < swh <= {major_storm_threshold_swh:.1f}m)')
                    lines.append(line_sig_marker)
                    labeled_significant_peak = True
        
        # If no peaks were labeled as "major" or "significant", but some general peaks were found
        if not labeled_major_storm and not labeled_significant_peak:
            line_minor_peaks, = ax1.plot([],[], 'co', markersize=3, label=f'{len(picos_tempestades_all_plot)} other peaks found (swh <= {label_significant_peaks_threshold_swh:.1f}m)', zorder=3)
            lines.append(line_minor_peaks)
            
    else: # No peaks at all in this period
        line_no_peaks, = ax1.plot([],[], 'x', color='gray', markersize=5, label='No peaks found in period', zorder=3)
        lines.append(line_no_peaks)

    # Set x-axis limits to the data range for the current plot
    ax1.set_xlim([df_plot_data.index.min(), df_plot_data.index.max()])

    # Dynamically adjust x-axis date locators and formatters based on the time range of the plot
    time_delta_days = (df_plot_data.index.max() - df_plot_data.index.min()).days
    if time_delta_days <= 0: # Edge case: single data point or invalid range
        major_locator = mdates.AutoDateLocator()
        major_formatter = mdates.ConciseDateFormatter(major_locator) # Handles single points well
        minor_locator = mdates.AutoDateLocator()
    elif time_delta_days <= 30: # For plots up to 1 month long
        major_locator = mdates.DayLocator(interval=5) # Ticks every 5 days
        major_formatter = mdates.DateFormatter('%Y-%m-%d')
        minor_locator = mdates.DayLocator() # Minor ticks daily
    elif time_delta_days <= 366 : # For plots up to 1 year long
        major_locator = mdates.MonthLocator(interval=1) # Ticks every month
        major_formatter = mdates.DateFormatter('%Y-%m')
        minor_locator = mdates.WeekdayLocator(byweekday=mdates.MO) # Minor ticks every Monday
    elif time_delta_days <= 366 * 5: # For plots up to 5 years long
        major_locator = mdates.MonthLocator(interval=6) # Ticks every 6 months
        major_formatter = mdates.DateFormatter('%Y-%m')
        minor_locator = mdates.MonthLocator(interval=1) # Minor ticks every month
    elif time_delta_days <= 366 * 15: # For plots up to 15 years long
        major_locator = mdates.YearLocator(1) # Ticks every year
        major_formatter = mdates.DateFormatter('%Y')
        minor_locator = mdates.MonthLocator(interval=3) # Minor ticks quarterly
    else: # For longer periods
        major_locator = mdates.YearLocator(2) # Ticks every 2 years
        major_formatter = mdates.DateFormatter('%Y')
        minor_locator = mdates.YearLocator(1) # Minor ticks every year
        
    ax1.xaxis.set_major_locator(major_locator)
    ax1.xaxis.set_major_formatter(major_formatter)
    ax1.xaxis.set_minor_locator(minor_locator)
    
    # Add the legend to the plot
    ax1.legend(handles=lines, loc='upper left')
    # Set the main title for the plot
    plt.suptitle(f'Storm Peaks Analysis: {title_suffix}', fontsize=20)

    # Configure grid lines
    ax1.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on() # Enable minor ticks
    ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2) 

    fig.autofmt_xdate() # Automatically format x-axis date labels for better readability
    fig.tight_layout()  # Adjust plot to ensure everything fits without overlapping
    
    # Save the plot to a PNG file
    try:
        plt.savefig(output_plot_png_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_plot_png_file}")
    except Exception as e:
        print(f"Error saving plot {output_plot_png_file}: {e}")
    
    plt.close(fig) # Close the figure to free up memory


def main():
    """
    Main function to orchestrate the storm peak analysis.
    It handles configuration, data loading, preprocessing, peak identification,
    and the generation of plots and a summary CSV file.
    """
    # --- User Configuration Section ---
    # This section contains parameters that can be easily modified by the user.

    # Configuration for different plot periods
    # 'suffix': String appended to the base filename for distinction (e.g., '-last-10years').
    # 'years_ago': Number of years to look back from the end of the data. 
    #              Set to None to plot the full available period.
    # 'title': Descriptive title for the plot period.
    plot_periods_config = [
        {'suffix': '', 'years_ago': None, 'title': 'Full Period'},
        {'suffix': '-last-10years', 'years_ago': 10, 'title': 'Last 10 Years'},
        {'suffix': '-last-5years', 'years_ago': 5, 'title': 'Last 5 Years'},
        {'suffix': '-last-2years', 'years_ago': 2, 'title': 'Last 2 Years'},
        {'suffix': '-last-year', 'years_ago': 1, 'title': 'Last Year'}
    ]

    # File paths
    input_csv_path = 'input.csv'        # Path to the storm input data CSV file
    tide_csv_path = 'tide-levels.csv'   # Path to the tide level data CSV file
    
    base_output_plot_name = 'storm-peaks' # Base name for generated plot files
    output_storms_txt = 'storm-peaks.txt' # Name of the output CSV for major storms (full period)
    plots_output_dir = 'plots'            # Subdirectory to save generated plots

    # Peak detection parameters
    peak_height_threshold_swh = 2.0     # Minimum SWH for a data point to be considered part of a peak
    peak_distance_hours = 72            # Minimum horizontal distance (in hours) between identified peaks
    major_storm_threshold_swh = 6.0     # SWH value above which a peak is classified as a "major storm"
    label_significant_peaks_threshold_swh = 4.0 # SWH value above which peaks (not necessarily major) are labeled on plots
    # --- End of User Configuration Section ---

    # --- Step 1: Load and Preprocess Storm Data (from input_csv_path) ---
    print(f"Loading storm data from {input_csv_path}...")
    try:
        # Read storm data, parse the 'datetime' column as dates, and set it as the index
        df_inputs = pd.read_csv(input_csv_path, parse_dates=['datetime'], index_col='datetime')
        if df_inputs.empty:
            print(f"Error: Storm data file {input_csv_path} is empty or could not be read.")
            return
    except FileNotFoundError:
        print(f"Error: File not found - {input_csv_path}. Please check the file path.")
        return
    except Exception as e:
        print(f"Error reading storm data file {input_csv_path}: {e}")
        return

    print("Preprocessing storm data (resampling to hourly and interpolating missing values)...")
    # Resample data to hourly frequency, calculating the mean for each hour
    df_inputs_resampled = df_inputs.resample('h').mean()
    # Interpolate missing values using spline interpolation (order 2)
    # 'limit_direction="both"' fills NaNs at the beginning and end as well
    df_inputs_interpolated = df_inputs_resampled.interpolate(method='spline', order=2, limit_direction='both')
    # Drop any rows where 'swh' is still NaN after interpolation (essential for peak finding)
    df_inputs_interpolated.dropna(subset=['swh'], inplace=True)
    if df_inputs_interpolated.empty:
        print("Error: Storm data is empty after preprocessing (resampling/interpolation).")
        return

    # --- Step 2: Load and Preprocess Tide Data (from tide_csv_path) ---
    print(f"Loading tide data from {tide_csv_path}...")
    df_tides = pd.DataFrame() # Initialize an empty DataFrame for tide data
    try:
        # Read tide data: skip the header row, assume no header in data, assign column names
        df_tides_raw = pd.read_csv(
            tide_csv_path, skiprows=1, header=None, names=['date_str', 'time_str', 'tide_val'], 
            dtype=str, keep_default_na=False, na_filter=False # Read all as strings initially
        )
        if not df_tides_raw.empty:
            # Create a new DataFrame with parsed datetime and numeric tide values
            df_tides = pd.DataFrame({
                'date_str': df_tides_raw['date_str'], 
                'time_str': df_tides_raw['time_str'],
                'tide': pd.to_numeric(df_tides_raw['tide_val'], errors='coerce') # Convert tide to numeric, coercing errors
            })
            # Create a 'datetime_col' by parsing date and time strings
            df_tides['datetime_col'] = df_tides.apply(lambda r: parse_tide_datetime(r['date_str'], r['time_str']), axis=1)
            # Drop rows where datetime parsing failed or tide value is not numeric
            df_tides.dropna(subset=['datetime_col', 'tide'], inplace=True) 
            
            if not df_tides.empty: 
                # Handle duplicate timestamps by averaging tide values
                if df_tides.duplicated(subset=['datetime_col']).any():
                    print(f"Warning: Duplicate timestamps found in {tide_csv_path}. Averaging 'tide' values for these duplicates.")
                    # Group by datetime and calculate mean tide, then reset index to make 'datetime_col' a column again
                    df_tides = df_tides.groupby('datetime_col', as_index=False).agg({'tide': 'mean'})
                
                # Set the 'datetime_col' as the index for time series operations
                df_tides.set_index('datetime_col', inplace=True)
                # Keep only the 'tide' column
                if 'tide' in df_tides.columns:
                     df_tides = df_tides[['tide']] 
                else: # Should not happen if groupby was successful
                     print(f"Error: 'tide' column unexpectedly lost after processing {tide_csv_path}.")
                     df_tides = pd.DataFrame() # Reset to empty if error occurs
        
        if df_tides.empty: # Check if df_tides is empty after all parsing attempts
            print(f"Warning: Tide data is effectively empty after parsing and cleaning from {tide_csv_path}.")      
    
    except FileNotFoundError: 
        print(f"Error: Tide data file not found - {tide_csv_path}.")
    except Exception as e: 
        print(f"Error processing tide data file {tide_csv_path}: {e}")

    # Resample tide data to hourly frequency and interpolate, similar to storm data
    df_tides_interpolated = pd.DataFrame() # Initialize
    if not df_tides.empty:
        print("Preprocessing tide data (resampling to hourly and interpolating missing values)...")
        try:
            df_tides_resampled = df_tides.resample('h').mean()
            df_tides_interpolated = df_tides_resampled.interpolate(method='linear', limit_direction='both')
            df_tides_interpolated.dropna(inplace=True) # Drop any remaining NaNs in tide column
            if df_tides_interpolated.empty: 
                print(f"Warning: Tide data became empty after resampling/interpolation.")
        except Exception as e: 
            print(f"Error during tide data resampling/interpolation: {e}")
            df_tides_interpolated = pd.DataFrame() # Reset to empty on error
            
    if df_tides_interpolated.empty:
        print("Error: Processed tide data is empty. Cannot proceed as tide data is required for the analysis.")
        return 

    # --- Step 3: Determine Common Timeframe and Merge Data ---
    print("Determining common timeframe for storm and tide data...")
    if df_inputs_interpolated.empty: # Should have been caught earlier, but as a safeguard
        print("Error: Storm data is empty, cannot determine common timeframe."); return
    
    # Find the intersection of the time ranges of the two datasets
    common_start_time = max(df_inputs_interpolated.index.min(), df_tides_interpolated.index.min())
    common_end_time = min(df_inputs_interpolated.index.max(), df_tides_interpolated.index.max())

    # Check if a valid common timeframe exists
    if pd.isna(common_start_time) or pd.isna(common_end_time) or common_start_time >= common_end_time:
        print("Error: No overlapping timeframe found between input storm data and processed tide data.")
        return
    
    print(f"Overall common timeframe for analysis: {common_start_time} to {common_end_time}")
    
    # Join the two DataFrames using an inner join to keep only common timestamps
    df_merged_full_common_period = df_inputs_interpolated.join(df_tides_interpolated, how='inner')
    # Explicitly slice to the calculated common start and end times to be absolutely sure
    df_merged_full_common_period = df_merged_full_common_period[common_start_time:common_end_time] 
    # Drop any rows where 'swh' or 'tide' might still be NaN (e.g., if join or slicing created them)
    df_merged_full_common_period.dropna(subset=['swh', 'tide'], inplace=True) 
            
    if df_merged_full_common_period.empty:
        print("Error: Overall merged DataFrame is empty after joining and filtering for the common period. No data to analyze.")
        return

    # --- Step 4: Identify and Save Major Storms for the FULL Common Period (for CSV output) ---
    print(f"\nIdentifying storm peaks for the full common period ({common_start_time} to {common_end_time}) for CSV output...")
    # Find peaks in SWH data for the entire common period
    full_peaks_indices, _ = find_peaks(
        df_merged_full_common_period['swh'], 
        height=peak_height_threshold_swh, 
        distance=peak_distance_hours
    )
    
    if len(full_peaks_indices) > 0:
        # DataFrame containing all peaks found in the full period
        picos_tempestades_all_full = df_merged_full_common_period.iloc[full_peaks_indices]
        # Filter these peaks to get "major storms"
        principais_tempestades_full = picos_tempestades_all_full[picos_tempestades_all_full['swh'] > major_storm_threshold_swh].copy()

        if not principais_tempestades_full.empty:
            print(f"Found {len(principais_tempestades_full)} major storm peaks in the full common period.")
            print(f"Saving these major storms to {output_storms_txt}...")
            # Define columns for the output CSV
            output_columns = ['swh', 'tide'] 
            for col in ['mwd', 'pp1d', 'wind', 'dwi']: # Add other available data columns
                if col in principais_tempestades_full.columns:
                    output_columns.insert(output_columns.index('tide'), col) # Insert before 'tide'
            columns_to_save = [col for col in output_columns if col in principais_tempestades_full.columns]
            
            # Ensure the index has a name for the CSV output
            if principais_tempestades_full.index.name is None:
                principais_tempestades_full.index.name = 'datetime'
            
            # Save to CSV
            principais_tempestades_full[columns_to_save].to_csv(
                output_storms_txt, sep=',', index=True, index_label='datetime', header=True, 
                float_format='%.2f', na_rep='NaN'
            )
        else:
            print(f"No major storms (swh > {major_storm_threshold_swh}m) found in the full common period for CSV output.")
            with open(output_storms_txt, 'w') as f: f.write(f"No major storms (swh > {major_storm_threshold_swh}m) found in the full common period.\n")
    else:
        print(f"No storm peaks found in the full common period for CSV output.")
        with open(output_storms_txt, 'w') as f: f.write("No storm peaks found in the full common period.\n")
    
    # --- Step 5: Loop through plot_periods_config to generate plots for each specified period ---
    if df_merged_full_common_period.empty: # Should be caught earlier, but as a safeguard
        print("Cannot generate plots as there is no merged data for the common period.")
        return

    # Get the latest date in the entire common dataset to calculate relative periods (e.g., last N years)
    latest_date_in_data = df_merged_full_common_period.index.max()

    for period_conf in plot_periods_config:
        current_df_period = pd.DataFrame() # DataFrame for the current plotting period
        period_start_date = None
        period_end_date = latest_date_in_data # End date is always the latest available date

        if period_conf['years_ago'] is None: # Indicates the full common period
            current_df_period = df_merged_full_common_period.copy()
            period_start_date = df_merged_full_common_period.index.min()
        else: # For relative periods like "last N years"
            try:
                # Calculate the start date by subtracting years from the latest date
                period_start_date = latest_date_in_data - DateOffset(years=period_conf['years_ago'])
                # Ensure the calculated start date is not earlier than the actual start of the common data
                period_start_date = max(period_start_date, df_merged_full_common_period.index.min())
                
                # Slice the full dataset to get data for the current period
                current_df_period = df_merged_full_common_period[period_start_date:period_end_date].copy()
            except Exception as e:
                print(f"Error calculating date range for '{period_conf['title']}': {e}. Skipping this plot.")
                continue # Move to the next plot configuration
        
        # If no data falls into the calculated period, skip plotting for it
        if current_df_period.empty:
            print(f"No data available for the period '{period_conf['title']}' (from {period_start_date} to {period_end_date}). Skipping this plot.")
            continue
        
        print(f"\nProcessing plot for: {period_conf['title']} (Data from {current_df_period.index.min()} to {current_df_period.index.max()})")

        # IMPORTANT: Re-identify peaks specifically for THIS current_df_period
        # This ensures that peak characteristics (like distance) are relative to the zoomed-in period
        peaks_indices_period, _ = find_peaks(
            current_df_period['swh'], 
            height=peak_height_threshold_swh, 
            distance=peak_distance_hours
        )
        
        picos_all_period_df = pd.DataFrame()    # All peaks for this period
        principais_period_df = pd.DataFrame() # Major storms for this period

        if len(peaks_indices_period) > 0:
            picos_all_period_df = current_df_period.iloc[peaks_indices_period]
            principais_period_df = picos_all_period_df[picos_all_period_df['swh'] > major_storm_threshold_swh].copy()
        
        # Call the plotting function for the current period's data
        generate_storm_plot(
            df_plot_data=current_df_period,
            picos_tempestades_all_plot=picos_all_period_df,
            principais_tempestades_plot=principais_period_df,
            base_output_plot_name=base_output_plot_name,
            period_config=period_conf,
            major_storm_threshold_swh=major_storm_threshold_swh,
            label_significant_peaks_threshold_swh=label_significant_peaks_threshold_swh,
            plots_output_dir=plots_output_dir 
        )

    print("\nScript finished successfully.")

if __name__ == '__main__':
    main()
