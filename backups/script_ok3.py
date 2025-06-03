import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime # Import datetime directly
from scipy.signal import find_peaks
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

def parse_tide_datetime(date_str, time_str):
    """Helper function to parse date and time strings for tide data."""
    try:
        return pd.to_datetime(f"{str(date_str).strip()} {str(time_str).strip()}")
    except Exception:
        return pd.NaT

def main():
    # --- Configuration ---
    inputs_csv_path = 'input.csv'       
    tide_csv_path = 'tide-levels.csv'
    
    output_plot_png = 'storm-peaks.png'
    output_plot_pdf = 'storm-peaks.pdf'
    output_storms_txt = 'storm-peaks.txt'
    
    peak_height_threshold_swh = 2.0 
    peak_distance_hours = 72 
    major_storm_threshold_swh = 6.0 

    # --- Load and Preprocess Storm Data (input.csv) ---
    print(f"Loading storm data from {inputs_csv_path}...")
    try:
        df_inputs = pd.read_csv(inputs_csv_path, parse_dates=['datetime'], index_col='datetime')
        if df_inputs.empty:
            print(f"Error: Storm data file {inputs_csv_path} is empty or could not be read.")
            return
    except FileNotFoundError:
        print(f"Error: File not found - {inputs_csv_path}. Please ensure the file exists or update the path.")
        return
    except Exception as e:
        print(f"Error reading {inputs_csv_path}: {e}")
        return

    print("Preprocessing storm data (resampling and interpolating)...")
    # Changed 'H' to 'h' for resampling storm data
    df_inputs_resampled = df_inputs.resample('h').mean()
    df_inputs_interpolated = df_inputs_resampled.interpolate(method='spline', order=2, limit_direction='both')
    df_inputs_interpolated.dropna(subset=['swh'], inplace=True)
    if df_inputs_interpolated.empty:
        print("Error: Storm data is empty after preprocessing.")
        return

    # --- Load and Preprocess Tide Data (tide-levels.csv) ---
    print(f"Loading tide data from {tide_csv_path}...")
    df_tides = pd.DataFrame()
    try:
        df_tides_raw = pd.read_csv(
            tide_csv_path, 
            skiprows=1,      
            header=None,     
            names=['date_str', 'time_str', 'tide_val'], 
            dtype=str,       
            keep_default_na=False, 
            na_filter=False
        )
        
        if not df_tides_raw.empty:
            df_tides = pd.DataFrame({
                'date_str': df_tides_raw['date_str'],
                'time_str': df_tides_raw['time_str'],
                'tide': pd.to_numeric(df_tides_raw['tide_val'], errors='coerce')
            })
            df_tides['datetime_col'] = df_tides.apply(lambda r: parse_tide_datetime(r['date_str'], r['time_str']), axis=1)
            df_tides.dropna(subset=['datetime_col', 'tide'], inplace=True) 

            if not df_tides.empty: 
                if df_tides.duplicated(subset=['datetime_col']).any():
                    print(f"Warning: Duplicate timestamps found in {tide_csv_path}. Averaging 'tide' values.")
                    df_tides['tide'] = pd.to_numeric(df_tides['tide'], errors='coerce')
                    df_tides.dropna(subset=['datetime_col', 'tide'], inplace=True) 
                    if not df_tides.empty:
                        df_tides = df_tides.groupby('datetime_col', as_index=False).agg({'tide': 'mean'})
                    else:
                        print(f"Warning: Tide data empty after trying to handle duplicates in {tide_csv_path}.")
                
                if not df_tides.empty:
                    df_tides.set_index('datetime_col', inplace=True)
                    if 'tide' in df_tides.columns:
                         df_tides = df_tides[['tide']] 
                    else:
                         print(f"Error: 'tide' column lost after processing {tide_csv_path}.")
                         df_tides = pd.DataFrame()
            else: 
                print(f"Warning: Tide data empty after initial parsing from {tide_csv_path}.")
        else: 
            print(f"Warning: Tide data file {tide_csv_path} is empty or only contained a header.")
            
    except FileNotFoundError: 
        print(f"Error: File not found - {tide_csv_path}")
    except Exception as e: 
        print(f"Error processing {tide_csv_path}: {e}")

    # --- Resample tide data to hourly mean and interpolate ---
    df_tides_interpolated = pd.DataFrame()
    if not df_tides.empty:
        print("Preprocessing tide data (resampling and interpolating)...")
        try:
            # Changed 'H' to 'h' for resampling tide data
            df_tides_resampled = df_tides.resample('h').mean()
            df_tides_interpolated = df_tides_resampled.interpolate(method='linear', limit_direction='both')
            df_tides_interpolated.dropna(inplace=True)
            if df_tides_interpolated.empty: 
                print(f"Warning: Tide data is empty after resampling/interpolation.")
        except Exception as e:
            print(f"Error during resampling/interpolation of tide data: {e}")
            df_tides_interpolated = pd.DataFrame()
    else:
        print(f"Info: No tide data to process further (initial loading failed or file was empty).")

    # --- Determine Common Timeframe & Filter by Tide Availability ---
    print("Determining common timeframe and filtering by tide availability...")
    if df_inputs_interpolated.empty:
        print("Error: Storm data is empty, cannot proceed.")
        return
    
    df_merged = pd.DataFrame()
    if df_tides_interpolated.empty:
        print("Error: Processed tide data is empty. Cannot proceed as tide data is required for analysis.")
        return 
    else:
        common_start_time = max(df_inputs_interpolated.index.min(), df_tides_interpolated.index.min())
        common_end_time = min(df_inputs_interpolated.index.max(), df_tides_interpolated.index.max())

        if pd.isna(common_start_time) or pd.isna(common_end_time) or common_start_time >= common_end_time:
            print("Error: No overlapping timeframe found between input storm data and processed tide data.")
            return
        else:
            print(f"Common timeframe: {common_start_time} to {common_end_time}")
            df_inputs_common = df_inputs_interpolated[common_start_time:common_end_time].copy()
            df_tides_common = df_tides_interpolated[common_start_time:common_end_time].copy()
            df_merged_temp = df_inputs_common.join(df_tides_common, how='inner') 
            df_merged = df_merged_temp.dropna(subset=['swh', 'tide']).copy() 
            
            if df_merged.empty:
                print("Error: Merged DataFrame is empty. No data points had both swh and tide after merging for the common period.")
                return

    if 'swh' not in df_merged.columns or df_merged['swh'].isnull().all():
        print("Error: No valid 'swh' data available in the final merged and filtered data.")
        return

    # --- Identify Storm Peaks (only from the common, tide-available period) ---
    print("Identifying storm peaks based on 'swh' from the common, tide-available period...")
    peaks_indices, _ = find_peaks(df_merged['swh'], height=peak_height_threshold_swh, distance=peak_distance_hours)
    
    picos_tempestades_all = pd.DataFrame() 
    principais_tempestades = pd.DataFrame() 

    if len(peaks_indices) == 0:
        print(f"No storm peaks found with swh > {peak_height_threshold_swh}m (distance {peak_distance_hours}hrs) in the common, tide-available period.")
        with open(output_storms_txt, 'w') as f: f.write("No storm peaks found matching criteria in the common, tide-available period.\n")
        print(f"Empty storm list saved to {output_storms_txt}")
    else:
        picos_tempestades_all = df_merged.iloc[peaks_indices]
        principais_tempestades = picos_tempestades_all[picos_tempestades_all['swh'] > major_storm_threshold_swh].copy()

        if principais_tempestades.empty:
            print(f"No major storms (swh > {major_storm_threshold_swh}m) found in the common, tide-available period.")
            with open(output_storms_txt, 'w') as f:
                f.write(f"No major storms (swh > {major_storm_threshold_swh}m) found in the common, tide-available period.\n")
                if not picos_tempestades_all.empty:
                     f.write(f"(Found {len(picos_tempestades_all)} general peaks with swh > {peak_height_threshold_swh}m, but none classified as 'major'.)\n")
            print(f"Storm list (no major storms) saved to {output_storms_txt}")
        else:
            print(f"Found {len(principais_tempestades)} major storm peaks in the common, tide-available period.")
            print(f"Saving major storms to {output_storms_txt}...")
            output_columns = ['swh', 'tide'] 
            for col in ['mwd', 'pp1d', 'wind', 'dwi']: 
                if col in principais_tempestades.columns:
                    output_columns.insert(output_columns.index('tide'), col)
            
            columns_to_save = [col for col in output_columns if col in principais_tempestades.columns]
            # Save as CSV (comma-separated)
            principais_tempestades[columns_to_save].to_csv(
                output_storms_txt, 
                sep=',', 
                index=True, 
                header=True, 
                float_format='%.2f', 
                na_rep='NaN'
            )

    # --- Plot Storm Peaks (based on the common, tide-available period) ---
    print("Generating storm peaks plot...")
    fig, ax1 = plt.subplots(figsize=(15, 7.5)) 

    lines = [] 

    if not df_merged.empty:
        # Plot SWH on the primary y-axis (ax1)
        # zorder=2 to ensure it's above the tide level line
        line_swh, = ax1.plot(df_merged.index, df_merged['swh'], linestyle='-', color='black', label='Significant Wave Height (swh)', linewidth=0.9, zorder=2)
        lines.append(line_swh)
    else:
        print("Warning: No data in df_merged to plot for the base swh timeseries.")

    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Wave Height, swh (m)', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for the tide level
    ax2 = ax1.twinx()
    if not df_merged.empty and 'tide' in df_merged.columns and not df_merged['tide'].dropna().empty:
        # Plot Tide Level on the secondary y-axis (ax2)
        # zorder=1 to place it in the background
        line_tide, = ax2.plot(df_merged.index, df_merged['tide'], linestyle='-', color='lightgray', label='Tide Level', linewidth=1.2, alpha=0.5, zorder=1) 
        lines.append(line_tide)
        
        # Adjust y-limits for the tide axis (ax2)
        tide_min = df_merged['tide'].min()
        tide_max = df_merged['tide'].max()
        tide_range = tide_max - tide_min
        padding = tide_range * 1.0
        ax2.set_ylim(tide_min - padding, tide_max + padding)

    # Configure the secondary y-axis (tide)
    ax2.set_ylabel('Tide Level (m)', fontsize=14, color='dimgray') 
    ax2.tick_params(axis='y', labelcolor='dimgray') 
    ax2.grid(False) 

    # Annotate major storm peaks on ax1 (swh axis)
    if not principais_tempestades.empty:
        # zorder=3 for legend marker, zorder=4 for text, to be above swh and tide lines
        line_peak_marker, = ax1.plot([], [], 'bo', markersize=5, label=f'Major Storm Peaks (swh > {major_storm_threshold_swh:.1f}m)', zorder=3)
        lines.append(line_peak_marker)
        for idx, row in principais_tempestades.iterrows():
            ax1.text(idx, row['swh'], f"{row['swh']:.2f}", size=9, color='blue', ha='center', va='bottom', zorder=4)
    elif not picos_tempestades_all.empty : 
        # zorder=3 for legend marker
        line_minor_peaks, = ax1.plot([],[], 'co', markersize=3, label=f'{len(picos_tempestades_all)} peaks found (none > {major_storm_threshold_swh:.1f}m swh)', zorder=3)
        lines.append(line_minor_peaks)


    if not df_merged.empty:
        ax1.set_xlim([df_merged.index.min(), df_merged.index.max()])
    elif not df_inputs_interpolated.empty: 
        ax1.set_xlim([df_inputs_interpolated.index.min(), df_inputs_interpolated.index.max()])

    two_year_locator = mdates.MonthLocator(interval=24) 
    ax1.xaxis.set_major_locator(two_year_locator)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    monthly_locator = mdates.MonthLocator() 
    ax1.xaxis.set_minor_locator(monthly_locator)
    
    ax1.legend(handles=lines, loc='upper left')
    plt.suptitle('Storm Peaks Analysis', fontsize=20)

    ax1.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on() 
    ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2) 

    fig.autofmt_xdate() 
    fig.tight_layout() 
    
    try:
        plt.savefig(output_plot_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_plot_png}")
        plt.savefig(output_plot_pdf, bbox_inches='tight')
        print(f"Plot saved to {output_plot_pdf}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)
    print("Script finished.")

if __name__ == '__main__':
    main()
