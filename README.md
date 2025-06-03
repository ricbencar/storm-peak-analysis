# Storm Peak Analysis Script

## Purpose

This script performs an analysis of storm peak events by combining storm wave data with tide level data. It identifies significant storm peaks based on wave height, filters for major storms, and generates plots visualizing these events over various time periods. It also outputs a CSV file listing the major storm events found over the entire common data period.

---

## Input Files

1.  **Storm Data CSV (e.g., 'input.csv'):**
    * **Expected format:** CSV file with a `datetime` column and a `swh` (Significant Wave Height) column, among others (like `mwd`, `pp1d`, `wind`, `dwi`).
    * The `datetime` column should be parsable by pandas (e.g., 'YYYY-MM-DD HH:MM:SS').
    * **Example:**
        ```csv
        datetime,swh,mwd,pp1d,wind,dwi
        1940-01-01 00:00:00,4.318,176.48,9.43,14.81,128.85
        ...
        ```

2.  **Tide Level CSV (e.g., 'tide-levels.csv'):**
    * **Expected format:** CSV file. The script expects data rows to represent date, time, and tide value.
    * It specifically skips the first row (assumed to be a header like "datetime,tide") and then expects three comma-separated values per data line: Date (YYYY-MM-DD), Time (HH:MM), Tide Value (float).
    * **Example (after skipping header):**
        ```csv
        1980-01-01,12:00,3.061828
        1980-01-01,01:00,3.483248
        ...
        ```

---

## Output Files

1.  **PNG Plot(s) (e.g., 'storm-peaks.png', 'storm-peaks-last-10years.png'):**
    * Visualizations of significant wave height (`swh`) and tide levels over specified periods.
    * Storm peaks above certain thresholds are annotated on the plots.
    * Plots are saved in a subdirectory specified by `plots_output_dir` (default: 'plots').

2.  **CSV Data File (e.g., 'storm-peaks.txt'):**
    * A comma-separated values file listing details of "major storms" identified over the entire common period where both storm and tide data are available.
    * Includes `datetime`, `swh`, `tide`, and other available storm parameters.

---

## User Configuration

* File paths for input and output.
* Thresholds for peak detection (general peak height, distance between peaks, major storm height).
* Periods for which plots should be generated (e.g., full period, last 10 years, last 1 year).
* Threshold for labeling significant peaks (default is 5.0m).

---

## Installation of Required Packages

This script requires the following Python packages. You can install them using pip:

```bash
pip install pandas numpy matplotlib scipy
```

---

## Running the Script

1.  Ensure Python is installed on your system.
2.  Install the required packages as listed above.
3.  Place the input CSV files (**'input.csv'**, **'tide-levels.csv'**) in the same directory as the script, or update the `inputs_csv_path` and `tide_csv_path` variables in the script.
4.  Run the script from your terminal: `python your_script_name.py` (e.g., `python storm_analysis_script.py`)
5.  Outputs will be generated in the script's directory and the **'plots'** subdirectory.
