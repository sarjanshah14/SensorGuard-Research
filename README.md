# SensorGuard-Research

SensorGuard-Research is focused on anomaly detection and localization in cyber-physical systems using the Skoltech Anomaly Benchmark (SKAB) dataset. This repository contains raw data runs, statistical analysis utilities, and visualization modules to study anomalies and changepoints on physical testbeds.

---

## Dataset: SKAB (Skoltech Anomaly Benchmark)

The `SKAB/` directory contains multivariate time-series sensor readings collected from a water circulation testbed. The system monitors parameters such as vibration, flow rates, temperature, and current across the testbed, with artificially induced valve anomalies:

- **`SKAB/anomaly-free/anomaly-free.csv`**: Baseline run under normal conditions (does not contain label columns).
- **`SKAB/valve1/*.csv`**: 16 runs with various anomalies induced at valve 1.
- **`SKAB/valve2/*.csv`**: 4 runs with various anomalies induced at valve 2.
- **`SKAB/other/*.csv`**: 14 runs with other miscellaneous anomalies.

### Column Specifications
The CSV files are semicolon-delimited (`;`) and record data at **1 Hz (1 reading per second)**. The columns are:

1. `datetime`: Timestamp of the sensor reading.
2. `Accelerometer1RMS` & `Accelerometer2RMS`: Vibration measurements (RMS).
3. `Current`: Electrical current draw of the pump system.
4. `Pressure`: System pressure.
5. `Temperature`: Water/system temperature.
6. `Thermocouple`: Thermocouple temperature reading.
7. `Voltage`: Operating voltage.
8. `Volume Flow RateRMS`: Flow rate (RMS).
9. `anomaly`: *[Labeled files only]* Binary label (`0.0` = normal operation, `1.0` = anomalous state).
10. `changepoint`: *[Labeled files only]* Binary label (`0.0` = normal state, `1.0` = transition/changepoint event).

---

## Getting Started

### 1. Environment Setup
The repository includes a pre-configured Python 3.14 virtual environment located in `env/`. 

To activate the environment on macOS/Linux:
```bash
source env/bin/activate
```

*(Ensure your package manager is activated and packages like pandas, matplotlib, and seaborn are available, which are already pre-installed in the `env` folder.)*

---

## Exploring the Data

### 🖥️ Interactive Command-Line Tool (`visualize.py`)
We provide a standalone CLI tool to quickly review, summarize, and visualize any of the SKAB datasets.

#### Interactive Selection
Simply run the script to interactively choose a category and a file from a menu:
```bash
python visualize.py
```

#### Command-Line Arguments
Alternatively, inspect files directly by passing paths:
```bash
# List all discovered SKAB files
python visualize.py --list

# View and generate plots for a specific CSV file
python visualize.py --file SKAB/valve1/0.csv
```

By default, plots are rendered and saved as PNG images under a newly created `plots/` directory:
- **`plots/<category>_<filename>_signals.png`**: Multi-panel signal plots displaying all sensor channels with labeled anomaly periods shaded in red/coral and changepoint markers.
- **`plots/<category>_<filename>_correlation.png`**: Sensor feature correlation heatmaps.

To view the plots on-screen instead of saving them:
```bash
python visualize.py --file SKAB/valve1/0.csv --no-save
```

---

## 📓 Jupyter Notebook Analysis (`explore.ipynb`)

For a richer, cell-by-cell visual exploration, open the `explore.ipynb` notebook. It contains cells configured for:
- Loading dataset groups.
- Displaying summary tables and label statistics.
- Generating interactive stacked plots with highlight spans.
- Comparing correlations during normal operation vs. anomaly events.

Launch JupyterLab to interact with the notebook:
```bash
jupyter lab
# Or open directly in your IDE (such as VS Code or PyCharm)
```
