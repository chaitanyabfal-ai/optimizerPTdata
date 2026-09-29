# Gas MOC Optimizer — WSL Runbook

This guide runs the Tkinter desktop application from Ubuntu on WSL, using the project's Python 3.14.4. The project root is `/mnt/c/Users/Swati/Desktop/iPTran Train data`.

## 1. Prerequisites

- Ubuntu under WSL 2, with WSLg or another configured X/Wayland display. This application opens a GUI; it is not a command-line-only program.
- Python 3.14.4 available as `python3` in the Ubuntu terminal.
- The project files, including `Input/` and `requirements.txt`.

From PowerShell, you can open the project in Ubuntu with:

```powershell
wsl.exe -d Ubuntu --cd "/mnt/c/Users/Swati/Desktop/iPTran Train data"
```

Or in an already-open WSL terminal:

```bash
cd "/mnt/c/Users/Swati/Desktop/iPTran Train data"
```

Confirm the current directory and Python version:

```bash
pwd
python3 --version
```

Expected Python output: `Python 3.14.4`.

## 2. Activate the project virtual environment

There is already a WSL virtual environment named `stream` in the project folder, created with Python 3.14.4. Activate it from the project root:

```bash
source stream/bin/activate
```

The shell prompt should show `(stream)`. Confirm the environment is active:

```bash
which python
python --version
```

Use `python` after activation for all remaining Python and pip commands. The existing environment is for Ubuntu/WSL; do not use a Windows virtual environment from WSL or vice versa. If `stream` is missing or damaged, recreate it with `python3 -m venv stream`, then activate it.

If creating the environment reports that `venv` or `ensurepip` is unavailable, install the venv package matching the Python interpreter used for this project (for example, the Ubuntu `python3.14-venv` package if available), then recreate `stream`.

## 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file includes the application's direct third-party packages: NumPy, pandas, Matplotlib, and SciPy. Pip installs their compatible transitive dependencies as well.

## 4. Check GUI/Tk support

The application uses Tkinter and Matplotlib's `TkAgg` backend. Tkinter is not a pip dependency. Test it from the activated environment:

```bash
python -m tkinter
```

A small Tk window should appear. Close that test window before continuing. If the module is missing, install the Tk package compatible with the Python 3.14.4 interpreter. The Ubuntu `python3-tk` package may be built for Ubuntu's system Python rather than a separately installed Python 3.14, so verify it matches your interpreter. If the test window does not appear, resolve the WSLg/display setup before launching the app.

## 5. Launch the application

Keep the virtual environment activated and run:

```bash
python "gas_moc_smart_optimizer_v5 (1).py"
```

The main window is titled **Gas MOC Optimizer**. No command-line arguments are required. If the GUI does not appear, check the Tk test above and the WSLg display configuration.

## 6. Load the project input data

1. In the left panel, click **Load PT CSV (time, pressure_bar)**, or use **File → Load PT Data**.
2. Select `Input/B178_set_1_filt500.csv`. The required column names are exactly `time` and `pressure_bar`.
3. Optionally click **Load Elevation CSV (distance_m, elevation_m)** and select `Input/elevation.csv`. Its columns are `distance_m` and `elevation_m`. Without this optional file, the model assumes a flat pipeline.
4. Check the loaded-data labels and the **Log** tab for confirmation.

Input files are selected through file dialogs; the app does not automatically load them just because they are in the `Input` directory.

## 7. Review parameters before simulation

The left panel contains pipeline geometry, gas/pressure properties, valve timing, numerical settings, diameter-profile choices, and optimizer settings. Confirm these against the experiment before running. In particular:

- `T_total` must exceed the sum of valve open-start, opening, hold-open, and closing durations.
- Keep `beta` within the UI's stated range of 0.5–0.9 unless you intentionally choose otherwise.
- The pressure CSV starts around 36.26 bar. Verify the upstream/downstream boundary settings and sensor interpretation against the measurement setup.
- The supplied config sets roughness to 2 mm, while the UI default is 0.045 mm. Verify the correct physical value.
- With the supplied config, the valve timing totals 219 s (`28 + 8 + 180 + 3`) while `T_total` is 200 s; resolve this mismatch before using those values.
- The supplied config puts the PT at 7950 m and the valve at 8000 m with 100 m grid spacing. The PT is at/adjacent to the valve grid node, which can make diameter optimization insensitive. Verify the actual PT location and use a grid/location that represents it adequately.

To load the supplied settings, choose **File → Load Configuration** and select `Input/common_config_set1.json`; inspect and correct the values above before continuing. Loading this JSON changes parameter fields, not the PT/elevation data selection. It is also fine to use the UI defaults and enter validated experimental values manually.

## 8. Run a forward simulation

1. Ensure the PT CSV is loaded; the app will not run a simulation without it.
2. Select **Uniform** or **Probable Profile** in the optimization settings.
3. Click **Run Simulation**.
4. Review the success dialog and the **Pressure Match**, **Transient Overview**, **Initial Conditions**, and **Log** tabs. Check that the simulated pressure trace, units, valve timing, and RMSE are plausible before attempting a fit.

## 9. Run diameter optimization

1. Verify PT data and all model parameters once more.
2. Set the diameter multiplier bounds, iteration count, population size, optimization method, and worker count.
3. On the first run, use **Serial (1)** workers to reduce multiprocessing complications and CPU load. Larger problems may take significantly longer.
4. Select an optimizer method. `de_then_lbfgsb` is labeled as the recommended global-plus-local method in the UI; `differential_evolution` is the default.
5. Click **Start Optimization**. Follow progress in the status label and **Log** tab. **Stop Optimization** requests a stop; avoid closing the window while it is running unless you intend to terminate it.
6. When complete, review **Pressure Match**, **Convergence**, **Diameter Profile**, **Error Analysis**, and **Segment Data**. Treat the fitted diameters as model estimates and validate them against engineering constraints and the experiment.

## 10. Export results

After optimization completes, use the **File** menu:

- **Export Results** saves measured/model pressure and error to CSV.
- **Export Segment Diameters** saves segment location, elevation, nominal/initial/optimized diameter, and percentage change to CSV.
- **Generate PDF Report** creates a report with plots and a summary.

Choose the `Output/` directory in each save dialog and use distinct filenames if you want to preserve earlier results. Exports are not automatic, and the existing files in `Output/` are not proof that the current run has completed.

## 11. Close the application

Close the GUI normally. To leave the virtual environment, return to the WSL terminal and run:

```bash
deactivate
```

## Quick start (after the first setup)

```bash
cd "/mnt/c/Users/Swati/Desktop/iPTran Train data"
source stream/bin/activate
python "gas_moc_smart_optimizer_v5 (1).py"
```
