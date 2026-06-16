# Racing Sim

Simpel autonomous vehicle racing simulation with PID, pure pursuit, and acados-based MPC controllers.

## Setup

Run commands from the project root unless noted otherwise.

1. Install acados by following the official installation guide:

   https://docs.acados.org/installation/index.html

2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv tswr_venv
   source tswr_venv/bin/activate
   ```

3. Install Python dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

4. Edit the paths in `activate_acados_example.sh`:

   ```bash
   source /absolute/path/to/your/.venv/bin/activate
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/absolute/path/to/acados/lib"
   export ACADOS_SOURCE_DIR="/absolute/path/to/acados"
   ```

   Point the virtual environment path to your project venv, and point the acados paths to your local acados installation.

5. Source the acados activation script:

   ```bash
   source activate_acados_example.sh
   ```

6. Install this package in editable mode:

   ```bash
   pip3 install -e .
   ```

## Running

Run scripts from the project root, otherwise acados C generated code will generate in multiple places.

Examples:

```bash
python scripts/progress_mpc.py
```
