# AMCharge

**Amorphous Material Generation with Charge Balanced Constraint**

AMCharge is a generative inverse design method for amorphous materials that guarantees the generation of charge balanced samples.
It builds on conditional flow-matching with an E(n)-equivariant graph neural network (EGNN) as the velocity predictor.

## Project Structure

- `main.py` - Entry point: loads a run config, trains/tests/analyzes
- `db.py` - Experiment configuration schema ([MongoDB](https://www.mongodb.com/))
- `data.py` - Sample, Batch, and Dataset classes; ghost atom padding
- `nn/`
  - `velocity_net.py` - EGNN-based velocity predictor
  - `layers.py` - Equivariant graph convolution layers
  - `charge.py` - Charge modules: soft projection (PCFM) and discrete projection (DP)
- `pipeline/`
  - `flow_matching.py` - Conditional flow-matching: flow paths, noise sampling, Euler integration
  - `trainer.py` - Training loop with checkpoint management
  - `tester.py` - Inference with optional charge projections
  - `analyzer.py` - Charge trajectory analysis and visualization
- `helper/`
  - `add_run.py` - [Streamlit](https://streamlit.io/) GUI for creating experiment configurations
  - `find_run.py` - Streamlit GUI for querying and inspecting results
- `scripts/` - [Slurm](https://slurm.schedmd.com/) job submission scripts
- `runtime/` - [Nix](https://nixos.org/) flake and [Singularity](https://sylabs.io/singularity/) container definition

## Setup

### Prerequisites

- Python 3.12
- MongoDB instance (for experiment tracking)

### Installation

With Nix and [direnv](https://direnv.net/) (recommended), entering the project directory automatically sets up the environment.

For SLURM clusters, a Singularity container definition is provided at `runtime/charge-bal.def`.
Build the container image and place it at `runtime/charge-bal.sif`, which the job script uses to run experiments.

For other environments, install dependencies manually:

```sh
uv sync
source .venv/bin/activate
```

### Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```sh
cp .env.example .env
```

`MONGO_URI` is the MongoDB connection string.
`REMOTE_HOSTS` is optional, used by the `sync-remote` helper to sync the project to remote machines.

## Usage

### 1. Create a run

Launch the Streamlit GUI to configure an experiment (model architecture, dataset, training/testing parameters, charge projection settings):

```sh
add-run
```

This opens a web interface on port 8010.

### 2. Execute a run

```sh
python main.py <RUN_ID>
```

On a SLURM cluster:

```sh
./scripts/job <RUN_ID>
```

### 3. Inspect results

```sh
find-run
```

This opens a web interface on port 8011 for querying runs, viewing logs, and downloading outputs.

## Datasets

The paper evaluates AMCharge on two datasets:

- **a-SiO2**: amorphous silica samples (80-250 atoms), generated via MD simulation with the Tersoff potential.
- **MEG**: multi-element glass samples (~800 atoms, 11 element types), generated via melt-quench with the BMP potential.

A small subset of the MEG dataset is included at `data/bmp-sample/` for quick testing.

Datasets are stored as ASE extended XYZ files (`atoms.extxyz`).
Each sample is represented as a tuple of lattice, atomic positions, and element types.
Ghost atoms are used to pad samples to a uniform density, allowing the model to control sample density without modifying the lattice or atom count.
