"""Streamlit GUI for finding and managing Run entries in the database."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from db import Run, connect_db


class RunStatus(Enum):
    """Run status derived from timestamps and message fields."""

    NOT_STARTED = "Not Started"
    RUNNING = "Running"
    FAILED = "Failed"
    FINISHED = "Finished"


def get_run_status(run: Run) -> RunStatus:
    """Derive run status from timestamps and message fields."""
    if run.started_at is None:
        return RunStatus.NOT_STARTED
    if run.message is not None:
        return RunStatus.FAILED
    if run.completed_at is not None:
        return RunStatus.FINISHED
    return RunStatus.RUNNING


def get_status_color(status: RunStatus) -> str:
    """Return color for status badge."""
    return {
        RunStatus.NOT_STARTED: "gray",
        RunStatus.RUNNING: "blue",
        RunStatus.FAILED: "red",
        RunStatus.FINISHED: "green",
    }[status]


def format_datetime(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@st.cache_resource
def init_db():
    """Initialize database connection."""
    connect_db()


@st.cache_data(ttl=60)
def get_unique_model_names() -> list[str]:
    """Get unique model names from all runs."""
    return sorted(Run.objects.distinct("model.name"))


@st.cache_data(ttl=60)
def get_unique_dataset_paths() -> list[str]:
    """Get unique dataset paths from trainer and tester datasets."""
    trainer_paths = Run.objects.distinct("trainer.dataset.path")
    tester_paths = Run.objects.distinct("tester.dataset.path")
    all_paths = set(trainer_paths) | set(tester_paths)
    all_paths.discard(None)
    return sorted(all_paths)


@st.cache_data(ttl=60)
def get_trainer_ranges() -> dict:
    """Get min/max ranges for trainer numeric fields."""
    pipeline = [
        {"$match": {"trainer": {"$exists": True, "$ne": None}}},
        {
            "$group": {
                "_id": None,
                "min_epoch": {"$min": "$trainer.train_epoch"},
                "max_epoch": {"$max": "$trainer.train_epoch"},
                "min_lr": {"$min": "$trainer.lr"},
                "max_lr": {"$max": "$trainer.lr"},
            }
        },
    ]
    result = list(Run.objects.aggregate(pipeline))
    if result:
        r = result[0]
        return {
            "min_epoch": r.get("min_epoch") or 1,
            "max_epoch": r.get("max_epoch") or 1000,
            "min_lr": r.get("min_lr") or 1e-6,
            "max_lr": r.get("max_lr") or 1e-2,
        }
    return {"min_epoch": 1, "max_epoch": 1000, "min_lr": 1e-6, "max_lr": 1e-2}


@st.cache_data(ttl=60)
def get_tester_ranges() -> dict:
    """Get min/max ranges for tester numeric fields."""
    pipeline = [
        {"$match": {"tester": {"$exists": True, "$ne": None}}},
        {
            "$group": {
                "_id": None,
                "min_n_steps": {"$min": "$tester.n_steps"},
                "max_n_steps": {"$max": "$tester.n_steps"},
            }
        },
    ]
    result = list(Run.objects.aggregate(pipeline))
    if result:
        r = result[0]
        return {
            "min_n_steps": r.get("min_n_steps") or 1,
            "max_n_steps": r.get("max_n_steps") or 200,
        }
    return {"min_n_steps": 1, "max_n_steps": 200}


def render_sidebar_filters() -> dict:
    """Render sidebar filters and return filter values."""
    st.sidebar.header("Filters")

    # Status multiselect
    st.sidebar.subheader("Status")
    all_statuses = [s.value for s in RunStatus]
    selected_status_values = st.sidebar.multiselect(
        "Select Status",
        options=all_statuses,
        default=all_statuses,
        key="status_filter",
    )
    selected_statuses = [s for s in RunStatus if s.value in selected_status_values]

    # Model filter
    st.sidebar.subheader("Model Configuration")
    model_names = get_unique_model_names()
    selected_models = st.sidebar.multiselect(
        "Model Name",
        options=model_names,
        default=None,
        key="model_filter",
    )

    # Dataset filter
    st.sidebar.subheader("Dataset Configuration")
    dataset_paths = get_unique_dataset_paths()
    selected_paths = st.sidebar.multiselect(
        "Dataset Path",
        options=dataset_paths,
        default=None,
        key="dataset_path_filter",
    )

    # Trainer filters
    st.sidebar.subheader("Trainer Configuration")
    trainer_ranges = get_trainer_ranges()

    use_epoch_filter = st.sidebar.toggle("Filter by Epoch Range", key="use_epoch_filter")
    epoch_range = None
    if use_epoch_filter:
        min_epoch = trainer_ranges["min_epoch"]
        max_epoch = trainer_ranges["max_epoch"]
        if min_epoch >= max_epoch:
            max_epoch = min_epoch + 1
        epoch_range = st.sidebar.slider(
            "Train Epoch",
            min_value=min_epoch,
            max_value=max_epoch,
            value=(min_epoch, max_epoch),
            key="epoch_range",
        )

    use_lr_filter = st.sidebar.toggle("Filter by Learning Rate", key="use_lr_filter")
    lr_range = None
    if use_lr_filter:
        min_lr = float(trainer_ranges["min_lr"])
        max_lr = float(trainer_ranges["max_lr"])
        if min_lr >= max_lr:
            max_lr = min_lr + 1e-6
        lr_range = st.sidebar.slider(
            "Learning Rate",
            min_value=min_lr,
            max_value=max_lr,
            value=(min_lr, max_lr),
            format="%.6f",
            key="lr_range",
        )

    # Tester filters
    st.sidebar.subheader("Tester Configuration")
    tester_ranges = get_tester_ranges()

    use_n_steps_filter = st.sidebar.toggle("Filter by N Steps", key="use_n_steps_filter")
    n_steps_range = None
    if use_n_steps_filter:
        min_n_steps = tester_ranges["min_n_steps"]
        max_n_steps = tester_ranges["max_n_steps"]
        if min_n_steps >= max_n_steps:
            max_n_steps = min_n_steps + 1
        n_steps_range = st.sidebar.slider(
            "N Steps",
            min_value=min_n_steps,
            max_value=max_n_steps,
            value=(min_n_steps, max_n_steps),
            key="n_steps_range",
        )

    # Refresh button
    st.sidebar.divider()
    if st.sidebar.button("Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    return {
        "statuses": selected_statuses,
        "model_names": selected_models or None,
        "epoch_range": epoch_range,
        "lr_range": lr_range,
        "n_steps_range": n_steps_range,
        "dataset_paths": selected_paths or None,
    }


def build_query(filters: dict) -> list[Run]:
    """Build and execute MongoDB query based on filters."""
    queryset = Run.objects.all()

    # Filter by model name
    if filters["model_names"]:
        queryset = queryset.filter(model__name__in=filters["model_names"])

    # Filter by trainer configs
    if filters["epoch_range"]:
        queryset = queryset.filter(
            trainer__train_epoch__gte=filters["epoch_range"][0],
            trainer__train_epoch__lte=filters["epoch_range"][1],
        )
    if filters["lr_range"]:
        queryset = queryset.filter(
            trainer__lr__gte=filters["lr_range"][0],
            trainer__lr__lte=filters["lr_range"][1],
        )

    # Filter by tester configs
    if filters["n_steps_range"]:
        queryset = queryset.filter(
            tester__n_steps__gte=filters["n_steps_range"][0],
            tester__n_steps__lte=filters["n_steps_range"][1],
        )

    # Get all runs first
    runs = list(queryset.order_by("-created_at"))

    # Post-filter by computed status
    if filters["statuses"]:
        status_set = set(filters["statuses"])
        runs = [r for r in runs if get_run_status(r) in status_set]

    # Post-filter by dataset configs (match if trainer OR tester dataset matches)
    if filters["dataset_paths"]:
        path_set = set(filters["dataset_paths"])
        runs = [
            r
            for r in runs
            if (r.trainer and r.trainer.dataset and r.trainer.dataset.path in path_set)
            or (r.tester and r.tester.dataset and r.tester.dataset.path in path_set)
        ]

    return runs


def render_dataset_section(dataset):
    """Render dataset configuration details."""
    if not dataset:
        st.text("No dataset configured")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Path: {dataset.path or '-'}")
        st.text(f"Batch Size: {dataset.batch_size}")
    with col2:
        st.text(f"Init R Cut: {dataset.init_r_cut or '-'}")
        st.text(f"Charge Module: {dataset.charge_module or '-'}")

    # Properties
    if dataset.properties:
        props_data = [
            {"Name": p.name, "Offset": f"{p.offset:.4f}", "Scale": f"{p.scale:.4f}"}
            for p in dataset.properties
        ]
        st.table(props_data)


def render_run_card(run: Run, index: int):
    """Render a single run as an expandable card."""
    status = get_run_status(run)
    status_color = get_status_color(status)

    # Build expander label with summary
    created_str = run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "-"

    # Build train/test summary
    train_str = ""
    if run.do_train and run.trainer and run.trainer.dataset:
        dataset_name = Path(run.trainer.dataset.path).name if run.trainer.dataset.path else "-"
        train_str = f"Train({dataset_name})"
    test_str = ""
    if run.do_test and run.tester and run.tester.dataset:
        dataset_name = Path(run.tester.dataset.path).name if run.tester.dataset.path else "-"
        test_str = f"Test({dataset_name})"
    task_str = " | ".join(filter(None, [train_str, test_str])) or "No task"

    label = f"`{run.id}` | :{status_color}[{status.value}] | {run.model.name} | {task_str} | {created_str}"

    with st.expander(label, expanded=False):
        # Row 1: Timestamps & Actions
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
        with col1:
            st.text(f"Created: {format_datetime(run.created_at)}")
        with col2:
            st.text(f"Started: {format_datetime(run.started_at)}")
        with col3:
            st.text(f"Completed: {format_datetime(run.completed_at)}")
        with col4:
            st.text(f"Host: {run.run_on or '-'}")
        with col5:
            if status in (RunStatus.NOT_STARTED, RunStatus.RUNNING):
                if st.button("Discard", key=f"discard_{run.id}_{index}", type="secondary"):
                    if run.started_at is None:
                        run.started_at = datetime.now(timezone.utc)
                    run.message = "Manually discarded"
                    run.save()
                    st.cache_data.clear()
                    st.rerun()

        st.divider()

        # Row 2: Model Configuration
        st.markdown("##### Model Configuration")
        st.markdown(f"**Model:** `{run.model.name}`")
        if run.model.kwargs:
            st.json(run.model.kwargs, expanded=1)

        # Row 3: Trainer Config
        if run.do_train and run.trainer:
            st.divider()
            st.markdown("##### Trainer Configuration")
            t = run.trainer
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(f"Epochs: {t.train_epoch}")
                st.text(f"Learning Rate: {t.lr:.6f}")
            with col2:
                st.text(f"Save Per Epoch: {t.save_per_epoch}")
                st.text(f"Save Trajectory: {t.save_trajectory}")
                if t.save_trajectory:
                    st.text(f"Traj N Steps: {t.traj_n_steps}")
            with col3:
                st.text(f"Use Checkpoint: {t.use_checkpoint}")
                if t.use_checkpoint:
                    st.text(f"Checkpoint Epoch: {t.checkpoint_epoch or '-'}")
                    st.text(f"Checkpoint Run ID: {t.checkpoint_run_id or '-'}")

            with st.expander("Trainer Dataset", expanded=False):
                render_dataset_section(t.dataset)

        # Row 4: Tester Config
        if run.do_test and run.tester:
            st.divider()
            st.markdown("##### Tester Configuration")
            t = run.tester
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"N Steps: {t.n_steps}")
                st.text(f"Save Trajectory: {t.save_trajectory}")
                st.text(f"Analyze Trajectory: {t.analyze_trajectory}")
            with col2:
                st.text(f"Use Checkpoint: {t.use_checkpoint}")
                if t.use_checkpoint:
                    st.text(f"Checkpoint Epoch: {t.checkpoint_epoch or '-'}")
                    st.text(f"Checkpoint Run ID: {t.checkpoint_run_id or '-'}")

            with st.expander("Tester Dataset", expanded=False):
                render_dataset_section(t.dataset)

        # Row 5: Logs & Results
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            logs_count = len(run.logs) if run.logs else 0
            st.markdown(f"##### Logs ({logs_count} entries)")
            if run.logs and len(run.logs) > 0:
                with st.expander("View Logs"):
                    show_log_data = st.checkbox("Show data", key=f"show_log_data_{run.id}_{index}")
                    for log in run.logs:
                        epoch_str = f"Epoch {log.epoch}" if log.epoch is not None else "-"
                        loss_str = f"Loss {log.loss:.6f}" if log.loss is not None else "-"
                        st.text(f"{epoch_str} | {loss_str} | {format_datetime(log.timestamp)}")
                        if show_log_data and log.data:
                            st.json(log.data)

        with col2:
            results_count = len(run.results) if run.results else 0
            st.markdown(f"##### Results ({results_count} entries)")
            if run.results and len(run.results) > 0:
                with st.expander("View Results"):
                    for i, result in enumerate(run.results):
                        st.text(f"Result {i+1} - {format_datetime(result.timestamp)}")
                        if result.metrics:
                            st.json(result.metrics)
                        if result.outputs:
                            st.json(result.outputs)

        # Row 6: Error Message (if failed)
        if status == RunStatus.FAILED and run.message:
            st.divider()
            st.markdown("##### Error Message")
            st.error(run.message)


def main():
    st.set_page_config(page_title="Find Runs", layout="wide")
    st.title("Find and Manage Runs")

    init_db()

    # Render sidebar filters
    filters = render_sidebar_filters()

    # Build and execute query
    runs = build_query(filters)

    # Display results
    st.subheader(f"Results ({len(runs)} runs)")

    if not runs:
        st.info("No runs match the current filters.")
    else:
        for i, run in enumerate(runs):
            render_run_card(run, i)


if __name__ == "__main__":
    main()
