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
    """Render dataset configuration as a compact bordered card."""
    if not dataset:
        st.caption("No dataset configured")
        return

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
        with c1:
            st.markdown(f"**Path** `{dataset.path or '-'}`")
        with c2:
            st.markdown(f"**Index** `{dataset.index or 'all'}`")
        with c3:
            st.markdown(f"**Batch** `{dataset.batch_size}`")
        with c4:
            st.markdown(f"**R Cut** `{dataset.init_r_cut or '-'}`")
        with c5:
            st.markdown(f"**Charge** `{dataset.charge_module or '-'}`")

        if dataset.properties:
            props_data = [
                {"Name": p.name, "Offset": f"{p.offset:.4f}", "Scale": f"{p.scale:.4f}"}
                for p in dataset.properties
            ]
            st.dataframe(
                props_data,
                width="stretch",
                hide_index=True,
                height=min(35 + 35 * len(props_data), 200),
            )


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
        # Header: compact metadata line + discard button
        meta_col, action_col = st.columns([9, 1])
        with meta_col:
            st.markdown(
                f"**Created** `{format_datetime(run.created_at)}` · "
                f"**Started** `{format_datetime(run.started_at)}` · "
                f"**Completed** `{format_datetime(run.completed_at)}` · "
                f"**Host** `{run.run_on or '-'}`"
            )
        with action_col:
            if status in (RunStatus.NOT_STARTED, RunStatus.RUNNING):
                if st.button("Discard", key=f"discard_{run.id}_{index}", type="secondary"):
                    if run.started_at is None:
                        run.started_at = datetime.now(timezone.utc)
                    run.message = "Manually discarded"
                    run.save()
                    st.cache_data.clear()
                    st.rerun()

        # Error message shown immediately for failed runs
        if status == RunStatus.FAILED and run.message:
            st.error(run.message)

        # Build dynamic tab list
        tab_names = ["Configuration"]
        if run.do_train and run.trainer:
            tab_names.append("Training")
        if run.do_test and run.tester:
            tab_names.append("Testing")
        tab_names.append("Output")

        tabs = st.tabs(tab_names)
        tab_idx = 0

        # --- Configuration tab ---
        with tabs[tab_idx]:
            tab_idx += 1
            model_col, fm_col = st.columns(2)
            with model_col:
                with st.container(border=True):
                    st.markdown(f"**Model** · `{run.model.name}`")
                    if run.model.kwargs:
                        st.json(run.model.kwargs, expanded=False)
            with fm_col:
                with st.container(border=True):
                    if run.flow_matcher and run.flow_matcher.element_dist:
                        st.markdown("**Flow Matcher**")
                        st.markdown(f"Element Distribution: `{run.flow_matcher.element_dist}`")
                    else:
                        st.markdown("**Flow Matcher** · *not configured*")

        # --- Training tab ---
        if run.do_train and run.trainer:
            with tabs[tab_idx]:
                tab_idx += 1
                t = run.trainer
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"**Epochs** `{t.train_epoch}`  \n**LR** `{t.lr:.6f}`")
                    with c2:
                        st.markdown(f"**Save/Epoch** `{t.save_per_epoch}`")
                    with c3:
                        traj = f"**Analyze Traj** `{'Yes' if t.analyze_trajectory else 'No'}`"
                        if t.analyze_trajectory:
                            traj += f"  \n**Traj Steps** `{t.traj_n_steps}`"
                            traj += f"  \n**Analysis Temp** `{getattr(t, 'analysis_temperature', 0.1):.4f}`"
                        st.markdown(traj)
                    with c4:
                        ckpt = f"**Checkpoint** `{'Yes' if t.use_checkpoint else 'No'}`"
                        if t.use_checkpoint:
                            ckpt += f"  \n**Ckpt Epoch** `{t.checkpoint_epoch or '-'}`"
                            ckpt += f"  \n**Ckpt Run** `{t.checkpoint_run_id or '-'}`"
                        st.markdown(ckpt)

                st.caption("Dataset")
                render_dataset_section(t.dataset)

        # --- Testing tab ---
        if run.do_test and run.tester:
            with tabs[tab_idx]:
                tab_idx += 1
                t = run.tester
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"**N Steps** `{t.n_steps}`")
                    with c2:
                        lines = [f"**Save Traj** `{'Yes' if t.save_trajectory else 'No'}`"]
                        if getattr(t, "use_pcfm", False):
                            lines.append(f"**PCFM** `Yes`  \n**PCFM Temp** `{getattr(t, 'pcfm_temperature', 0.1):.4f}`")
                        else:
                            lines.append("**PCFM** `No`")
                        st.markdown("  \n".join(lines))
                    with c3:
                        lines = [f"**Analyze Traj** `{'Yes' if t.analyze_trajectory else 'No'}`"]
                        if t.analyze_trajectory:
                            lines.append(f"**Analysis Temp** `{getattr(t, 'analysis_temperature', 0.1):.4f}`")
                        st.markdown("  \n".join(lines))
                    with c4:
                        ckpt = f"**Checkpoint** `{'Yes' if t.use_checkpoint else 'No'}`"
                        if t.use_checkpoint:
                            ckpt += f"  \n**Ckpt Epoch** `{t.checkpoint_epoch or '-'}`"
                            ckpt += f"  \n**Ckpt Run** `{t.checkpoint_run_id or '-'}`"
                        st.markdown(ckpt)

                st.caption("Dataset")
                render_dataset_section(t.dataset)

        # --- Output tab ---
        with tabs[tab_idx]:
            logs_col, results_col = st.columns(2)
            with logs_col:
                with st.container(border=True):
                    logs_count = len(run.logs) if run.logs else 0
                    st.markdown(f"**Logs** ({logs_count} entries)")
                    if run.logs and len(run.logs) > 0:
                        log_rows = [
                            {
                                "Epoch": log.epoch if log.epoch is not None else "-",
                                "Loss": f"{log.loss:.6f}" if log.loss is not None else "-",
                                "Time": format_datetime(log.timestamp),
                            }
                            for log in run.logs
                        ]
                        st.dataframe(log_rows, width="stretch", hide_index=True)
                        if st.checkbox("Show log data", key=f"log_data_{run.id}_{index}"):
                            for log in run.logs:
                                if log.data:
                                    st.json(log.data, expanded=False)

            with results_col:
                with st.container(border=True):
                    results_count = len(run.results) if run.results else 0
                    st.markdown(f"**Results** ({results_count} entries)")
                    if run.results and len(run.results) > 0:
                        for i, result in enumerate(run.results):
                            st.markdown(f"*Result {i + 1}* · {format_datetime(result.timestamp)}")
                            if result.metrics:
                                st.json(result.metrics, expanded=False)
                            if result.outputs:
                                st.json(result.outputs, expanded=False)


def main():
    st.set_page_config(page_title="Charge-Bal - Find Runs", layout="wide", page_icon=":mag:")
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
