"""Streamlit GUI for adding new Run configurations to the database."""

from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from db import (
    Analyzer,
    Dataset,
    FlowMatcherConfig,
    Model,
    Property,
    Run,
    Tester,
    Trainer,
    connect_db,
    generate_experiment_id,
)
from nn import MODEL_REGISTRY


@st.cache_resource
def init_db():
    """Initialize database connection."""
    connect_db()


def get_model_defaults(model_class) -> dict:
    """Extract default kwargs from model class __init__ signature."""
    sig = inspect.signature(model_class.__init__)
    defaults = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            value = param.default
            if callable(value):
                continue
            defaults[name] = value
        else:
            defaults[name] = None
    return defaults


def get_document_defaults(doc_class) -> dict:
    """Extract default values from MongoEngine EmbeddedDocument fields."""
    defaults = {}
    for field_name, field_obj in doc_class._fields.items():
        if field_name == "_cls":
            continue
        if field_obj.default is not None:
            defaults[field_name] = field_obj.default
    return defaults


def parse_json_kwargs(text: str) -> tuple[dict | None, str | None]:
    """Parse JSON kwargs with error handling.

    Returns (parsed_dict, error_message).
    """
    if not text.strip():
        return {}, None
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            return None, "JSON must be an object/dict"
        return result, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def render_properties_section(prefix: str) -> list[dict]:
    """Render dynamic property list editor. Returns list of {name, offset, scale} dicts."""
    count_key = f"{prefix}_prop_count"
    if count_key not in st.session_state:
        st.session_state[count_key] = 0

    def _add():
        st.session_state[count_key] += 1

    def _remove(idx):
        n = st.session_state[count_key]
        # Shift widget values from idx+1..n-1 down to idx..n-2
        for j in range(idx, n - 1):
            for field in ("name", "offset", "scale"):
                src_key = f"{prefix}_prop_{field}_{j + 1}"
                dst_key = f"{prefix}_prop_{field}_{j}"
                if src_key in st.session_state:
                    st.session_state[dst_key] = st.session_state[src_key]
        # Clean up last entry keys
        for field in ("name", "offset", "scale"):
            last_key = f"{prefix}_prop_{field}_{n - 1}"
            if last_key in st.session_state:
                del st.session_state[last_key]
        st.session_state[count_key] -= 1

    st.button("Add Property", on_click=_add, key=f"{prefix}_add_prop")

    n = st.session_state[count_key]
    results = []
    for i in range(n):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            name = st.text_input(
                "Name", value="", key=f"{prefix}_prop_name_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
            )
        with col2:
            offset = st.number_input(
                "Offset", value=0.0, format="%.6f",
                key=f"{prefix}_prop_offset_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
            )
        with col3:
            scale = st.number_input(
                "Scale", value=1.0, format="%.6f",
                key=f"{prefix}_prop_scale_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
            )
        with col4:
            if i == 0:
                st.write("")  # spacer for label alignment
            st.button("X", on_click=_remove, args=(i,), key=f"{prefix}_remove_prop_{i}")
        results.append({"name": name, "offset": offset, "scale": scale})

    return results


def render_dataset_section(prefix: str, dataset_defaults: dict) -> dict:
    """Render dataset configuration fields."""
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        path = st.text_input("Path", value="data/bmp-sample", key=f"{prefix}_path")
    with col2:
        index = st.text_input(
            "Index", value="", key=f"{prefix}_index",
            help="ASE index string for frame selection, e.g. ':1000'. Leave empty for all frames.",
        )
    index = index.strip() or None
    with col3:
        batch_size = st.number_input(
            "Batch", value=dataset_defaults.get("batch_size", 4),
            min_value=1, step=1, key=f"{prefix}_batch_size",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        use_init_r_cut = st.toggle(
            "Use Init R Cut",
            value=dataset_defaults.get("init_r_cut") is not None,
            key=f"{prefix}_use_init_r_cut",
        )
    init_r_cut = None
    if use_init_r_cut:
        with col2:
            init_r_cut = st.number_input(
                "R Cut", value=dataset_defaults.get("init_r_cut") or 7.0,
                format="%.2f", key=f"{prefix}_init_r_cut",
                label_visibility="collapsed",
            )
    with col3:
        charge_module = st.text_input(
            "Charge Module", value="",
            key=f"{prefix}_charge_module",
            help="e.g. 'bmp'. Leave empty for none.",
        )
    charge_module = charge_module.strip() or None

    st.divider()
    st.caption("Properties (condition normalization)")
    properties = render_properties_section(prefix)

    return {
        "path": path,
        "index": index,
        "batch_size": batch_size,
        "init_r_cut": init_r_cut,
        "charge_module": charge_module,
        "properties": properties,
    }


def main():
    st.set_page_config(page_title="Charge-Bal - Add Run", page_icon=":test_tube:")
    st.title("Add New Run Configuration")

    init_db()

    # Load defaults
    dataset_defaults = get_document_defaults(Dataset)
    trainer_defaults = get_document_defaults(Trainer)
    tester_defaults = get_document_defaults(Tester)

    # Model & Flow Matcher
    st.header("Model & Flow Matcher")
    model_col, fm_col = st.columns(2)
    with model_col:
        with st.container(border=True):
            model_names = list(MODEL_REGISTRY.keys())
            selected_model = st.selectbox("Model", options=model_names)
            model_cls = MODEL_REGISTRY[selected_model]
            model_defaults = get_model_defaults(model_cls)
            model_kwargs_text = st.text_area(
                "Model Kwargs (JSON)",
                value=json.dumps(model_defaults, indent=2),
                height=250,
            )
    with fm_col:
        with st.container(border=True):
            use_element_dist = st.toggle(
                "Use Element Distribution", value=False, key="use_element_dist",
            )
            element_dist_text = ""
            if use_element_dist:
                element_dist_text = st.text_area(
                    "Element Distribution (JSON list of probabilities, aligned with model element list)",
                    value="[]",
                    height=80,
                    key="element_dist_input",
                )
                noise_sigma = st.number_input(
                    "Noise Sigma",
                    value=1.0,
                    format="%.4f",
                    step=0.1,
                    min_value=0.0,
                    key="noise_sigma",
                    help="Scale of Gaussian noise added to element centers. Default 1.0.",
                )
            else:
                st.caption("Element distribution not enabled")

    # Training Configuration
    st.header("Training Configuration")
    do_train = st.toggle("Enable Training", value=False)

    train_config = {}
    if do_train:
        st.caption("Dataset")
        with st.container(border=True):
            train_dataset = render_dataset_section("train", dataset_defaults)

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                train_epoch = st.number_input(
                    "Epochs", value=trainer_defaults.get("train_epoch") or 100,
                    min_value=0, step=10,
                )
            with col2:
                lr = st.number_input(
                    "Learning Rate", value=trainer_defaults.get("lr") or 1e-4,
                    format="%.6f", step=1e-5,
                )
            with col3:
                save_per_epoch = st.number_input(
                    "Save Per Epoch", value=trainer_defaults.get("save_per_epoch", 10),
                    min_value=1, step=5,
                )

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                analyze_trajectory = st.toggle(
                    "Analyze Traj", value=trainer_defaults.get("analyze_trajectory", False),
                    key="train_analyze_traj",
                )
                traj_n_steps = None
                analysis_temperature = trainer_defaults.get("analysis_temperature", 0.1)
                if analyze_trajectory:
                    traj_n_steps = st.number_input(
                        "Traj Steps", value=trainer_defaults.get("traj_n_steps", 50),
                        min_value=1, step=10, key="train_traj_steps",
                    )
                    analysis_temperature = st.number_input(
                        "Analysis Temperature",
                        value=trainer_defaults.get("analysis_temperature", 0.1),
                        format="%.4f", step=0.01, min_value=0.001,
                        key="train_analysis_temperature",
                        help="Softmax temperature for forward trajectory charge analysis.",
                    )
        with right:
            with st.container(border=True):
                use_checkpoint = st.toggle(
                    "Use Ckpt", value=trainer_defaults.get("use_checkpoint", False),
                    key="train_use_ckpt",
                )
                checkpoint_epoch, checkpoint_run_id = None, None
                if use_checkpoint:
                    checkpoint_epoch = st.number_input(
                        "Ckpt Epoch", value=0, min_value=0, step=1,
                        key="train_ckpt_epoch",
                    )
                    checkpoint_run_id = st.text_input(
                        "Checkpoint Run ID", key="train_ckpt_run_id",
                    )

        train_config = {
            "dataset": train_dataset, "train_epoch": train_epoch, "lr": lr,
            "save_per_epoch": save_per_epoch, "analyze_trajectory": analyze_trajectory,
            "traj_n_steps": traj_n_steps, "analysis_temperature": analysis_temperature,
            "use_checkpoint": use_checkpoint,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_run_id": checkpoint_run_id,
        }

    # Testing Configuration
    st.header("Testing Configuration")
    do_test = st.toggle("Enable Testing", value=False)

    test_config = {}
    if do_test:
        st.caption("Dataset")
        with st.container(border=True):
            test_dataset = render_dataset_section("test", dataset_defaults)

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                n_steps = st.number_input(
                    "N Steps", value=tester_defaults.get("n_steps") or 50,
                    min_value=1, step=10,
                )
        with right:
            with st.container(border=True):
                save_trajectory_test = st.toggle(
                    "Save Traj", value=tester_defaults.get("save_trajectory", False),
                    key="test_save_traj",
                )
                save_index = ""
                if save_trajectory_test:
                    save_index = st.text_input(
                        "Save Index", value=":500",
                        key="test_save_index",
                        help="ASE-style index for per-sample logit saving. e.g. ':500' (first 500), ':' (all).",
                    )

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                use_pcfm = st.toggle(
                    "PCFM Projection", value=tester_defaults.get("use_pcfm", False),
                    key="test_use_pcfm",
                )
                pcfm_temperature = tester_defaults.get("pcfm_temperature", 0.1)
                if use_pcfm:
                    pcfm_temperature = st.number_input(
                        "PCFM Temperature",
                        value=tester_defaults.get("pcfm_temperature", 0.1),
                        format="%.4f", step=0.01, min_value=0.001,
                        key="test_pcfm_temperature",
                    )
        with right:
            with st.container(border=True):
                use_checkpoint_test = st.toggle(
                    "Use Ckpt", value=tester_defaults.get("use_checkpoint", False),
                    key="test_use_ckpt",
                )
                checkpoint_epoch_test, checkpoint_run_id_test = None, None
                if use_checkpoint_test:
                    checkpoint_epoch_test = st.number_input(
                        "Ckpt Epoch", value=0, min_value=0, step=1,
                        key="test_ckpt_epoch",
                    )
                    checkpoint_run_id_test = st.text_input(
                        "Checkpoint Run ID", key="test_ckpt_run_id",
                    )

        test_config = {
            "dataset": test_dataset, "n_steps": n_steps,
            "save_trajectory": save_trajectory_test,
            "save_index": save_index.strip() or None,
            "use_checkpoint": use_checkpoint_test,
            "checkpoint_epoch": checkpoint_epoch_test,
            "checkpoint_run_id": checkpoint_run_id_test,
            "use_pcfm": use_pcfm,
            "pcfm_temperature": pcfm_temperature,
        }

    # Analysis Configuration
    st.header("Analysis Configuration")
    do_analyze = st.toggle("Enable Analysis", value=False)

    analyze_config = {}
    if do_analyze:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                source_run_id = st.text_input(
                    "Source Run ID",
                    key="analyze_source_run_id",
                    help="Run ID to load trajectory data from. Leave empty to use current run.",
                )
            with col2:
                analyzer_defaults = get_document_defaults(Analyzer)
                analyze_temperature = st.number_input(
                    "Analysis Temperature",
                    value=analyzer_defaults.get("analysis_temperature", 0.1),
                    format="%.4f", step=0.01, min_value=0.001,
                    key="analyze_temperature",
                    help="Softmax temperature τ for computing soft charges from raw logits.",
                )

        analyze_config = {
            "source_run_id": source_run_id.strip() or None,
            "analysis_temperature": analyze_temperature,
        }

    # Submit
    st.divider()

    if st.button("Create Run", type="primary"):
        errors = []

        # Parse model kwargs
        model_kwargs, err = parse_json_kwargs(model_kwargs_text)
        if err:
            errors.append(f"Model kwargs: {err}")

        # Parse element distribution
        element_dist = None
        if use_element_dist:
            try:
                element_dist = json.loads(element_dist_text)
                if not isinstance(element_dist, list) or not all(isinstance(v, (int, float)) for v in element_dist):
                    errors.append("Element distribution must be a JSON list of numbers")
                    element_dist = None
                elif len(element_dist) == 0:
                    errors.append("Element distribution list cannot be empty")
                    element_dist = None
                elif any(v < 0 for v in element_dist):
                    errors.append("Element distribution values must be non-negative")
                    element_dist = None
            except json.JSONDecodeError as e:
                errors.append(f"Element distribution: Invalid JSON: {e}")
                element_dist = None

        # Validate required fields
        if do_train and not train_config.get("dataset", {}).get("path"):
            errors.append("Training dataset path is required")
        if do_test and not test_config.get("dataset", {}).get("path"):
            errors.append("Testing dataset path is required")
        if do_test and test_config.get("use_pcfm") and not test_config.get("dataset", {}).get("charge_module"):
            errors.append("PCFM Projection requires a Charge Module to be set in the test dataset")
        if do_analyze and not analyze_config.get("source_run_id"):
            if not do_test:
                errors.append("Analysis without source_run_id requires Testing to be enabled")
            elif not test_config.get("save_trajectory"):
                errors.append("Analysis without source_run_id requires Save Trajectory to be enabled in Testing")
        # Validate properties
        for section_name, config in [("Training", train_config), ("Testing", test_config)]:
            props = config.get("dataset", {}).get("properties", [])
            for i, p in enumerate(props):
                if not p.get("name", "").strip():
                    errors.append(f"{section_name} property {i + 1}: name is required")
                if p.get("scale", 0.0) == 0.0:
                    errors.append(f"{section_name} property {i + 1}: scale cannot be zero")

        if errors:
            for error in errors:
                st.error(error)
        else:
            # Build embedded documents
            def build_properties(prop_list: list[dict]) -> list[Property]:
                return [
                    Property(name=p["name"], offset=p["offset"], scale=p["scale"])
                    for p in prop_list
                ]

            def build_dataset_doc(values: dict) -> Dataset:
                return Dataset(
                    path=values["path"],
                    index=values.get("index"),
                    batch_size=values["batch_size"],
                    init_r_cut=values["init_r_cut"],
                    charge_module=values.get("charge_module"),
                    properties=build_properties(values.get("properties", [])),
                )

            # Build Trainer
            trainer = None
            if do_train:
                trainer = Trainer(
                    dataset=build_dataset_doc(train_config["dataset"]),
                    train_epoch=train_config["train_epoch"],
                    lr=train_config["lr"],
                    save_per_epoch=train_config["save_per_epoch"],
                    analyze_trajectory=train_config["analyze_trajectory"],
                    traj_n_steps=train_config.get("traj_n_steps"),
                    analysis_temperature=train_config.get("analysis_temperature", 0.1),
                    use_checkpoint=train_config["use_checkpoint"],
                    checkpoint_epoch=train_config.get("checkpoint_epoch"),
                    checkpoint_run_id=train_config.get("checkpoint_run_id") or None,
                )

            # Build Tester
            tester = None
            if do_test:
                tester = Tester(
                    dataset=build_dataset_doc(test_config["dataset"]),
                    n_steps=test_config["n_steps"],
                    save_trajectory=test_config["save_trajectory"],
                    save_index=test_config.get("save_index"),
                    use_checkpoint=test_config["use_checkpoint"],
                    checkpoint_epoch=test_config.get("checkpoint_epoch"),
                    checkpoint_run_id=test_config.get("checkpoint_run_id") or None,
                    use_pcfm=test_config.get("use_pcfm", False),
                    pcfm_temperature=test_config.get("pcfm_temperature", 0.1),
                )

            # Build Analyzer
            analyzer = None
            if do_analyze:
                analyzer = Analyzer(
                    source_run_id=analyze_config.get("source_run_id"),
                    analysis_temperature=analyze_config.get("analysis_temperature", 0.1),
                )

            # Build FlowMatcherConfig
            fm_config = None
            if element_dist is not None:
                fm_config = FlowMatcherConfig(
                    element_dist=[float(v) for v in element_dist],
                    noise_sigma=noise_sigma,
                )

            # Create Run
            run = Run(
                id=generate_experiment_id(),
                created_at=datetime.now(timezone.utc),
                model=Model(name=selected_model, kwargs=model_kwargs),
                flow_matcher=fm_config,
                do_train=do_train,
                trainer=trainer,
                do_test=do_test,
                tester=tester,
                do_analyze=do_analyze,
                analyzer=analyzer,
            )

            run.save()

            st.success("Run created successfully!")
            st.code(run.id, language=None)
            st.caption("Copy this Run ID to execute: `python main.py <run_id>`")


if __name__ == "__main__":
    main()
