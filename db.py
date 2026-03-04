from __future__ import annotations

import argparse
import logging
import os
import secrets
import socket
import time
from datetime import datetime

from dotenv import load_dotenv
from mongoengine import (
    Document,
    EmbeddedDocument,
    connect,
    disconnect,
)
from pymongo.errors import AutoReconnect, ConnectionFailure, NetworkTimeout
from mongoengine.fields import (
    StringField,
    IntField,
    FloatField,
    BooleanField,
    DictField,
    DateTimeField,
    ListField,
    EmbeddedDocumentField,
    EmbeddedDocumentListField,
)


class Property(EmbeddedDocument):
    """Property statistics."""

    name = StringField(required=True)
    offset = FloatField(required=True)
    scale = FloatField(required=True)


class Dataset(EmbeddedDocument):
    """Dataset configuration for one run."""

    path = StringField(required=True)
    index = StringField()  # ASE index string for frame selection, e.g. ":1000"
    batch_size = IntField(default=4)
    properties = EmbeddedDocumentListField(Property)
    init_r_cut = FloatField()  # padded cutoff for neighbor list construction
    charge_module = StringField()  # charge computation module name, e.g. "bmp"
    max_density = FloatField()  # target atoms/A^3; fills with ghost atoms (X, Z=0)


class FlowMatcherConfig(EmbeddedDocument):
    """Flow matching configuration."""

    element_dist = ListField(FloatField())  # categorical probs aligned with model element list
    noise_sigma = FloatField(default=1.0)  # noise scale for element flow matching


class Model(EmbeddedDocument):
    """Vector prediction model configuration."""

    name = StringField(required=True)
    kwargs = DictField()


class ResultEntry(EmbeddedDocument):
    """Results of a single experiment run."""

    timestamp = DateTimeField(required=True)
    metrics = DictField()
    outputs = DictField()


class LogEntry(EmbeddedDocument):
    """A single log entry with timestamp and data."""

    timestamp = DateTimeField(required=True)
    epoch = IntField()
    loss = FloatField()
    data = DictField()


class Trainer(EmbeddedDocument):
    meta = {"strict": False}

    dataset = EmbeddedDocumentField(Dataset)
    train_epoch = IntField(required=True)  # Total number of epoches to train
    lr = FloatField(required=True)
    save_per_epoch = IntField(default=10)
    analyze_trajectory = BooleanField(default=False)  # Save forward trajectory + run charge analysis
    traj_n_steps = IntField(default=50)  # Number of steps for forward trajectory
    analysis_temperature = FloatField(default=0.1)  # Softmax temperature for charge analysis
    use_checkpoint = BooleanField(default=False)  # Whether to load from checkpoint
    checkpoint_epoch = IntField()  # The epoch number to load from
    checkpoint_run_id = StringField()  # Load checkpoint from another run


class Tester(EmbeddedDocument):
    meta = {"strict": False}

    dataset = EmbeddedDocumentField(Dataset)
    n_steps = IntField(required=True)  # Number of denoising steps
    save_trajectory = BooleanField(default=False)
    save_index = StringField()  # ASE-style index string for per-sample logit saving, e.g. ":500"
    use_checkpoint = BooleanField(default=False)  # Whether to load from checkpoint
    checkpoint_epoch = IntField()  # The epoch number to load from
    checkpoint_run_id = StringField()  # Load checkpoint from another run
    use_pcfm = BooleanField(default=False)
    pcfm_temperature = FloatField(default=0.1)
    use_discrete_project = BooleanField(default=False)


class Analyzer(EmbeddedDocument):
    meta = {"strict": False}

    source_run_id = StringField()  # Load saved logits from another run (default: current)


class Run(Document):
    """Run tracking document."""

    meta = {"collection": "runs"}

    id = StringField(primary_key=True)
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    message = StringField()  # Error message for failed runs
    run_on = StringField()  # hostname of machine

    model = EmbeddedDocumentField(Model, required=True)
    flow_matcher = EmbeddedDocumentField(FlowMatcherConfig)
    do_train = BooleanField(default=False)  # Whether to run training
    trainer = EmbeddedDocumentField(Trainer)
    do_test = BooleanField(default=False)  # Whether to run testing
    tester = EmbeddedDocumentField(Tester)
    do_analyze = BooleanField(default=False)  # Whether to run analysis
    analyzer = EmbeddedDocumentField(Analyzer)

    results = EmbeddedDocumentListField(ResultEntry)
    logs = EmbeddedDocumentListField(LogEntry)
    completed_at = DateTimeField()


def connect_db() -> None:
    """Connect to MongoDB using MONGO_URI from .env file."""
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not found in environment variables")
    connect(host=mongo_uri)


def disconnect_db() -> None:
    disconnect()


logger = logging.getLogger(__name__)


def save_run(run: Run, retries: int = 5, base_delay: float = 5.0) -> None:
    """Save a Run document with retry on transient MongoDB errors."""
    transient = (AutoReconnect, ConnectionFailure, NetworkTimeout, OSError)
    for attempt in range(1, retries + 1):
        try:
            run.save()
            return
        except transient as e:
            if attempt == retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("MongoDB save failed (attempt %d/%d): %s. Retrying in %ds...", attempt, retries, e, delay)
            time.sleep(delay)


def generate_experiment_id() -> str:
    """Generate a random 16-character hex ID."""
    return secrets.token_hex(8)
