from __future__ import annotations

import argparse
import socket
import sys
import traceback
from datetime import datetime, timezone

import torch

from db import Run, connect_db, disconnect_db
from nn import create_model
from nn.layers import OneHotElementEmbedding
from pipeline import FlowMatcher, train, test
from data import create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Charge Balanced Amorphous Material Generation")
    parser.add_argument("run_id", type=str, help="Run ID to execute")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    connect_db()

    run = None
    try:
        run = Run.objects(id=args.run_id).first()
        if run is None:
            print(f"Error: Run {args.run_id!r} not found", file=sys.stderr)
            return 1

        if run.started_at is not None:
            print(f"Error: Run {run.id} already started/finished.", file=sys.stderr)
            return 1

        run.started_at = datetime.now(timezone.utc)
        run.run_on = socket.gethostname()
        run.save()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model_kwargs = dict(run.model.kwargs or {})
        model = create_model(run.model.name, model_kwargs)

        element_embedding = model.element_embedding
        flow_matcher = FlowMatcher(element_embedding=element_embedding)
        print(f"Finished initialization for run {run.id}")

        if run.do_train:
            print(f"Starting training for run {run.id}")
            train_dl = create_dataloader(run.trainer.dataset, device, shuffle=True)
            train(model, flow_matcher, train_dl, run, device)
            print(f"Training completed for run {run.id}")

        if run.do_test:
            print(f"Starting testing for run {run.id}")
            test_dl = create_dataloader(run.tester.dataset, device, shuffle=False)
            test(model, flow_matcher, test_dl, run, device)
            print(f"Testing completed for run {run.id}")

        run.completed_at = datetime.now(timezone.utc)
        run.save()
        print(f"Run {run.id} finished successfully")
        return 0

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if run is not None:
            try:
                run.message = error_msg[:2000]
                run.save()
            except Exception:
                pass
        print(f"Error: {error_msg}", file=sys.stderr)
        return 1

    finally:
        disconnect_db()


if __name__ == "__main__":
    sys.exit(main())
