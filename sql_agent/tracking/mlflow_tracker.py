"""MLflow-backed tracker.

Design notes:
    * Uses ``MlflowClient`` with explicit ``run_id`` everywhere, NOT the
      ``mlflow.start_run()`` context manager. ``start_run`` sets a thread-
      local active run which can surprise concurrent callers — explicit
      run ids are safer under the gRPC thread pool.
    * Runs are nested under a configurable experiment (default: ``sql_agent``).
      The experiment is created on first use if absent.
    * Tracking URI defaults to the project-local filesystem
      (``file:<project_root>/logs/tracking/mlruns``) — no external service.
    * All tracking calls are wrapped in try/except; MLflow failures NEVER
      bubble out of ``start`` / ``finish`` / ``finish_error``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sql_agent.config import get_logger, settings

from .base import summarize_state


_log = get_logger("tracking.mlflow")


@dataclass
class _MLflowHandle:
    start_time_ns: int
    session_id: str
    user_query: str
    run_id: Optional[str]  # None => start() failed silently; finish/finish_error no-op


class MLflowTracker:
    name: str = "mlflow"

    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                "MLflowTracker requires mlflow. Install via: "
                "pip install -r requirements-tracking.txt"
            ) from exc

        uri = (tracking_uri if tracking_uri is not None else settings.mlflow_tracking_uri).strip()
        if not uri:
            # MLflow 3.x deprecated the FileStore backend (Feb 2026). Default
            # to SQLite for production-grade lifecycle tracking. The DB lives
            # under logs/tracking/ next to the run artifacts.
            default_dir = settings.resolved_path("logs/tracking")
            default_dir.mkdir(parents=True, exist_ok=True)
            # Forward-slash URI avoids Windows backslash issues in SQLAlchemy.
            db_path = (default_dir / "mlflow.db").as_posix()
            uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(uri)

        experiment_name = experiment_name or settings.mlflow_experiment_name
        self._client = MlflowClient(tracking_uri=uri)
        # set_experiment creates if missing and is idempotent.
        try:
            mlflow.set_experiment(experiment_name)
            exp = self._client.get_experiment_by_name(experiment_name)
            self._experiment_id: Optional[str] = (
                exp.experiment_id if exp is not None else None
            )
        except Exception as exc:  # pragma: no cover — MLflow race/permissions
            _log.warning("MLflow set_experiment failed (%s); tracker disabled.", exc)
            self._experiment_id = None

        self._uri = uri
        self._experiment_name = experiment_name
        _log.info(
            "MLflowTracker ready: uri=%s experiment=%s experiment_id=%s",
            uri,
            experiment_name,
            self._experiment_id,
        )

    @property
    def tracking_uri(self) -> str:
        return self._uri

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    # ------------------------------------------------------------------

    def start(self, *, session_id: str, user_query: str) -> _MLflowHandle:
        if self._experiment_id is None:
            return _MLflowHandle(
                start_time_ns=time.perf_counter_ns(),
                session_id=session_id,
                user_query=user_query,
                run_id=None,
            )
        try:
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                run_name=f"turn-{session_id[:8]}",
                tags={
                    "session_id": session_id,
                    "sql_agent.tracker": "mlflow",
                },
            )
            return _MLflowHandle(
                start_time_ns=time.perf_counter_ns(),
                session_id=session_id,
                user_query=user_query,
                run_id=run.info.run_id,
            )
        except Exception as exc:
            _log.warning("MLflow create_run failed: %s", exc)
            return _MLflowHandle(
                start_time_ns=time.perf_counter_ns(),
                session_id=session_id,
                user_query=user_query,
                run_id=None,
            )

    def finish(self, handle: _MLflowHandle, final_state: Dict[str, Any]) -> None:
        if handle.run_id is None:
            return
        latency_ms = (time.perf_counter_ns() - handle.start_time_ns) / 1_000_000.0
        summary = summarize_state(
            final_state,
            user_query=handle.user_query,
            session_id=handle.session_id,
            query_max_chars=settings.tracking_query_max_chars,
        )
        summary["metrics"]["latency_ms"] = latency_ms
        status = "FINISHED" if summary["metrics"].get("success", 0.0) >= 1.0 else "FAILED"
        self._log_and_close(handle.run_id, summary, status=status)

    def finish_error(self, handle: _MLflowHandle, exc: BaseException) -> None:
        if handle.run_id is None:
            return
        latency_ms = (time.perf_counter_ns() - handle.start_time_ns) / 1_000_000.0
        summary = summarize_state(
            {},  # empty state; user_query survives via handle
            user_query=handle.user_query,
            session_id=handle.session_id,
            query_max_chars=settings.tracking_query_max_chars,
        )
        summary["metrics"]["latency_ms"] = latency_ms
        summary["metrics"]["success"] = 0.0
        summary["tags"]["error_type"] = "exception"
        summary["artifacts"]["error.txt"] = f"{type(exc).__name__}: {exc}"
        self._log_and_close(handle.run_id, summary, status="FAILED")

    # ------------------------------------------------------------------

    def _log_and_close(
        self,
        run_id: str,
        summary: Dict[str, Any],
        *,
        status: str,
    ) -> None:
        client = self._client
        try:
            for k, v in summary["params"].items():
                # MLflow enforces non-empty param values; skip empties.
                if v is None:
                    continue
                s = str(v)
                if s == "":
                    continue
                try:
                    client.log_param(run_id, k, s)
                except Exception:  # pragma: no cover — MLflow rejects weird chars occasionally
                    pass
            for k, v in summary["metrics"].items():
                try:
                    client.log_metric(run_id, k, float(v))
                except Exception:  # pragma: no cover
                    pass
            for k, v in summary["tags"].items():
                if not v:
                    continue
                try:
                    client.set_tag(run_id, k, str(v))
                except Exception:  # pragma: no cover
                    pass

            # Artifacts: write each as a JSON file under the run's artifact dir.
            import json
            import tempfile
            from pathlib import Path

            for artifact_name, obj in summary["artifacts"].items():
                try:
                    with tempfile.TemporaryDirectory() as td:
                        local_path = Path(td) / artifact_name
                        if artifact_name.endswith(".txt"):
                            local_path.write_text(str(obj), encoding="utf-8")
                        else:
                            local_path.write_text(
                                json.dumps(obj, ensure_ascii=False, default=str, indent=2),
                                encoding="utf-8",
                            )
                        client.log_artifact(run_id, str(local_path))
                except Exception:  # pragma: no cover
                    pass
        finally:
            try:
                client.set_terminated(run_id, status=status)
            except Exception as exc:  # pragma: no cover
                _log.warning("MLflow set_terminated failed: %s", exc)
