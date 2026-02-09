from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    eta_seconds: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    logs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    log_queue: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)
    subscribers: list["queue.Queue[dict[str, Any]]"] = field(default_factory=list)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        with self._lock:
            job.status = status
            job.updated_at = time.time()

    def add_log(self, job_id: str, message: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        event = {"type": "log", "message": message, "ts": time.time()}
        with self._lock:
            job.logs.append(message)
            job.updated_at = time.time()
            subscribers = list(job.subscribers)
        self._publish_event(job, event, subscribers)

    def update_progress(self, job_id: str, progress: float, eta_seconds: Optional[float] = None) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        progress = max(0.0, min(1.0, float(progress)))
        with self._lock:
            job.progress = progress
            job.eta_seconds = eta_seconds
            job.updated_at = time.time()
            subscribers = list(job.subscribers)
        event = {
            "type": "progress",
            "progress": job.progress,
            "eta_seconds": job.eta_seconds,
            "ts": time.time(),
        }
        self._publish_event(job, event, subscribers)

    def set_error(self, job_id: str, message: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        with self._lock:
            job.status = "error"
            job.error = message
            job.updated_at = time.time()
            subscribers = list(job.subscribers)
        event = {"type": "error", "message": message, "ts": time.time()}
        self._publish_event(job, event, subscribers)

    def set_result(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        with self._lock:
            job.status = "completed"
            job.result = result
            job.updated_at = time.time()
            subscribers = list(job.subscribers)
        event = {"type": "completed", "result": result, "ts": time.time()}
        self._publish_event(job, event, subscribers)

    def subscribe(self, job_id: str) -> Optional["queue.Queue[dict[str, Any]]"]:
        job = self.get_job(job_id)
        if job is None:
            return None
        q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        with self._lock:
            job.subscribers.append(q)
        return q

    def unsubscribe(self, job_id: str, q: "queue.Queue[dict[str, Any]]") -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        with self._lock:
            if q in job.subscribers:
                job.subscribers.remove(q)

    def _publish_event(
        self,
        job: Job,
        event: Dict[str, Any],
        subscribers: Optional[list["queue.Queue[dict[str, Any]]"]] = None,
    ) -> None:
        job.log_queue.put(event)
        for subscriber in subscribers or []:
            subscriber.put(event)
