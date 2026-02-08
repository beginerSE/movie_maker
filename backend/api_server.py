from __future__ import annotations

import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.job_manager import JobManager
from new_video_gui11 import generate_video

app = FastAPI(title="News Short Generator Studio API")
manager = JobManager()


class VideoGenerateRequest(BaseModel):
    api_key: str
    script_path: str
    image_paths: List[str]
    use_bgm: bool = False
    bgm_path: str = ""
    bgm_gain_db: float = 0.0
    output_dir: str
    width: int = 1080
    height: int = 1920
    fps: int = 30
    voice_name: str = "Kore"
    tts_engine: str = "Gemini"
    vv_mode: str = "rotation"
    vv_rotation_labels: Optional[List[str]] = None
    vv_caster_label: str = "四国めたん"
    vv_analyst_label: str = "ずんだもん"
    vv_base_url: str = "http://127.0.0.1:50021"
    vv_speed_scale: float = 1.0
    caption_font_size: int = 36
    speaker_font_size: int = 30
    caption_max_chars: int = 22
    caption_box_alpha: int = 170
    caption_box_enabled: bool = True
    caption_box_height: int = 420
    bg_off_style: str = "shadow"
    caption_text_color: str = "#FFFFFF"


class JobResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    eta_seconds: Optional[float]
    error: Optional[str]
    result: dict


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/video/generate", response_model=JobResponse)
async def generate_video_job(payload: VideoGenerateRequest) -> JobResponse:
    job = manager.create_job()
    manager.update_status(job.job_id, "running")

    def log_fn(message: str) -> None:
        manager.add_log(job.job_id, message)

    def progress_fn(value: float, eta_seconds: Optional[float] = None) -> None:
        manager.update_progress(job.job_id, value, eta_seconds)

    def worker() -> None:
        try:
            generate_video(
                api_key=payload.api_key,
                script_path=payload.script_path,
                image_paths=payload.image_paths,
                use_bgm=payload.use_bgm,
                bgm_path=payload.bgm_path,
                bgm_gain_db=payload.bgm_gain_db,
                output_dir=payload.output_dir,
                width=payload.width,
                height=payload.height,
                fps=payload.fps,
                voice_name=payload.voice_name,
                log_fn=log_fn,
                progress_fn=progress_fn,
                tts_engine=payload.tts_engine,
                vv_mode=payload.vv_mode,
                vv_rotation_labels=payload.vv_rotation_labels,
                vv_caster_label=payload.vv_caster_label,
                vv_analyst_label=payload.vv_analyst_label,
                vv_base_url=payload.vv_base_url,
                vv_speed_scale=payload.vv_speed_scale,
                caption_font_size=payload.caption_font_size,
                speaker_font_size=payload.speaker_font_size,
                caption_max_chars=payload.caption_max_chars,
                caption_box_alpha=payload.caption_box_alpha,
                caption_box_enabled=payload.caption_box_enabled,
                caption_box_height=payload.caption_box_height,
                bg_off_style=payload.bg_off_style,
                caption_text_color=payload.caption_text_color,
            )
            manager.set_result(job.job_id, {"message": "completed"})
        except Exception as exc:
            manager.set_error(job.job_id, str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return JobResponse(job_id=job.job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        eta_seconds=job.eta_seconds,
        error=job.error,
        result=job.result,
    )


@app.websocket("/ws/jobs/{job_id}")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    job = manager.get_job(job_id)
    if job is None:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            event = job.log_queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
