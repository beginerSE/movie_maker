from __future__ import annotations

import base64
import asyncio
import configparser
import copy
import json
import logging
import mimetypes
import os
import pathlib
import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path


from pathlib import Path

def app_root() -> Path:
    # PyInstaller exe で動いているときは exe のあるフォルダ
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # 通常実行（python api_server.py）のときはこのファイルのフォルダ
    return Path(__file__).resolve().parent

APP_ROOT = app_root()

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.job_manager import JobManager
from backend.video_core import (
    CAPTION_MARGIN_BOTTOM,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_PONCHI_GEMINI_MODEL,
    DEFAULT_SCRIPT_GEMINI_MODEL,
    DEFAULT_SCRIPT_OPENAI_MODEL,
    DEFAULT_TITLE_MAX_TOKENS,
    GEMINI_MATERIAL_DEFAULT_MODEL,
    extract_script_text,
    generate_materials_with_gemini,
    generate_ponchi_suggestions_with_gemini,
    generate_script_with_claude,
    generate_script_with_gemini,
    generate_script_with_openai,
    generate_video,
    parse_srt_file,
    resolve_gemini_material_model,
    fetch_voicevox_speakers,
)

class StripApiPrefixMiddleware:
    """Allow clients to access endpoints with or without an `/api` prefix."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if path == "/api" or path.startswith("/api/"):
            stripped = path[4:] or "/"
            scope = {**scope, "path": stripped}
        await self.app(scope, receive, send)


app = FastAPI(title="News Short Generator Studio API")
app.add_middleware(StripApiPrefixMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = JobManager()
logger = logging.getLogger("movie_maker.api")
if not logger.handlers:
    log_path = ROOT_DIR / "backend" / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)  # ★追加：backend/ を作る

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

APP_DATA_DIR_NAME = "NewsShortGeneratorStudio"
DEFAULT_PROJECT_ID = "default"


def user_data_root() -> Path:
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / APP_DATA_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / APP_DATA_DIR_NAME
    return Path.home() / ".local" / "share" / APP_DATA_DIR_NAME


DATA_ROOT = user_data_root()
PROJECTS_ROOT = DATA_ROOT / "projects"


def _slugify_project_id(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or "project"


def _project_dir(project_id: str) -> Path:
    safe_id = _slugify_project_id(project_id)
    return PROJECTS_ROOT / safe_id


def _project_meta_path(project_id: str) -> Path:
    return _project_dir(project_id) / "project.json"


def _project_settings_path(project_id: str) -> Path:
    return _project_dir(project_id) / "settings.json"


def _ensure_project_structure(
    project_id: str,
    display_name: Optional[str] = None,
    project_type: Optional[str] = None,
) -> Path:
    project_id = _slugify_project_id(project_id)
    normalized_type = _normalize_project_type(project_type)
    project_dir = _project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    for name in ["scripts", "materials", "outputs", "logs"]:
        (project_dir / name).mkdir(parents=True, exist_ok=True)
    meta_path = _project_meta_path(project_id)
    if not meta_path.exists():
        meta = {
            "id": project_id,
            "name": display_name or ("Default" if project_id == DEFAULT_PROJECT_ID else project_id),
            "project_type": normalized_type,
            "flow_state": _default_flow_state() if normalized_type == "flow" else {},
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        meta = _read_project_meta(project_id)
        should_write = False
        if project_type is not None and meta.get("project_type") != normalized_type:
            meta["project_type"] = normalized_type
            should_write = True
        if meta.get("project_type") == "flow":
            normalized_flow_state = _normalize_flow_state(meta.get("flow_state"))
            if meta.get("flow_state") != normalized_flow_state:
                meta["flow_state"] = normalized_flow_state
                should_write = True
        if should_write:
            _write_project_meta(project_id, meta)
    settings_path = _project_settings_path(project_id)
    if not settings_path.exists():
        settings_path.write_text("{}", encoding="utf-8")
    return project_dir


def _read_project_meta(project_id: str) -> dict[str, Any]:
    meta_path = _project_meta_path(project_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="project not found")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data.setdefault("id", _slugify_project_id(project_id))
    data.setdefault("name", data["id"])
    data["project_type"] = _normalize_project_type(data.get("project_type"))
    if data["project_type"] == "flow":
        data["flow_state"] = _normalize_flow_state(data.get("flow_state"))
    else:
        data["flow_state"] = {}
    return data


def _write_project_meta(project_id: str, data: dict[str, Any]) -> None:
    data = copy.deepcopy(data)
    data["id"] = _slugify_project_id(project_id)
    data["project_type"] = _normalize_project_type(data.get("project_type"))
    if data["project_type"] == "flow":
        data["flow_state"] = _normalize_flow_state(data.get("flow_state"))
    else:
        data["flow_state"] = {}
    data["updated_at"] = int(time.time())
    _project_meta_path(project_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_default_project() -> None:
    _ensure_project_structure(DEFAULT_PROJECT_ID, "Default")


def _resolve_project_id(value: Optional[str]) -> str:
    project_id = _slugify_project_id(value or DEFAULT_PROJECT_ID)
    _ensure_project_structure(project_id)
    return project_id


def _project_output_dir(project_id: str) -> str:
    return str(_project_dir(project_id) / "outputs")


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_model_config() -> dict[str, Any]:
    default_thumbnail = [GEMINI_MATERIAL_DEFAULT_MODEL]
    default_ponchi = [DEFAULT_PONCHI_GEMINI_MODEL]
    config_paths = [
        ROOT_DIR / "condig.ini",
        ROOT_DIR / "config.ini",
        APP_ROOT / "condig.ini",
        APP_ROOT / "config.ini",
    ]
    parser = configparser.ConfigParser()
    loaded = parser.read([str(path) for path in config_paths if path.exists()], encoding="utf-8")
    if not loaded:
        return {
            "thumbnail_models": default_thumbnail,
            "thumbnail_default": default_thumbnail[0],
            "ponchi_models": default_ponchi,
            "ponchi_default": default_ponchi[0],
        }

    thumb_models = _split_csv(parser.get("thumbnail", "models", fallback=""))
    thumb_default = parser.get("thumbnail", "default_model", fallback="").strip()
    ponchi_models = _split_csv(parser.get("ponchi", "models", fallback=""))
    ponchi_default = parser.get("ponchi", "default_model", fallback="").strip()
    if not thumb_models:
        thumb_models = default_thumbnail
    if not ponchi_models:
        ponchi_models = default_ponchi
    if thumb_default not in thumb_models:
        thumb_default = thumb_models[0]
    if ponchi_default not in ponchi_models:
        ponchi_default = ponchi_models[0]
    return {
        "thumbnail_models": thumb_models,
        "thumbnail_default": thumb_default,
        "ponchi_models": ponchi_models,
        "ponchi_default": ponchi_default,
    }


MODEL_CONFIG = _load_model_config()

FLOW_STEPS = [
    "script",
    "base_video",
    "title_description",
    "thumbnail",
    "ponchi",
    "final_edit",
]
FLOW_STEP_STATUSES = {"未着手", "編集中", "完了"}


def _default_flow_state() -> dict[str, str]:
    return {step: "未着手" for step in FLOW_STEPS}


def _normalize_project_type(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower()
    return "flow" if normalized == "flow" else "standard"


def _normalize_flow_state(value: Any) -> dict[str, str]:
    default_state = _default_flow_state()
    if not isinstance(value, dict):
        return default_state
    normalized: dict[str, str] = {}
    for step in FLOW_STEPS:
        status = value.get(step)
        normalized[step] = status if status in FLOW_STEP_STATUSES else "未着手"
    return normalized


@app.middleware("http")
async def log_request_errors(request: Request, call_next) -> Response:
    try:
        response = await call_next(request)
    except HTTPException as exc:
        logger.warning(
            "HTTP error on %s %s: %s",
            request.method,
            request.url.path,
            exc.detail,
        )
        raise
    except Exception as exc:
        logger.exception(
            "Unhandled error on %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "path": request.url.path,
                "error": str(exc),
            },
        )
    if response.status_code >= 500:
        logger.error(
            "Server error response on %s %s: status=%s",
            request.method,
            request.url.path,
            response.status_code,
        )
    return response


def _normalize_engine(value: str) -> str:
    return (value or "").strip().lower()


def _build_title_desc_prompt(script_text: str, count: int, extra: str) -> str:
    extra_block = f"\n[追加指示]\n{extra.strip()}\n" if extra.strip() else ""
    return (
        "あなたはYouTube動画の企画編集者です。\n"
        "以下の台本をもとに、YouTubeでバズりそうでクリックされやすいタイトル案と説明文を作成してください。\n\n"
        "[条件]\n"
        f"- タイトル案は {count} 個\n"
        "- 日本語\n"
        "- 誇張しすぎや誤解を招く表現は避ける\n"
        "- 文字数は短め（30〜45文字程度）\n"
        "- 台本の内容が伝わるように\n\n"
        "[出力形式]\n"
        "タイトル案:\n"
        "1. ...\n"
        "2. ...\n"
        "...\n"
        "説明文:\n"
        "- 2〜4段落で読みやすく\n"
        "- 冒頭にフック\n"
        "- 最後にチャンネル登録の一言\n"
        f"{extra_block}\n"
        "[台本]\n"
        f"{script_text}\n"
    )


def _normalize_ponchi_suggestions(
    suggestions: List[dict],
    items: List[dict],
) -> List[dict]:
    normalized: List[dict] = []
    for idx, item in enumerate(items, 1):
        suggestion = suggestions[idx - 1] if idx - 1 < len(suggestions) else {}
        start = suggestion.get("start") or item.get("start")
        end = suggestion.get("end") or item.get("end")
        text = suggestion.get("text") or item.get("text", "")
        visual = suggestion.get("visual_suggestion") or ""
        prompt = suggestion.get("image_prompt") or visual or text
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "visual_suggestion": visual,
                "image_prompt": prompt,
            }
        )
    return normalized


def _ponchi_ideas_path(output_dir: Optional[str], srt_path: str) -> Optional[pathlib.Path]:
    if not output_dir:
        return None
    srt_stem = pathlib.Path(srt_path).stem
    return pathlib.Path(output_dir) / srt_stem / f"{srt_stem}_ponchi_ideas.json"


def _validate_video_request(payload: "VideoGenerateRequest") -> None:
    if not payload.script_path:
        raise HTTPException(status_code=400, detail="script_path が空です。")
    script_path = pathlib.Path(payload.script_path)
    if not script_path.exists():
        raise HTTPException(status_code=400, detail=f"script_path が見つかりません: {payload.script_path}")
    if not payload.image_paths:
        raise HTTPException(status_code=400, detail="image_paths が空です。")
    missing_images = [path for path in payload.image_paths if not pathlib.Path(path).exists()]
    if missing_images:
        detail = "image_paths が見つかりません: " + ", ".join(missing_images[:5])
        if len(missing_images) > 5:
            detail += " ..."
        raise HTTPException(status_code=400, detail=detail)


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
    caption_margin_bottom: int = CAPTION_MARGIN_BOTTOM
    caption_box_alpha: int = 170
    caption_box_enabled: bool = True
    caption_box_height: int = 420
    bg_off_style: str = "shadow"
    caption_text_color: str = "#FFFFFF"
    project_id: Optional[str] = None


class ScriptGenerateRequest(BaseModel):
    api_key: str
    provider: str = "Gemini"
    prompt: str
    model: str = DEFAULT_SCRIPT_GEMINI_MODEL
    max_tokens: Optional[int] = None


class ScriptGenerateResponse(BaseModel):
    text: str


class TitleGenerateRequest(BaseModel):
    api_key: str
    provider: str = "Gemini"
    script_path: str
    count: int = 5
    extra: str = ""
    model: str = DEFAULT_SCRIPT_GEMINI_MODEL
    max_tokens: Optional[int] = None


class TitleGenerateResponse(BaseModel):
    text: str


class MaterialsGenerateRequest(BaseModel):
    api_key: str
    prompt: str
    model: str = GEMINI_MATERIAL_DEFAULT_MODEL
    output_dir: Optional[str] = None
    project_id: Optional[str] = None


class MaterialsGenerateResponse(BaseModel):
    image_path: Optional[str]
    image_base64: Optional[str]
    mime_type: str
    model_note: Optional[str]


class PonchiIdeasRequest(BaseModel):
    api_key: str
    engine: str = "Gemini"
    srt_path: str
    output_dir: Optional[str] = None
    gemini_model: str = DEFAULT_PONCHI_GEMINI_MODEL
    project_id: Optional[str] = None


class PonchiIdeasResponse(BaseModel):
    items: List[dict]
    json_path: Optional[str]


class PonchiImagesRequest(BaseModel):
    api_key: str
    srt_path: str
    output_dir: str
    suggestions: Optional[List[dict]] = None
    model: str = GEMINI_MATERIAL_DEFAULT_MODEL
    project_id: Optional[str] = None


class ProjectCreateRequest(BaseModel):
    name: str
    id: Optional[str] = None
    project_type: str = "standard"


class ProjectUpdateRequest(BaseModel):
    name: str


class ProjectCloneRequest(BaseModel):
    name: Optional[str] = None
    id: Optional[str] = None


class ProjectFlowStateUpdateRequest(BaseModel):
    flow_state: Optional[dict[str, str]] = None
    step: Optional[str] = None
    status: Optional[str] = None


class ProjectSettingsUpdateRequest(BaseModel):
    settings: dict[str, Any]


class PonchiImagesResponse(BaseModel):
    items: List[dict]
    output_dir: str
    json_path: str


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
    _ensure_default_project()
    return {"status": "ok"}


@app.get("/projects")
async def list_projects() -> dict:
    _ensure_default_project()
    projects: list[dict[str, Any]] = []
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    for child in sorted(PROJECTS_ROOT.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        meta_path = child / "project.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            project_id = str(meta.get("id", child.name)).strip() or child.name
            project_name = str(meta.get("name", child.name)).strip() or project_id
            projects.append({"id": project_id, "name": project_name, "project_type": _normalize_project_type(meta.get("project_type"))})
        except Exception:
            logger.warning("Skipping broken project metadata: %s", meta_path)
    return {"projects": projects, "default_project_id": DEFAULT_PROJECT_ID}


@app.post("/projects")
async def create_project(payload: ProjectCreateRequest) -> dict:
    project_id = _slugify_project_id(payload.id or payload.name)
    project_dir = _project_dir(project_id)
    if project_dir.exists():
        raise HTTPException(status_code=409, detail="project already exists")
    _ensure_project_structure(project_id, payload.name.strip(), payload.project_type)
    return {"project_id": project_id, "project": _read_project_meta(project_id)}


@app.put("/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectUpdateRequest) -> dict:
    project_id = _slugify_project_id(project_id)
    meta = _read_project_meta(project_id)
    meta["name"] = payload.name.strip() or meta.get("name") or project_id
    _write_project_meta(project_id, meta)
    return {"project": _read_project_meta(project_id)}


@app.get("/projects/{project_id}/flow")
async def get_project_flow(project_id: str) -> dict:
    resolved = _resolve_project_id(project_id)
    meta = _read_project_meta(resolved)
    if meta.get("project_type") != "flow":
        raise HTTPException(status_code=400, detail="project is not flow type")
    return {
        "project_id": resolved,
        "project_type": "flow",
        "flow_state": _normalize_flow_state(meta.get("flow_state")),
        "steps": FLOW_STEPS,
        "statuses": sorted(FLOW_STEP_STATUSES),
    }


@app.put("/projects/{project_id}/flow")
async def put_project_flow(project_id: str, payload: ProjectFlowStateUpdateRequest) -> dict:
    resolved = _resolve_project_id(project_id)
    meta = _read_project_meta(resolved)
    if meta.get("project_type") != "flow":
        raise HTTPException(status_code=400, detail="project is not flow type")
    flow_state = _normalize_flow_state(meta.get("flow_state"))

    if payload.flow_state is not None:
        flow_state = _normalize_flow_state(payload.flow_state)
    elif payload.step is not None:
        step = (payload.step or "").strip()
        status = (payload.status or "").strip()
        if step not in FLOW_STEPS:
            raise HTTPException(status_code=400, detail=f"unknown flow step: {step}")
        if status not in FLOW_STEP_STATUSES:
            raise HTTPException(status_code=400, detail=f"unknown flow status: {status}")
        flow_state[step] = status
    else:
        raise HTTPException(status_code=400, detail="flow_state or step/status is required")

    meta["flow_state"] = flow_state
    _write_project_meta(resolved, meta)
    return {
        "project_id": resolved,
        "project_type": "flow",
        "flow_state": flow_state,
    }


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> dict:
    project_id = _slugify_project_id(project_id)
    if project_id == DEFAULT_PROJECT_ID:
        raise HTTPException(status_code=400, detail="default project cannot be deleted")
    target = _project_dir(project_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="project not found")
    shutil.rmtree(target)
    return {"deleted": True, "project_id": project_id}


@app.post("/projects/{project_id}/clone")
async def clone_project(project_id: str, payload: ProjectCloneRequest) -> dict:
    source_id = _resolve_project_id(project_id)
    source_dir = _project_dir(source_id)
    if not source_dir.exists():
        raise HTTPException(status_code=404, detail="project not found")
    clone_id = _slugify_project_id(payload.id or f"{source_id}-copy")
    clone_dir = _project_dir(clone_id)
    if clone_dir.exists():
        raise HTTPException(status_code=409, detail="clone project already exists")
    _ensure_project_structure(
        clone_id,
        payload.name or f"{source_id} copy",
        _read_project_meta(source_id).get("project_type"),
    )
    shutil.copy2(_project_settings_path(source_id), _project_settings_path(clone_id))
    for subdir in ["scripts", "materials"]:
        src = source_dir / subdir
        dst = clone_dir / subdir
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    return {"project": _read_project_meta(clone_id)}


@app.get("/projects/{project_id}/settings")
async def get_project_settings(project_id: str) -> dict:
    resolved = _resolve_project_id(project_id)
    path = _project_settings_path(resolved)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"project_id": resolved, "settings": data}


@app.put("/projects/{project_id}/settings")
async def put_project_settings(project_id: str, payload: ProjectSettingsUpdateRequest) -> dict:
    resolved = _resolve_project_id(project_id)
    path = _project_settings_path(resolved)
    path.write_text(
        json.dumps(payload.settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"project_id": resolved, "settings": payload.settings}


@app.get("/settings/ai-models")
async def get_ai_models() -> dict:
    return {
        "thumbnail": {
            "provider": "Gemini",
            "models": MODEL_CONFIG["thumbnail_models"],
            "default_model": MODEL_CONFIG["thumbnail_default"],
        },
        "ponchi": {
            "provider": "Gemini",
            "models": MODEL_CONFIG["ponchi_models"],
            "default_model": MODEL_CONFIG["ponchi_default"],
        },
    }


@app.post("/script/generate", response_model=ScriptGenerateResponse)
async def generate_script(payload: ScriptGenerateRequest) -> ScriptGenerateResponse:
    provider = _normalize_engine(payload.provider)
    model = payload.model or DEFAULT_SCRIPT_GEMINI_MODEL
    try:
        if provider == "gemini":
            text = generate_script_with_gemini(
                api_key=payload.api_key,
                prompt=payload.prompt,
                model=model,
            )
        elif provider == "chatgpt":
            text = generate_script_with_openai(
                api_key=payload.api_key,
                prompt=payload.prompt,
                model=model or DEFAULT_SCRIPT_OPENAI_MODEL,
                max_tokens=payload.max_tokens,
            )
        elif provider == "claudecode":
            text = generate_script_with_claude(
                api_key=payload.api_key,
                prompt=payload.prompt,
                model=model or DEFAULT_CLAUDE_MODEL,
                max_tokens=payload.max_tokens or 20000,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {payload.provider}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Script generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScriptGenerateResponse(text=text)


@app.post("/title/generate", response_model=TitleGenerateResponse)
async def generate_title(payload: TitleGenerateRequest) -> TitleGenerateResponse:
    provider = _normalize_engine(payload.provider)
    try:
        script_text = extract_script_text(payload.script_path)
        prompt = _build_title_desc_prompt(
            script_text=script_text,
            count=payload.count,
            extra=payload.extra,
        )
        model = payload.model or DEFAULT_SCRIPT_GEMINI_MODEL
        if provider == "gemini":
            text = generate_script_with_gemini(
                api_key=payload.api_key,
                prompt=prompt,
                model=model,
            )
        elif provider == "chatgpt":
            text = generate_script_with_openai(
                api_key=payload.api_key,
                prompt=prompt,
                model=model or DEFAULT_SCRIPT_OPENAI_MODEL,
                max_tokens=payload.max_tokens or DEFAULT_TITLE_MAX_TOKENS,
            )
        elif provider == "claudecode":
            text = generate_script_with_claude(
                api_key=payload.api_key,
                prompt=prompt,
                model=model or DEFAULT_CLAUDE_MODEL,
                max_tokens=payload.max_tokens or DEFAULT_TITLE_MAX_TOKENS,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {payload.provider}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Title generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TitleGenerateResponse(text=text)


@app.post("/materials/generate", response_model=MaterialsGenerateResponse)
async def generate_materials(payload: MaterialsGenerateRequest) -> MaterialsGenerateResponse:
    try:
        project_id = _resolve_project_id(payload.project_id)
        resolved_model, model_note = resolve_gemini_material_model(payload.model)
        image_bytes, mime_type = generate_materials_with_gemini(
            api_key=payload.api_key,
            prompt=payload.prompt,
            model=resolved_model,
        )
        image_path: Optional[str] = None
        image_b64: Optional[str] = None
        output_dir = pathlib.Path(payload.output_dir) if payload.output_dir else (_project_dir(project_id) / "materials")
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = mimetypes.guess_extension(mime_type) or ".png"
        filename = f"material_{int(time.time())}{ext}"
        target = output_dir / filename
        target.write_bytes(image_bytes)
        image_path = str(target)
        if not image_path:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return MaterialsGenerateResponse(
            image_path=image_path,
            image_base64=image_b64,
            mime_type=mime_type,
            model_note=model_note,
        )
    except Exception as exc:
        logger.exception("Materials generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ponchi/ideas", response_model=PonchiIdeasResponse)
async def generate_ponchi_ideas(payload: PonchiIdeasRequest) -> PonchiIdeasResponse:
    engine = _normalize_engine(payload.engine)
    try:
        project_id = _resolve_project_id(payload.project_id)
        items = parse_srt_file(payload.srt_path)
        if not items:
            raise RuntimeError("SRTから字幕が見つかりませんでした。")
        if engine and engine != "gemini":
            raise HTTPException(status_code=400, detail="ポンチ絵の案出しは現在 Gemini のみ対応です。")
        model = payload.gemini_model.strip() or MODEL_CONFIG["ponchi_default"]
        if model not in MODEL_CONFIG["ponchi_models"]:
            model = MODEL_CONFIG["ponchi_default"]
        suggestions = generate_ponchi_suggestions_with_gemini(
            api_key=payload.api_key,
            items=items,
            model=model,
        )
        normalized = _normalize_ponchi_suggestions(suggestions, items)
        json_path: Optional[str] = None
        output_dir = payload.output_dir or str(_project_dir(project_id) / "materials")
        ideas_path = _ponchi_ideas_path(output_dir, payload.srt_path)
        if ideas_path:
            ideas_path.parent.mkdir(parents=True, exist_ok=True)
            ideas_path.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            json_path = str(ideas_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ponchi idea generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PonchiIdeasResponse(items=normalized, json_path=json_path)


@app.post("/ponchi/images", response_model=PonchiImagesResponse)
async def generate_ponchi_images(payload: PonchiImagesRequest) -> PonchiImagesResponse:
    try:
        project_id = _resolve_project_id(payload.project_id)
        items = parse_srt_file(payload.srt_path)
        if not items:
            raise RuntimeError("SRTから字幕が見つかりませんでした。")
        suggestions = payload.suggestions or []
        target_output_dir = payload.output_dir or str(_project_dir(project_id) / "materials")
        if not suggestions:
            ideas_path = _ponchi_ideas_path(target_output_dir, payload.srt_path)
            if ideas_path and ideas_path.exists():
                suggestions = json.loads(ideas_path.read_text(encoding="utf-8"))
        if not suggestions:
            raise RuntimeError("案が見つかりません。先に「案出し」を実行してください。")
        normalized = _normalize_ponchi_suggestions(suggestions, items)
        output_dir_path = pathlib.Path(target_output_dir) / pathlib.Path(payload.srt_path).stem
        output_dir_path.mkdir(parents=True, exist_ok=True)
        images: List[dict[str, Any]] = []
        for idx, item in enumerate(normalized, 1):
            prompt = item.get("image_prompt") or item.get("visual_suggestion") or item.get("text")
            if not prompt:
                prompt = "シンプルで分かりやすい図解のイラスト"
            image_bytes, mime_type = generate_materials_with_gemini(
                api_key=payload.api_key,
                prompt=prompt,
                model=payload.model,
            )
            ext = mimetypes.guess_extension(mime_type) or ".png"
            image_path = output_dir_path / f"ponchi_{idx:03d}{ext}"
            image_path.write_bytes(image_bytes)
            images.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "visual_suggestion": item["visual_suggestion"],
                    "image_prompt": prompt,
                    "image": image_path.name,
                    "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                }
            )
        json_path = output_dir_path / f"{pathlib.Path(payload.srt_path).stem}_ponchi.json"
        json_path.write_text(
            json.dumps(images, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.exception("Ponchi image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PonchiImagesResponse(
        items=images,
        output_dir=str(output_dir_path),
        json_path=str(json_path),
    )


def _build_base_video_output_stem() -> str:
    now = datetime.now()
    return f"{now.strftime('%Y%m%d%H%M')}{int(time.time() * 1000)}"


@app.post("/video/generate", response_model=JobResponse)
async def generate_video_job(payload: VideoGenerateRequest) -> JobResponse:
    project_id = _resolve_project_id(payload.project_id)
    if not payload.output_dir.strip():
        payload.output_dir = _project_output_dir(project_id)
    _validate_video_request(payload)
    try:
        job = manager.create_job()
        manager.update_status(job.job_id, "running")
    except Exception as exc:
        logger.exception("Failed to create job")
        raise HTTPException(status_code=500, detail=f"ジョブ作成に失敗しました: {exc}") from exc

    def log_fn(message: str) -> None:
        manager.add_log(job.job_id, message)

    def progress_fn(value: float, eta_seconds: Optional[float] = None) -> None:
        manager.update_progress(job.job_id, value, eta_seconds)

    output_stem = _build_base_video_output_stem()

    def worker() -> None:
        try:
            video_path, srt_path = generate_video(
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
                caption_margin_bottom=payload.caption_margin_bottom,
                caption_box_alpha=payload.caption_box_alpha,
                caption_box_enabled=payload.caption_box_enabled,
                caption_box_height=payload.caption_box_height,
                bg_off_style=payload.bg_off_style,
                caption_text_color=payload.caption_text_color,
                output_stem=output_stem,
            )
            manager.set_result(
                job.job_id,
                {
                    "message": "completed",
                    "video_path": video_path,
                    "srt_path": srt_path,
                },
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            traceback_text = traceback.format_exc()
            logger.exception("Video generation failed: %s", error_message)
            log_fn(f"[error] {error_message}")
            log_fn(traceback_text)
            manager.set_error(job.job_id, error_message)

    threading.Thread(target=worker, daemon=True).start()
    return JobResponse(job_id=job.job_id)


@app.get("/video/preview")
async def video_preview(path: str = Query(..., description="Path to generated video file")) -> FileResponse:
    raise HTTPException(
        status_code=410,
        detail=(
            "Desktop preview now uses local files directly via VideoPlayerController.file. "
            "HTTP /video/preview is deprecated and must not be used."
        ),
    )


@app.get("/voicevox/speakers")
async def list_voicevox_speakers(base_url: str = Query(...)) -> Any:
    try:
        speakers = fetch_voicevox_speakers(base_url)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    names: List[str] = []
    seen = set()
    for speaker in speakers:
        name = speaker.get("name") if isinstance(speaker, dict) else None
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return {"speakers": names}


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


@app.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str) -> dict:
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "logs": list(job.logs),
    }


@app.websocket("/ws/jobs/{job_id}")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    job = manager.get_job(job_id)
    if job is None:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    subscriber = manager.subscribe(job_id)
    if subscriber is None:
        await websocket.close(code=1008)
        return
    try:
        for message in job.logs:
            await websocket.send_json({"type": "log", "message": message, "ts": job.updated_at})
        if job.progress > 0.0 or job.eta_seconds is not None:
            await websocket.send_json(
                {
                    "type": "progress",
                    "progress": job.progress,
                    "eta_seconds": job.eta_seconds,
                    "ts": job.updated_at,
                }
            )
        if job.status == "error" and job.error:
            await websocket.send_json({"type": "error", "message": job.error, "ts": job.updated_at})
        if job.status == "completed" and job.result:
            await websocket.send_json({"type": "completed", "result": job.result, "ts": job.updated_at})
        while True:
            event = await asyncio.to_thread(subscriber.get)
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    finally:
        manager.unsubscribe(job_id, subscriber)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
