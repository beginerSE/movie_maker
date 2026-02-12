from __future__ import annotations

import base64
import json
import re
import time
import traceback
import wave
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
from google import genai
from google.genai import types
from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    concatenate_audioclips,
)
from moviepy.audio.AudioClip import CompositeAudioClip, AudioArrayClip
from PIL import Image, ImageDraw, ImageFont
from proglog import ProgressBarLogger

import anthropic

# Windows でよくある日本語フォント候補
FONT_PATHS = [
    r"C:\Windows\Fonts\YuGothR.ttc",  # 游ゴシック
    r"C:\Windows\Fonts\meiryo.ttc",  # メイリオ
    r"C:\Windows\Fonts\msgothic.ttc",  # MS ゴシック
]

# デフォルト値
FONT_SIZE = 36
SPEAKER_FONT_SIZE = 30

CAPTION_MAX_CHARS_PER_LINE = 22
CAPTION_MARGIN_X = 60
CAPTION_MARGIN_BOTTOM = 40  # (互換用。固定高さ設計では基本使わない)
CAPTION_LINE_SPACING = 6
CAPTION_BOX_ALPHA = 170
CAPTION_TEXT_STROKE = 2

# 字幕背景（黒幕）固定高さ + ON/OFF
DEFAULT_CAPTION_BOX_ENABLED = True
DEFAULT_CAPTION_BOX_HEIGHT = 420  # px

# 背景OFF時のデザイン
DEFAULT_BG_OFF_STYLE = "shadow"  # "shadow" / "rounded_panel" / "none"
DEFAULT_BG_OFF_PANEL_ALPHA = 140
DEFAULT_BG_OFF_PANEL_RADIUS = 22

# 字幕文字色（#RRGGBB）
DEFAULT_CAPTION_TEXT_COLOR = "#FFFFFF"

# 背景OFF時の影（薄い影）
DEFAULT_TEXT_SHADOW_ALPHA = 120
DEFAULT_TEXT_SHADOW_OFFSET = (2, 2)

# VOICEVOX 関連デフォルト
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"
DEFAULT_VV_ROTATION = [1, 3]  # 環境依存
DEFAULT_VV_CASTER_LABEL = "四国めたん"
DEFAULT_VV_ANALYST_LABEL = "ずんだもん"
DEFAULT_VV_SPEED = 1.0  # 話速

# Claude default
DEFAULT_CLAUDE_MODEL = "claude-opus-4-5-20251101"
DEFAULT_CLAUDE_MAX_TOKENS = 20000
DEFAULT_SCRIPT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_SCRIPT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_TITLE_MAX_TOKENS = 2000

DEFAULT_PONCHI_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_PONCHI_OPENAI_MODEL = "gpt-4.1-mini"

SPEAKER_ALIASES = {
    "キャスター": ["キャスター", "司会", "アナウンサー", "MC", "Caster"],
    "アナリスト": ["アナリスト", "解説", "専門家", "Analyst"],
}
SPEAKER_KEYS = list(SPEAKER_ALIASES.keys())


def normalize_voicevox_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return DEFAULT_VOICEVOX_URL
    if not re.match(r"^https?://", base_url):
        base_url = f"http://{base_url}"
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/speakers"):
        path = path[: -len("/speakers")]
    elif path.endswith("/speaker"):
        path = path[: -len("/speaker")]
    normalized = parsed._replace(path=path)
    return urlunparse(normalized)


def split_sentences_jp(text: str) -> List[str]:
    """「。」「．」「！」「？」などの直後で文を分割（句点は残す）"""
    parts = re.split(r"(?<=[。．！？?!])", text)
    return [p.strip() for p in parts if p.strip()]


def allocate_durations_by_length(sentences: List[str], total_duration: float) -> List[float]:
    """文の文字数比で自然に時間配分"""
    lens = [max(1, len(s)) for s in sentences]
    total = sum(lens)
    if total == 0:
        return [total_duration]
    return [total_duration * (length / total) for length in lens]


def parse_hex_color(hex_str: str, default: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Tuple[int, int, int, int]:
    """
    "#RRGGBB" or "RRGGBB" を (r,g,b,a) に変換。失敗したら default。
    """
    if not hex_str:
        return default
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return default
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b, 255)
    except Exception:
        return default


def detect_speaker(line: str) -> Tuple[str, str]:
    line = line.strip()
    for spk in SPEAKER_KEYS:
        for alias in SPEAKER_ALIASES[spk]:
            if line.startswith(f"{alias}：") or line.startswith(f"{alias}:"):
                if "：" in line:
                    text = line.split("：", 1)[1]
                else:
                    text = line.split(":", 1)[1]
                return spk, text.strip()
    return "", line


def parse_script(path: str) -> List[Tuple[str, str]]:
    """dialogue_input.txt を パースして (話者, テキスト) のリストを返す"""
    lines = []
    prev = "キャスター"
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        spk, body = detect_speaker(s)
        if not body:
            continue
        if not spk:
            spk = prev
        lines.append((spk, body))
        prev = spk
    return lines


def build_tts_prompt(speaker: str, text: str) -> str:
    """Gemini 用: 話者ごとにしゃべり方を変えるプロンプト"""
    if speaker == "キャスター":
        style = "calm, clear, professional news anchor, natural pace"
    else:
        style = "confident, informative, analytical tone, natural pace"
    return f"Read in Japanese with {style}. Text: {text}"


def pick_font(size: int) -> ImageFont.ImageFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_japanese(text: str, max_chars: int) -> List[str]:
    """スペースに依存せず字数ベースで折り返し"""
    lines = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= max_chars:
            lines.append(buf)
            buf = ""
    if buf:
        lines.append(buf)
    return lines


def letterbox_fit(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """画像を縦横比維持でリサイズし、黒背景にレターボックス"""
    tw, th = target_size
    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", target_size, (0, 0, 0))
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(img, offset)
    return canvas


def draw_caption_overlay(
    speaker: str,
    text: str,
    target_size: Tuple[int, int],
    caption_font_size: int,
    speaker_font_size: int,
    caption_max_chars: int,
    caption_margin_bottom: int,  # 互換用（固定高さ設計では基本使わない）
    caption_box_alpha: int,
    caption_box_enabled: bool = True,
    caption_box_height: int = DEFAULT_CAPTION_BOX_HEIGHT,
    bg_off_style: str = DEFAULT_BG_OFF_STYLE,  # "shadow" / "rounded_panel" / "none"
    bg_off_panel_alpha: int = DEFAULT_BG_OFF_PANEL_ALPHA,
    bg_off_panel_radius: int = DEFAULT_BG_OFF_PANEL_RADIUS,
    text_color_hex: str = DEFAULT_CAPTION_TEXT_COLOR,
) -> Image.Image:
    """字幕だけを描いた半透明オーバーレイ（RGBA）を返す（背景は固定高さで任意ON/OFF + 背景OFFデザイン）"""
    tw, th = target_size
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    font = pick_font(caption_font_size)
    spk_font = pick_font(speaker_font_size)

    caption_lines = wrap_japanese(text, caption_max_chars)

    # 行高さ計算
    line_heights = []
    for ln in caption_lines:
        bbox = draw.textbbox((0, 0), ln, font=font, stroke_width=CAPTION_TEXT_STROKE)
        line_heights.append(bbox[3] - bbox[1])
    lh = max(line_heights) if line_heights else font.size

    spk_label = f"{speaker}"
    spk_bbox = draw.textbbox((0, 0), spk_label, font=spk_font, stroke_width=CAPTION_TEXT_STROKE)
    spk_h = (spk_bbox[3] - spk_bbox[1])

    text_rgba = parse_hex_color(text_color_hex, default=(255, 255, 255, 255))

    pad_top = 12
    gap = 10
    pad_bottom = 12

    if caption_box_enabled:
        box_h = int(max(1, caption_box_height))
        box_y0 = th - box_h

        box = Image.new("RGBA", (tw, box_h), (0, 0, 0, int(caption_box_alpha)))
        overlay.alpha_composite(box, (0, box_y0))

        text_top_y = box_y0 + pad_top
        stroke_w = CAPTION_TEXT_STROKE
        use_shadow = False
    else:
        total_text_h = spk_h + gap + len(caption_lines) * (lh + CAPTION_LINE_SPACING) - CAPTION_LINE_SPACING
        text_top_y = th - (pad_bottom + total_text_h)

        stroke_w = 1
        use_shadow = (bg_off_style == "shadow")

    x = CAPTION_MARGIN_X
    y0 = text_top_y

    if (not caption_box_enabled) and (bg_off_style == "rounded_panel"):
        pad_x = 18
        pad_y = 14

        max_w = 0
        sb = draw.textbbox((0, 0), spk_label, font=spk_font, stroke_width=stroke_w)
        max_w = max(max_w, sb[2] - sb[0])

        for ln in caption_lines:
            bb = draw.textbbox((0, 0), ln, font=font, stroke_width=stroke_w)
            max_w = max(max_w, bb[2] - bb[0])

        total_h = spk_h + gap + len(caption_lines) * (lh + CAPTION_LINE_SPACING) - CAPTION_LINE_SPACING

        panel_x0 = x - pad_x
        panel_y0 = y0 - pad_y
        panel_x1 = x + max_w + pad_x
        panel_y1 = y0 + total_h + pad_y

        panel_x0 = max(0, panel_x0)
        panel_y0 = max(0, panel_y0)
        panel_x1 = min(tw, panel_x1)
        panel_y1 = min(th, panel_y1)

        draw.rounded_rectangle(
            [panel_x0, panel_y0, panel_x1, panel_y1],
            radius=int(bg_off_panel_radius),
            fill=(0, 0, 0, int(bg_off_panel_alpha)),
        )

    shadow_offset = DEFAULT_TEXT_SHADOW_OFFSET
    shadow_fill = (0, 0, 0, DEFAULT_TEXT_SHADOW_ALPHA)

    def draw_text_with_optional_shadow(pos, text_, font_, fill_, stroke_width_, stroke_fill_):
        if use_shadow:
            draw.text(
                (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]),
                text_,
                font=font_,
                fill=shadow_fill,
                stroke_width=0,
            )
        draw.text(
            pos,
            text_,
            font=font_,
            fill=fill_,
            stroke_width=stroke_width_,
            stroke_fill=stroke_fill_,
        )

    y = y0
    draw_text_with_optional_shadow(
        (x, y),
        spk_label,
        spk_font,
        text_rgba,
        stroke_w,
        (0, 0, 0, 255),
    )
    y += spk_h + gap

    for ln in caption_lines:
        draw_text_with_optional_shadow(
            (x, y),
            ln,
            font,
            text_rgba,
            stroke_w,
            (0, 0, 0, 255),
        )
        y += lh + CAPTION_LINE_SPACING

    return overlay


def save_wave(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def image_to_clip_with_audio(
    image_path: Path,
    speaker: str,
    text: str,
    audio_path: Path,
    target_size: Tuple[int, int],
    fps: int = 30,
    caption_font_size: int = FONT_SIZE,
    speaker_font_size: int = SPEAKER_FONT_SIZE,
    caption_max_chars: int = CAPTION_MAX_CHARS_PER_LINE,
    caption_margin_bottom: int = CAPTION_MARGIN_BOTTOM,
    caption_box_alpha: int = CAPTION_BOX_ALPHA,
    caption_box_enabled: bool = DEFAULT_CAPTION_BOX_ENABLED,
    caption_box_height: int = DEFAULT_CAPTION_BOX_HEIGHT,
    bg_off_style: str = DEFAULT_BG_OFF_STYLE,
    text_color_hex: str = DEFAULT_CAPTION_TEXT_COLOR,
) -> VideoFileClip:
    """
    1セリフを「文ごとに字幕を切り替える」クリップへ変換。
    音声はセリフ全体をそのまま載せ、見た目だけ文ごとに切り替える。
    """
    base = Image.open(str(image_path)).convert("RGB")
    bg_rgba = letterbox_fit(base, target_size).convert("RGBA")

    audio_full = AudioFileClip(str(audio_path))
    try:
        total_dur = audio_full.duration or 0.0
        fps_audio = 48000
        snd = audio_full.to_soundarray(fps=fps_audio)
        audio_mem = AudioArrayClip(snd, fps=fps_audio).with_duration(total_dur)
    finally:
        audio_full.close()

    sentences = split_sentences_jp(text) or [text]
    durations = allocate_durations_by_length(sentences, total_dur)

    sentence_clips = []
    for idx, (sent, dur) in enumerate(zip(sentences, durations), 1):
        overlay_text = f"{speaker}：{sent}" if idx == 1 else sent
        overlay_rgba = draw_caption_overlay(
            speaker,
            overlay_text,
            target_size,
            caption_font_size,
            speaker_font_size,
            caption_max_chars,
            caption_margin_bottom,
            caption_box_alpha,
            caption_box_enabled=caption_box_enabled,
            caption_box_height=caption_box_height,
            bg_off_style=bg_off_style,
            text_color_hex=text_color_hex,
        )
        combined_rgb = Image.alpha_composite(bg_rgba, overlay_rgba).convert("RGB")
        frame_np = np.array(combined_rgb)
        c = ImageClip(frame_np, duration=dur)
        sentence_clips.append(c)

    line_visual = concatenate_videoclips(sentence_clips, method="chain")
    for c in sentence_clips:
        c.close()
    line_visual.audio = audio_mem
    return line_visual


def write_srt(lines: List[Tuple[str, str]], out_srt: Path, per_line_secs: List[float]) -> None:
    def fmt_t(ms):
        s, ms = divmod(int(ms), 1000)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    t = 0
    idx = 1
    with out_srt.open("w", encoding="utf-8") as f:
        for (speaker, text), sec in zip(lines, per_line_secs):
            start = t
            end = t + int(sec * 1000)
            f.write(f"{idx}\n{fmt_t(start)} --> {fmt_t(end)}\n{speaker}：{text}\n\n")
            t = end
            idx += 1


def fetch_voicevox_speakers(base_url: str) -> List[Dict[str, Any]]:
    base_url = normalize_voicevox_url(base_url).rstrip("/")
    url = f"{base_url}/speakers"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "VOICEVOX サーバーに接続できません。起動しているかURLを確認してください。"
        ) from exc
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(
            "VOICEVOX の /speakers から有効なJSONを取得できませんでした。"
        ) from exc


def resolve_voicevox_speaker_label(
    label: str,
    speakers: List[Dict[str, Any]],
    default_id: Optional[int] = None,
) -> int:
    label = (label or "").strip()
    if not label:
        if default_id is not None:
            return default_id
        raise RuntimeError("VOICEVOX 話者ラベルが空です。")

    try:
        numeric_id = int(label)
        style_ids = {
            int(style.get("id"))
            for sp in speakers
            for style in sp.get("styles", [])
            if isinstance(style.get("id"), (int, float, str))
        }
        if numeric_id in style_ids:
            return numeric_id
        if default_id is not None:
            return default_id
        raise RuntimeError(f"VOICEVOX 話者ID '{numeric_id}' が /speakers に存在しません。")
    except ValueError:
        pass

    exact_candidates = []
    partial_candidates = []
    for sp in speakers:
        name = sp.get("name", "")
        if not name:
            continue
        if name == label:
            exact_candidates.append(sp)
        elif label in name:
            partial_candidates.append(sp)

    candidates = exact_candidates or partial_candidates
    if not candidates:
        if default_id is not None:
            return default_id
        raise RuntimeError(f"VOICEVOX 話者 '{label}' が /speakers から見つかりませんでした。")

    sp = candidates[0]
    styles = sp.get("styles", [])
    if not styles:
        if default_id is not None:
            return default_id
        raise RuntimeError(f"VOICEVOX 話者 '{label}' に styles がありません。")

    for st in styles:
        if st.get("name") == "ノーマル":
            return int(st.get("id"))

    return int(styles[0].get("id"))


def collect_voicevox_style_ids(speakers: List[Dict[str, Any]]) -> List[int]:
    ids: List[int] = []
    for sp in speakers:
        for style in sp.get("styles", []):
            try:
                style_id = int(style.get("id"))
            except (TypeError, ValueError):
                continue
            if style_id not in ids:
                ids.append(style_id)
    return ids


def tts_with_voicevox(
    text: str,
    speaker_id: int,
    out_wav: Path,
    base_url: str = DEFAULT_VOICEVOX_URL,
    speed_scale: float = DEFAULT_VV_SPEED,
) -> None:
    base_url = normalize_voicevox_url(base_url).rstrip("/")
    audio_query_url = f"{base_url}/audio_query"
    synthesis_url = f"{base_url}/synthesis"

    try:
        q_res = requests.post(
            audio_query_url,
            params={"text": text, "speaker": speaker_id},
            timeout=10,
        )
        q_res.raise_for_status()
        query = q_res.json()

        try:
            ss = float(speed_scale)
            if ss > 0:
                query["speedScale"] = ss
        except Exception:
            pass

        s_res = requests.post(
            synthesis_url,
            params={"speaker": speaker_id},
            json=query,
            timeout=60,
        )
        s_res.raise_for_status()

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with out_wav.open("wb") as f:
            f.write(s_res.content)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "VOICEVOX サーバーに接続できません。起動しているかURLを確認してください。"
        ) from exc
    except ValueError as exc:
        raise RuntimeError("VOICEVOX の応答を解析できませんでした。") from exc
    except Exception as exc:
        raise RuntimeError(f"VOICEVOX 合成に失敗しました (speaker_id={speaker_id})。") from exc


class TkMoviePyLogger(ProgressBarLogger):
    """
    MoviePy の logger callback (frame_index) を受けて GUI 進捗へ反映。
    base〜base+span を使って「工程内の進捗」を表現する。
    """

    def __init__(self, progress_fn, base: float = 0.8, span: float = 0.2):
        super().__init__()
        self.progress_fn = progress_fn
        self.base = float(base)
        self.span = float(span)
        self._last_progress: Optional[float] = None
        self._last_emit_time = 0.0
        self._min_step = 0.005
        self._min_interval = 0.2

    def bars_callback(self, bar, attr, value, old_value=None):
        if bar != "frame_index" or attr != "index":
            return

        try:
            total = float(self.bars[bar].get("total") or 0)
            idx = float(value or 0)
            if total <= 0:
                return

            ratio = max(0.0, min(1.0, idx / total))
            v = self.base + self.span * ratio

            elapsed = self.bars[bar].get("elapsed")  # seconds
            rate = self.bars[bar].get("rate")  # it/s

            eta = None
            if rate and isinstance(rate, (int, float)) and rate > 0:
                remaining = max(0.0, total - idx)
                eta = remaining / float(rate)
            elif elapsed and isinstance(elapsed, (int, float)) and ratio > 0.01:
                total_est = float(elapsed) / ratio
                eta = max(0.0, total_est - float(elapsed))

            if self.progress_fn:
                now = time.monotonic()
                if (
                    self._last_progress is None
                    or abs(v - self._last_progress) >= self._min_step
                    or (now - self._last_emit_time) >= self._min_interval
                    or v >= 1.0
                ):
                    self._last_progress = v
                    self._last_emit_time = now
                    try:
                        self.progress_fn(v, eta)
                    except TypeError:
                        self.progress_fn(v)

        except Exception:
            pass


def generate_video(
    api_key: str,
    script_path: str,
    image_paths: List[str],
    use_bgm: bool,
    bgm_path: str,
    bgm_gain_db: float,
    output_dir: str,
    width: int,
    height: int,
    fps: int,
    voice_name: str,
    log_fn,
    progress_fn=None,
    tts_engine: str = "Gemini",  # "Gemini" or "VOICEVOX"
    vv_mode: str = "rotation",  # "rotation" or "two_person"
    vv_rotation_labels: Optional[List[str]] = None,
    vv_caster_label: str = DEFAULT_VV_CASTER_LABEL,
    vv_analyst_label: str = DEFAULT_VV_ANALYST_LABEL,
    vv_base_url: str = DEFAULT_VOICEVOX_URL,
    vv_speed_scale: float = DEFAULT_VV_SPEED,
    caption_font_size: int = FONT_SIZE,
    speaker_font_size: int = SPEAKER_FONT_SIZE,
    caption_max_chars: int = CAPTION_MAX_CHARS_PER_LINE,
    caption_margin_bottom: int = CAPTION_MARGIN_BOTTOM,
    caption_box_alpha: int = CAPTION_BOX_ALPHA,
    caption_box_enabled: bool = DEFAULT_CAPTION_BOX_ENABLED,
    caption_box_height: int = DEFAULT_CAPTION_BOX_HEIGHT,
    bg_off_style: str = DEFAULT_BG_OFF_STYLE,
    caption_text_color: str = DEFAULT_CAPTION_TEXT_COLOR,
) -> None:
    last_progress: Optional[float] = None
    last_emit_time = 0.0
    min_step = 0.005
    min_interval = 0.2

    def update_progress(v: float):
        nonlocal last_progress, last_emit_time
        if progress_fn is None:
            return
        value = max(0.0, min(1.0, float(v)))
        now = time.monotonic()
        if (
            last_progress is None
            or abs(value - last_progress) >= min_step
            or (now - last_emit_time) >= min_interval
            or value >= 1.0
        ):
            last_progress = value
            last_emit_time = now
            progress_fn(value)

    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        engine = (tts_engine or "Gemini").strip().lower()
        vv_base_url = normalize_voicevox_url(vv_base_url)
        vv_mode_normalized = (vv_mode or "rotation").strip().lower()
        log_fn(f"TTS エンジン: {tts_engine}")

        client = None
        tts_model = None
        vv_speakers = None

        if engine == "gemini":
            log_fn("Google Gemini クライアントを初期化中...")
            client = genai.Client(api_key=api_key)
            tts_model = "gemini-2.5-flash-preview-tts"
        elif engine == "voicevox":
            log_fn(f"VOICEVOX エンジンで TTS を行います。({vv_base_url}) が起動している必要があります。")
            vv_speakers = fetch_voicevox_speakers(vv_base_url)
            log_fn(f"/speakers から {len(vv_speakers)} 件の話者を取得しました。")
        else:
            raise RuntimeError(f"サポートされていない TTS エンジンです: {tts_engine}")

        vv_mode_int = "rotation" if vv_mode_normalized == "rotation" else "two_person"
        vv_rotation_ids: List[int] = DEFAULT_VV_ROTATION

        if engine == "voicevox":
            available_style_ids = collect_voicevox_style_ids(vv_speakers)
            if available_style_ids:
                vv_rotation_ids = available_style_ids[:2]
            if vv_rotation_labels:
                resolved_ids = []
                for lbl in vv_rotation_labels:
                    sid = resolve_voicevox_speaker_label(
                        lbl,
                        vv_speakers,
                        vv_rotation_ids[0] if vv_rotation_ids else None,
                    )
                    resolved_ids.append(sid)
                if resolved_ids:
                    vv_rotation_ids = resolved_ids

            vv_caster_id = resolve_voicevox_speaker_label(vv_caster_label, vv_speakers, vv_rotation_ids[0])
            vv_analyst_id = resolve_voicevox_speaker_label(
                vv_analyst_label,
                vv_speakers,
                vv_rotation_ids[1] if len(vv_rotation_ids) > 1 else vv_rotation_ids[0],
            )
            log_fn(
                f"VOICEVOX 話者ID: rotation={vv_rotation_ids} caster={vv_caster_id} analyst={vv_analyst_id}"
            )
        else:
            vv_caster_id = vv_analyst_id = vv_rotation_ids[0]

        log_fn(f"原稿を読み込み中: {script_path}")
        lines = parse_script(script_path)
        if not lines:
            raise RuntimeError("原稿からセリフが1つも見つかりませんでした。")

        total_lines = len(lines)
        log_fn(f"{total_lines} 行のセリフを検出しました。")
        update_progress(0.0)

        if not image_paths:
            raise RuntimeError("画像が1枚も指定されていません。")
        img_cycle = cycle([Path(p) for p in image_paths])

        # ===== TTS ===== (0%〜40%)
        audio_paths: List[Path] = []
        for i, (speaker, text) in enumerate(lines, 1):
            log_fn(f"[TTS] {i}/{total_lines} 生成中... (話者: {speaker})")

            if engine == "gemini":
                prompt = build_tts_prompt(speaker, text)
                resp = client.models.generate_content(
                    model=tts_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name
                                )
                            )
                        ),
                    ),
                )
                pcm = resp.candidates[0].content.parts[0].inline_data.data
                out_wav = output_dir_path / f"{i:03d}_{speaker}_gemini.wav"
                save_wave(str(out_wav), pcm)

            else:
                if vv_mode_int == "two_person":
                    spk_id = vv_analyst_id if speaker == "アナリスト" else vv_caster_id
                else:
                    spk_id = vv_rotation_ids[(i - 1) % len(vv_rotation_ids)]

                out_wav = output_dir_path / f"{i:03d}_{speaker}_vv{spk_id}.wav"
                tts_with_voicevox(
                    text=text,
                    speaker_id=spk_id,
                    out_wav=out_wav,
                    base_url=vv_base_url,
                    speed_scale=vv_speed_scale,
                )

            audio_paths.append(out_wav)
            update_progress(0.4 * i / total_lines)

        log_fn("TTS生成完了。")

        # ===== BGM（全体用の事前チェック）===== (40%〜60%)
        if use_bgm and bgm_path:
            log_fn("BGM は動画全体にループ適用されます。（後工程で合成）")
        else:
            log_fn("BGMは使用されません。")
        update_progress(0.6)

        # ===== クリップ生成 ===== (60%〜80%)
        target_resolution = (width, height)
        clips = []
        per_line_secs = []

        for i, ((speaker, text), ap) in enumerate(zip(lines, audio_paths), 1):
            img_path = next(img_cycle)
            log_fn(f"[CLIP] {i}/{total_lines} 画像='{img_path.name}' からクリップ生成中...")
            clip = image_to_clip_with_audio(
                image_path=img_path,
                speaker=speaker,
                text=text,
                audio_path=ap,
                target_size=target_resolution,
                fps=fps,
                caption_font_size=caption_font_size,
                speaker_font_size=speaker_font_size,
                caption_max_chars=caption_max_chars,
                caption_margin_bottom=caption_margin_bottom,
                caption_box_alpha=caption_box_alpha,
                caption_box_enabled=caption_box_enabled,
                caption_box_height=caption_box_height,
                bg_off_style=bg_off_style,
                text_color_hex=caption_text_color,
            )
            per_line_secs.append(clip.duration)
            clips.append(clip)
            update_progress(0.6 + 0.2 * i / total_lines)

        log_fn("クリップの連結中...")
        final = concatenate_videoclips(clips, method="chain")
        update_progress(0.8)

        # ===== BGM を動画全体にループ適用 =====
        if use_bgm and bgm_path:
            try:
                log_fn("動画全体にBGMをループ適用中...")
                bgm_clip = AudioFileClip(bgm_path)
                try:
                    fps_audio = 48000
                    bgm_snd = bgm_clip.to_soundarray(fps=fps_audio)
                    bgm_mem = AudioArrayClip(bgm_snd, fps=fps_audio).with_duration(bgm_clip.duration)
                finally:
                    bgm_clip.close()

                bgm_duration = bgm_mem.duration
                if not bgm_duration or bgm_duration <= 0:
                    raise RuntimeError("BGM の長さが 0 秒です。")

                loops = int(final.duration // bgm_duration) + 1
                bgm_long = concatenate_audioclips([bgm_mem] * loops)
                bgm_looped = bgm_long.with_duration(final.duration)

                gain = 10 ** (bgm_gain_db / 20.0)
                bgm_looped = bgm_looped.with_volume_scaled(gain)

                voice_audio = final.audio
                final.audio = CompositeAudioClip([voice_audio, bgm_looped])

                log_fn("BGM 合成完了。")
            except Exception as exc:
                log_fn(f"BGM 合成中にエラーが発生しました（BGMなしで続行します）: {exc}")

        # ===== 動画書き出し ===== (80%〜100%)
        txt_stem = Path(script_path).stem
        final_out = output_dir_path / f"{txt_stem}.mp4"
        srt_path = output_dir_path / f"{txt_stem}.srt"
        log_fn(f"動画を書き出し中: {final_out}")

        mp_logger = TkMoviePyLogger(progress_fn=progress_fn, base=0.8, span=0.2)

        final.write_videofile(
            str(final_out),
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            logger=mp_logger,
        )

        update_progress(1.0)

        log_fn(f"SRT を出力中: {srt_path}")
        write_srt(lines, srt_path, per_line_secs)

        for c in clips:
            c.close()
        final.close()

        log_fn("✅ 全処理が完了しました。")

    except Exception:
        tb = traceback.format_exc()
        log_fn("❌ エラーが発生しました:\n" + tb)
        raise


def generate_script_with_claude(
    api_key: str,
    prompt: str,
    model: str,
    max_tokens: int,
) -> str:
    if not api_key:
        raise RuntimeError("Claude APIキーが空です。")
    if not prompt.strip():
        raise RuntimeError("プロンプトが空です。")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=int(max_tokens),
        messages=[{"role": "user", "content": prompt}],
    )

    chunks = []
    try:
        for part in response.content:
            txt = getattr(part, "text", None)
            if txt:
                chunks.append(txt)
    except Exception:
        pass

    if chunks:
        return "\n".join(chunks).strip()

    s = str(response)
    return s.strip()


def generate_script_with_gemini(
    api_key: str,
    prompt: str,
    model: str,
) -> str:
    if not api_key:
        raise RuntimeError("Gemini APIキーが空です。")
    if not prompt.strip():
        raise RuntimeError("プロンプトが空です。")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    text = getattr(resp, "text", "") or ""
    if not text:
        raise RuntimeError("Geminiから台本の取得に失敗しました。")
    return text.strip()


def _post_openai_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: Tuple[int, int] = (10, 120),
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> requests.Response:
    retry_statuses = {429, 500, 502, 503, 504}
    last_response: Optional[requests.Response] = None
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_base * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(
                "ChatGPT APIへの接続がタイムアウトしました。時間を空けて再試行してください。"
            ) from exc

        last_response = response
        if response.status_code in retry_statuses and attempt < max_retries:
            time.sleep(backoff_base * (2 ** (attempt - 1)))
            continue
        return response

    if last_response is not None:
        raise RuntimeError(_format_openai_error(last_response))
    if last_exc is not None:
        raise RuntimeError(
            "ChatGPT APIへの接続に失敗しました。ネットワークを確認してください。"
        ) from last_exc
    raise RuntimeError("ChatGPT APIへのリクエストに失敗しました。")


def _format_openai_error(response: requests.Response) -> str:
    status = response.status_code
    if status == 401:
        return "ChatGPT APIキーが無効です。APIキーを確認してください。"
    if status == 429:
        return "ChatGPT APIのレート制限に達しました。時間を空けて再試行してください。"
    if status in {500, 502, 503, 504}:
        return (
            "ChatGPT APIのサーバーが一時的に不安定です。"
            "時間を空けて再試行してください。"
        )

    message = None
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
    except ValueError:
        message = None
    if message:
        return f"ChatGPT APIエラー: {status} {message}"
    return f"ChatGPT APIエラー: {status}"


def _extract_openai_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or ""
    return str(payload.get("error", {}).get("message") or response.text or "")


def _clamp_openai_max_tokens(model: str, requested: int) -> int:
    """モデルごとの差異を吸収しつつ、過大な max_tokens を抑制する。"""
    cleaned = (model or "").strip().lower()
    # 既知の上限。未知モデルは安全側で既定値をそのまま使う。
    known_limits = {
        "gpt-4.1": 32768,
        "gpt-4.1-mini": 16384,
        "gpt-4.1-nano": 16384,
        "gpt-4o": 16384,
        "gpt-4o-mini": 16384,
        "o1": 65536,
        "o1-mini": 65536,
    }
    for prefix, limit in known_limits.items():
        if cleaned.startswith(prefix):
            return max(1, min(int(requested), limit))
    return max(1, int(requested))


def _openai_max_token_keys_for_model(model: str) -> List[str]:
    """モデルによって max_tokens / max_completion_tokens の受け口が違うため順に試す。"""
    cleaned = (model or "").strip().lower()
    if cleaned.startswith(("gpt-4.1", "gpt-4o", "o1", "o3", "gpt-5")):
        return ["max_completion_tokens", "max_tokens"]
    return ["max_tokens", "max_completion_tokens"]


def generate_script_with_openai(
    api_key: str,
    prompt: str,
    model: str,
    max_tokens: int | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("ChatGPT APIキーが空です。")
    if not prompt.strip():
        raise RuntimeError("プロンプトが空です。")

    payload = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    token_keys = _openai_max_token_keys_for_model(model)
    bounded_tokens = _clamp_openai_max_tokens(model, max_tokens) if max_tokens is not None else None
    if bounded_tokens is not None:
        payload[token_keys[0]] = bounded_tokens

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = _post_openai_with_retry(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        payload=payload,
    )

    # モデル差異による引数エラーや上限超過を吸収して1回だけ再試行する。
    if not resp.ok and resp.status_code == 400 and bounded_tokens is not None:
        err_msg = _extract_openai_error_message(resp)
        lowered = err_msg.lower()
        retry_payload = dict(payload)

        unsupported_key = None
        for key in token_keys:
            if f"'{key}'" in lowered and "unsupported" in lowered:
                unsupported_key = key
                break

        if unsupported_key:
            retry_payload.pop(unsupported_key, None)
            for candidate in token_keys:
                if candidate != unsupported_key:
                    retry_payload[candidate] = bounded_tokens
                    break
            resp = _post_openai_with_retry(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                payload=retry_payload,
            )
        else:
            limit_match = re.search(r"at most\s+(\d+)\s+completion tokens", lowered)
            if limit_match:
                detected_limit = int(limit_match.group(1))
                for key in token_keys:
                    retry_payload[key] = max(1, min(bounded_tokens, detected_limit))
                resp = _post_openai_with_retry(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    payload=retry_payload,
                )

    if not resp.ok:
        raise RuntimeError(_format_openai_error(resp))
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("ChatGPTから台本の取得に失敗しました。")
    return content.strip()


GEMINI_MATERIAL_DEFAULT_MODEL = "gemini-2.5-flash-image"
GEMINI_MATERIAL_MODEL_ALIASES = {
    "nanobanana": GEMINI_MATERIAL_DEFAULT_MODEL,
}


def resolve_gemini_material_model(model: str) -> Tuple[str, Optional[str]]:
    cleaned = (model or "").strip()
    if not cleaned:
        return GEMINI_MATERIAL_DEFAULT_MODEL, (
            f"ℹ️ モデル未指定のため {GEMINI_MATERIAL_DEFAULT_MODEL} を使用します。"
        )
    alias = GEMINI_MATERIAL_MODEL_ALIASES.get(cleaned)
    if alias:
        return alias, f"ℹ️ {cleaned} は画像生成では {alias} を使用します。"
    return cleaned, None


def generate_materials_with_gemini(
    api_key: str,
    prompt: str,
    model: str = GEMINI_MATERIAL_DEFAULT_MODEL,
) -> Tuple[bytes, str]:
    if not api_key:
        raise RuntimeError("Gemini APIキーが空です。")
    if not prompt.strip():
        raise RuntimeError("プロンプトが空です。")

    resolved_model, _ = resolve_gemini_material_model(model)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{resolved_model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if (
        not response.ok
        and response.status_code == 400
        and resolved_model != GEMINI_MATERIAL_DEFAULT_MODEL
        and "response modalities" in response.text.lower()
    ):
        fallback_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MATERIAL_DEFAULT_MODEL}:generateContent"
        )
        response = requests.post(fallback_url, headers=headers, json=payload, timeout=60)
    if not response.ok:
        raise RuntimeError(f"Gemini APIエラー: {response.status_code} {response.text}")
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Gemini APIエラー: {data['error']}")

    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                mime_type = inline.get("mimeType", "image/png")
                image_bytes = base64.b64decode(inline["data"])
                return image_bytes, mime_type

    raise RuntimeError("Geminiから画像が取得できませんでした。")


def format_seconds_to_timecode(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_ponchi_suggestions_with_gemini(
    api_key: str,
    items: List[Dict[str, Any]],
    model: str = DEFAULT_PONCHI_GEMINI_MODEL,
) -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("Gemini APIキーを入力してください。")
    if not items:
        raise RuntimeError("SRT字幕が空です。")

    client = genai.Client(api_key=api_key)
    prompt_lines = [
        "あなたは動画用の補足イラスト（ポンチ絵）を設計する担当者です。",
        "次の字幕情報をもとに、時間帯ごとのポンチ絵案を作成してください。",
        "出力は必ずMarkdownの表のみで、表の前後に文章を一切つけないでください。",
        "列名は必ず次の4つだけをこの順で使用してください:",
        "start_time, end_time, illustration_prompt, note",
        "時刻は hh:mm:ss 形式で、字幕の範囲に合わせてください。",
        "",
        "出力例:",
        "| start_time | end_time | illustration_prompt | note |",
        "|------------|----------|---------------------|------|",
        "| 00:00:05 | 00:00:12 | 金価格上昇を示す上向き矢印と金塊のイラスト | 導入部 |",
        "",
        "字幕一覧:",
    ]
    for idx, item in enumerate(items, 1):
        start = format_seconds_to_timecode(item.get("start", 0))
        end = format_seconds_to_timecode(item.get("end", 0))
        prompt_lines.append(f"{idx}. {start}〜{end} {item.get('text', '')}")

    table_text = ""
    for _ in range(2):
        resp = client.models.generate_content(
            model=model,
            contents="\n".join(prompt_lines),
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if all(line.strip().startswith("|") and line.strip().endswith("|") for line in lines):
            table_text = "\n".join(lines)
            break
    if not table_text:
        raise RuntimeError("Markdown表の提案取得に失敗しました。")

    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    header = [cell.strip().lower() for cell in lines[0].strip("|").split("|")]
    expected = ["start_time", "end_time", "illustration_prompt", "note"]
    if header != expected:
        raise RuntimeError("提案表ヘッダーが不正です。")
    rows: List[Dict[str, Any]] = []
    for raw_line in lines[2:]:
        cells = [cell.strip() for cell in raw_line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        rows.append(
            {
                "start": cells[0],
                "end": cells[1],
                "visual_suggestion": cells[2],
                "image_prompt": cells[2],
                "note": cells[3],
            }
        )
    if not rows:
        raise RuntimeError("提案表の行を解析できませんでした。")
    return rows


def generate_ponchi_suggestions_with_openai(
    api_key: str,
    items: List[Dict[str, Any]],
    model: str = DEFAULT_PONCHI_OPENAI_MODEL,
) -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("ChatGPT APIキーを入力してください。")
    if not items:
        raise RuntimeError("SRT字幕が空です。")

    prompt_lines = [
        "You are planning storyboard-like illustrations (ponchi-e) for a video.",
        "For each subtitle, suggest what should be displayed and provide an image generation prompt in Japanese.",
        "Return ONLY a JSON array. Each item must include:",
        "start, end, text, visual_suggestion, image_prompt",
        "",
        "Subtitles:",
    ]
    for idx, item in enumerate(items, 1):
        start = format_seconds_to_timecode(item.get("start", 0))
        end = format_seconds_to_timecode(item.get("end", 0))
        prompt_lines.append(f"{idx}. {start}〜{end} {item.get('text', '')}")

    payload = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": "You output strictly valid JSON and nothing else.",
            },
            {"role": "user", "content": "\n".join(prompt_lines)},
        ],
    }
    resp = _post_openai_with_retry(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    if not resp.ok:
        raise RuntimeError(f"ChatGPT APIエラー: {resp.status_code} {resp.text}")
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("ChatGPTから提案の取得に失敗しました。")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise RuntimeError("ChatGPTのJSON解析に失敗しました。")
    if isinstance(parsed, dict) and "items" in parsed:
        parsed = parsed["items"]
    if not isinstance(parsed, list):
        raise RuntimeError("提案は配列形式で返してください。")
    return parsed


def parse_timecode_to_seconds(s: str) -> float:
    """
    "mm:ss" / "hh:mm:ss" / "ss" を秒に。
    例: "00:12" -> 12, "1:02" -> 62, "00:01:30" -> 90
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("時間が空です。")

    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    parts = s.split(":")
    if len(parts) == 2:  # mm:ss
        m = int(parts[0])
        sec = float(parts[1])
        return m * 60 + sec
    if len(parts) == 3:  # hh:mm:ss
        h = int(parts[0])
        m = int(parts[1])
        sec = float(parts[2])
        return h * 3600 + m * 60 + sec

    raise ValueError(f"時間形式が不正です: {s}")


def parse_srt_timecode_to_seconds(s: str) -> float:
    """SRT形式 (00:00:01,234) を秒に変換する。"""
    if not s:
        raise ValueError("SRT 時間が空です。")
    cleaned = s.replace(",", ".")
    return parse_timecode_to_seconds(cleaned)


def parse_srt_file(path: str) -> List[Dict[str, Any]]:
    """SRT を読み込んで start/end/text の配列にする。"""
    content = Path(path).read_text(encoding="utf-8").splitlines()
    items: List[Dict[str, Any]] = []
    i = 0
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    while i < len(content):
        line = content[i].strip()
        if not line:
            i += 1
            continue
        if line.isdigit():
            i += 1
            if i >= len(content):
                break
            line = content[i].strip()

        match = time_re.search(line)
        if not match:
            i += 1
            continue

        start_s, end_s = match.groups()
        i += 1
        text_lines = []
        while i < len(content) and content[i].strip():
            text_lines.append(content[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        if text:
            items.append(
                {
                    "start": parse_srt_timecode_to_seconds(start_s),
                    "end": parse_srt_timecode_to_seconds(end_s),
                    "text": text,
                }
            )
        i += 1

    return items


def extract_script_text(path: str) -> str:
    if not path:
        raise ValueError("台本ファイルのパスが空です。")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError("台本ファイルが見つかりません。")
    suffix = target.suffix.lower()
    if suffix == ".srt":
        items = parse_srt_file(str(target))
        if not items:
            raise ValueError("SRTに字幕が見つかりませんでした。")
        text = "\n".join(item.get("text", "") for item in items if item.get("text"))
    else:
        text = target.read_text(encoding="utf-8")
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        raise ValueError("台本テキストが空です。")
    return cleaned
