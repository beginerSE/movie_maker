# -*- coding: utf-8 -*-
"""
News Short Generator Studio（Windows向け）
- 左：React風サイドバー（アイコン＋メニュー）
- 中央：フォーム（「動画生成」「台本生成」「動画タイトル・説明作成」「資料作成」「動画編集」「詳細動画編集」ページ切替）
- 右：ログ（+ 進捗）

[動画編集（NEW）]
- MP4 を読み込み
- 指定時間帯（例: 00:12〜00:18）に、選択した画像を指定座標（例: x=100,y=200）へ重ねる
- 複数オーバーレイに対応（リストに追加して一括書き出し）
- 書き出しは moviepy の CompositeVideoClip を使用
"""

import os
import re
import json
import threading
import traceback
import tkinter as tk
import mimetypes
import base64
from tkinter import filedialog, messagebox, simpledialog
from proglog import ProgressBarLogger
import tkinter.font as tkfont
import time
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
from itertools import cycle
from urllib.parse import urlparse, unquote

import customtkinter as ctk

import requests  # VOICEVOX 用

# Gemini
from google import genai
from google.genai import types

# Claude (Anthropic)
import anthropic

# OpenAI (ChatGPT)
from openai import OpenAI

from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    concatenate_audioclips,
)
# from moviepy.audio.fx import all as afx
import moviepy.audio.fx as afx

from moviepy.audio.AudioClip import CompositeAudioClip

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import wave

from tkinter import ttk
from PIL import ImageTk


class VideoEditorPage(ttk.Frame):
    def __init__(self, master, app_context=None):
        super().__init__(master)
        self.app_context = app_context

        ttk.Label(self, text="動画編集", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 6))
        self.overlay_frame.pack(fill="both", expand=True)

    def get_overlay_config(self):
        return self.overlay_frame.export_overlays()


# ==========================
# 設定保存ファイル
# ==========================
CONFIG_PATH = Path.home() / "Config.json"
LEGACY_CONFIG_PATH = Path.home() / ".news_short_generator_studio.json"


# ==========================
# 文字列処理／字幕関連のユーティリティ
# ==========================
def split_sentences_jp(text: str):
    """「。」「．」「！」「？」などの直後で文を分割（句点は残す）"""
    parts = re.split(r"(?<=[。．！？?!])", text)
    return [p.strip() for p in parts if p.strip()]


def allocate_durations_by_length(sentences, total_duration: float):
    """文の文字数比で自然に時間配分"""
    lens = [max(1, len(s)) for s in sentences]
    s = sum(lens)
    if s == 0:
        return [total_duration]
    return [total_duration * (L / s) for L in lens]


def parse_hex_color(hex_str: str, default=(255, 255, 255, 255)) -> Tuple[int, int, int, int]:
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

# 動画編集：JSONインポート時の既定値
DEFAULT_EDIT_IMPORT_X = 100
DEFAULT_EDIT_IMPORT_Y = 200
DEFAULT_EDIT_IMPORT_W = 0
DEFAULT_EDIT_IMPORT_H = 0
DEFAULT_EDIT_IMPORT_OPACITY = 1.0
DEFAULT_IMAGE_SEARCH_PROVIDER = "Google"

# 詳細動画編集（プロジェクト）
DETAILED_PROJECT_EXT = ".mmproj"
DETAILED_AUTOSAVE_INTERVAL_MS = 30000

DEFAULT_PONCHI_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_PONCHI_OPENAI_MODEL = "gpt-4.1-mini"

SCRIPT_MODEL_MASTER = {
    "Gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "ChatGPT": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
    ],
    "ClaudeCode": [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
}


SPEAKER_ALIASES = {
    "キャスター": ["キャスター", "司会", "アナウンサー", "MC", "Caster"],
    "アナリスト": ["アナリスト", "解説", "専門家", "Analyst"],
}
SPEAKER_KEYS = list(SPEAKER_ALIASES.keys())


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


def pick_font(size: int):
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


def save_wave(filename: str, pcm: bytes, channels=1, rate=24000, sample_width=2):
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
):
    """
    1セリフを「文ごとに字幕を切り替える」クリップへ変換。
    音声はセリフ全体をそのまま載せ、見た目だけ文ごとに切り替える。
    """
    base = Image.open(str(image_path)).convert("RGB")
    bg_rgba = letterbox_fit(base, target_size).convert("RGBA")

    audio_full = AudioFileClip(str(audio_path))
    total_dur = audio_full.duration

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

    line_visual = concatenate_videoclips(sentence_clips, method="compose")
    line_visual.audio = audio_full
    return line_visual


def write_srt(lines: List[Tuple[str, str]], out_srt: Path, per_line_secs: List[float]):
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


# ==========================
# VOICEVOX ヘルパー
# ==========================
def fetch_voicevox_speakers(base_url: str) -> List[Dict[str, Any]]:
    base_url = base_url.rstrip("/")
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
        return int(label)
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


def tts_with_voicevox(
    text: str,
    speaker_id: int,
    out_wav: Path,
    base_url: str = DEFAULT_VOICEVOX_URL,
    speed_scale: float = DEFAULT_VV_SPEED,
):
    base_url = base_url.rstrip("/")
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

    def __init__(self, progress_fn, base=0.8, span=0.2):
        super().__init__()
        self.progress_fn = progress_fn
        self.base = float(base)
        self.span = float(span)

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
            rate = self.bars[bar].get("rate")        # it/s

            eta = None
            if rate and isinstance(rate, (int, float)) and rate > 0:
                remaining = max(0.0, total - idx)
                eta = remaining / float(rate)
            elif elapsed and isinstance(elapsed, (int, float)) and ratio > 0.01:
                total_est = float(elapsed) / ratio
                eta = max(0.0, total_est - float(elapsed))

            if self.progress_fn:
                try:
                    self.progress_fn(v, eta)
                except TypeError:
                    self.progress_fn(v)

        except Exception:
            pass


# ==========================
# 動画生成 メイン処理
# ==========================
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
):
    def update_progress(v: float):
        if progress_fn is not None:
            progress_fn(max(0.0, min(1.0, float(v))))

    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        engine = (tts_engine or "Gemini").strip().lower()
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

        vv_mode_int = "rotation" if vv_mode == "rotation" else "two_person"
        vv_rotation_ids: List[int] = DEFAULT_VV_ROTATION

        if engine == "voicevox":
            if vv_rotation_labels:
                resolved_ids = []
                for lbl in vv_rotation_labels:
                    sid = resolve_voicevox_speaker_label(lbl, vv_speakers, None)
                    resolved_ids.append(sid)
                if resolved_ids:
                    vv_rotation_ids = resolved_ids

            vv_caster_id = resolve_voicevox_speaker_label(vv_caster_label, vv_speakers, vv_rotation_ids[0])
            vv_analyst_id = resolve_voicevox_speaker_label(
                vv_analyst_label,
                vv_speakers,
                vv_rotation_ids[1] if len(vv_rotation_ids) > 1 else vv_rotation_ids[0],
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
        audio_paths = []
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
        TARGET_RESOLUTION = (width, height)
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
                target_size=TARGET_RESOLUTION,
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
        final = concatenate_videoclips(clips, method="compose")
        update_progress(0.8)

        # ===== BGM を動画全体にループ適用 =====
        if use_bgm and bgm_path:
            try:
                log_fn("動画全体にBGMをループ適用中...")
                bgm_clip = AudioFileClip(bgm_path)

                bgm_duration = bgm_clip.duration
                if not bgm_duration or bgm_duration <= 0:
                    raise RuntimeError("BGM の長さが 0 秒です。")

                loops = int(final.duration // bgm_duration) + 1
                bgm_long = concatenate_audioclips([bgm_clip] * loops)
                bgm_looped = bgm_long.with_duration(final.duration)

                gain = 10 ** (bgm_gain_db / 20.0)
                bgm_looped = bgm_looped.with_volume_scaled(gain)

                voice_audio = final.audio
                final.audio = CompositeAudioClip([voice_audio, bgm_looped])

                log_fn("BGM 合成完了。")
            except Exception as e:
                log_fn(f"BGM 合成中にエラーが発生しました（BGMなしで続行します）: {e}")

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


# ==========================
# Claude 台本生成
# ==========================
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


# ==========================
# Gemini 台本生成
# ==========================
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


# ==========================
# ChatGPT 台本生成
# ==========================
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
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens

    resp = _post_openai_with_retry(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    if not resp.ok:
        raise RuntimeError(_format_openai_error(resp))
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("ChatGPTから台本の取得に失敗しました。")
    return content.strip()


# ==========================
# Gemini 画像生成
# ==========================
# GEMINI_MATERIAL_DEFAULT_MODEL = "gemini-2.5-flash-image"
GEMINI_MATERIAL_DEFAULT_MODEL = "gemini-3-pro-image-preview"
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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["Image"]},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
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


# ==========================
# 動画編集（NEW）
# ==========================
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
        "あなたは動画用の資料（ポンチ絵）を企画する担当者です。",
        "次の字幕ごとに、表示すると良い資料の内容を提案してください。",
        "各項目で「visual_suggestion（表示内容の説明）」と「image_prompt（画像生成用の日本語プロンプト）」を作ってください。",
        "出力は JSON 配列のみで、各要素は次のキーを含めてください:",
        'start, end, text, visual_suggestion, image_prompt',
        "",
        "字幕一覧:",
    ]
    for idx, item in enumerate(items, 1):
        start = format_seconds_to_timecode(item.get("start", 0))
        end = format_seconds_to_timecode(item.get("end", 0))
        prompt_lines.append(f"{idx}. {start}〜{end} {item.get('text', '')}")

    resp = client.models.generate_content(
        model=model,
        contents="\n".join(prompt_lines),
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    text = getattr(resp, "text", "") or ""
    if not text:
        raise RuntimeError("Geminiから提案の取得に失敗しました。")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise RuntimeError("提案JSONの解析に失敗しました。")
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise RuntimeError("提案は配列形式で返してください。")
    return data


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


def search_images_serpapi(api_key: str, query: str, provider: str = "Google") -> List[str]:
    if api_key.startswith("AIza"):
        raise RuntimeError(
            "SerpAPIのAPIキーではなくGoogle APIキーが入力されています。"
            "SerpAPIの管理画面で発行したAPIキーを使用してください。"
        )
    engine = "google_images" if provider == "Google" else "bing_images"
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": engine,
            "q": query,
            "api_key": api_key,
            "num": 5,
        },
        timeout=20,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "SerpAPIの認証に失敗しました。APIキーが無効か、SerpAPIの画像検索が有効化されていません。"
        )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("images_results", []) or []
    urls = []
    for item in results:
        url = item.get("original") or item.get("thumbnail")
        if url:
            urls.append(url)
    return urls


def download_image(url: str, output_dir: Path, basename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    ext = Path(name).suffix.lower()
    if not ext or len(ext) > 5:
        ext = ""

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if not ext and content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    if not ext:
        ext = ".jpg"

    safe_base = re.sub(r"[^\w\-]+", "_", basename).strip("_") or "image"
    filename = f"{safe_base}{ext}"
    out_path = output_dir / filename
    counter = 1
    while out_path.exists():
        out_path = output_dir / f"{safe_base}_{counter}{ext}"
        counter += 1
    out_path.write_bytes(resp.content)
    return out_path


def apply_image_overlays_to_video(
    input_mp4: str,
    overlays: List[Dict[str, Any]],
    output_mp4: str,
    log_fn,
    progress_fn=None,
):
    """
    overlays: [
      {
        "image_path": "...png",
        "start": 12.0,
        "end": 18.0,
        "x": 100,
        "y": 200,
        "w": 400,     # optional
        "h": 0,       # optional
        "opacity": 1.0
      }, ...
    ]
    """
    def update_progress(v: float, eta: float | None = None):
        if progress_fn is not None:
            try:
                progress_fn(v, eta)
            except TypeError:
                progress_fn(v)

    if not input_mp4 or not Path(input_mp4).exists():
        raise RuntimeError("入力MP4が見つかりません。")
    if not overlays:
        raise RuntimeError("オーバーレイが1件もありません。")
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_fn(f"[EDIT] 入力: {input_mp4}")
    log_fn(f"[EDIT] 出力: {output_mp4}")
    update_progress(0.05, None)

    base = VideoFileClip(input_mp4)
    try:
        dur = float(base.duration or 0)
        if dur <= 0:
            raise RuntimeError("入力動画のdurationが取得できません。")

        comps = [base]
        n = len(overlays)

        for i, ov in enumerate(overlays, 1):
            img_path = Path(ov["image_path"])
            if not img_path.exists():
                raise RuntimeError(f"画像が見つかりません: {img_path}")

            start = float(ov["start"])
            end = float(ov["end"])
            if end <= start:
                raise RuntimeError(f"時間帯が不正です（end<=start）: {start}〜{end}")

            if start < 0 or end < 0:
                raise RuntimeError("時間は0以上で指定してください。")
            if start > dur:
                raise RuntimeError(f"start が動画長を超えています: start={start}, duration={dur}")
            end = min(end, dur)

            x = int(ov.get("x", 0))
            y = int(ov.get("y", 0))
            w = int(ov.get("w", 0) or 0)
            h = int(ov.get("h", 0) or 0)
            opacity = float(ov.get("opacity", 1.0) or 1.0)
            opacity = max(0.0, min(1.0, opacity))

            log_fn(f"[EDIT] {i}/{n} overlay: {img_path.name}  t={start:.2f}〜{end:.2f}  pos=({x},{y})  size=({w},{h})  op={opacity}")

            ic = ImageClip(str(img_path))
            if w > 0 and h > 0:
                ic = ic.resized(new_size=(w, h))
            elif w > 0:
                ic = ic.resized(width=w)
            elif h > 0:
                ic = ic.resized(height=h)

            ic = ic.with_start(start).with_end(end).with_position((x, y)).with_opacity(opacity)
            comps.append(ic)

            update_progress(0.05 + 0.60 * (i / n), None)

        final = CompositeVideoClip(comps, size=base.size)
        final = final.with_duration(dur)
        update_progress(0.70, None)

        log_fn("[EDIT] 書き出し開始...")
        mp_logger = TkMoviePyLogger(progress_fn=progress_fn, base=0.70, span=0.30)

        final.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            fps=int(base.fps or 30),
            logger=mp_logger,
        )
        update_progress(1.0, 0)

        log_fn("[EDIT] ✅ 完了しました。")

        final.close()

    finally:
        base.close()


# ==========================
# GUI
# ==========================
class NewsShortGeneratorStudio(ctk.CTk):
    # --- Theme ---
    COL_BG = "#0b0f1a"
    COL_PANEL = "#121826"
    COL_PANEL2 = "#0f172a"
    COL_CARD = "#172033"
    COL_CARD_SOFT = "#1b2438"
    COL_BORDER = "#25314a"
    COL_TEXT = "#e8edf7"
    COL_MUTED = "#9aa9c2"
    COL_ACCENT = "#6366f1"
    COL_ACCENT_HOVER = "#4f46e5"
    COL_ACCENT_SOFT = "#1f2440"
    COL_DANGER = "#ef5350"
    COL_DANGER_HOVER = "#d64543"
    COL_OK = "#10b981"
    COL_OK_HOVER = "#0f9e6d"

    SIDEBAR_W = 240
    LOG_W = 320
    EDIT_PREVIEW_MAX = (640, 360)

    def __init__(self):
        super().__init__()

        self.title("Studio - News Short Generator")
        self.geometry("1220x760")
        self.minsize(1100, 680)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color=self.COL_BG)

        # ==========================
        # GUI用フォントを統一
        # ==========================
        self.FONT_FAMILY = self._pick_font_family([
            "Noto Sans JP",
            "BIZ UDGothic",
            "BIZ UDゴシック",
            "Yu Gothic UI",
            "Yu Gothic",
            "Meiryo UI",
            "Meiryo",
            "Segoe UI",
        ])

        self.FONT_TITLE = ctk.CTkFont(family=self.FONT_FAMILY, size=20, weight="bold")
        self.FONT_SUB = ctk.CTkFont(family=self.FONT_FAMILY, size=12)
        self.FONT_NAV = ctk.CTkFont(family=self.FONT_FAMILY, size=13, weight="bold")
        self.FONT_PILL = ctk.CTkFont(family=self.FONT_FAMILY, size=12, weight="bold")
        self._apply_theme_overrides()
        # ==========================

        # state
        self.image_paths: List[str] = []
        self.active_page = "video"

        self.page_title_labels: Dict[str, ctk.CTkLabel] = {}
        self.page_pills: Dict[str, ctk.CTkLabel] = {}

        self.prompt_templates: Dict[str, str] = {}
        self.prompt_template_names: List[str] = []
        self.prompt_template_var = ctk.StringVar(value="")
        self.script_engine_var = ctk.StringVar(value="ClaudeCode")
        self.script_gemini_model_var = ctk.StringVar(value=DEFAULT_SCRIPT_GEMINI_MODEL)
        self.script_chatgpt_model_var = ctk.StringVar(value=DEFAULT_SCRIPT_OPENAI_MODEL)
        self.script_claude_model_var = ctk.StringVar(value=DEFAULT_CLAUDE_MODEL)
        self.title_engine_var = ctk.StringVar(value="Gemini")
        self.title_gemini_model_var = ctk.StringVar(value=DEFAULT_SCRIPT_GEMINI_MODEL)
        self.title_chatgpt_model_var = ctk.StringVar(value=DEFAULT_SCRIPT_OPENAI_MODEL)
        self.title_claude_model_var = ctk.StringVar(value=DEFAULT_CLAUDE_MODEL)

        self.edit_overlays: List[Dict[str, Any]] = []
        self._edit_preview_base: Image.Image | None = None
        self._edit_preview_overlay: Image.Image | None = None
        self._edit_preview_imgtk = None
        self._edit_overlay_original_size = None
        self._edit_preview_size = None
        self._edit_cell_entry = None
        self._edit_detail_labels: Dict[str, ctk.CTkLabel] = {}
        self.detailed_project_path: Optional[Path] = None
        self.detailed_project_data: Dict[str, Any] = self._default_detailed_project()
        self.detailed_assets: List[Dict[str, Any]] = []
        self.detailed_timeline: List[Dict[str, Any]] = []
        self.detailed_overlays: List[Dict[str, Any]] = []
        self._detailed_preview_clip: Optional[VideoFileClip] = None
        self._detailed_preview_playing = False
        self._detailed_preview_job = None
        self._detailed_preview_time = 0.0
        self._detailed_preview_imgtk = None
        self._detailed_asset_imgtk = None
        self._detailed_drag_timeline_iid = None
        self._detailed_autosave_job = None
        self._detailed_dirty = False
        self.ponchi_suggestions: List[Dict[str, Any]] = []

        # build UI
        self._build_layout()
        self._build_sidebar()
        self._build_center_pages()
        self._build_log_panel()

        self.load_config()
        self.switch_page("video")
        self._edit_thumb_imgtk = None
        self._sync_edit_preview_state()
        self._schedule_detailed_autosave()

    def _pick_font_family(self, candidates: list[str]) -> str:
        try:
            fams = set(tkfont.families())
        except Exception:
            fams = set()
        for name in candidates:
            if name in fams:
                return name
        # 最後の砦
        return "TkDefaultFont"

    def _apply_theme_overrides(self) -> None:
        theme = ctk.ThemeManager.theme

        def _set(widget: str, key: str, value) -> None:
            if widget not in theme:
                return
            theme[widget][key] = value

        _set("CTkButton", "corner_radius", 16)
        _set("CTkButton", "border_width", 1)
        _set("CTkButton", "border_color", self.COL_BORDER)
        _set("CTkButton", "fg_color", self.COL_CARD)
        _set("CTkButton", "hover_color", self.COL_CARD_SOFT)
        _set("CTkButton", "text_color", self.COL_TEXT)
        _set("CTkButton", "font", (self.FONT_FAMILY, 12, "bold"))

        _set("CTkEntry", "corner_radius", 14)
        _set("CTkEntry", "border_width", 1)
        _set("CTkEntry", "border_color", self.COL_BORDER)
        _set("CTkEntry", "fg_color", self.COL_PANEL)
        _set("CTkEntry", "text_color", self.COL_TEXT)
        _set("CTkEntry", "font", (self.FONT_FAMILY, 11))

        _set("CTkOptionMenu", "corner_radius", 14)
        _set("CTkOptionMenu", "button_color", self.COL_CARD)
        _set("CTkOptionMenu", "button_hover_color", self.COL_CARD_SOFT)
        _set("CTkOptionMenu", "fg_color", self.COL_PANEL)
        _set("CTkOptionMenu", "text_color", self.COL_TEXT)
        _set("CTkOptionMenu", "font", (self.FONT_FAMILY, 11))

        _set("CTkLabel", "text_color", self.COL_TEXT)
        _set("CTkLabel", "font", (self.FONT_FAMILY, 11))

        _set("CTkTextbox", "corner_radius", 14)
        _set("CTkTextbox", "border_width", 1)
        _set("CTkTextbox", "border_color", self.COL_BORDER)
        _set("CTkTextbox", "fg_color", self.COL_PANEL)
        _set("CTkTextbox", "text_color", self.COL_TEXT)
        _set("CTkTextbox", "font", (self.FONT_FAMILY, 11))
        _set("CTkCheckBox", "font", (self.FONT_FAMILY, 11))
        _set("CTkCheckBox", "text_color", self.COL_TEXT)

        _set("CTkSlider", "button_corner_radius", 10)
        _set("CTkSlider", "button_color", self.COL_ACCENT)
        _set("CTkSlider", "button_hover_color", self.COL_ACCENT_HOVER)

    def _sync_option_menu_values(
        self,
        menu: ctk.CTkOptionMenu,
        var: ctk.StringVar,
        values: list[str],
        selected: str,
        fallback: str,
    ) -> None:
        next_value = selected or fallback
        options = list(values)
        if next_value and next_value not in options:
            options = [next_value] + options
        menu.configure(values=options)
        if next_value:
            var.set(next_value)

    def _fetch_openai_models(self, api_key: str) -> list[str]:
        if not api_key:
            return []
        client = OpenAI(api_key=api_key)
        resp = client.models.list()
        models = [getattr(item, "id", "") for item in getattr(resp, "data", [])]
        filtered = [m for m in models if m and (m.startswith("gpt") or m.startswith("o"))]
        return sorted(set(filtered))

    def _fetch_gemini_models(self, api_key: str) -> list[str]:
        if not api_key:
            return []
        client = genai.Client(api_key=api_key)
        models = []
        for item in client.models.list():
            name = getattr(item, "name", "") or getattr(item, "model", "")
            if not name:
                continue
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if "gemini" not in name:
                continue
            models.append(name)
        return sorted(set(models))

    def _fetch_claude_models(self, api_key: str) -> list[str]:
        if not api_key:
            return []
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.models.list()
        models = [getattr(item, "id", "") for item in getattr(resp, "data", [])]
        filtered = [m for m in models if m and m.startswith("claude")]
        return sorted(set(filtered))

    def _apply_script_model_options(
        self,
        gemini_models: list[str] | None,
        chatgpt_models: list[str] | None,
        claude_models: list[str] | None,
    ) -> None:
        if hasattr(self, "script_gemini_model_menu"):
            values = gemini_models or SCRIPT_MODEL_MASTER["Gemini"]
            self._sync_option_menu_values(
                self.script_gemini_model_menu,
                self.script_gemini_model_var,
                values,
                self.script_gemini_model_var.get(),
                DEFAULT_SCRIPT_GEMINI_MODEL,
            )
        if hasattr(self, "script_chatgpt_model_menu"):
            values = chatgpt_models or SCRIPT_MODEL_MASTER["ChatGPT"]
            self._sync_option_menu_values(
                self.script_chatgpt_model_menu,
                self.script_chatgpt_model_var,
                values,
                self.script_chatgpt_model_var.get(),
                DEFAULT_SCRIPT_OPENAI_MODEL,
            )
        if hasattr(self, "script_claude_model_menu"):
            values = claude_models or SCRIPT_MODEL_MASTER["ClaudeCode"]
            self._sync_option_menu_values(
                self.script_claude_model_menu,
                self.script_claude_model_var,
                values,
                self.script_claude_model_var.get(),
                DEFAULT_CLAUDE_MODEL,
            )
        if hasattr(self, "title_gemini_model_menu"):
            values = gemini_models or SCRIPT_MODEL_MASTER["Gemini"]
            self._sync_option_menu_values(
                self.title_gemini_model_menu,
                self.title_gemini_model_var,
                values,
                self.title_gemini_model_var.get(),
                DEFAULT_SCRIPT_GEMINI_MODEL,
            )
        if hasattr(self, "title_chatgpt_model_menu"):
            values = chatgpt_models or SCRIPT_MODEL_MASTER["ChatGPT"]
            self._sync_option_menu_values(
                self.title_chatgpt_model_menu,
                self.title_chatgpt_model_var,
                values,
                self.title_chatgpt_model_var.get(),
                DEFAULT_SCRIPT_OPENAI_MODEL,
            )
        if hasattr(self, "title_claude_model_menu"):
            values = claude_models or SCRIPT_MODEL_MASTER["ClaudeCode"]
            self._sync_option_menu_values(
                self.title_claude_model_menu,
                self.title_claude_model_var,
                values,
                self.title_claude_model_var.get(),
                DEFAULT_CLAUDE_MODEL,
            )

    def _refresh_script_model_options(self) -> None:
        def worker():
            gemini_models = None
            chatgpt_models = None
            claude_models = None
            try:
                gemini_key = self._get_gemini_api_key()
                chatgpt_key = self._get_chatgpt_api_key()
                claude_key = self._get_claude_api_key()
                if gemini_key:
                    gemini_models = self._fetch_gemini_models(gemini_key)
                if chatgpt_key:
                    chatgpt_models = self._fetch_openai_models(chatgpt_key)
                if claude_key:
                    claude_models = self._fetch_claude_models(claude_key)
            except Exception as exc:
                self.log(f"⚠️ モデル一覧の取得に失敗しました: {exc}")
            self.after(0, lambda: self._apply_script_model_options(
                gemini_models,
                chatgpt_models,
                claude_models,
            ))

        threading.Thread(target=worker, daemon=True).start()

    def _default_detailed_project(self) -> Dict[str, Any]:
        return {
            "name": "",
            "root_dir": "",
            "input_dir": "",
            "output_dir": "",
            "assets": [],
            "main_video": "",
            "timeline": [],
            "overlays": [],
            "audio": {
                "bgm_path": "",
                "bgm_volume": 0.7,
                "video_audio": True,
                "video_volume": 1.0,
                "fade_in": 0.0,
                "fade_out": 0.0,
            },
            "export": {
                "output_path": "",
                "resolution": "1920x1080",
                "fps": 30,
            },
        }

    



    # ==========================
    # 追加：ステータス更新
    # ==========================
    def set_status(self, text: str, ok: bool = True):
        def _apply():
            if ok:
                self.status_pill.configure(
                    text=f" {text} ",
                    fg_color="#0f2a20",
                    text_color="#c8ffe7",
                )
            else:
                self.status_pill.configure(
                    text=f" {text} ",
                    fg_color="#3a1b24",
                    text_color="#ffd4df",
                )
        self.after(0, _apply)
    def _get_selected_tree_iid(self) -> str | None:
        if not hasattr(self, "edit_tree"):
            return None
        sel = self.edit_tree.selection()
        if not sel:
            return None
        return sel[0]

    def _selected_overlay_index(self) -> int | None:
        iid = self._get_selected_tree_iid()
        if iid is None:
            return None
        try:
            # iid は "0","1","2"...（index）として扱う
            return int(iid)
        except Exception:
            return None

    def _setup_edit_tree_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Overlay.Treeview",
            background=self.COL_PANEL2,
            fieldbackground=self.COL_PANEL2,
            foreground=self.COL_TEXT,
            bordercolor=self.COL_BORDER,
            rowheight=30,
            font=(self.FONT_FAMILY, 11),
        )
        style.configure(
            "Overlay.Treeview.Heading",
            background=self.COL_CARD,
            foreground=self.COL_TEXT,
            relief="flat",
            font=(self.FONT_FAMILY, 11, "bold"),
        )
        style.map(
            "Overlay.Treeview",
            background=[("selected", "#1c3a68")],
            foreground=[("selected", "#ffffff")],
        )
        style.map(
            "Overlay.Treeview.Heading",
            background=[("active", "#1b2a44")],
            foreground=[("active", "#ffffff")],
        )

    def _setup_detailed_asset_tree_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Detailed.Treeview",
            background=self.COL_PANEL2,
            fieldbackground=self.COL_PANEL2,
            foreground=self.COL_TEXT,
            bordercolor=self.COL_BORDER,
            rowheight=28,
            font=(self.FONT_FAMILY, 10),
        )
        style.configure(
            "Detailed.Treeview.Heading",
            background=self.COL_CARD,
            foreground=self.COL_TEXT,
            relief="flat",
            font=(self.FONT_FAMILY, 10, "bold"),
        )
        style.map(
            "Detailed.Treeview",
            background=[("selected", "#1c3a68")],
            foreground=[("selected", "#ffffff")],
        )
        style.map(
            "Detailed.Treeview.Heading",
            background=[("active", "#1b2a44")],
            foreground=[("active", "#ffffff")],
        )

    def _setup_detailed_timeline_tree_style(self):
        self._setup_detailed_asset_tree_style()


    def refresh_edit_overlay_table(self):
        if not hasattr(self, "edit_tree"):
            return

        # 選択解除
        try:
            self.edit_tree.selection_remove(self.edit_tree.selection())
        except Exception:
            pass

        # 全削除
        for item in self.edit_tree.get_children():
            self.edit_tree.delete(item)

        # 再投入
        for idx, ov in enumerate(self.edit_overlays):
            img_path = str(ov.get("image_path", ""))
            img_name = Path(img_path).name if img_path else ""

            start = float(ov.get("start", 0.0) or 0.0)
            end = float(ov.get("end", 0.0) or 0.0)
            x = int(ov.get("x", 0) or 0)
            y = int(ov.get("y", 0) or 0)
            w = int(ov.get("w", 0) or 0)
            h = int(ov.get("h", 0) or 0)
            op = float(ov.get("opacity", 1.0) or 1.0)

            tag = "even" if idx % 2 == 0 else "odd"
            self.edit_tree.insert(
                "", "end",
                iid=str(idx),
                tags=(tag,),
                values=(img_name, f"{start:.2f}", f"{end:.2f}", x, y, w, h, f"{op:.2f}")
            )

        # 旧リスト（残すなら同期）
        self.refresh_edit_overlay_list()

        # プレビュー初期化
        if hasattr(self, "edit_thumb_label"):
            self._edit_thumb_imgtk = None
            self.edit_thumb_label.configure(image="", text="（行を選択すると表示）")
        if hasattr(self, "edit_thumb_path"):
            self.edit_thumb_path.configure(text="")
        if self._edit_detail_labels:
            for label in self._edit_detail_labels.values():
                label.configure(text="--")

        self._sync_edit_preview_state()


    def _apply_overlay_to_form(self, ov: Dict[str, Any], *, log_message: bool = False):
        self.edit_overlay_img_entry.delete(0, "end")
        self.edit_overlay_img_entry.insert(0, ov.get("image_path", ""))
        self._load_edit_overlay_preview(ov.get("image_path", ""))

        self.edit_start_entry.delete(0, "end")
        self.edit_start_entry.insert(0, str(ov.get("start", 0)))

        self.edit_end_entry.delete(0, "end")
        self.edit_end_entry.insert(0, str(ov.get("end", 0)))

        self.edit_x_entry.delete(0, "end")
        self.edit_x_entry.insert(0, str(ov.get("x", 0)))

        self.edit_y_entry.delete(0, "end")
        self.edit_y_entry.insert(0, str(ov.get("y", 0)))

        self.edit_w_entry.delete(0, "end")
        self.edit_w_entry.insert(0, str(ov.get("w", 0)))

        self.edit_h_entry.delete(0, "end")
        self.edit_h_entry.insert(0, str(ov.get("h", 0)))

        self.edit_opacity_entry.delete(0, "end")
        self.edit_opacity_entry.insert(0, str(ov.get("opacity", 1.0)))

        if hasattr(self, "edit_preview_x_slider"):
            self.edit_preview_x_slider.set(float(ov.get("x", 0)))
        if hasattr(self, "edit_preview_y_slider"):
            self.edit_preview_y_slider.set(float(ov.get("y", 0)))
        if hasattr(self, "edit_preview_scale_slider"):
            target_w = int(ov.get("w", 0) or 0)
            if target_w > 0 and self._edit_overlay_original_size:
                base_w = self._edit_overlay_original_size[0]
                scale = max(1, min(300, int((target_w / base_w) * 100)))
                self.edit_preview_scale_slider.set(scale)
            else:
                self.edit_preview_scale_slider.set(100)
        self._sync_edit_preview_from_sliders()

        if log_message:
            self.log("✏️ 選択行をフォームへ反映しました。")

    def _update_edit_detail_panel(self, ov: Dict[str, Any]):
        if not self._edit_detail_labels:
            return
        self._edit_detail_labels["start"].configure(text=f"{float(ov.get('start', 0.0)):.2f}s")
        self._edit_detail_labels["end"].configure(text=f"{float(ov.get('end', 0.0)):.2f}s")
        self._edit_detail_labels["x"].configure(text=str(int(ov.get("x", 0) or 0)))
        self._edit_detail_labels["y"].configure(text=str(int(ov.get("y", 0) or 0)))
        self._edit_detail_labels["w"].configure(text=str(int(ov.get("w", 0) or 0)))
        self._edit_detail_labels["h"].configure(text=str(int(ov.get("h", 0) or 0)))
        self._edit_detail_labels["opacity"].configure(
            text=f"{float(ov.get('opacity', 1.0)):.2f}"
        )


    def on_edit_overlay_select(self, _event=None):
        idx = self._selected_overlay_index()
        if idx is None:
            return
        if idx < 0 or idx >= len(self.edit_overlays):
            return

        ov = self.edit_overlays[idx]
        self._update_edit_detail_panel(ov)
        self._apply_overlay_to_form(ov)
        img_path = ov.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            if hasattr(self, "edit_thumb_label"):
                self._edit_thumb_imgtk = None
                self.edit_thumb_label.configure(text="画像が見つかりません", image="")
            if hasattr(self, "edit_thumb_path"):
                self.edit_thumb_path.configure(text=str(img_path))
            self._edit_thumb_imgtk = None
            return

        # サムネ生成
        try:
            im = Image.open(img_path).convert("RGBA")
            im.thumbnail((360, 360))
            self._edit_thumb_imgtk = ImageTk.PhotoImage(im)
            self.edit_thumb_label.configure(text="", image=self._edit_thumb_imgtk)
            self.edit_thumb_path.configure(text=str(img_path))
        except Exception as e:
            self._edit_thumb_imgtk = None
            self.edit_thumb_label.configure(text=f"サムネ生成失敗: {e}", image="")
            self.edit_thumb_path.configure(text=str(img_path))

    def on_edit_tree_double_click(self, event):
        if not hasattr(self, "edit_tree"):
            return
        if self._edit_cell_entry is not None:
            self._edit_cell_entry.destroy()
            self._edit_cell_entry = None

        region = self.edit_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = self.edit_tree.identify_row(event.y)
        column_id = self.edit_tree.identify_column(event.x)
        if not row_id or not column_id:
            return

        try:
            row_index = int(row_id)
        except ValueError:
            return
        if row_index < 0 or row_index >= len(self.edit_overlays):
            return

        columns = self.edit_tree["columns"]
        col_index = int(column_id.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(columns):
            return
        column_key = columns[col_index]

        ov = self.edit_overlays[row_index]
        current_value = ov.get("image_path", "") if column_key == "image" else str(ov.get(column_key, ""))

        bbox = self.edit_tree.bbox(row_id, column_id)
        if not bbox:
            return
        x, y, w, h = bbox
        entry = ttk.Entry(self.edit_tree)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, current_value)
        entry.focus_set()
        entry.select_range(0, tk.END)
        self._edit_cell_entry = entry

        def _commit(_event=None):
            new_value = entry.get().strip()
            if self._apply_tree_cell_edit(row_index, column_key, new_value):
                entry.destroy()
                self._edit_cell_entry = None
                self.refresh_edit_overlay_table()
                self.edit_tree.selection_set(str(row_index))
                self.edit_tree.see(str(row_index))
                self.on_edit_overlay_select()
            else:
                entry.focus_set()
                entry.select_range(0, tk.END)

        def _cancel(_event=None):
            entry.destroy()
            self._edit_cell_entry = None

        entry.bind("<Return>", _commit)
        entry.bind("<Escape>", _cancel)
        entry.bind("<FocusOut>", _commit)

    def _parse_timecode_value(self, value: str) -> float:
        if not value:
            raise ValueError("時間を入力してください。")
        if ":" in value:
            return float(parse_timecode_to_seconds(value))
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("時間は mm:ss / hh:mm:ss か秒数で入力してください。") from exc

    def _apply_tree_cell_edit(self, idx: int, key: str, value: str) -> bool:
        ov = self.edit_overlays[idx]
        try:
            if key == "image":
                if not value:
                    raise ValueError("画像パスを入力してください。")
                if not Path(value).exists():
                    raise ValueError("画像ファイルが見つかりません。")
                ov["image_path"] = value
            elif key in ("start", "end"):
                time_value = self._parse_timecode_value(value)
                start = time_value if key == "start" else float(ov.get("start", 0.0))
                end = time_value if key == "end" else float(ov.get("end", 0.0))
                if end <= start:
                    raise ValueError("終了時間は開始時間より大きくしてください。")
                ov[key] = float(time_value)
            elif key in ("x", "y", "w", "h"):
                ov[key] = int(value or "0")
            elif key == "opacity":
                opacity = float(value or "1.0")
                if not (0.0 <= opacity <= 1.0):
                    raise ValueError("不透明度は 0.0〜1.0 で指定してください。")
                ov[key] = opacity
            else:
                return False
        except Exception as exc:
            messagebox.showerror("編集エラー", str(exc))
            return False

        self.edit_overlays[idx] = ov
        self.save_config()
        return True

    def load_selected_overlay_to_form(self):
        """選択行の値を、上のフォーム入力欄へ反映（編集しやすくする）"""
        idx = self._selected_overlay_index()
        if idx is None:
            messagebox.showinfo("選択", "編集したい行を選択してください。")
            return
        if idx < 0 or idx >= len(self.edit_overlays):
            return

        ov = self.edit_overlays[idx]
        self._apply_overlay_to_form(ov, log_message=True)

    def update_selected_overlay_from_form(self):
        """フォーム入力欄の値で、選択行を上書き更新"""
        idx = self._selected_overlay_index()
        if idx is None:
            messagebox.showinfo("更新", "更新したい行を選択してください。")
            return
        if idx < 0 or idx >= len(self.edit_overlays):
            return

        try:
            img = self.edit_overlay_img_entry.get().strip()
            if not img or not Path(img).exists():
                raise ValueError("有効な画像ファイルを選択してください。")

            start_s = self.edit_start_entry.get().strip()
            end_s = self.edit_end_entry.get().strip()
            start = parse_timecode_to_seconds(start_s)
            end = parse_timecode_to_seconds(end_s)
            if end <= start:
                raise ValueError("終了時間は開始時間より大きくしてください。")

            x = int(self.edit_x_entry.get().strip() or "0")
            y = int(self.edit_y_entry.get().strip() or "0")
            w = int(self.edit_w_entry.get().strip() or "0")
            h = int(self.edit_h_entry.get().strip() or "0")

            opacity = float(self.edit_opacity_entry.get().strip() or "1.0")
            if not (0.0 <= opacity <= 1.0):
                raise ValueError("不透明度は 0.0〜1.0 で指定してください。")

            self.edit_overlays[idx] = {
                "image_path": img,
                "start": float(start),
                "end": float(end),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "opacity": float(opacity),
            }

            self.save_config()
            self.refresh_edit_overlay_table()

            # 更新した行を選択状態に戻す
            if hasattr(self, "edit_tree"):
                self.edit_tree.selection_set(str(idx))
                self.edit_tree.see(str(idx))

            self.log(f"✅ 選択行を更新しました: row={idx+1}")

        except Exception as e:
            messagebox.showerror("更新エラー", str(e))

    def delete_selected_overlay(self):
        idx = self._selected_overlay_index()
        if idx is None:
            messagebox.showinfo("削除", "削除したい行を選択してください。")
            return
        if idx < 0 or idx >= len(self.edit_overlays):
            return

        if not messagebox.askyesno("削除確認", f"選択行（{idx+1}行目）を削除しますか？"):
            return

        del self.edit_overlays[idx]
        self.save_config()
        self.refresh_edit_overlay_table()
        self.log(f"🗑️ 行を削除しました: row={idx+1}")


        # 削除後にできれば近い行を選択
        if hasattr(self, "edit_tree") and self.edit_overlays:
            new_idx = min(idx, len(self.edit_overlays) - 1)
            self.edit_tree.selection_set(str(new_idx))
            self.edit_tree.see(str(new_idx))
            self.on_edit_overlay_select()







    
    def _default_prompt_templates(self) -> Dict[str, str]:
        return {
            "ニュース対談（標準）": (
                "あなたはベテランのニュース番組の台本作家です。\n"
                "以下の条件で「キャスター」と「アナリスト」による対談形式の台本を日本語で作成してください。\n\n"
                "[話者と形式]\n"
                "- 行頭に必ず「キャスター：」「アナリスト：」のどちらかを付ける（全角コロン）\n"
                "- 顔文字・絵文字・記号多用は不可\n"
                "- 具体的な数値や根拠を交えて\n\n"
                "[トピック]\n"
                "（ここにテーマを記入）\n\n"
                "[追加要望]\n"
                "（必要なら）\n"
            ),
            "投資ニュース（強め断言）": (
                "あなたは投資ニュース番組の台本作家です。\n"
                "テーマについて、投資判断に役立つ具体例・数値・リスクを織り込みつつ、\n"
                "最後は“断定的なポジショントーク”として結論をはっきり言い切ってください。\n\n"
                "[話者と形式]\n"
                "- 行頭に必ず「キャスター：」「アナリスト：」\n"
                "- 顔文字・絵文字は禁止\n\n"
                "[トピック]\n"
                "（ここにテーマ）\n\n"
                "[必須要素]\n"
                "- 強気シナリオ/弱気シナリオ\n"
                "- 直近ニュース想定（何が材料か）\n"
                "- 投資家の行動案（分割・損切り・利確目安など）\n"
            ),
            "日本語学習（文法ポンチ絵用）": (
                "あなたは日本語教師です。\n"
                "N2/N1 学習者向けに、文法項目をポンチ絵の構成（見出し＋例文＋注意点）で説明してください。\n\n"
                "[条件]\n"
                "- 見出し→意味→使い方→例文3つ→よくある間違い→類似表現との違い\n"
                "- 例文は短く自然な日本語\n\n"
                "[文法項目]\n"
                "（ここに記入）\n"
            ),
        }

    def _refresh_template_menu(self):
        names = sorted(self.prompt_templates.keys())
        if not names:
            names = ["（テンプレなし）"]

        self.prompt_template_names = names
        if hasattr(self, "prompt_template_menu") and self.prompt_template_menu is not None:
            self.prompt_template_menu.configure(values=names)

        cur = self.prompt_template_var.get()
        if cur not in names:
            self.prompt_template_var.set(names[0])

    def apply_selected_template(self):
        name = self.prompt_template_var.get()
        if not name or name == "（テンプレなし）":
            messagebox.showinfo("テンプレート", "テンプレートがありません。")
            return
        txt = self.prompt_templates.get(name, "")
        self._set_textbox(self.claude_prompt_text, txt)
        self.log(f"✅ テンプレ適用: {name}")

    def save_current_prompt_as_template(self):
        name = self.prompt_template_var.get().strip()
        if not name or name == "（テンプレなし）":
            messagebox.showerror("エラー", "保存先テンプレ名を選択してください。")
            return

        body = self._get_textbox(self.claude_prompt_text).strip()
        if not body:
            messagebox.showerror("エラー", "プロンプトが空です。")
            return

        self.prompt_templates[name] = body
        self._refresh_template_menu()
        self.save_config()
        self.log(f"✅ テンプレ上書き保存: {name}")
        messagebox.showinfo("保存", f"テンプレートを保存しました:\n{name}")

    def create_new_template(self):
        body = self._get_textbox(self.claude_prompt_text).strip()
        if not body:
            messagebox.showerror("エラー", "プロンプトが空です。先にプロンプトを入力してください。")
            return

        name = simpledialog.askstring("新規テンプレート", "テンプレート名を入力してください：")
        if not name:
            return
        name = name.strip()
        if not name:
            return

        if name in self.prompt_templates:
            if not messagebox.askyesno("確認", f"同名テンプレート '{name}' が存在します。上書きしますか？"):
                return

        self.prompt_templates[name] = body
        self._refresh_template_menu()
        self.prompt_template_var.set(name)
        self.save_config()
        self.log(f"✅ 新規テンプレ作成: {name}")

    def delete_selected_template(self):
        name = self.prompt_template_var.get()
        if not name or name == "（テンプレなし）":
            return
        if name not in self.prompt_templates:
            return

        if not messagebox.askyesno("削除確認", f"テンプレート '{name}' を削除しますか？"):
            return

        del self.prompt_templates[name]
        self._refresh_template_menu()
        self.save_config()
        self.log(f"🗑️ テンプレ削除: {name}")

    # --------------------------
    # Layout root
    # --------------------------
    def _build_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # sidebar
        self.grid_columnconfigure(1, weight=1)  # center
        self.grid_columnconfigure(2, weight=0)  # log

        self.sidebar = ctk.CTkFrame(
            self,
            width=self.SIDEBAR_W,
            corner_radius=20,
            fg_color=self.COL_PANEL,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(14, 8), pady=14)
        self.sidebar.grid_propagate(False)

        self.center = ctk.CTkFrame(
            self,
            corner_radius=20,
            fg_color=self.COL_PANEL2,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.center.grid(row=0, column=1, sticky="nsew", padx=8, pady=14)
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(0, weight=1)

        self.log_panel = ctk.CTkFrame(
            self,
            width=self.LOG_W,
            corner_radius=20,
            fg_color=self.COL_PANEL,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.log_panel.grid(row=0, column=2, sticky="nse", padx=(8, 14), pady=14)
        self.log_panel.grid_propagate(False)

    # --------------------------
    # Sidebar
    # --------------------------
    def _build_sidebar(self):
        top = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        self.sidebar.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            top, text="STUDIO",
            # font=ctk.CTkFont(size=20, weight="bold"),
            font=self.FONT_TITLE,
            text_color=self.COL_TEXT
        )
        title.grid(row=0, column=0, sticky="w")

        sub = ctk.CTkLabel(
            top, text="Movie Maker",
            # font=ctk.CTkFont(size=12),
            font=self.FONT_SUB,
            text_color=self.COL_MUTED
        )
        sub.grid(row=1, column=0, sticky="w", pady=(2, 0))

        menu = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        menu.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8, 10))
        menu.grid_columnconfigure(0, weight=1)

        self.btn_video = self._nav_button(menu, "🎬 動画生成", lambda: self.switch_page("video"))
        self.btn_video.grid(row=0, column=0, sticky="ew", pady=6)

        self.btn_script = self._nav_button(menu, "✍️動画台本生成", lambda: self.switch_page("script"))
        self.btn_script.grid(row=1, column=0, sticky="ew", pady=6)

        self.btn_title_desc = self._nav_button(
            menu, "🏷️ 動画タイトル・説明作成", lambda: self.switch_page("title_desc")
        )
        self.btn_title_desc.grid(row=2, column=0, sticky="ew", pady=6)

        self.btn_material = self._nav_button(menu, "📚 サムネイル作成", lambda: self.switch_page("material"))
        self.btn_material.grid(row=3, column=0, sticky="ew", pady=6)

        self.btn_ponchi = self._nav_button(menu, "📝 ポンチ絵作成", lambda: self.switch_page("ponchi"))
        self.btn_ponchi.grid(row=4, column=0, sticky="ew", pady=6)

        # NEW: 動画編集
        self.btn_edit = self._nav_button(menu, "🧩 動画編集", lambda: self.switch_page("edit"))
        self.btn_edit.grid(row=5, column=0, sticky="ew", pady=6)

        self.btn_detailed_edit = self._nav_button(menu, "🎛️ 詳細動画編集", lambda: self.switch_page("detailed_edit"))
        self.btn_detailed_edit.grid(row=6, column=0, sticky="ew", pady=6)

        self.btn_settings = self._nav_button(menu, "⚙️ 設定", lambda: self.switch_page("settings"))
        self.btn_settings.grid(row=7, column=0, sticky="ew", pady=6)

        self.btn_about = self._nav_button(menu, "ℹ️ About", lambda: self.switch_page("about"))
        self.btn_about.grid(row=8, column=0, sticky="ew", pady=6)

        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom.grid_columnconfigure(0, weight=1)

        self.status_pill = ctk.CTkLabel(
            bottom,
            text=" 準備中 ",
            corner_radius=14,
            fg_color="#0f2a20",
            text_color="#c8ffe7",
            font=self.FONT_PILL,
            anchor="w",
        )
        self.status_pill.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

    def _nav_button(self, parent, text, cmd):
        return ctk.CTkButton(
            parent,
            text=text,
            command=cmd,
            height=44,
            corner_radius=16,
            fg_color=self.COL_CARD,
            hover_color=self.COL_CARD_SOFT,
            border_width=1,
            border_color=self.COL_BORDER,
            text_color=self.COL_TEXT,
            anchor="w",
            font=self.FONT_NAV,
        )

    def _set_active_nav(self, key: str):
        def style(btn, active: bool):
            if active:
                btn.configure(
                    fg_color=self.COL_ACCENT_SOFT,
                    hover_color="#2b3360",
                    border_color=self.COL_ACCENT,
                    text_color="#eef1ff",
                )
            else:
                btn.configure(
                    fg_color=self.COL_CARD,
                    hover_color=self.COL_CARD_SOFT,
                    border_color=self.COL_BORDER,
                    text_color=self.COL_TEXT,
                )

        style(self.btn_video, key == "video")
        style(self.btn_script, key == "script")
        style(self.btn_title_desc, key == "title_desc")
        style(self.btn_material, key == "material")
        style(self.btn_ponchi, key == "ponchi")
        style(self.btn_edit, key == "edit")
        style(self.btn_detailed_edit, key == "detailed_edit")
        style(self.btn_settings, key == "settings")
        style(self.btn_about, key == "about")

    # --------------------------
    # Center pages
    # --------------------------
    def _build_center_pages(self):
        self.pages: Dict[str, ctk.CTkFrame] = {}

        self.page_container = ctk.CTkFrame(self.center, fg_color="transparent")
        self.page_container.grid(row=0, column=0, sticky="nsew")
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        self.pages["video"] = self._make_page(self.page_container)
        self.pages["script"] = self._make_page(self.page_container)
        self.pages["title_desc"] = self._make_page(self.page_container)
        self.pages["material"] = self._make_page(self.page_container)
        self.pages["ponchi"] = self._make_page(self.page_container)
        self.pages["edit"] = self._make_page(self.page_container)  # NEW
        self.pages["detailed_edit"] = self._make_page(self.page_container)
        self.pages["settings"] = self._make_page(self.page_container)
        self.pages["about"] = self._make_page(self.page_container)

        self._build_video_page(self.pages["video"])
        self._build_script_page(self.pages["script"])
        self._build_title_desc_page(self.pages["title_desc"])
        self._build_material_page(self.pages["material"])
        self._build_ponchi_page(self.pages["ponchi"])
        self._build_edit_page(self.pages["edit"])  # NEW
        self._build_detailed_edit_page(self.pages["detailed_edit"])
        self._build_settings_page(self.pages["settings"])
        self._build_about_page(self.pages["about"])

    def _make_page(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)
        return f

    def switch_page(self, key: str):
        self.active_page = key
        self._set_active_nav(key)

        for k, page in self.pages.items():
            if k == key:
                page.tkraise()

        title_map = {
            "video": "動画生成",
            "script": "台本生成",
            "title_desc": "動画タイトル・説明作成",
            "material": "資料作成",
            "ponchi": "ポンチ絵作成",
            "edit": "動画編集",
            "detailed_edit": "詳細動画編集",
            "settings": "設定",
            "about": "About",
        }
        self.log(f"--- ページ切替: {title_map.get(key, key)} ---")

    def _build_page_header(self, page_key: str, page: ctk.CTkFrame, title: str):
        header = ctk.CTkFrame(
            page,
            corner_radius=20,
            fg_color=self.COL_PANEL,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        header.grid_rowconfigure(1, weight=1)
        header.grid_columnconfigure(0, weight=1)

        accent = ctk.CTkFrame(header, height=3, fg_color=self.COL_ACCENT)
        accent.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 0))

        title_lbl = ctk.CTkLabel(
            header,
            text=title,
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color=self.COL_TEXT,
            anchor="w",
        )
        title_lbl.grid(row=1, column=0, sticky="w", padx=14, pady=12)

        pill = ctk.CTkLabel(
            header,
            text="  Studio  ",
            corner_radius=14,
            fg_color=self.COL_ACCENT_SOFT,
            text_color="#dbe1ff",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        pill.grid(row=1, column=1, sticky="e", padx=14, pady=12)

        self.page_title_labels[page_key] = title_lbl
        self.page_pills[page_key] = pill

    # --------------------------
    # Common: form builders (grid only)
    # --------------------------
    def _make_scroll_form(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        form = ctk.CTkScrollableFrame(
            parent,
            corner_radius=20,
            fg_color=self.COL_PANEL,
            label_text="",
            border_width=1,
            border_color=self.COL_BORDER,
        )
        form.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        form.grid_columnconfigure(0, weight=1)
        return form

    def _v_label(self, parent, text: str):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COL_TEXT,
            anchor="w",
        )

    def _v_hint(self, parent, text: str):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=11),
            text_color=self.COL_MUTED,
            anchor="w",
            justify="left",
        )

    def _v_entry(self, parent, show: Optional[str] = None):
        return ctk.CTkEntry(parent, height=34, corner_radius=12, show=show)

    def _v_path_row(self, parent, button_text: str, button_cmd, *, show: Optional[str] = None):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=0)

        entry = ctk.CTkEntry(row, height=34, corner_radius=12, show=show)
        entry.grid(row=0, column=0, sticky="ew")

        btn = ctk.CTkButton(
            row,
            text=button_text,
            command=button_cmd,
            height=34,
            corner_radius=14,
            fg_color=self.COL_CARD,
            hover_color=self.COL_CARD_SOFT,
            width=110,
        )
        btn.grid(row=0, column=1, sticky="e", padx=(10, 0))
        return row, entry

    def _v_two_buttons_row(self, parent, left_text, left_cmd, right_text, right_cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)

        b1 = ctk.CTkButton(
            row,
            text=left_text,
            command=left_cmd,
            height=38,
            corner_radius=12,
            fg_color="#1f5d8f",
            hover_color="#1b527f",
        )
        b1.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        b2 = ctk.CTkButton(
            row,
            text=right_text,
            command=right_cmd,
            height=38,
            corner_radius=12,
            fg_color="#a80d0d",
            hover_color="#8f0b0b",
        )
        b2.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        return row

    # --------------------------
    # Video page
    # --------------------------
    def _build_video_page(self, page):
        self._build_page_header("video", page, "動画生成")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0

        self._v_label(form, "TTS / 原稿").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(
            form,
            "Gemini利用時は設定タブでAPIキーを入力してください。VOICEVOX利用時は不要です。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        self._v_label(form, "原稿ファイル (dialogue_input.txt)").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        path_row, self.script_entry = self._v_path_row(form, "選択", self.browse_script)
        path_row.grid(row=r, column=0, sticky="ew", pady=(0, 14)); r += 1

        self._v_label(form, "画像リスト（1セリフごとに循環使用）").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.img_listbox = ctk.CTkTextbox(
            form,
            height=120,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.img_listbox.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        btn_row = self._v_two_buttons_row(form, "画像を追加", self.add_images, "全削除", self.clear_images)
        btn_row.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "BGM").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        self.use_bgm_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            form,
            text="BGMを使用する",
            variable=self.use_bgm_var,
            text_color=self.COL_TEXT,
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        self._v_label(form, "BGMファイル").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        bgm_row, self.bgm_entry = self._v_path_row(form, "選択", self.browse_bgm)
        bgm_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "BGM 音量(dB, マイナスで小さく)").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.bgm_gain_slider = ctk.CTkSlider(form, from_=-30, to=5, number_of_steps=35)
        self.bgm_gain_slider.set(-18)
        self.bgm_gain_slider.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "TTSエンジン").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.tts_engine_var = ctk.StringVar(value="VOICEVOX")
        self.tts_engine_menu = ctk.CTkOptionMenu(
            form,
            values=["Gemini", "VOICEVOX"],
            variable=self.tts_engine_var,
            command=self.on_tts_engine_change,
            corner_radius=12,
            height=34,
        )
        self.tts_engine_menu.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self.gemini_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.gemini_frame.grid(row=r, column=0, sticky="ew", pady=(0, 14))
        self.gemini_frame.grid_columnconfigure(0, weight=1)

        gr = 0
        self._v_label(self.gemini_frame, "Gemini 音声").grid(row=gr, column=0, sticky="w", padx=12, pady=(12, 6)); gr += 1
        self.voice_entry = ctk.CTkEntry(self.gemini_frame, height=34, corner_radius=12)
        self.voice_entry.insert(0, "Kore")
        self.voice_entry.grid(row=gr, column=0, sticky="ew", padx=12, pady=(0, 12)); gr += 1

        self.voicevox_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.voicevox_frame.grid(row=r + 1, column=0, sticky="ew", pady=(0, 14))
        self.voicevox_frame.grid_columnconfigure(0, weight=1)

        vr = 0
        self._v_label(self.voicevox_frame, "VOICEVOX 設定").grid(row=vr, column=0, sticky="w", padx=12, pady=(12, 6)); vr += 1

        self.vv_baseurl_entry = ctk.CTkEntry(self.voicevox_frame, height=34, corner_radius=12)
        self.vv_baseurl_entry.insert(0, DEFAULT_VOICEVOX_URL)
        self._v_label(self.voicevox_frame, "エンジンURL").grid(row=vr, column=0, sticky="w", padx=12, pady=(0, 6)); vr += 1
        self.vv_baseurl_entry.grid(row=vr, column=0, sticky="ew", padx=12, pady=(0, 12)); vr += 1

        self.vv_mode_var = ctk.StringVar(value="ローテーション")
        self._v_label(self.voicevox_frame, "話者モード").grid(row=vr, column=0, sticky="w", padx=12, pady=(0, 6)); vr += 1
        ctk.CTkOptionMenu(
            self.voicevox_frame,
            values=["ローテーション", "2人対談"],
            variable=self.vv_mode_var,
            corner_radius=12,
            height=34,
        ).grid(row=vr, column=0, sticky="ew", padx=12, pady=(0, 12)); vr += 1

        self.vv_rotation_entry = ctk.CTkEntry(self.voicevox_frame, height=34, corner_radius=12)
        self.vv_rotation_entry.insert(0, ",".join(str(x) for x in DEFAULT_VV_ROTATION))
        self._v_label(self.voicevox_frame, "ローテーション話者(カンマ)").grid(row=vr, column=0, sticky="w", padx=12, pady=(0, 6)); vr += 1
        self.vv_rotation_entry.grid(row=vr, column=0, sticky="ew", padx=12, pady=(0, 12)); vr += 1

        self.vv_caster_entry = ctk.CTkEntry(self.voicevox_frame, height=34, corner_radius=12)
        self.vv_caster_entry.insert(0, DEFAULT_VV_CASTER_LABEL)
        self._v_label(self.voicevox_frame, "キャスター話者").grid(row=vr, column=0, sticky="w", padx=12, pady=(0, 6)); vr += 1
        self.vv_caster_entry.grid(row=vr, column=0, sticky="ew", padx=12, pady=(0, 12)); vr += 1

        self.vv_analyst_entry = ctk.CTkEntry(self.voicevox_frame, height=34, corner_radius=12)
        self.vv_analyst_entry.insert(0, DEFAULT_VV_ANALYST_LABEL)
        self._v_label(self.voicevox_frame, "アナリスト話者").grid(row=vr, column=0, sticky="w", padx=12, pady=(0, 6)); vr += 1
        self.vv_analyst_entry.grid(row=vr, column=0, sticky="ew", padx=12, pady=(0, 12)); vr += 1

        self._v_label(self.voicevox_frame, "話速(0.5〜2.0)").grid(row=vr, column=0, sticky="w", padx=12, pady=(0, 6)); vr += 1
        self.vv_speed_slider = ctk.CTkSlider(self.voicevox_frame, from_=0.5, to=2.0, number_of_steps=30)
        self.vv_speed_slider.set(DEFAULT_VV_SPEED)
        self.vv_speed_slider.grid(row=vr, column=0, sticky="ew", padx=12, pady=(0, 14)); vr += 1

        r += 2

        self._v_label(form, "字幕フォントサイズ").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.caption_font_entry = self._v_entry(form)
        self.caption_font_entry.insert(0, str(FONT_SIZE))
        self.caption_font_entry.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "字幕背景の透明度(alpha 0-255)").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.caption_alpha_entry = self._v_entry(form)
        self.caption_alpha_entry.insert(0, str(CAPTION_BOX_ALPHA))
        self.caption_alpha_entry.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "背景OFF時のデザイン").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.bg_off_style_var = ctk.StringVar(value="影")
        ctk.CTkOptionMenu(
            form,
            values=["影", "角丸パネル", "なし"],
            variable=self.bg_off_style_var,
            corner_radius=12,
            height=34,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "字幕文字色（#RRGGBB）").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.caption_text_color_entry = self._v_entry(form)
        self.caption_text_color_entry.insert(0, DEFAULT_CAPTION_TEXT_COLOR)
        self.caption_text_color_entry.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "話者名フォントサイズ").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.speaker_font_entry = self._v_entry(form)
        self.speaker_font_entry.insert(0, str(SPEAKER_FONT_SIZE))
        self.speaker_font_entry.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "1行あたり最大文字数").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.caption_width_entry = self._v_entry(form)
        self.caption_width_entry.insert(0, str(CAPTION_MAX_CHARS_PER_LINE))
        self.caption_width_entry.grid(row=r, column=0, sticky="ew", pady=(0, 14)); r += 1

        self.caption_box_enabled_var = ctk.BooleanVar(value=DEFAULT_CAPTION_BOX_ENABLED)
        ctk.CTkCheckBox(
            form,
            text="字幕背景（黒幕）を表示する（固定高さ）",
            variable=self.caption_box_enabled_var,
            text_color=self.COL_TEXT,
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        self._v_label(form, "字幕背景の高さ(px, 固定)").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.caption_box_height_entry = self._v_entry(form)
        self.caption_box_height_entry.insert(0, str(DEFAULT_CAPTION_BOX_HEIGHT))
        self.caption_box_height_entry.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

        self._v_label(form, "出力フォルダ").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        out_row, self.output_entry = self._v_path_row(form, "選択", self.browse_output)
        out_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "解像度").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        wh_row = ctk.CTkFrame(form, fg_color="transparent")
        wh_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        wh_row.grid_columnconfigure(0, weight=1)
        wh_row.grid_columnconfigure(1, weight=1)

        w_box = ctk.CTkFrame(wh_row, fg_color="transparent")
        w_box.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        w_box.grid_columnconfigure(0, weight=1)

        self._v_hint(w_box, "幅(px)").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.width_entry = self._v_entry(w_box)
        self.width_entry.insert(0, "1080")
        self.width_entry.grid(row=1, column=0, sticky="ew")

        h_box = ctk.CTkFrame(wh_row, fg_color="transparent")
        h_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        h_box.grid_columnconfigure(0, weight=1)

        self._v_hint(h_box, "高さ(px)").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.height_entry = self._v_entry(h_box)
        self.height_entry.insert(0, "1920")
        self.height_entry.grid(row=1, column=0, sticky="ew")

        self._v_label(form, "FPS").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.fps_entry = self._v_entry(form)
        self.fps_entry.insert(0, "30")
        self.fps_entry.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

        self.run_button = ctk.CTkButton(
            form,
            text="▶ 動画を生成する",
            command=self.on_run_clicked,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
            height=46,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.run_button.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

        self.on_tts_engine_change(self.tts_engine_var.get())

    def on_tts_engine_change(self, value: str):
        if value == "Gemini":
            self.gemini_frame.grid()
            self.voicevox_frame.grid_remove()
        else:
            self.voicevox_frame.grid()
            self.gemini_frame.grid_remove()

    def on_script_engine_change(self, value: str):
        engine = value or "ClaudeCode"
        if hasattr(self, "script_gemini_frame"):
            if engine == "Gemini":
                self.script_gemini_frame.grid()
            else:
                self.script_gemini_frame.grid_remove()
        if hasattr(self, "script_chatgpt_frame"):
            if engine == "ChatGPT":
                self.script_chatgpt_frame.grid()
            else:
                self.script_chatgpt_frame.grid_remove()
        if hasattr(self, "script_claude_frame"):
            if engine == "ClaudeCode":
                self.script_claude_frame.grid()
            else:
                self.script_claude_frame.grid_remove()
        if hasattr(self, "btn_generate_script"):
            self.btn_generate_script.configure(text=f"▶ {engine}で台本生成")

    def on_title_engine_change(self, value: str):
        engine = value or "Gemini"
        if hasattr(self, "title_gemini_frame"):
            if engine == "Gemini":
                self.title_gemini_frame.grid()
            else:
                self.title_gemini_frame.grid_remove()
        if hasattr(self, "title_chatgpt_frame"):
            if engine == "ChatGPT":
                self.title_chatgpt_frame.grid()
            else:
                self.title_chatgpt_frame.grid_remove()
        if hasattr(self, "title_claude_frame"):
            if engine == "ClaudeCode":
                self.title_claude_frame.grid()
            else:
                self.title_claude_frame.grid_remove()
        if hasattr(self, "btn_generate_title_desc"):
            self.btn_generate_title_desc.configure(text=f"▶ {engine}で生成")

    # --------------------------
    # Script page
    # --------------------------
    def _build_script_page(self, page):
        self._build_page_header("script", page, "台本生成")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0

        self._v_label(form, "台本作成 API").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(
            form,
            "使用するAPIを選択します。APIキーは設定タブで一元管理します。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 12)); r += 1

        ctk.CTkOptionMenu(
            form,
            values=["Gemini", "ChatGPT", "ClaudeCode"],
            variable=self.script_engine_var,
            corner_radius=12,
            height=34,
            command=self.on_script_engine_change,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self.script_gemini_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.script_gemini_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.script_gemini_frame.grid_columnconfigure(0, weight=1)
        gr = 0
        self._v_label(self.script_gemini_frame, "Gemini モデル").grid(
            row=gr, column=0, sticky="w", padx=12, pady=(12, 6)
        ); gr += 1
        self.script_gemini_model_menu = ctk.CTkOptionMenu(
            self.script_gemini_frame,
            values=SCRIPT_MODEL_MASTER["Gemini"],
            variable=self.script_gemini_model_var,
            height=34,
            corner_radius=12,
        )
        self.script_gemini_model_menu.grid(row=gr, column=0, sticky="ew", padx=12, pady=(0, 12)); gr += 1

        self.script_chatgpt_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.script_chatgpt_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.script_chatgpt_frame.grid_columnconfigure(0, weight=1)
        cr = 0
        self._v_label(self.script_chatgpt_frame, "ChatGPT モデル").grid(
            row=cr, column=0, sticky="w", padx=12, pady=(12, 6)
        ); cr += 1
        self.script_chatgpt_model_menu = ctk.CTkOptionMenu(
            self.script_chatgpt_frame,
            values=SCRIPT_MODEL_MASTER["ChatGPT"],
            variable=self.script_chatgpt_model_var,
            height=34,
            corner_radius=12,
        )
        self.script_chatgpt_model_menu.grid(row=cr, column=0, sticky="ew", padx=12, pady=(0, 12)); cr += 1

        self.script_claude_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.script_claude_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.script_claude_frame.grid_columnconfigure(0, weight=1)
        ar = 0
        self._v_label(self.script_claude_frame, "ClaudeCode モデル").grid(
            row=ar, column=0, sticky="w", padx=12, pady=(12, 6)
        ); ar += 1
        self.script_claude_model_menu = ctk.CTkOptionMenu(
            self.script_claude_frame,
            values=SCRIPT_MODEL_MASTER["ClaudeCode"],
            variable=self.script_claude_model_var,
            height=34,
            corner_radius=12,
        )
        self.script_claude_model_menu.grid(row=ar, column=0, sticky="ew", padx=12, pady=(0, 12)); ar += 1

        self._v_label(self.script_claude_frame, "max_tokens").grid(
            row=ar, column=0, sticky="w", padx=12, pady=(0, 6)
        ); ar += 1
        self.claude_max_tokens_entry = ctk.CTkEntry(self.script_claude_frame, height=34, corner_radius=12)
        self.claude_max_tokens_entry.insert(0, str(DEFAULT_CLAUDE_MAX_TOKENS))
        self.claude_max_tokens_entry.grid(row=ar, column=0, sticky="ew", padx=12, pady=(0, 12)); ar += 1

        self._v_label(form, "テンプレート").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self._v_hint(form, "保存したテンプレートを選択して呼び出し/上書き/削除できます。").grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        tpl_row = ctk.CTkFrame(form, fg_color="transparent")
        tpl_row.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1
        tpl_row.grid_columnconfigure(0, weight=1)
        tpl_row.grid_columnconfigure(1, weight=0)

        self.prompt_template_menu = ctk.CTkOptionMenu(
            tpl_row,
            values=["（テンプレなし）"],
            variable=self.prompt_template_var,
            corner_radius=12,
            height=34,
        )
        self.prompt_template_menu.grid(row=0, column=0, sticky="ew")

        ctk.CTkButton(
            tpl_row,
            text="適用",
            command=self.apply_selected_template,
            height=34,
            corner_radius=12,
            width=110,
            fg_color="#172238",
            hover_color="#1b2a44",
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        tpl_btns = ctk.CTkFrame(form, fg_color="transparent")
        tpl_btns.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1
        tpl_btns.grid_columnconfigure(0, weight=1)
        tpl_btns.grid_columnconfigure(1, weight=1)
        tpl_btns.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(
            tpl_btns,
            text="新規作成",
            command=self.create_new_template,
            height=38,
            corner_radius=12,
            fg_color="#172238",
            hover_color="#1b2a44",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            tpl_btns,
            text="上書き保存",
            command=self.save_current_prompt_as_template,
            height=38,
            corner_radius=12,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 8))

        ctk.CTkButton(
            tpl_btns,
            text="削除",
            command=self.delete_selected_template,
            height=38,
            corner_radius=12,
            fg_color="#3b1d1d",
            hover_color="#4a2323",
        ).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        self._v_label(form, "プロンプト").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self._v_hint(
            form,
            "ここに台本生成用プロンプトを書きます。生成結果は下の出力欄に表示されます。"
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        self.claude_prompt_text = ctk.CTkTextbox(
            form,
            height=240,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.claude_prompt_text.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        prompt_btn_row = ctk.CTkFrame(form, fg_color="transparent")
        prompt_btn_row.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1
        prompt_btn_row.grid_columnconfigure(0, weight=1)
        prompt_btn_row.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            prompt_btn_row,
            text="テンプレ挿入",
            command=self.insert_prompt_template,
            height=38,
            corner_radius=12,
            fg_color="#172238",
            hover_color="#1b2a44",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            prompt_btn_row,
            text="クリア",
            command=lambda: self._set_textbox(self.claude_prompt_text, ""),
            height=38,
            corner_radius=12,
            fg_color="#3b1d1d",
            hover_color="#4a2323",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self._v_label(form, "生成").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        gen_row = ctk.CTkFrame(form, fg_color="transparent")
        gen_row.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1
        gen_row.grid_columnconfigure(0, weight=1)
        gen_row.grid_columnconfigure(1, weight=0)

        self.btn_generate_script = ctk.CTkButton(
            gen_row,
            text="▶ ClaudeCodeで台本生成",
            command=self.on_generate_script_clicked,
            fg_color=self.COL_ACCENT,
            hover_color=self.COL_ACCENT_HOVER,
            height=44,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_generate_script.grid(row=0, column=0, sticky="ew")

        self.btn_copy_script = ctk.CTkButton(
            gen_row,
            text="コピー",
            command=self.copy_generated_script,
            fg_color="#172238",
            hover_color="#1b2a44",
            height=44,
            corner_radius=14,
            width=120,
        )
        self.btn_copy_script.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self._v_label(form, "生成結果").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        self.claude_output_text = ctk.CTkTextbox(
            form,
            height=280,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.claude_output_text.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "保存").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        save_row, self.script_save_path_entry = self._v_path_row(
            form, "保存先選択", self.browse_script_save_path
        )
        self.script_save_path_entry.delete(0, "end")
        self.script_save_path_entry.insert(0, str(Path.home() / "dialogue_input.txt"))
        save_row.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        ctk.CTkButton(
            form,
            text="生成結果を保存",
            command=self.save_generated_script,
            height=40,
            corner_radius=12,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

        self.on_script_engine_change(self.script_engine_var.get())

        self._refresh_template_menu()

    # --------------------------
    # Title/Description page
    # --------------------------
    def _build_title_desc_page(self, page):
        self._build_page_header("title_desc", page, "動画タイトル・説明作成")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0

        self._v_label(form, "台本ファイル (SRT / TXT)").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(
            form,
            "動画台本（字幕）を読み込み、クリックされそうなタイトル案と説明文を生成します。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        file_row, self.title_script_entry = self._v_path_row(form, "選択", self.browse_title_script)
        file_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "生成AI").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        ctk.CTkOptionMenu(
            form,
            values=["Gemini", "ChatGPT", "ClaudeCode"],
            variable=self.title_engine_var,
            corner_radius=12,
            height=34,
            command=self.on_title_engine_change,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self.title_gemini_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.title_gemini_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.title_gemini_frame.grid_columnconfigure(0, weight=1)
        gr = 0
        self._v_label(self.title_gemini_frame, "Gemini モデル").grid(
            row=gr, column=0, sticky="w", padx=12, pady=(12, 6)
        ); gr += 1
        self.title_gemini_model_menu = ctk.CTkOptionMenu(
            self.title_gemini_frame,
            values=SCRIPT_MODEL_MASTER["Gemini"],
            variable=self.title_gemini_model_var,
            height=34,
            corner_radius=12,
        )
        self.title_gemini_model_menu.grid(row=gr, column=0, sticky="ew", padx=12, pady=(0, 12)); gr += 1

        self.title_chatgpt_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.title_chatgpt_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.title_chatgpt_frame.grid_columnconfigure(0, weight=1)
        cr = 0
        self._v_label(self.title_chatgpt_frame, "ChatGPT モデル").grid(
            row=cr, column=0, sticky="w", padx=12, pady=(12, 6)
        ); cr += 1
        self.title_chatgpt_model_menu = ctk.CTkOptionMenu(
            self.title_chatgpt_frame,
            values=SCRIPT_MODEL_MASTER["ChatGPT"],
            variable=self.title_chatgpt_model_var,
            height=34,
            corner_radius=12,
        )
        self.title_chatgpt_model_menu.grid(row=cr, column=0, sticky="ew", padx=12, pady=(0, 12)); cr += 1

        self.title_claude_frame = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        self.title_claude_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.title_claude_frame.grid_columnconfigure(0, weight=1)
        ar = 0
        self._v_label(self.title_claude_frame, "ClaudeCode モデル").grid(
            row=ar, column=0, sticky="w", padx=12, pady=(12, 6)
        ); ar += 1
        self.title_claude_model_menu = ctk.CTkOptionMenu(
            self.title_claude_frame,
            values=SCRIPT_MODEL_MASTER["ClaudeCode"],
            variable=self.title_claude_model_var,
            height=34,
            corner_radius=12,
        )
        self.title_claude_model_menu.grid(row=ar, column=0, sticky="ew", padx=12, pady=(0, 12)); ar += 1

        self._v_label(form, "タイトル案の数").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.title_count_entry = self._v_entry(form)
        self.title_count_entry.insert(0, "5")
        self.title_count_entry.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "追加指示（任意）").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self._v_hint(form, "例：ターゲット層、キーワード、トーンなど。").grid(
            row=r, column=0, sticky="w", pady=(0, 10)
        ); r += 1
        self.title_extra_text = ctk.CTkTextbox(
            form,
            height=140,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.title_extra_text.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        gen_row = ctk.CTkFrame(form, fg_color="transparent")
        gen_row.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1
        gen_row.grid_columnconfigure(0, weight=1)
        gen_row.grid_columnconfigure(1, weight=0)

        self.btn_generate_title_desc = ctk.CTkButton(
            gen_row,
            text="▶ Geminiで生成",
            command=self.on_generate_title_desc_clicked,
            fg_color=self.COL_ACCENT,
            hover_color=self.COL_ACCENT_HOVER,
            height=44,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_generate_title_desc.grid(row=0, column=0, sticky="ew")

        self.btn_copy_title_desc = ctk.CTkButton(
            gen_row,
            text="コピー",
            command=self.copy_generated_title_desc,
            fg_color="#172238",
            hover_color="#1b2a44",
            height=44,
            corner_radius=14,
            width=120,
        )
        self.btn_copy_title_desc.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self._v_label(form, "生成結果").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.title_output_text = ctk.CTkTextbox(
            form,
            height=280,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.title_output_text.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self.on_title_engine_change(self.title_engine_var.get())

    # --------------------------
    # Material page (Gemini Image)
    # --------------------------
    def _build_material_page(self, page):
        self._build_page_header("material", page, "資料作成")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0

        self._v_label(form, "Gemini API (Image)").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(
            form,
            "Gemini APIを使って、プロンプトから画像を生成します。APIキーは設定タブで管理します。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 12)); r += 1

        self._v_label(form, "モデル").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.material_model_entry = self._v_entry(form)
        self.material_model_entry.insert(0, GEMINI_MATERIAL_DEFAULT_MODEL)
        self.material_model_entry.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "プロンプト").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self._v_hint(form, "生成したい画像の被写体・背景・雰囲気などを記入してください。").grid(
            row=r, column=0, sticky="w", pady=(0, 10)
        ); r += 1

        self.material_prompt_text = ctk.CTkTextbox(
            form,
            height=220,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.material_prompt_text.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        prompt_btn_row = ctk.CTkFrame(form, fg_color="transparent")
        prompt_btn_row.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1
        prompt_btn_row.grid_columnconfigure(0, weight=1)
        prompt_btn_row.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            prompt_btn_row,
            text="雛形を挿入",
            command=self.insert_material_prompt_template,
            height=38,
            corner_radius=12,
            fg_color="#172238",
            hover_color="#1b2a44",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            prompt_btn_row,
            text="クリア",
            command=lambda: self._set_textbox(self.material_prompt_text, ""),
            height=38,
            corner_radius=12,
            fg_color="#3b1d1d",
            hover_color="#4a2323",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        gen_row = ctk.CTkFrame(form, fg_color="transparent")
        gen_row.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1
        gen_row.grid_columnconfigure(0, weight=1)
        gen_row.grid_columnconfigure(1, weight=0)

        self.btn_generate_material = ctk.CTkButton(
            gen_row,
            text="▶ Geminiで画像生成",
            command=self.on_generate_material_clicked,
            fg_color=self.COL_ACCENT,
            hover_color=self.COL_ACCENT_HOVER,
            height=44,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_generate_material.grid(row=0, column=0, sticky="ew")

        self.btn_copy_material = ctk.CTkButton(
            gen_row,
            text="パスをコピー",
            command=self.copy_generated_material,
            fg_color="#172238",
            hover_color="#1b2a44",
            height=44,
            corner_radius=14,
            width=120,
        )
        self.btn_copy_material.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self._v_label(form, "生成画像").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        self.material_output_frame = ctk.CTkFrame(
            form,
            height=280,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.material_output_frame.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.material_output_frame.grid_propagate(False)
        self.material_output_frame.grid_columnconfigure(0, weight=1)
        self.material_output_frame.grid_rowconfigure(0, weight=1)

        self.material_output_label = ctk.CTkLabel(
            self.material_output_frame,
            text="画像がここに表示されます",
            text_color=self.COL_MUTED,
            anchor="center",
        )
        self.material_output_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self._v_label(form, "保存").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        save_row, self.material_save_path_entry = self._v_path_row(
            form, "保存先フォルダ", self.browse_material_save_path
        )
        self.material_save_path_entry.delete(0, "end")
        self.material_save_path_entry.insert(0, str(self._default_material_save_dir()))
        save_row.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        ctk.CTkButton(
            form,
            text="生成結果を保存",
            command=self.save_generated_material,
            height=40,
            corner_radius=12,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

    # --------------------------
    # Ponchi page (NEW)
    # --------------------------
    def _build_ponchi_page(self, page):
        self._build_page_header("ponchi", page, "ポンチ絵作成")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0

        self._v_label(form, "SRTファイル").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(
            form,
            "字幕に沿って案出しを行い、その案に基づいて画像（ポンチ絵）を生成します。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        srt_row, self.ponchi_srt_entry = self._v_path_row(form, "SRT選択", self.browse_ponchi_srt)
        srt_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "出力フォルダ").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        out_row, self.ponchi_output_dir_entry = self._v_path_row(
            form, "保存先", self.browse_ponchi_output_dir
        )
        out_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        self.ponchi_output_dir_entry.insert(0, str(Path.home() / "ponchi_images"))

        self._v_label(form, "提案生成エンジン").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.ponchi_suggestion_engine_var = ctk.StringVar(value="Gemini")
        ctk.CTkOptionMenu(
            form,
            values=["Gemini", "ChatGPT"],
            variable=self.ponchi_suggestion_engine_var,
            corner_radius=12,
            height=34,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        api_wrap = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_CARD)
        api_wrap.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1
        api_wrap.grid_columnconfigure(0, weight=1)

        ar = 0
        self._v_hint(
            api_wrap,
            "APIキーは設定タブで一元管理します。",
        ).grid(row=ar, column=0, sticky="w", padx=12, pady=(12, 12)); ar += 1

        self._v_label(api_wrap, "Gemini 提案モデル").grid(
            row=ar, column=0, sticky="w", padx=12, pady=(0, 6)
        ); ar += 1
        self.ponchi_gemini_model_entry = ctk.CTkEntry(api_wrap, height=34, corner_radius=12)
        self.ponchi_gemini_model_entry.insert(0, DEFAULT_PONCHI_GEMINI_MODEL)
        self.ponchi_gemini_model_entry.grid(row=ar, column=0, sticky="ew", padx=12, pady=(0, 12)); ar += 1

        self._v_label(api_wrap, "ChatGPT モデル").grid(
            row=ar, column=0, sticky="w", padx=12, pady=(0, 6)
        ); ar += 1
        self.ponchi_openai_model_entry = ctk.CTkEntry(api_wrap, height=34, corner_radius=12)
        self.ponchi_openai_model_entry.insert(0, DEFAULT_PONCHI_OPENAI_MODEL)
        self.ponchi_openai_model_entry.grid(row=ar, column=0, sticky="ew", padx=12, pady=(0, 12)); ar += 1

        self._v_label(form, "生成").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        gen_row = ctk.CTkFrame(form, fg_color="transparent")
        gen_row.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1
        gen_row.grid_columnconfigure(0, weight=1)
        gen_row.grid_columnconfigure(1, weight=1)
        gen_row.grid_columnconfigure(2, weight=0)

        self.btn_generate_ponchi_ideas = ctk.CTkButton(
            gen_row,
            text="▶ 案出し",
            command=self.on_generate_ponchi_ideas_clicked,
            fg_color=self.COL_ACCENT,
            hover_color=self.COL_ACCENT_HOVER,
            height=44,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_generate_ponchi_ideas.grid(row=0, column=0, sticky="ew")

        self.btn_generate_ponchi_images = ctk.CTkButton(
            gen_row,
            text="▶ ポンチ絵作成",
            command=self.on_generate_ponchi_images_clicked,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
            height=44,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_generate_ponchi_images.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        ctk.CTkButton(
            gen_row,
            text="クリア",
            command=lambda: self._set_textbox(self.ponchi_output_text, ""),
            fg_color="#172238",
            hover_color="#1b2a44",
            height=44,
            corner_radius=14,
            width=120,
        ).grid(row=0, column=2, sticky="e", padx=(12, 0))

        self._v_label(form, "生成結果").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self._v_hint(
            form,
            f"画像生成は {GEMINI_MATERIAL_DEFAULT_MODEL} を使用します。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        self.ponchi_output_text = ctk.CTkTextbox(
            form,
            height=260,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.ponchi_output_text.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

    # --------------------------
    # Edit page (NEW)
    # --------------------------
    def _build_edit_page(self, page):
        self._build_page_header("edit", page, "動画編集")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0
        # Preview area (center top)
        preview_wrap = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        preview_wrap.grid(row=r, column=0, sticky="ew", padx=10, pady=(10, 18)); r += 1
        preview_wrap.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview_wrap, text="プレビュー",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))

        self.edit_preview_label = ctk.CTkLabel(
            preview_wrap,
            text="動画を選択してください",
            text_color=self.COL_MUTED,
            width=self.EDIT_PREVIEW_MAX[0],
            height=self.EDIT_PREVIEW_MAX[1],
            anchor="center",
        )
        self.edit_preview_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))

        preview_controls = ctk.CTkFrame(preview_wrap, fg_color="transparent")
        preview_controls.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        preview_controls.grid_columnconfigure(1, weight=1)

        # X slider
        ctk.CTkLabel(
            preview_controls, text="画像X",
            text_color=self.COL_MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.edit_preview_x_slider = ctk.CTkSlider(
            preview_controls, from_=0, to=1920,
            command=lambda _v: self._sync_edit_preview_from_sliders(),
        )
        self.edit_preview_x_slider.set(0)
        self.edit_preview_x_slider.grid(row=0, column=1, sticky="ew", padx=(12, 8))
        self.edit_preview_x_value = ctk.CTkLabel(
            preview_controls, text="0", text_color=self.COL_TEXT, width=60, anchor="e"
        )
        self.edit_preview_x_value.grid(row=0, column=2, sticky="e")

        # Y slider
        ctk.CTkLabel(
            preview_controls, text="画像Y",
            text_color=self.COL_MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.edit_preview_y_slider = ctk.CTkSlider(
            preview_controls, from_=0, to=1080,
            command=lambda _v: self._sync_edit_preview_from_sliders(),
        )
        self.edit_preview_y_slider.set(0)
        self.edit_preview_y_slider.grid(row=1, column=1, sticky="ew", padx=(12, 8), pady=(6, 0))
        self.edit_preview_y_value = ctk.CTkLabel(
            preview_controls, text="0", text_color=self.COL_TEXT, width=60, anchor="e"
        )
        self.edit_preview_y_value.grid(row=1, column=2, sticky="e", pady=(6, 0))

        # Scale slider
        ctk.CTkLabel(
            preview_controls, text="画像スケール(%)",
            text_color=self.COL_MUTED,
            anchor="w",
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.edit_preview_scale_slider = ctk.CTkSlider(
            preview_controls, from_=10, to=300,
            command=lambda _v: self._sync_edit_preview_from_sliders(),
        )
        self.edit_preview_scale_slider.set(100)
        self.edit_preview_scale_slider.grid(row=2, column=1, sticky="ew", padx=(12, 8), pady=(6, 0))
        self.edit_preview_scale_value = ctk.CTkLabel(
            preview_controls, text="100%", text_color=self.COL_TEXT, width=60, anchor="e"
        )
        self.edit_preview_scale_value.grid(row=2, column=2, sticky="e", pady=(6, 0))

        self._v_label(form, "入力動画").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(form, "MP4 を読み込んで、指定時間帯に画像を重ねて加工します。").grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        in_row, self.edit_input_entry = self._v_path_row(form, "選択", self.browse_edit_input_mp4)
        in_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        self._v_label(form, "出力動画").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        out_row, self.edit_output_entry = self._v_path_row(form, "選択", self.browse_edit_output_mp4)
        out_row.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "オーバーレイ設定（1件ずつ追加）").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self._v_hint(form, "時間は mm:ss / hh:mm:ss で指定。例: 00:12〜00:18").grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        # overlay image
        self._v_label(form, "画像").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        img_row, self.edit_overlay_img_entry = self._v_path_row(form, "選択", self.browse_edit_overlay_image)
        img_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        # time range row
        tr = ctk.CTkFrame(form, fg_color="transparent")
        tr.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        tr.grid_columnconfigure(0, weight=1)
        tr.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(tr, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(tr, fg_color="transparent")
        right.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        right.grid_columnconfigure(0, weight=1)

        self._v_hint(left, "開始 (mm:ss)").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_start_entry = self._v_entry(left)
        self.edit_start_entry.insert(0, "00:00")
        self.edit_start_entry.grid(row=1, column=0, sticky="ew")

        self._v_hint(right, "終了 (mm:ss)").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_end_entry = self._v_entry(right)
        self.edit_end_entry.insert(0, "00:05")
        self.edit_end_entry.grid(row=1, column=0, sticky="ew")

        # position row
        pr = ctk.CTkFrame(form, fg_color="transparent")
        pr.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        pr.grid_columnconfigure(0, weight=1)
        pr.grid_columnconfigure(1, weight=1)

        px = ctk.CTkFrame(pr, fg_color="transparent")
        px.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        px.grid_columnconfigure(0, weight=1)

        py = ctk.CTkFrame(pr, fg_color="transparent")
        py.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        py.grid_columnconfigure(0, weight=1)

        self._v_hint(px, "X (px)").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_x_entry = self._v_entry(px)
        self.edit_x_entry.insert(0, "100")
        self.edit_x_entry.grid(row=1, column=0, sticky="ew")

        self._v_hint(py, "Y (px)").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_y_entry = self._v_entry(py)
        self.edit_y_entry.insert(0, "200")
        self.edit_y_entry.grid(row=1, column=0, sticky="ew")

        # size row
        sr = ctk.CTkFrame(form, fg_color="transparent")
        sr.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        sr.grid_columnconfigure(0, weight=1)
        sr.grid_columnconfigure(1, weight=1)

        sw = ctk.CTkFrame(sr, fg_color="transparent")
        sw.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        sw.grid_columnconfigure(0, weight=1)

        sh = ctk.CTkFrame(sr, fg_color="transparent")
        sh.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        sh.grid_columnconfigure(0, weight=1)

        self._v_hint(sw, "幅 w (px) ※0で元サイズ").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_w_entry = self._v_entry(sw)
        self.edit_w_entry.insert(0, "0")
        self.edit_w_entry.grid(row=1, column=0, sticky="ew")

        self._v_hint(sh, "高さ h (px) ※0で元サイズ").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_h_entry = self._v_entry(sh)
        self.edit_h_entry.insert(0, "0")
        self.edit_h_entry.grid(row=1, column=0, sticky="ew")

        # opacity
        self._v_label(form, "不透明度 (0.0〜1.0)").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.edit_opacity_entry = self._v_entry(form)
        self.edit_opacity_entry.insert(0, "1.0")
        self.edit_opacity_entry.grid(row=r, column=0, sticky="ew", pady=(0, 14)); r += 1

        # add overlay buttons
        add_row = ctk.CTkFrame(form, fg_color="transparent")
        add_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        add_row.grid_columnconfigure(0, weight=1)
        add_row.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            add_row,
            text="＋ 追加",
            command=self.add_edit_overlay,
            height=38,
            corner_radius=12,
            fg_color="#1f5d8f",
            hover_color="#1b527f",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            add_row,
            text="リスト全消去",
            command=self.clear_edit_overlays,
            height=38,
            corner_radius=12,
            fg_color="#a80d0d",
            hover_color="#8f0b0b",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # ==========================
        # NEW: スプレッドシート風テーブル + サムネプレビュー
        # ==========================
        self._v_label(form, "オーバーレイ一覧（表）").grid(row=r, column=0, sticky="w", pady=(4, 6)); r += 1

        table_wrap = ctk.CTkFrame(form, fg_color="transparent")
        table_wrap.grid(row=r, column=0, sticky="nsew", pady=(0, 12)); r += 1
        table_wrap.grid_columnconfigure(0, weight=3)  # table
        table_wrap.grid_columnconfigure(1, weight=2)  # preview
        table_wrap.grid_rowconfigure(0, weight=1)

        # ---- left: Treeview table ----
        tv_frame = ctk.CTkFrame(table_wrap, corner_radius=14, fg_color=self.COL_BG)
        tv_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tv_frame.grid_rowconfigure(0, weight=1)
        tv_frame.grid_columnconfigure(0, weight=1)

        self._setup_edit_tree_style()

        columns = ("image", "start", "end", "x", "y", "w", "h", "opacity")
        self.edit_tree = ttk.Treeview(
            tv_frame,
            columns=columns,
            show="headings",
            height=10,
            style="Overlay.Treeview",
        )

        # ヘッダ
        self.edit_tree.heading("image", text="image")
        self.edit_tree.heading("start", text="start")
        self.edit_tree.heading("end", text="end")
        self.edit_tree.heading("x", text="x")
        self.edit_tree.heading("y", text="y")
        self.edit_tree.heading("w", text="w")
        self.edit_tree.heading("h", text="h")
        self.edit_tree.heading("opacity", text="opacity")

        # 列幅（好みで調整OK）
        self.edit_tree.column("image", width=220, anchor="w")
        self.edit_tree.column("start", width=80, anchor="e")
        self.edit_tree.column("end", width=80, anchor="e")
        self.edit_tree.column("x", width=60, anchor="e")
        self.edit_tree.column("y", width=60, anchor="e")
        self.edit_tree.column("w", width=60, anchor="e")
        self.edit_tree.column("h", width=60, anchor="e")
        self.edit_tree.column("opacity", width=80, anchor="e")

        # スクロール
        vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=self.edit_tree.yview)
        self.edit_tree.configure(yscrollcommand=vsb.set)

        self.edit_tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        vsb.grid(row=0, column=1, sticky="ns", pady=10)

        # 選択イベント
        self.edit_tree.bind("<<TreeviewSelect>>", self.on_edit_overlay_select)
        self.edit_tree.bind("<Double-1>", self.on_edit_tree_double_click)
        self.edit_tree.tag_configure("even", background=self.COL_PANEL2)
        self.edit_tree.tag_configure("odd", background="#0f1e34")

        # ---- right: thumbnail preview ----
        pv = ctk.CTkFrame(table_wrap, corner_radius=14, fg_color=self.COL_BG)
        pv.grid(row=0, column=1, sticky="nsew")
        pv.grid_columnconfigure(0, weight=1)
        pv.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            pv, text="サムネプレビュー",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COL_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        self.edit_thumb_label = ctk.CTkLabel(
            pv, text="（行を選択すると表示）",
            text_color=self.COL_MUTED,
            anchor="center",
        )
        self.edit_thumb_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        self.edit_thumb_path = ctk.CTkLabel(
            pv, text="",
            text_color=self.COL_MUTED,
            anchor="w",
            justify="left",
        )
        self.edit_thumb_path.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        detail = ctk.CTkFrame(pv, corner_radius=12, fg_color=self.COL_PANEL2)
        detail.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        detail.grid_columnconfigure(0, weight=1)
        detail.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            detail, text="選択中の設定",
            text_color=self.COL_TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))

        detail_fields = [
            ("start", "開始"),
            ("end", "終了"),
            ("x", "X"),
            ("y", "Y"),
            ("w", "幅"),
            ("h", "高さ"),
            ("opacity", "不透明度"),
        ]
        row = 1
        for idx, (key, label) in enumerate(detail_fields):
            col = idx % 2
            if col == 0 and idx > 0:
                row += 1
            wrap = ctk.CTkFrame(detail, fg_color="transparent")
            wrap.grid(row=row, column=col, sticky="ew", padx=10, pady=(2, 2))
            wrap.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                wrap, text=label,
                text_color=self.COL_MUTED,
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            value_label = ctk.CTkLabel(
                wrap, text="--",
                text_color=self.COL_TEXT,
                anchor="e",
            )
            value_label.grid(row=0, column=1, sticky="e")
            self._edit_detail_labels[key] = value_label

        # ---- row actions (edit/delete/update) ----
        act = ctk.CTkFrame(form, fg_color="transparent")
        act.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        act.grid_columnconfigure(0, weight=1)
        act.grid_columnconfigure(1, weight=1)
        act.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(
            act, text="選択行をフォームへ反映",
            command=self.load_selected_overlay_to_form,
            height=38, corner_radius=12,
            fg_color="#172238", hover_color="#1b2a44",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            act, text="選択行を更新",
            command=self.update_selected_overlay_from_form,
            height=38, corner_radius=12,
            fg_color=self.COL_OK, hover_color=self.COL_OK_HOVER,
        ).grid(row=0, column=1, sticky="ew", padx=8)

        ctk.CTkButton(
            act, text="選択行を削除",
            command=self.delete_selected_overlay,
            height=38, corner_radius=12,
            fg_color="#3b1d1d", hover_color="#4a2323",
        ).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        # ---- SRT -> 画像収集 + JSON ----
        self._v_label(form, "SRTから画像収集").grid(row=r, column=0, sticky="w", pady=(6, 6)); r += 1
        self._v_hint(form, "字幕ごとに検索キーワードを生成し、Google/Bing画像検索から取得します。").grid(
            row=r, column=0, sticky="w", pady=(0, 8)
        ); r += 1

        srt_row, self.edit_srt_entry = self._v_path_row(form, "SRT選択", self.browse_edit_srt)
        srt_row.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        out_row, self.edit_image_output_entry = self._v_path_row(
            form, "保存先", self.browse_edit_image_output_dir
        )
        out_row.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1
        self.edit_image_output_entry.insert(0, str(Path.home() / "srt_images"))

        api_row = ctk.CTkFrame(form, fg_color="transparent")
        api_row.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        api_row.grid_columnconfigure(0, weight=1)
        api_row.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(api_row, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(api_row, fg_color="transparent")
        right.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        right.grid_columnconfigure(0, weight=1)

        self._v_hint(left, "画像検索プロバイダ").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_search_provider_var = ctk.StringVar(value=DEFAULT_IMAGE_SEARCH_PROVIDER)
        self.edit_search_provider_menu = ctk.CTkOptionMenu(
            left,
            values=["Google", "Bing"],
            variable=self.edit_search_provider_var,
        )
        self.edit_search_provider_menu.grid(row=1, column=0, sticky="ew")

        self._v_hint(right, "画像検索 APIキー（SerpAPI）").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.edit_search_api_key_entry = self._v_entry(right, show="*")
        self.edit_search_api_key_entry.grid(row=1, column=0, sticky="ew")

        defaults = ctk.CTkFrame(form, fg_color="transparent")
        defaults.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1
        defaults.grid_columnconfigure(0, weight=1)
        defaults.grid_columnconfigure(1, weight=1)
        defaults.grid_columnconfigure(2, weight=1)
        defaults.grid_columnconfigure(3, weight=1)
        defaults.grid_columnconfigure(4, weight=1)

        self._v_hint(defaults, "既定X").grid(row=0, column=0, sticky="w")
        self.edit_default_x_entry = self._v_entry(defaults)
        self.edit_default_x_entry.insert(0, str(DEFAULT_EDIT_IMPORT_X))
        self.edit_default_x_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        self._v_hint(defaults, "既定Y").grid(row=0, column=1, sticky="w")
        self.edit_default_y_entry = self._v_entry(defaults)
        self.edit_default_y_entry.insert(0, str(DEFAULT_EDIT_IMPORT_Y))
        self.edit_default_y_entry.grid(row=1, column=1, sticky="ew", padx=(6, 6))

        self._v_hint(defaults, "既定W").grid(row=0, column=2, sticky="w")
        self.edit_default_w_entry = self._v_entry(defaults)
        self.edit_default_w_entry.insert(0, str(DEFAULT_EDIT_IMPORT_W))
        self.edit_default_w_entry.grid(row=1, column=2, sticky="ew", padx=(6, 6))

        self._v_hint(defaults, "既定H").grid(row=0, column=3, sticky="w")
        self.edit_default_h_entry = self._v_entry(defaults)
        self.edit_default_h_entry.insert(0, str(DEFAULT_EDIT_IMPORT_H))
        self.edit_default_h_entry.grid(row=1, column=3, sticky="ew", padx=(6, 6))


        self._v_hint(defaults, "既定Opacity").grid(row=0, column=4, sticky="w")
        self.edit_default_opacity_entry = self._v_entry(defaults)
        self.edit_default_opacity_entry.insert(0, str(DEFAULT_EDIT_IMPORT_OPACITY))
        self.edit_default_opacity_entry.grid(row=1, column=4, sticky="ew", padx=(6, 0))

        srt_action = ctk.CTkFrame(form, fg_color="transparent")
        srt_action.grid(row=r, column=0, sticky="ew", pady=(0, 14)); r += 1
        srt_action.grid_columnconfigure(0, weight=1)
        srt_action.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            srt_action,
            text="SRTから画像収集",
            command=self.on_collect_images_from_srt,
            height=40,
            corner_radius=12,
            fg_color="#1f5d8f",
            hover_color="#1b527f",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            srt_action,
            text="JSON読み込み",
            command=self.import_edit_overlays_from_json,
            height=40,
            corner_radius=12,
            fg_color="#172238",
            hover_color="#1b2a44",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # render button（既存のまま）
        self.edit_render_button = ctk.CTkButton(
            form,
            text="▶ オーバーレイを書き出す",
            command=self.on_render_edit_video_clicked,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
            height=46,
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.edit_render_button.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

        self.refresh_edit_overlay_table()


        # 多分要らない
        # # listbox
        # self._v_label(form, "オーバーレイ一覧").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        # self.edit_overlay_list = ctk.CTkTextbox(
        #     form,
        #     height=160,
        #     corner_radius=14,
        #     fg_color=self.COL_BG,
        #     border_width=1,
        #     border_color=self.COL_BORDER,
        # )
        # self.edit_overlay_list.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        # # render button
        # self.edit_render_button = ctk.CTkButton(
        #     form,
        #     text="▶ オーバーレイを書き出す",
        #     command=self.on_render_edit_video_clicked,
        #     fg_color=self.COL_OK,
        #     hover_color=self.COL_OK_HOVER,
        #     height=46,
        #     corner_radius=14,
        #     font=ctk.CTkFont(size=14, weight="bold"),
        # )
        # self.edit_render_button.grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

        # self.refresh_edit_overlay_table()

    # --------------------------
    # Detailed Edit page
    # --------------------------
    def _build_detailed_edit_page(self, page):
        self._build_page_header("detailed_edit", page, "詳細動画編集")
        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0
        # Project management
        proj = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        proj.grid(row=r, column=0, sticky="ew", padx=10, pady=(10, 16)); r += 1
        proj.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            proj, text="プロジェクト管理",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT,
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=14, pady=(12, 6))

        ctk.CTkLabel(proj, text="プロジェクト名", text_color=self.COL_MUTED, anchor="w").grid(
            row=1, column=0, sticky="w", padx=14, pady=(0, 6)
        )
        self.detailed_project_name_entry = ctk.CTkEntry(proj, height=34, corner_radius=12)
        self.detailed_project_name_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(0, 6))

        btn_row = ctk.CTkFrame(proj, fg_color="transparent")
        btn_row.grid(row=1, column=2, sticky="e", padx=14, pady=(0, 6))
        ctk.CTkButton(
            btn_row, text="新規", command=self.new_detailed_project,
            height=32, corner_radius=12, fg_color="#1f5d8f",
        ).grid(row=0, column=0, padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="保存", command=self.save_detailed_project,
            height=32, corner_radius=12, fg_color=self.COL_OK,
        ).grid(row=0, column=1, padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="開く", command=self.open_detailed_project,
            height=32, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=2)

        path_row, self.detailed_project_path_entry = self._v_path_row(
            proj, "保存先", self.browse_detailed_project_path
        )
        path_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=(0, 12))

        dirs = ctk.CTkFrame(proj, fg_color="transparent")
        dirs.grid(row=3, column=0, columnspan=3, sticky="ew", padx=14, pady=(0, 12))
        dirs.grid_columnconfigure((0, 1, 2), weight=1)

        self._v_hint(dirs, "プロジェクトルート").grid(row=0, column=0, sticky="w")
        self._v_hint(dirs, "入力素材フォルダ").grid(row=0, column=1, sticky="w")
        self._v_hint(dirs, "出力フォルダ").grid(row=0, column=2, sticky="w")

        self.detailed_root_entry = self._v_entry(dirs)
        self.detailed_root_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.detailed_input_entry = self._v_entry(dirs)
        self.detailed_input_entry.grid(row=1, column=1, sticky="ew", padx=8)
        self.detailed_output_entry = self._v_entry(dirs)
        self.detailed_output_entry.grid(row=1, column=2, sticky="ew", padx=(8, 0))

        self.detailed_autosave_label = ctk.CTkLabel(
            proj, text="自動保存: 未設定", text_color=self.COL_MUTED, anchor="w"
        )
        self.detailed_autosave_label.grid(row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 12))

        # Asset management
        assets = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        assets.grid(row=r, column=0, sticky="ew", padx=10, pady=(0, 16)); r += 1
        assets.grid_columnconfigure(0, weight=1)
        assets.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            assets, text="素材（アセット）管理",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        asset_body = ctk.CTkFrame(assets, fg_color="transparent")
        asset_body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))
        asset_body.grid_columnconfigure(0, weight=3)
        asset_body.grid_columnconfigure(1, weight=2)
        asset_body.grid_rowconfigure(0, weight=1)

        asset_table_frame = ctk.CTkFrame(asset_body, corner_radius=12, fg_color=self.COL_BG)
        asset_table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        asset_table_frame.grid_rowconfigure(0, weight=1)
        asset_table_frame.grid_columnconfigure(0, weight=1)

        self._setup_detailed_asset_tree_style()
        asset_columns = ("name", "type", "duration", "resolution", "status")
        self.detailed_asset_tree = ttk.Treeview(
            asset_table_frame,
            columns=asset_columns,
            show="headings",
            height=8,
            style="Detailed.Treeview",
        )
        for col, text in zip(asset_columns, ["ファイル", "種別", "長さ", "解像度", "状態"], strict=False):
            self.detailed_asset_tree.heading(col, text=text)
        self.detailed_asset_tree.column("name", width=260, anchor="w")
        self.detailed_asset_tree.column("type", width=80, anchor="center")
        self.detailed_asset_tree.column("duration", width=90, anchor="e")
        self.detailed_asset_tree.column("resolution", width=110, anchor="e")
        self.detailed_asset_tree.column("status", width=80, anchor="center")

        asset_scroll = ttk.Scrollbar(asset_table_frame, orient="vertical", command=self.detailed_asset_tree.yview)
        self.detailed_asset_tree.configure(yscrollcommand=asset_scroll.set)
        self.detailed_asset_tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        asset_scroll.grid(row=0, column=1, sticky="ns", pady=10)
        self.detailed_asset_tree.bind("<<TreeviewSelect>>", self.on_detailed_asset_select)

        asset_preview = ctk.CTkFrame(asset_body, corner_radius=12, fg_color=self.COL_BG)
        asset_preview.grid(row=0, column=1, sticky="nsew")
        asset_preview.grid_columnconfigure(0, weight=1)
        asset_preview.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            asset_preview, text="サムネイル",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        self.detailed_asset_thumb = ctk.CTkLabel(
            asset_preview, text="（素材を選択）", text_color=self.COL_MUTED, anchor="center"
        )
        self.detailed_asset_thumb.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self.detailed_asset_path_label = ctk.CTkLabel(
            asset_preview, text="", text_color=self.COL_MUTED, anchor="w", justify="left"
        )
        self.detailed_asset_path_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        asset_btns = ctk.CTkFrame(assets, fg_color="transparent")
        asset_btns.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        asset_btns.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(
            asset_btns, text="素材を追加", command=self.add_detailed_assets,
            height=36, corner_radius=12, fg_color="#1f5d8f",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            asset_btns, text="選択を削除", command=self.remove_detailed_asset,
            height=36, corner_radius=12, fg_color="#3b1d1d",
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(
            asset_btns, text="再リンク", command=self.relink_detailed_asset,
            height=36, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=2, sticky="ew", padx=6)
        ctk.CTkButton(
            asset_btns, text="状態更新", command=self.refresh_detailed_assets,
            height=36, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        # Timeline
        timeline = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        timeline.grid(row=r, column=0, sticky="ew", padx=10, pady=(0, 16)); r += 1
        timeline.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            timeline, text="タイムライン編集（基本）",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        main_row, self.detailed_main_video_entry = self._v_path_row(
            timeline, "読み込み", self.browse_detailed_main_video
        )
        main_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        self._setup_detailed_timeline_tree_style()
        clip_cols = ("clip", "start", "end", "duration")
        self.detailed_timeline_tree = ttk.Treeview(
            timeline,
            columns=clip_cols,
            show="headings",
            height=6,
            style="Detailed.Treeview",
        )
        for col, text in zip(clip_cols, ["クリップ", "In", "Out", "長さ"], strict=False):
            self.detailed_timeline_tree.heading(col, text=text)
        self.detailed_timeline_tree.column("clip", width=140, anchor="w")
        self.detailed_timeline_tree.column("start", width=90, anchor="e")
        self.detailed_timeline_tree.column("end", width=90, anchor="e")
        self.detailed_timeline_tree.column("duration", width=90, anchor="e")
        self.detailed_timeline_tree.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.detailed_timeline_tree.bind("<ButtonPress-1>", self.on_timeline_drag_start)
        self.detailed_timeline_tree.bind("<B1-Motion>", self.on_timeline_drag_motion)
        self.detailed_timeline_tree.bind("<<TreeviewSelect>>", self.on_timeline_select)

        trim_row = ctk.CTkFrame(timeline, fg_color="transparent")
        trim_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        trim_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._v_hint(trim_row, "In").grid(row=0, column=0, sticky="w")
        self._v_hint(trim_row, "Out").grid(row=0, column=1, sticky="w")
        self._v_hint(trim_row, "Split位置").grid(row=0, column=2, sticky="w")

        self.detailed_trim_in_entry = self._v_entry(trim_row)
        self.detailed_trim_in_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.detailed_trim_out_entry = self._v_entry(trim_row)
        self.detailed_trim_out_entry.grid(row=1, column=1, sticky="ew", padx=6)
        self.detailed_split_entry = self._v_entry(trim_row)
        self.detailed_split_entry.grid(row=1, column=2, sticky="ew", padx=6)

        ctk.CTkButton(
            trim_row, text="適用", command=self.apply_timeline_trim,
            height=32, corner_radius=12, fg_color=self.COL_OK,
        ).grid(row=1, column=3, sticky="ew", padx=(6, 0))

        clip_btns = ctk.CTkFrame(timeline, fg_color="transparent")
        clip_btns.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        clip_btns.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(
            clip_btns, text="Split", command=self.split_timeline_clip,
            height=34, corner_radius=12, fg_color="#1f5d8f",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            clip_btns, text="削除", command=self.delete_timeline_clip,
            height=34, corner_radius=12, fg_color="#3b1d1d",
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(
            clip_btns, text="▲ 上へ", command=lambda: self.move_timeline_clip(-1),
            height=34, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=2, sticky="ew", padx=6)
        ctk.CTkButton(
            clip_btns, text="▼ 下へ", command=lambda: self.move_timeline_clip(1),
            height=34, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        # Overlay section
        overlays = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        overlays.grid(row=r, column=0, sticky="ew", padx=10, pady=(0, 16)); r += 1
        overlays.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            overlays, text="オーバーレイ（画像・テキスト）",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        overlay_form = ctk.CTkFrame(overlays, fg_color="transparent")
        overlay_form.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        overlay_form.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._v_hint(overlay_form, "種別").grid(row=0, column=0, sticky="w")
        self._v_hint(overlay_form, "開始").grid(row=0, column=1, sticky="w")
        self._v_hint(overlay_form, "終了").grid(row=0, column=2, sticky="w")
        self._v_hint(overlay_form, "X / Y").grid(row=0, column=3, sticky="w")

        self.detailed_overlay_type_var = ctk.StringVar(value="image")
        self.detailed_overlay_type_menu = ctk.CTkOptionMenu(
            overlay_form, values=["image", "text"], variable=self.detailed_overlay_type_var
        )
        self.detailed_overlay_type_menu.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.detailed_overlay_start_entry = self._v_entry(overlay_form)
        self.detailed_overlay_start_entry.grid(row=1, column=1, sticky="ew", padx=6)
        self.detailed_overlay_end_entry = self._v_entry(overlay_form)
        self.detailed_overlay_end_entry.grid(row=1, column=2, sticky="ew", padx=6)

        pos_wrap = ctk.CTkFrame(overlay_form, fg_color="transparent")
        pos_wrap.grid(row=1, column=3, sticky="ew", padx=(6, 0))
        pos_wrap.grid_columnconfigure((0, 1), weight=1)
        self.detailed_overlay_x_entry = self._v_entry(pos_wrap)
        self.detailed_overlay_x_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.detailed_overlay_y_entry = self._v_entry(pos_wrap)
        self.detailed_overlay_y_entry.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        overlay_form2 = ctk.CTkFrame(overlays, fg_color="transparent")
        overlay_form2.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        overlay_form2.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._v_hint(overlay_form2, "画像パス / テキスト").grid(row=0, column=0, sticky="w")
        self._v_hint(overlay_form2, "サイズ w/h").grid(row=0, column=1, sticky="w")
        self._v_hint(overlay_form2, "不透明度").grid(row=0, column=2, sticky="w")
        self._v_hint(overlay_form2, "文字サイズ/色/縁").grid(row=0, column=3, sticky="w")

        self.detailed_overlay_source_entry = self._v_entry(overlay_form2)
        self.detailed_overlay_source_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        size_wrap = ctk.CTkFrame(overlay_form2, fg_color="transparent")
        size_wrap.grid(row=1, column=1, sticky="ew", padx=6)
        size_wrap.grid_columnconfigure((0, 1), weight=1)
        self.detailed_overlay_w_entry = self._v_entry(size_wrap)
        self.detailed_overlay_w_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.detailed_overlay_h_entry = self._v_entry(size_wrap)
        self.detailed_overlay_h_entry.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.detailed_overlay_opacity_entry = self._v_entry(overlay_form2)
        self.detailed_overlay_opacity_entry.grid(row=1, column=2, sticky="ew", padx=6)

        text_wrap = ctk.CTkFrame(overlay_form2, fg_color="transparent")
        text_wrap.grid(row=1, column=3, sticky="ew", padx=(6, 0))
        text_wrap.grid_columnconfigure((0, 1, 2), weight=1)
        self.detailed_overlay_font_entry = self._v_entry(text_wrap)
        self.detailed_overlay_font_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.detailed_overlay_color_entry = self._v_entry(text_wrap)
        self.detailed_overlay_color_entry.grid(row=0, column=1, sticky="ew", padx=4)
        self.detailed_overlay_outline_entry = self._v_entry(text_wrap)
        self.detailed_overlay_outline_entry.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        overlay_btns = ctk.CTkFrame(overlays, fg_color="transparent")
        overlay_btns.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        overlay_btns.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(
            overlay_btns, text="追加", command=self.add_detailed_overlay,
            height=34, corner_radius=12, fg_color="#1f5d8f",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            overlay_btns, text="複製", command=self.duplicate_detailed_overlay,
            height=34, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(
            overlay_btns, text="削除", command=self.delete_detailed_overlay,
            height=34, corner_radius=12, fg_color="#3b1d1d",
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        overlay_columns = ("type", "start", "end", "x", "y", "w", "h", "opacity", "source")
        self.detailed_overlay_tree = ttk.Treeview(
            overlays,
            columns=overlay_columns,
            show="headings",
            height=6,
            style="Detailed.Treeview",
        )
        for col, text in zip(
            overlay_columns,
            ["種別", "開始", "終了", "X", "Y", "W", "H", "透過", "内容"],
            strict=False,
        ):
            self.detailed_overlay_tree.heading(col, text=text)
        self.detailed_overlay_tree.column("type", width=70, anchor="center")
        self.detailed_overlay_tree.column("start", width=80, anchor="e")
        self.detailed_overlay_tree.column("end", width=80, anchor="e")
        self.detailed_overlay_tree.column("x", width=60, anchor="e")
        self.detailed_overlay_tree.column("y", width=60, anchor="e")
        self.detailed_overlay_tree.column("w", width=60, anchor="e")
        self.detailed_overlay_tree.column("h", width=60, anchor="e")
        self.detailed_overlay_tree.column("opacity", width=70, anchor="e")
        self.detailed_overlay_tree.column("source", width=260, anchor="w")
        self.detailed_overlay_tree.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.detailed_overlay_tree.bind("<<TreeviewSelect>>", self.on_detailed_overlay_select)

        # Audio
        audio = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        audio.grid(row=r, column=0, sticky="ew", padx=10, pady=(0, 16)); r += 1
        audio.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            audio, text="音声",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        bgm_row, self.detailed_bgm_entry = self._v_path_row(audio, "BGM追加", self.browse_detailed_bgm)
        bgm_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        audio_row = ctk.CTkFrame(audio, fg_color="transparent")
        audio_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        audio_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._v_hint(audio_row, "BGM音量").grid(row=0, column=0, sticky="w")
        self._v_hint(audio_row, "動画音声").grid(row=0, column=1, sticky="w")
        self._v_hint(audio_row, "動画音量").grid(row=0, column=2, sticky="w")
        self._v_hint(audio_row, "フェード").grid(row=0, column=3, sticky="w")

        self.detailed_bgm_volume_slider = ctk.CTkSlider(audio_row, from_=0.0, to=1.0)
        self.detailed_bgm_volume_slider.set(0.7)
        self.detailed_bgm_volume_slider.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        self.detailed_video_audio_var = tk.BooleanVar(value=True)
        self.detailed_video_audio_check = ctk.CTkCheckBox(
            audio_row, text="ON", variable=self.detailed_video_audio_var
        )
        self.detailed_video_audio_check.grid(row=1, column=1, sticky="w", padx=6)

        self.detailed_video_volume_slider = ctk.CTkSlider(audio_row, from_=0.0, to=1.0)
        self.detailed_video_volume_slider.set(1.0)
        self.detailed_video_volume_slider.grid(row=1, column=2, sticky="ew", padx=6)

        fade_wrap = ctk.CTkFrame(audio_row, fg_color="transparent")
        fade_wrap.grid(row=1, column=3, sticky="ew", padx=(6, 0))
        fade_wrap.grid_columnconfigure((0, 1), weight=1)
        self.detailed_fade_in_entry = self._v_entry(fade_wrap)
        self.detailed_fade_in_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.detailed_fade_out_entry = self._v_entry(fade_wrap)
        self.detailed_fade_out_entry.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Preview
        preview = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        preview.grid(row=r, column=0, sticky="ew", padx=10, pady=(0, 16)); r += 1
        preview.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview, text="プレビュー",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        self.detailed_preview_label = ctk.CTkLabel(
            preview, text="タイムラインを読み込むと表示されます",
            text_color=self.COL_MUTED, anchor="center",
            width=640, height=360,
        )
        self.detailed_preview_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        preview_controls = ctk.CTkFrame(preview, fg_color="transparent")
        preview_controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        preview_controls.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            preview_controls, text="▶ 再生", command=self.start_detailed_preview,
            height=32, corner_radius=12, fg_color="#1f5d8f",
        ).grid(row=0, column=0, padx=(0, 6))
        ctk.CTkButton(
            preview_controls, text="■ 停止", command=self.stop_detailed_preview,
            height=32, corner_radius=12, fg_color="#3b1d1d",
        ).grid(row=0, column=1, padx=6)
        ctk.CTkButton(
            preview_controls, text="◀ 1フレ", command=lambda: self.step_preview(-1),
            height=32, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=2, padx=6)
        ctk.CTkButton(
            preview_controls, text="1フレ ▶", command=lambda: self.step_preview(1),
            height=32, corner_radius=12, fg_color="#172238",
        ).grid(row=0, column=3, padx=(6, 0))

        self.detailed_seek_slider = ctk.CTkSlider(
            preview, from_=0, to=1, command=self.on_preview_seek
        )
        self.detailed_seek_slider.set(0)
        self.detailed_seek_slider.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))

        # Export
        export = ctk.CTkFrame(form, corner_radius=16, fg_color=self.COL_PANEL)
        export.grid(row=r, column=0, sticky="ew", padx=10, pady=(0, 16)); r += 1
        export.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            export, text="書き出し",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COL_TEXT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        out_row, self.detailed_export_path_entry = self._v_path_row(
            export, "出力先", self.browse_detailed_export_path
        )
        out_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        export_row = ctk.CTkFrame(export, fg_color="transparent")
        export_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        export_row.grid_columnconfigure((0, 1, 2), weight=1)

        self._v_hint(export_row, "解像度").grid(row=0, column=0, sticky="w")
        self._v_hint(export_row, "FPS").grid(row=0, column=1, sticky="w")
        self._v_hint(export_row, "書き出し").grid(row=0, column=2, sticky="w")

        self.detailed_export_res_var = ctk.StringVar(value="1920x1080")
        self.detailed_export_res_menu = ctk.CTkOptionMenu(
            export_row,
            values=["1920x1080", "1080x1920", "1280x720", "720x1280", "元サイズ"],
            variable=self.detailed_export_res_var,
        )
        self.detailed_export_res_menu.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.detailed_export_fps_entry = self._v_entry(export_row)
        self.detailed_export_fps_entry.insert(0, "30")
        self.detailed_export_fps_entry.grid(row=1, column=1, sticky="ew", padx=6)
        ctk.CTkButton(
            export_row, text="書き出し開始", command=self.export_detailed_video,
            height=36, corner_radius=12, fg_color=self.COL_OK,
        ).grid(row=1, column=2, sticky="ew", padx=(6, 0))

        self.detailed_export_progress = ctk.CTkProgressBar(export)
        self.detailed_export_progress.set(0)
        self.detailed_export_progress.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))
        self.detailed_export_label = ctk.CTkLabel(
            export, text="進捗: 0%", text_color=self.COL_MUTED, anchor="w"
        )
        self.detailed_export_label.grid(row=4, column=0, sticky="w", padx=14, pady=(0, 12))

        self.detailed_trim_in_entry.insert(0, "00:00")
        self.detailed_trim_out_entry.insert(0, "00:05")
        self.detailed_split_entry.insert(0, "00:02")
        self.detailed_overlay_start_entry.insert(0, "00:00")
        self.detailed_overlay_end_entry.insert(0, "00:05")
        self.detailed_overlay_x_entry.insert(0, "0")
        self.detailed_overlay_y_entry.insert(0, "0")
        self.detailed_overlay_w_entry.insert(0, "0")
        self.detailed_overlay_h_entry.insert(0, "0")
        self.detailed_overlay_opacity_entry.insert(0, "1.0")
        self.detailed_overlay_font_entry.insert(0, "32")
        self.detailed_overlay_color_entry.insert(0, "#ffffff")
        self.detailed_overlay_outline_entry.insert(0, "2")
        self.detailed_fade_in_entry.insert(0, "0.0")
        self.detailed_fade_out_entry.insert(0, "0.0")

    def _sync_edit_preview_state(self):
        if not hasattr(self, "edit_preview_label"):
            return
        if not self._edit_preview_base:
            self.edit_preview_label.configure(text="動画を選択してください", image="")
            self._edit_preview_imgtk = None
            return
        self._update_edit_preview()

    def _sync_edit_preview_from_sliders(self):
        self._sync_edit_entries_from_sliders()
        self._update_edit_preview()

    def _sync_edit_entries_from_sliders(self):
        if not hasattr(self, "edit_preview_x_slider"):
            return
        x = int(self.edit_preview_x_slider.get())
        y = int(self.edit_preview_y_slider.get()) if hasattr(self, "edit_preview_y_slider") else 0
        scale = int(self.edit_preview_scale_slider.get()) if hasattr(self, "edit_preview_scale_slider") else 100

        if hasattr(self, "edit_preview_x_value"):
            self.edit_preview_x_value.configure(text=str(x))
        if hasattr(self, "edit_preview_y_value"):
            self.edit_preview_y_value.configure(text=str(y))
        if hasattr(self, "edit_preview_scale_value"):
            self.edit_preview_scale_value.configure(text=f"{scale}%")

        if hasattr(self, "edit_x_entry"):
            self.edit_x_entry.delete(0, "end")
            self.edit_x_entry.insert(0, str(x))
        if hasattr(self, "edit_y_entry"):
            self.edit_y_entry.delete(0, "end")
            self.edit_y_entry.insert(0, str(y))

        if self._edit_overlay_original_size and hasattr(self, "edit_w_entry") and hasattr(self, "edit_h_entry"):
            base_w, base_h = self._edit_overlay_original_size
            w = max(1, int(base_w * scale / 100))
            h = max(1, int(base_h * scale / 100))
            self.edit_w_entry.delete(0, "end")
            self.edit_w_entry.insert(0, str(w))
            self.edit_h_entry.delete(0, "end")
            self.edit_h_entry.insert(0, str(h))

    def _update_edit_preview_slider_ranges(self):
        if not self._edit_preview_size:
            return
        width, height = self._edit_preview_size
        if hasattr(self, "edit_preview_x_slider"):
            self.edit_preview_x_slider.configure(to=max(0, width))
        if hasattr(self, "edit_preview_y_slider"):
            self.edit_preview_y_slider.configure(to=max(0, height))

        if hasattr(self, "edit_preview_x_slider"):
            self.edit_preview_x_slider.set(min(self.edit_preview_x_slider.get(), width))
        if hasattr(self, "edit_preview_y_slider"):
            self.edit_preview_y_slider.set(min(self.edit_preview_y_slider.get(), height))

    def _load_edit_video_preview(self, path: str):
        if not path or not Path(path).exists():
            self._edit_preview_base = None
            self._edit_preview_size = None
            self._sync_edit_preview_state()
            return

        clip = None
        try:
            clip = VideoFileClip(path)
            t = 0.1 if clip.duration and clip.duration > 0.1 else 0
            frame = clip.get_frame(t)
            img = Image.fromarray(frame).convert("RGBA")
            self._edit_preview_base = img
            self._edit_preview_size = img.size
            self._update_edit_preview_slider_ranges()
            self._sync_edit_preview_from_sliders()
        except Exception as e:
            self._edit_preview_base = None
            self._edit_preview_size = None
            if hasattr(self, "edit_preview_label"):
                self.edit_preview_label.configure(text=f"プレビュー取得失敗: {e}", image="")
            self._edit_preview_imgtk = None
        finally:
            if clip:
                clip.close()

    def _load_edit_overlay_preview(self, path: str):
        if not path or not Path(path).exists():
            self._edit_preview_overlay = None
            self._edit_overlay_original_size = None
            self._sync_edit_preview_state()
            return
        try:
            overlay = Image.open(path).convert("RGBA")
            self._edit_preview_overlay = overlay
            self._edit_overlay_original_size = overlay.size
            if hasattr(self, "edit_preview_scale_slider"):
                self.edit_preview_scale_slider.set(100)
            self._sync_edit_preview_from_sliders()
        except Exception as e:
            self._edit_preview_overlay = None
            self._edit_overlay_original_size = None
            if hasattr(self, "edit_preview_label"):
                self.edit_preview_label.configure(text=f"画像読み込み失敗: {e}", image="")
            self._edit_preview_imgtk = None

    def _update_edit_preview(self):
        if not self._edit_preview_base or not hasattr(self, "edit_preview_label"):
            return
        base = self._edit_preview_base.copy().convert("RGBA")

        if self._edit_preview_overlay:
            scale = int(self.edit_preview_scale_slider.get()) if hasattr(self, "edit_preview_scale_slider") else 100
            x = int(self.edit_preview_x_slider.get()) if hasattr(self, "edit_preview_x_slider") else 0
            y = int(self.edit_preview_y_slider.get()) if hasattr(self, "edit_preview_y_slider") else 0
            ov = self._edit_preview_overlay
            if self._edit_overlay_original_size:
                ow, oh = self._edit_overlay_original_size
            else:
                ow, oh = ov.size
            nw = max(1, int(ow * scale / 100))
            nh = max(1, int(oh * scale / 100))
            ov_resized = ov.resize((nw, nh), Image.LANCZOS)
            base.alpha_composite(ov_resized, dest=(x, y))

        display = base.copy()
        display.thumbnail(self.EDIT_PREVIEW_MAX, Image.LANCZOS)
        self._edit_preview_imgtk = ImageTk.PhotoImage(display)
        self.edit_preview_label.configure(image=self._edit_preview_imgtk, text="")

    def browse_edit_input_mp4(self):
        path = filedialog.askopenfilename(
            title="入力MP4を選択",
            filetypes=[("MP4", "*.mp4"), ("動画ファイル", "*.mp4;*.mov;*.mkv;*.avi"), ("すべて", "*.*")],
        )
        if path:
            self.edit_input_entry.delete(0, "end")
            self.edit_input_entry.insert(0, path)
            self._load_edit_video_preview(path)
            self.save_config()

    def browse_edit_output_mp4(self):
        path = filedialog.asksaveasfilename(
            title="出力MP4の保存先を選択",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("すべて", "*.*")],
        )
        if path:
            self.edit_output_entry.delete(0, "end")
            self.edit_output_entry.insert(0, path)
            self.save_config()

    def browse_edit_overlay_image(self):
        path = filedialog.askopenfilename(
            title="重ねる画像を選択",
            filetypes=[("画像", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("すべて", "*.*")],
        )
        if path:
            self.edit_overlay_img_entry.delete(0, "end")
            self.edit_overlay_img_entry.insert(0, path)
            self._load_edit_overlay_preview(path)
            self.save_config()

    def browse_edit_srt(self):
        path = filedialog.askopenfilename(
            title="SRTファイルを選択",
            filetypes=[("SRT", "*.srt"), ("すべて", "*.*")],
        )
        if path and hasattr(self, "edit_srt_entry"):
            self.edit_srt_entry.delete(0, "end")
            self.edit_srt_entry.insert(0, path)
            self.save_config()

    def browse_ponchi_srt(self):
        path = filedialog.askopenfilename(
            title="SRTファイルを選択",
            filetypes=[("SRT", "*.srt"), ("すべて", "*.*")],
        )
        if path and hasattr(self, "ponchi_srt_entry"):
            self.ponchi_srt_entry.delete(0, "end")
            self.ponchi_srt_entry.insert(0, path)
            self.save_config()

    def browse_ponchi_output_dir(self):
        path = filedialog.askdirectory(title="出力フォルダを選択")
        if path and hasattr(self, "ponchi_output_dir_entry"):
            self.ponchi_output_dir_entry.delete(0, "end")
            self.ponchi_output_dir_entry.insert(0, path)
            self.save_config()

    def browse_edit_image_output_dir(self):
        path = filedialog.askdirectory(title="画像の保存先フォルダを選択")
        if path and hasattr(self, "edit_image_output_entry"):
            self.edit_image_output_entry.delete(0, "end")
            self.edit_image_output_entry.insert(0, path)
            self.save_config()

    def add_edit_overlay(self):
        try:
            img = self.edit_overlay_img_entry.get().strip()
            if not img or not Path(img).exists():
                messagebox.showerror("エラー", "有効な画像ファイルを選択してください。")
                return

            start_s = self.edit_start_entry.get().strip()
            end_s = self.edit_end_entry.get().strip()
            start = parse_timecode_to_seconds(start_s)
            end = parse_timecode_to_seconds(end_s)

            x = int(self.edit_x_entry.get().strip() or "0")
            y = int(self.edit_y_entry.get().strip() or "0")
            w = int(self.edit_w_entry.get().strip() or "0")
            h = int(self.edit_h_entry.get().strip() or "0")

            opacity = float(self.edit_opacity_entry.get().strip() or "1.0")
            if opacity < 0 or opacity > 1:
                messagebox.showerror("エラー", "不透明度は 0.0〜1.0 で指定してください。")
                return

            if end <= start:
                messagebox.showerror("エラー", "終了時間は開始時間より大きくしてください。")
                return

            ov = {
                "image_path": img,
                "start": float(start),
                "end": float(end),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "opacity": float(opacity),
            }
            self.edit_overlays.append(ov)
            self.refresh_edit_overlay_table()
            self.save_config()
            self.log(f"✅ オーバーレイ追加: {Path(img).name} {start_s}〜{end_s} pos=({x},{y}) size=({w},{h}) op={opacity}")

        except Exception as e:
            messagebox.showerror("エラー", f"オーバーレイ追加に失敗しました:\n{e}")

    def clear_edit_overlays(self):
        self.edit_overlays = []
        self.refresh_edit_overlay_table()
        self.save_config()
        self.log("🧹 オーバーレイ一覧をクリアしました。")

    def _get_edit_import_defaults(self) -> Dict[str, Any]:
        try:
            x = int(self.edit_default_x_entry.get().strip() or DEFAULT_EDIT_IMPORT_X)
        except Exception:
            x = DEFAULT_EDIT_IMPORT_X
        try:
            y = int(self.edit_default_y_entry.get().strip() or DEFAULT_EDIT_IMPORT_Y)
        except Exception:
            y = DEFAULT_EDIT_IMPORT_Y
        try:
            w = int(self.edit_default_w_entry.get().strip() or DEFAULT_EDIT_IMPORT_W)
        except Exception:
            w = DEFAULT_EDIT_IMPORT_W
        try:
            h = int(self.edit_default_h_entry.get().strip() or DEFAULT_EDIT_IMPORT_H)
        except Exception:
            h = DEFAULT_EDIT_IMPORT_H
        try:
            opacity = float(self.edit_default_opacity_entry.get().strip() or DEFAULT_EDIT_IMPORT_OPACITY)
        except Exception:
            opacity = DEFAULT_EDIT_IMPORT_OPACITY

        return {"x": x, "y": y, "w": w, "h": h, "opacity": opacity}

    def _generate_search_queries(self, items: List[Dict[str, Any]]) -> List[str]:
        api_key = self._get_gemini_api_key()
        if not api_key:
            raise RuntimeError("設定タブでGemini APIキーを入力してください。")

        client = genai.Client(api_key=api_key)
        prompt = (
            "あなたは動画編集の画像リサーチ担当です。\n"
            "次の字幕テキストごとに、Google/Bing画像検索向けの日本語検索クエリを1つずつ作成してください。\n"
            "出力は JSON 配列のみで、要素はクエリ文字列にしてください。\n"
            "字幕一覧:\n"
        )
        for idx, item in enumerate(items, 1):
            prompt += f"{idx}. {item.get('text', '')}\n"

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        text = getattr(resp, "text", "") or ""
        if not text:
            raise RuntimeError("検索クエリ生成に失敗しました。")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                raise RuntimeError("検索クエリのJSON解析に失敗しました。")
            data = json.loads(match.group(0))
        if not isinstance(data, list):
            raise RuntimeError("検索クエリは配列形式で返してください。")
        queries = [str(x).strip() for x in data if str(x).strip()]
        if not queries:
            raise RuntimeError("検索クエリが空でした。")
        return queries

    def _collect_images_from_srt_worker(
        self,
        srt_path: str,
        output_dir: str,
        provider: str,
        search_key: str,
    ):
        try:
            items = parse_srt_file(srt_path)
            if not items:
                raise RuntimeError("SRTから字幕が見つかりませんでした。")

            self.log(f"🔍 SRTから {len(items)} 件の字幕を読み込みました。")
            queries = self._generate_search_queries(items)
            if len(queries) < len(items):
                queries.extend(queries[-1:] * (len(items) - len(queries)))

            srt_stem = Path(srt_path).stem
            output_dir_path = Path(output_dir) / srt_stem
            output_dir_path.mkdir(parents=True, exist_ok=True)
            json_output_path = output_dir_path / f"{srt_stem}.json"
            results = []
            url_to_image: Dict[str, Path] = {}
            total = len(items)

            for idx, (item, query) in enumerate(zip(items, queries), 1):
                try:
                    self.log(f"収集中です・・・(検索ワード：{query} {len(results)}件取得済み)")
                    self.update_progress(idx / total)
                    urls = search_images_serpapi(search_key, query, provider=provider)
                    if not urls:
                        self.log(f"⚠️ 画像が見つかりませんでした: {query}")
                        continue

                    saved = None
                    for url in urls:
                        if url in url_to_image:
                            saved = url_to_image[url]
                            self.log(f"🔁 画像を再利用しました: {query} -> {saved.name}")
                            break

                    if saved is None:
                        selected_url = None
                        for url in urls:
                            if url not in url_to_image:
                                selected_url = url
                                break
                        if selected_url is None:
                            selected_url = urls[0]
                            saved = url_to_image.get(selected_url)
                        if saved is None:
                            saved = download_image(selected_url, output_dir_path, f"overlay_{idx:03d}")
                            url_to_image[selected_url] = saved

                    results.append(
                        {
                            "start": float(item["start"]),
                            "end": float(item["end"]),
                            "image": saved.name,
                        }
                    )
                except Exception as exc:
                    self.log(f"⚠️ 画像取得失敗: {query} ({exc})")
                    continue

            if not results:
                raise RuntimeError("画像を取得できませんでした。")

            json_output_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self.log(f"✅ 画像 {len(results)} 件を保存し、JSONを書き出しました: {json_output_path}")
            self.after(0, lambda: messagebox.showinfo("完了", "画像収集とJSON出力が完了しました。"))
            self.after(0, self.save_config)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("エラー", str(exc)))

    def on_collect_images_from_srt(self):
        try:
            srt_path = self.edit_srt_entry.get().strip()
            if not srt_path or not Path(srt_path).exists():
                raise RuntimeError("有効なSRTファイルを指定してください。")

            output_dir = self.edit_image_output_entry.get().strip()
            if not output_dir:
                raise RuntimeError("画像の保存先フォルダを指定してください。")

            provider = self.edit_search_provider_var.get() if hasattr(self, "edit_search_provider_var") else "Google"
            search_key = self.edit_search_api_key_entry.get().strip()
            if not search_key:
                raise RuntimeError("画像検索 APIキーを入力してください。")

            self.update_progress(0.0)
            worker = threading.Thread(
                target=self._collect_images_from_srt_worker,
                args=(srt_path, output_dir, provider, search_key),
                daemon=True,
            )
            worker.start()
        except Exception as exc:
            messagebox.showerror("エラー", str(exc))

    def import_edit_overlays_from_json(self):
        path = filedialog.askopenfilename(
            title="JSONを読み込む",
            filetypes=[("JSON", "*.json"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("エラー", f"JSONの読み込みに失敗しました: {exc}")
            return

        if isinstance(payload, dict):
            items = payload.get("overlays", [])
        else:
            items = payload

        if not isinstance(items, list):
            messagebox.showerror("エラー", "JSON形式が不正です。")
            return

        defaults = self._get_edit_import_defaults()
        base_dir = Path(path).parent
        imported = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            image_name = item.get("image") or item.get("image_path")
            if start is None or end is None or not image_name:
                continue
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            if not image_path.exists():
                continue

            self.edit_overlays.append(
                {
                    "image_path": str(image_path),
                    "start": float(start),
                    "end": float(end),
                    "x": defaults["x"],
                    "y": defaults["y"],
                    "w": defaults["w"],
                    "h": defaults["h"],
                    "opacity": defaults["opacity"],
                }
            )
            imported += 1

        if imported == 0:
            messagebox.showerror("エラー", "インポートできる行がありませんでした。")
            return

        self.refresh_edit_overlay_table()
        self.save_config()
        self.log(f"✅ JSONから {imported} 件をインポートしました。")
        messagebox.showinfo("完了", f"{imported} 件のオーバーレイをインポートしました。")

    def refresh_edit_overlay_list(self):
        if not hasattr(self, "edit_overlay_list"):
            return
        self.edit_overlay_list.delete("1.0", "end")
        for i, ov in enumerate(self.edit_overlays, 1):
            img = Path(ov["image_path"]).name
            start = ov["start"]
            end = ov["end"]
            x = ov.get("x", 0)
            y = ov.get("y", 0)
            w = ov.get("w", 0)
            h = ov.get("h", 0)
            op = ov.get("opacity", 1.0)
            self.edit_overlay_list.insert(
                "end",
                f"{i:02d}. {img}  t={start:.2f}〜{end:.2f}  pos=({x},{y})  size=({w},{h})  op={op}\n"
            )

    def on_render_edit_video_clicked(self):
        in_mp4 = self.edit_input_entry.get().strip()
        out_mp4 = self.edit_output_entry.get().strip()

        if not in_mp4 or not Path(in_mp4).exists():
            messagebox.showerror("エラー", "有効な入力MP4を選択してください。")
            return
        if not out_mp4:
            messagebox.showerror("エラー", "出力MP4の保存先を指定してください。")
            return
        if not self.edit_overlays:
            messagebox.showerror("エラー", "オーバーレイが1件もありません。")
            return

        self.save_config()
        self.log_text.delete("1.0", "end")
        self.update_progress(0.0)
        self.set_status("Working", ok=True)
        self.log("=== 動画編集（オーバーレイ）開始 ===")
        self.edit_render_button.configure(state="disabled", text="処理中...")

        def worker():
            try:
                apply_image_overlays_to_video(
                    input_mp4=in_mp4,
                    overlays=self.edit_overlays,
                    output_mp4=out_mp4,
                    log_fn=self.log,
                    progress_fn=self.update_progress,
                )
                self.set_status("Ready", ok=True)
                self.log("=== 動画編集 完了 ===")
                messagebox.showinfo("完了", "動画編集（オーバーレイ）の書き出しが完了しました。")
            except Exception as e:
                self.set_status("Error", ok=False)
                tb = traceback.format_exc()
                self.log("❌ 動画編集でエラー:\n" + tb)
                messagebox.showerror("エラー", f"動画編集に失敗しました:\n{e}")
            finally:
                self.after(0, lambda: self.edit_render_button.configure(state="normal", text="▶ オーバーレイを書き出す"))

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------
    # Settings page
    # --------------------------
    def _build_settings_page(self, page):
        self._build_page_header("settings", page, "設定")

        form = self._make_scroll_form(page)
        form.grid_columnconfigure(0, weight=1)

        r = 0

        self._v_label(form, "APIキー管理").grid(row=r, column=0, sticky="w", pady=(10, 6)); r += 1
        self._v_hint(
            form,
            "Gemini / ChatGPT / ClaudeCode のAPIキーを一元管理します。各機能のAPIキー欄はこの設定を参照します。",
        ).grid(row=r, column=0, sticky="w", pady=(0, 12)); r += 1

        self._v_label(form, "Gemini APIキー").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.gemini_api_key_entry = self._v_entry(form, show="*")
        self.gemini_api_key_entry.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "ChatGPT APIキー").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.chatgpt_api_key_entry = self._v_entry(form, show="*")
        self.chatgpt_api_key_entry.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        self._v_label(form, "ClaudeCode APIキー").grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1
        self.claude_api_key_entry = self._v_entry(form, show="*")
        self.claude_api_key_entry.grid(row=r, column=0, sticky="ew", pady=(0, 16)); r += 1

        ctk.CTkButton(
            form,
            text="設定を保存",
            command=self._save_settings_keys,
            fg_color=self.COL_OK,
            hover_color=self.COL_OK_HOVER,
            height=40,
            corner_radius=12,
        ).grid(row=r, column=0, sticky="ew", pady=(0, 18)); r += 1

    def _save_settings_keys(self):
        self.save_config()
        self.log("✅ 設定のAPIキーを保存しました。")
        self._refresh_script_model_options()
        messagebox.showinfo("保存完了", "APIキー設定を保存しました。")

    # --------------------------
    # About page
    # --------------------------
    def _build_about_page(self, page):
        self._build_page_header("about", page, "About")

        body = ctk.CTkFrame(page, corner_radius=18, fg_color=self.COL_PANEL)
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        txt = ctk.CTkTextbox(
            body, corner_radius=14, fg_color=self.COL_BG,
            border_width=1, border_color=self.COL_BORDER
        )
        txt.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        txt.insert(
            "end",
            "News Short Generator Studio\n\n"
            "- 左：サイドバー\n"
            "- 中央：フォーム（動画生成 / 台本生成 / 動画タイトル・説明作成 / 資料作成 / ポンチ絵作成 / 動画編集 / 詳細動画編集 / 設定）\n"
            "- 右：ログ（進捗）\n\n"
            "[動画編集]\n"
            "- 指定時間帯に画像を座標指定で重ねる（複数対応）\n"
            "- CompositeVideoClip により合成\n\n"
            "Tips:\n"
            "- 中央フォームが崩れる場合、ScrollableFrame内をgrid統一し、入力列の weight=1 を徹底してください。\n"
        )
        txt.configure(state="disabled")

    # --------------------------
    # Log panel
    # --------------------------
    def _build_log_panel(self):
        self.log_panel.grid_rowconfigure(2, weight=1)
        self.log_panel.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self.log_panel, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            head, text="ログ",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.COL_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.log_panel)
        self.progress_bar.configure(
            progress_color=self.COL_ACCENT,
            fg_color=self.COL_CARD_SOFT,
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=14, pady=(6, 6))
        self.progress_bar.set(0)

        self.log_text = ctk.CTkTextbox(
            self.log_panel,
            corner_radius=14,
            fg_color=self.COL_BG,
            border_width=1,
            border_color=self.COL_BORDER,
        )
        self.log_text.grid(row=2, column=0, sticky="nsew", padx=14, pady=(6, 6))

        self.progress_label = ctk.CTkLabel(self.log_panel, text="進捗: 0%", text_color=self.COL_MUTED, anchor="w")
        self.progress_label.grid(row=3, column=0, sticky="w", padx=14, pady=(6, 14))

    # --------------------------
    # Config
    # --------------------------
    def load_config(self):
        if not CONFIG_PATH.exists() and LEGACY_CONFIG_PATH.exists():
            try:
                LEGACY_CONFIG_PATH.replace(CONFIG_PATH)
            except Exception:
                pass
        if not CONFIG_PATH.exists():
            # 初回はテンプレだけ初期化
            self.prompt_templates = self._default_prompt_templates()
            self._refresh_template_menu()
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            self.prompt_templates = self._default_prompt_templates()
            self._refresh_template_menu()
            return

        gemini_key = (
            data.get("gemini_api_key")
            or data.get("material_api_key")
            or data.get("ponchi_gemini_api_key")
            or ""
        )
        chatgpt_key = data.get("chatgpt_api_key") or data.get("ponchi_openai_api_key") or ""
        claude_key = data.get("claude_api_key") or ""
        if hasattr(self, "gemini_api_key_entry"):
            self.gemini_api_key_entry.delete(0, "end")
            self.gemini_api_key_entry.insert(0, gemini_key)

        if hasattr(self, "chatgpt_api_key_entry"):
            self.chatgpt_api_key_entry.delete(0, "end")
            self.chatgpt_api_key_entry.insert(0, chatgpt_key)

        if hasattr(self, "claude_api_key_entry"):
            self.claude_api_key_entry.delete(0, "end")
            self.claude_api_key_entry.insert(0, claude_key)

        # video

        self.script_entry.delete(0, "end")
        self.script_entry.insert(0, data.get("script_path", ""))

        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, data.get("output_dir", ""))

        self.image_paths = data.get("image_paths", [])
        self.refresh_image_listbox()

        self.use_bgm_var.set(bool(data.get("use_bgm", False)))

        self.bgm_entry.delete(0, "end")
        self.bgm_entry.insert(0, data.get("bgm_path", ""))

        self.bgm_gain_slider.set(float(data.get("bgm_gain_db", -18)))

        self.fps_entry.delete(0, "end")
        self.fps_entry.insert(0, str(data.get("fps", 30)))

        self.tts_engine_var.set(data.get("tts_engine", "Gemini"))

        self.voice_entry.delete(0, "end")
        self.voice_entry.insert(0, data.get("voice_name", "Kore"))

        self.vv_baseurl_entry.delete(0, "end")
        self.vv_baseurl_entry.insert(0, data.get("vv_base_url", DEFAULT_VOICEVOX_URL))

        self.vv_mode_var.set(data.get("vv_mode", "ローテーション"))

        self.vv_rotation_entry.delete(0, "end")
        self.vv_rotation_entry.insert(0, data.get("vv_rotation_labels", ",".join(str(x) for x in DEFAULT_VV_ROTATION)))

        self.vv_caster_entry.delete(0, "end")
        self.vv_caster_entry.insert(0, data.get("vv_caster_label", DEFAULT_VV_CASTER_LABEL))

        self.vv_analyst_entry.delete(0, "end")
        self.vv_analyst_entry.insert(0, data.get("vv_analyst_label", DEFAULT_VV_ANALYST_LABEL))

        self.vv_speed_slider.set(float(data.get("vv_speed", DEFAULT_VV_SPEED)))

        width = data.get("width", 1080)
        height = data.get("height", 1920)
        self.width_entry.delete(0, "end")
        self.width_entry.insert(0, str(width))
        self.height_entry.delete(0, "end")
        self.height_entry.insert(0, str(height))

        self.caption_font_entry.delete(0, "end")
        self.caption_font_entry.insert(0, str(int(data.get("caption_font_size", FONT_SIZE))))

        self.speaker_font_entry.delete(0, "end")
        self.speaker_font_entry.insert(0, str(int(data.get("speaker_font_size", SPEAKER_FONT_SIZE))))

        self.caption_width_entry.delete(0, "end")
        self.caption_width_entry.insert(0, str(int(data.get("caption_max_chars", CAPTION_MAX_CHARS_PER_LINE))))

        self.caption_box_enabled_var.set(bool(data.get("caption_box_enabled", DEFAULT_CAPTION_BOX_ENABLED)))

        fallback_h = data.get("caption_margin_bottom", DEFAULT_CAPTION_BOX_HEIGHT)
        box_h = int(data.get("caption_box_height", fallback_h))
        self.caption_box_height_entry.delete(0, "end")
        self.caption_box_height_entry.insert(0, str(box_h))

        self.caption_alpha_entry.delete(0, "end")
        self.caption_alpha_entry.insert(0, str(int(data.get("caption_box_alpha", CAPTION_BOX_ALPHA))))

        self.bg_off_style_var.set(data.get("bg_off_style", "影"))

        self.caption_text_color_entry.delete(0, "end")
        self.caption_text_color_entry.insert(0, data.get("caption_text_color", DEFAULT_CAPTION_TEXT_COLOR))

        self.on_tts_engine_change(self.tts_engine_var.get())

        # script
        if hasattr(self, "script_engine_var"):
            self.script_engine_var.set(data.get("script_engine", "ClaudeCode"))

        if hasattr(self, "script_gemini_model_menu"):
            self._sync_option_menu_values(
                self.script_gemini_model_menu,
                self.script_gemini_model_var,
                SCRIPT_MODEL_MASTER["Gemini"],
                data.get("script_gemini_model"),
                DEFAULT_SCRIPT_GEMINI_MODEL,
            )

        if hasattr(self, "script_chatgpt_model_menu"):
            self._sync_option_menu_values(
                self.script_chatgpt_model_menu,
                self.script_chatgpt_model_var,
                SCRIPT_MODEL_MASTER["ChatGPT"],
                data.get("script_chatgpt_model"),
                DEFAULT_SCRIPT_OPENAI_MODEL,
            )

        if hasattr(self, "script_claude_model_menu"):
            self._sync_option_menu_values(
                self.script_claude_model_menu,
                self.script_claude_model_var,
                SCRIPT_MODEL_MASTER["ClaudeCode"],
                data.get("claude_model"),
                DEFAULT_CLAUDE_MODEL,
            )

        self.claude_max_tokens_entry.delete(0, "end")
        self.claude_max_tokens_entry.insert(0, str(data.get("claude_max_tokens", DEFAULT_CLAUDE_MAX_TOKENS)))

        self.script_save_path_entry.delete(0, "end")
        self.script_save_path_entry.insert(0, data.get("script_save_path", str(Path.home() / "dialogue_input.txt")))

        self._set_textbox(self.claude_prompt_text, data.get("claude_prompt", ""))
        self._set_textbox(self.claude_output_text, data.get("claude_output", ""))

        self.on_script_engine_change(self.script_engine_var.get())

        tpls = data.get("prompt_templates")
        if isinstance(tpls, dict) and tpls:
            self.prompt_templates = {str(k): str(v) for k, v in tpls.items()}
        else:
            self.prompt_templates = self._default_prompt_templates()

        selected = data.get("prompt_template_selected", "")
        if selected and selected in self.prompt_templates:
            self.prompt_template_var.set(selected)
        else:
            self.prompt_template_var.set(next(iter(self.prompt_templates.keys()), "（テンプレなし）"))

        # title/description
        if hasattr(self, "title_script_entry"):
            self.title_script_entry.delete(0, "end")
            self.title_script_entry.insert(0, data.get("title_script_path", ""))

        if hasattr(self, "title_engine_var"):
            self.title_engine_var.set(data.get("title_engine", "Gemini"))

        if hasattr(self, "title_gemini_model_menu"):
            self._sync_option_menu_values(
                self.title_gemini_model_menu,
                self.title_gemini_model_var,
                SCRIPT_MODEL_MASTER["Gemini"],
                data.get("title_gemini_model"),
                DEFAULT_SCRIPT_GEMINI_MODEL,
            )

        if hasattr(self, "title_chatgpt_model_menu"):
            self._sync_option_menu_values(
                self.title_chatgpt_model_menu,
                self.title_chatgpt_model_var,
                SCRIPT_MODEL_MASTER["ChatGPT"],
                data.get("title_chatgpt_model"),
                DEFAULT_SCRIPT_OPENAI_MODEL,
            )

        if hasattr(self, "title_claude_model_menu"):
            self._sync_option_menu_values(
                self.title_claude_model_menu,
                self.title_claude_model_var,
                SCRIPT_MODEL_MASTER["ClaudeCode"],
                data.get("title_claude_model"),
                DEFAULT_CLAUDE_MODEL,
            )

        if hasattr(self, "title_count_entry"):
            self.title_count_entry.delete(0, "end")
            self.title_count_entry.insert(0, str(data.get("title_count", 5)))

        if hasattr(self, "title_extra_text"):
            self._set_textbox(self.title_extra_text, data.get("title_extra", ""))

        if hasattr(self, "title_output_text"):
            self._set_textbox(self.title_output_text, data.get("title_output", ""))

        if hasattr(self, "title_engine_var"):
            self.on_title_engine_change(self.title_engine_var.get())

        # material
        if hasattr(self, "material_model_entry"):
            self.material_model_entry.delete(0, "end")
            self.material_model_entry.insert(0, data.get("material_model", GEMINI_MATERIAL_DEFAULT_MODEL))

        if hasattr(self, "material_prompt_text"):
            self._set_textbox(self.material_prompt_text, data.get("material_prompt", ""))

        if hasattr(self, "material_save_path_entry"):
            self.material_save_path_entry.delete(0, "end")
            default_dir = str(self._default_material_save_dir())
            self.material_save_path_entry.insert(0, data.get("material_save_path", default_dir))

        # ponchi (NEW)
        if hasattr(self, "ponchi_srt_entry"):
            self.ponchi_srt_entry.delete(0, "end")
            self.ponchi_srt_entry.insert(0, data.get("ponchi_srt_path", ""))

        if hasattr(self, "ponchi_output_dir_entry"):
            self.ponchi_output_dir_entry.delete(0, "end")
            self.ponchi_output_dir_entry.insert(0, data.get("ponchi_output_dir", ""))

        if hasattr(self, "ponchi_suggestion_engine_var"):
            self.ponchi_suggestion_engine_var.set(data.get("ponchi_engine", "Gemini"))

        if hasattr(self, "ponchi_gemini_model_entry"):
            self.ponchi_gemini_model_entry.delete(0, "end")
            self.ponchi_gemini_model_entry.insert(0, data.get("ponchi_gemini_model", DEFAULT_PONCHI_GEMINI_MODEL))

        if hasattr(self, "ponchi_openai_model_entry"):
            self.ponchi_openai_model_entry.delete(0, "end")
            self.ponchi_openai_model_entry.insert(0, data.get("ponchi_openai_model", DEFAULT_PONCHI_OPENAI_MODEL))

        # edit (NEW)
        if hasattr(self, "edit_input_entry"):
            self.edit_input_entry.delete(0, "end")
            self.edit_input_entry.insert(0, data.get("edit_input_mp4", ""))

        if hasattr(self, "edit_output_entry"):
            self.edit_output_entry.delete(0, "end")
            self.edit_output_entry.insert(0, data.get("edit_output_mp4", ""))

        if hasattr(self, "edit_srt_entry"):
            self.edit_srt_entry.delete(0, "end")
            self.edit_srt_entry.insert(0, data.get("edit_srt_path", "") or "")

        if hasattr(self, "edit_image_output_entry"):
            current = self.edit_image_output_entry.get()
            self.edit_image_output_entry.delete(0, "end")
            self.edit_image_output_entry.insert(0, data.get("edit_image_output_dir", "") or current)

        if hasattr(self, "edit_search_provider_var"):
            self.edit_search_provider_var.set(data.get("edit_search_provider", DEFAULT_IMAGE_SEARCH_PROVIDER))

        if hasattr(self, "edit_search_api_key_entry"):
            self.edit_search_api_key_entry.delete(0, "end")
            self.edit_search_api_key_entry.insert(0, data.get("edit_search_api_key", ""))

        if hasattr(self, "edit_default_x_entry"):
            self.edit_default_x_entry.delete(0, "end")
            self.edit_default_x_entry.insert(0, str(data.get("edit_default_x", DEFAULT_EDIT_IMPORT_X)))

        if hasattr(self, "edit_default_y_entry"):
            self.edit_default_y_entry.delete(0, "end")
            self.edit_default_y_entry.insert(0, str(data.get("edit_default_y", DEFAULT_EDIT_IMPORT_Y)))

        if hasattr(self, "edit_default_w_entry"):
            self.edit_default_w_entry.delete(0, "end")
            self.edit_default_w_entry.insert(0, str(data.get("edit_default_w", DEFAULT_EDIT_IMPORT_W)))

        if hasattr(self, "edit_default_h_entry"):
            self.edit_default_h_entry.delete(0, "end")
            self.edit_default_h_entry.insert(0, str(data.get("edit_default_h", DEFAULT_EDIT_IMPORT_H)))

        if hasattr(self, "edit_default_opacity_entry"):
            self.edit_default_opacity_entry.delete(0, "end")
            self.edit_default_opacity_entry.insert(0, str(data.get("edit_default_opacity", DEFAULT_EDIT_IMPORT_OPACITY)))

        ovs = data.get("edit_overlays", [])
        if isinstance(ovs, list):
            safe = []
            for ov in ovs:
                if not isinstance(ov, dict):
                    continue
                if "image_path" not in ov:
                    continue
                safe.append(ov)
            self.edit_overlays = safe
            self.refresh_edit_overlay_table()

        self._refresh_template_menu()
        self._refresh_script_model_options()

    def save_config(self):
        def _safe_int(val, default):
            try:
                return int(val)
            except Exception:
                return default

        def _safe_float(val, default):
            try:
                return float(val)
            except Exception:
                return default

        gemini_key = self._get_gemini_api_key()
        chatgpt_key = self._get_chatgpt_api_key()
        claude_key = self._get_claude_api_key()

        data = {
            "gemini_api_key": gemini_key,
            "chatgpt_api_key": chatgpt_key,
            "claude_api_key": claude_key,
            "script_path": self.script_entry.get().strip(),
            "output_dir": self.output_entry.get().strip(),
            "image_paths": self.image_paths,
            "use_bgm": bool(self.use_bgm_var.get()),
            "bgm_path": self.bgm_entry.get().strip(),
            "bgm_gain_db": float(self.bgm_gain_slider.get()),
            "fps": _safe_int(self.fps_entry.get() or 30, 30),
            "voice_name": self.voice_entry.get().strip() or "Kore",
            "width": _safe_int(self.width_entry.get() or 1920, 1920),
            "height": _safe_int(self.height_entry.get() or 1080, 1080),
            "caption_font_size": _safe_int(self.caption_font_entry.get() or FONT_SIZE, FONT_SIZE),
            "speaker_font_size": _safe_int(self.speaker_font_entry.get() or SPEAKER_FONT_SIZE, SPEAKER_FONT_SIZE),
            "caption_max_chars": _safe_int(self.caption_width_entry.get() or CAPTION_MAX_CHARS_PER_LINE, CAPTION_MAX_CHARS_PER_LINE),
            "caption_box_enabled": bool(self.caption_box_enabled_var.get()),
            "caption_box_height": _safe_int(self.caption_box_height_entry.get() or DEFAULT_CAPTION_BOX_HEIGHT, DEFAULT_CAPTION_BOX_HEIGHT),
            "caption_box_alpha": _safe_int(self.caption_alpha_entry.get() or CAPTION_BOX_ALPHA, CAPTION_BOX_ALPHA),
            "bg_off_style": self.bg_off_style_var.get(),
            "caption_text_color": self.caption_text_color_entry.get().strip() or DEFAULT_CAPTION_TEXT_COLOR,
            "tts_engine": self.tts_engine_var.get(),
            "vv_base_url": self.vv_baseurl_entry.get().strip() or DEFAULT_VOICEVOX_URL,
            "vv_mode": self.vv_mode_var.get(),
            "vv_rotation_labels": self.vv_rotation_entry.get().strip(),
            "vv_caster_label": self.vv_caster_entry.get().strip() or DEFAULT_VV_CASTER_LABEL,
            "vv_analyst_label": self.vv_analyst_entry.get().strip() or DEFAULT_VV_ANALYST_LABEL,
            "vv_speed": float(self.vv_speed_slider.get()),
            "script_engine": self.script_engine_var.get() if hasattr(self, "script_engine_var") else "ClaudeCode",
            "script_gemini_model": self.script_gemini_model_var.get().strip()
            if hasattr(self, "script_gemini_model_var")
            else DEFAULT_SCRIPT_GEMINI_MODEL,
            "script_chatgpt_model": self.script_chatgpt_model_var.get().strip()
            if hasattr(self, "script_chatgpt_model_var")
            else DEFAULT_SCRIPT_OPENAI_MODEL,
            "claude_model": self.script_claude_model_var.get().strip() or DEFAULT_CLAUDE_MODEL,
            "claude_max_tokens": _safe_int(self.claude_max_tokens_entry.get() or DEFAULT_CLAUDE_MAX_TOKENS, DEFAULT_CLAUDE_MAX_TOKENS),
            "script_save_path": self.script_save_path_entry.get().strip(),
            "claude_prompt": self._get_textbox(self.claude_prompt_text),
            "claude_output": self._get_textbox(self.claude_output_text),
            "prompt_templates": self.prompt_templates,
            "prompt_template_selected": self.prompt_template_var.get(),
            "title_script_path": getattr(self, "title_script_entry", None).get().strip()
            if hasattr(self, "title_script_entry")
            else "",
            "title_engine": self.title_engine_var.get() if hasattr(self, "title_engine_var") else "Gemini",
            "title_gemini_model": self.title_gemini_model_var.get().strip()
            if hasattr(self, "title_gemini_model_var")
            else DEFAULT_SCRIPT_GEMINI_MODEL,
            "title_chatgpt_model": self.title_chatgpt_model_var.get().strip()
            if hasattr(self, "title_chatgpt_model_var")
            else DEFAULT_SCRIPT_OPENAI_MODEL,
            "title_claude_model": self.title_claude_model_var.get().strip()
            if hasattr(self, "title_claude_model_var")
            else DEFAULT_CLAUDE_MODEL,
            "title_count": _safe_int(
                getattr(self, "title_count_entry", None).get().strip() if hasattr(self, "title_count_entry") else "5",
                5,
            ),
            "title_extra": self._get_textbox(self.title_extra_text)
            if hasattr(self, "title_extra_text")
            else "",
            "title_output": self._get_textbox(self.title_output_text)
            if hasattr(self, "title_output_text")
            else "",
            "material_model": getattr(self, "material_model_entry", None).get().strip()
            if hasattr(self, "material_model_entry")
            else GEMINI_MATERIAL_DEFAULT_MODEL,
            "material_prompt": self._get_textbox(self.material_prompt_text)
            if hasattr(self, "material_prompt_text")
            else "",
            "material_save_path": getattr(self, "material_save_path_entry", None).get().strip()
            if hasattr(self, "material_save_path_entry")
            else "",
            # ponchi (NEW)
            "ponchi_srt_path": getattr(self, "ponchi_srt_entry", None).get().strip()
            if hasattr(self, "ponchi_srt_entry")
            else "",
            "ponchi_output_dir": getattr(self, "ponchi_output_dir_entry", None).get().strip()
            if hasattr(self, "ponchi_output_dir_entry")
            else "",
            "ponchi_engine": self.ponchi_suggestion_engine_var.get()
            if hasattr(self, "ponchi_suggestion_engine_var")
            else "Gemini",
            "ponchi_gemini_model": getattr(self, "ponchi_gemini_model_entry", None).get().strip()
            if hasattr(self, "ponchi_gemini_model_entry")
            else DEFAULT_PONCHI_GEMINI_MODEL,
            "ponchi_openai_model": getattr(self, "ponchi_openai_model_entry", None).get().strip()
            if hasattr(self, "ponchi_openai_model_entry")
            else DEFAULT_PONCHI_OPENAI_MODEL,
            # edit (NEW)
            "edit_input_mp4": getattr(self, "edit_input_entry", None).get().strip() if hasattr(self, "edit_input_entry") else "",
            "edit_output_mp4": getattr(self, "edit_output_entry", None).get().strip() if hasattr(self, "edit_output_entry") else "",
            "edit_overlays": self.edit_overlays,
            "edit_srt_path": getattr(self, "edit_srt_entry", None).get().strip() if hasattr(self, "edit_srt_entry") else "",
            "edit_image_output_dir": getattr(self, "edit_image_output_entry", None).get().strip() if hasattr(self, "edit_image_output_entry") else "",
            "edit_search_provider": self.edit_search_provider_var.get() if hasattr(self, "edit_search_provider_var") else DEFAULT_IMAGE_SEARCH_PROVIDER,
            "edit_search_api_key": getattr(self, "edit_search_api_key_entry", None).get().strip() if hasattr(self, "edit_search_api_key_entry") else "",
            "edit_default_x": _safe_int(getattr(self, "edit_default_x_entry", None).get().strip() if hasattr(self, "edit_default_x_entry") else DEFAULT_EDIT_IMPORT_X, DEFAULT_EDIT_IMPORT_X),
            "edit_default_y": _safe_int(getattr(self, "edit_default_y_entry", None).get().strip() if hasattr(self, "edit_default_y_entry") else DEFAULT_EDIT_IMPORT_Y, DEFAULT_EDIT_IMPORT_Y),
            "edit_default_w": _safe_int(getattr(self, "edit_default_w_entry", None).get().strip() if hasattr(self, "edit_default_w_entry") else DEFAULT_EDIT_IMPORT_W, DEFAULT_EDIT_IMPORT_W),
            "edit_default_h": _safe_int(getattr(self, "edit_default_h_entry", None).get().strip() if hasattr(self, "edit_default_h_entry") else DEFAULT_EDIT_IMPORT_H, DEFAULT_EDIT_IMPORT_H),
            "edit_default_opacity": _safe_float(getattr(self, "edit_default_opacity_entry", None).get().strip() if hasattr(self, "edit_default_opacity_entry") else DEFAULT_EDIT_IMPORT_OPACITY, DEFAULT_EDIT_IMPORT_OPACITY),
        }
        try:
            CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # --------------------------
    # UI helpers
    # --------------------------
    def _get_gemini_api_key(self) -> str:
        if hasattr(self, "gemini_api_key_entry"):
            return self.gemini_api_key_entry.get().strip()
        return ""

    def _get_chatgpt_api_key(self) -> str:
        if hasattr(self, "chatgpt_api_key_entry"):
            return self.chatgpt_api_key_entry.get().strip()
        return ""

    def _get_claude_api_key(self) -> str:
        if hasattr(self, "claude_api_key_entry"):
            return self.claude_api_key_entry.get().strip()
        return ""

    def _get_textbox(self, tb: ctk.CTkTextbox) -> str:
        try:
            return tb.get("1.0", "end").rstrip("\n")
        except Exception:
            return ""

    def _set_textbox(self, tb: ctk.CTkTextbox, text: str):
        try:
            tb.delete("1.0", "end")
            tb.insert("end", text or "")
        except Exception:
            pass

    def _set_material_preview(self, image_bytes: bytes):
        if not hasattr(self, "material_output_label"):
            return
        try:
            image = Image.open(BytesIO(image_bytes))
            image.thumbnail((720, 360), Image.LANCZOS)
        except Exception:
            self.material_output_label.configure(text="画像の表示に失敗しました", image=None)
            return
        self._material_preview_imgtk = ImageTk.PhotoImage(image)
        self.material_output_label.configure(image=self._material_preview_imgtk, text="")

    def _default_material_save_dir(self) -> Path:
        downloads = Path.home() / "Downloads"
        return downloads if downloads.exists() else Path.home()

    def _resolve_material_save_dir(self) -> Optional[Path]:
        raw = self.material_save_path_entry.get().strip()
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.is_dir():
                return candidate
            if candidate.suffix:
                return candidate.parent if candidate.parent.exists() else None
            return candidate if candidate.exists() else None
        return self._default_material_save_dir()

    def log(self, text: str):
        def _append():
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
        self.after(0, _append)

    def update_progress(self, value: float, eta_seconds: float | None = None):
        def _set():
            v = max(0.0, min(1.0, float(value)))
            self.progress_bar.set(v)

            percent = int(v * 100)

            if eta_seconds is None or eta_seconds < 0 or eta_seconds != eta_seconds:
                self.progress_label.configure(text=f"進捗: {percent}%")
                return

            sec = int(eta_seconds + 0.5)
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            if h > 0:
                eta_txt = f"{h:02}:{m:02}:{s:02}"
            else:
                eta_txt = f"{m:02}:{s:02}"

            self.progress_label.configure(text=f"進捗: {percent}%（残り {eta_txt}）")

        self.after(0, _set)

    # --------------------------
    # 詳細動画編集: プロジェクト管理
    # --------------------------
    def browse_detailed_project_path(self):
        path = filedialog.asksaveasfilename(
            title="プロジェクト保存先",
            defaultextension=DETAILED_PROJECT_EXT,
            filetypes=[("Movie Maker Project", f"*{DETAILED_PROJECT_EXT}"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.detailed_project_path_entry.delete(0, "end")
            self.detailed_project_path_entry.insert(0, path)
            self.detailed_project_path = Path(path)
            self._mark_detailed_dirty()
            self._schedule_detailed_autosave()

    def _project_root_dir(self) -> str:
        return self.detailed_root_entry.get().strip()

    def _to_project_relative(self, path: str, root_dir: str) -> str:
        if not path:
            return ""
        root = Path(root_dir).expanduser() if root_dir else None
        p = Path(path).expanduser()
        if root:
            try:
                return f"$ROOT/{p.relative_to(root).as_posix()}"
            except Exception:
                return str(p)
        return str(p)

    def _from_project_relative(self, path: str, root_dir: str) -> str:
        if not path:
            return ""
        if path.startswith("$ROOT/") and root_dir:
            suffix = path.replace("$ROOT/", "", 1)
            return str(Path(root_dir).expanduser() / suffix)
        return path

    def _collect_detailed_project_data(self) -> Dict[str, Any]:
        root_dir = self._project_root_dir()
        data = {
            "name": self.detailed_project_name_entry.get().strip(),
            "root_dir": root_dir,
            "input_dir": self.detailed_input_entry.get().strip(),
            "output_dir": self.detailed_output_entry.get().strip(),
            "assets": [
                {
                    **asset,
                    "path": self._to_project_relative(asset.get("path", ""), root_dir),
                }
                for asset in self.detailed_assets
            ],
            "main_video": self._to_project_relative(self.detailed_main_video_entry.get().strip(), root_dir),
            "timeline": self.detailed_timeline,
            "overlays": [
                {
                    **ov,
                    "source": self._to_project_relative(ov.get("source", ""), root_dir)
                    if ov.get("type") == "image"
                    else ov.get("source", ""),
                }
                for ov in self.detailed_overlays
            ],
            "audio": {
                "bgm_path": self._to_project_relative(self.detailed_bgm_entry.get().strip(), root_dir),
                "bgm_volume": float(self.detailed_bgm_volume_slider.get()),
                "video_audio": bool(self.detailed_video_audio_var.get()),
                "video_volume": float(self.detailed_video_volume_slider.get()),
                "fade_in": float(self._safe_float(self.detailed_fade_in_entry.get().strip(), 0.0)),
                "fade_out": float(self._safe_float(self.detailed_fade_out_entry.get().strip(), 0.0)),
            },
            "export": {
                "output_path": self._to_project_relative(self.detailed_export_path_entry.get().strip(), root_dir),
                "resolution": self.detailed_export_res_var.get(),
                "fps": int(self._safe_int(self.detailed_export_fps_entry.get().strip(), 30)),
            },
        }
        return data

    def _apply_detailed_project_data(self, data: Dict[str, Any]):
        self.detailed_project_data = data
        self.detailed_project_name_entry.delete(0, "end")
        self.detailed_project_name_entry.insert(0, data.get("name", ""))
        self.detailed_root_entry.delete(0, "end")
        self.detailed_root_entry.insert(0, data.get("root_dir", ""))
        self.detailed_input_entry.delete(0, "end")
        self.detailed_input_entry.insert(0, data.get("input_dir", ""))
        self.detailed_output_entry.delete(0, "end")
        self.detailed_output_entry.insert(0, data.get("output_dir", ""))

        root_dir = data.get("root_dir", "")
        self.detailed_assets = []
        for asset in data.get("assets", []):
            asset_path = self._from_project_relative(asset.get("path", ""), root_dir)
            self.detailed_assets.append({**asset, "path": asset_path})
        self.refresh_detailed_assets()

        main_video = self._from_project_relative(data.get("main_video", ""), root_dir)
        self.detailed_main_video_entry.delete(0, "end")
        self.detailed_main_video_entry.insert(0, main_video)
        self.detailed_timeline = data.get("timeline", [])
        self.refresh_detailed_timeline()

        self.detailed_overlays = []
        for ov in data.get("overlays", []):
            source = ov.get("source", "")
            if ov.get("type") == "image":
                source = self._from_project_relative(source, root_dir)
            self.detailed_overlays.append({**ov, "source": source})
        self.refresh_detailed_overlays()

        audio = data.get("audio", {})
        self.detailed_bgm_entry.delete(0, "end")
        self.detailed_bgm_entry.insert(0, self._from_project_relative(audio.get("bgm_path", ""), root_dir))
        self.detailed_bgm_volume_slider.set(float(audio.get("bgm_volume", 0.7)))
        self.detailed_video_audio_var.set(bool(audio.get("video_audio", True)))
        self.detailed_video_volume_slider.set(float(audio.get("video_volume", 1.0)))
        self.detailed_fade_in_entry.delete(0, "end")
        self.detailed_fade_in_entry.insert(0, str(audio.get("fade_in", 0.0)))
        self.detailed_fade_out_entry.delete(0, "end")
        self.detailed_fade_out_entry.insert(0, str(audio.get("fade_out", 0.0)))

        export = data.get("export", {})
        self.detailed_export_path_entry.delete(0, "end")
        self.detailed_export_path_entry.insert(
            0, self._from_project_relative(export.get("output_path", ""), root_dir)
        )
        self.detailed_export_res_var.set(export.get("resolution", "1920x1080"))
        self.detailed_export_fps_entry.delete(0, "end")
        self.detailed_export_fps_entry.insert(0, str(export.get("fps", 30)))

        self._mark_detailed_dirty(False)
        self._schedule_detailed_autosave()

    def new_detailed_project(self):
        self.detailed_project_path = None
        self.detailed_project_path_entry.delete(0, "end")
        self._apply_detailed_project_data(self._default_detailed_project())
        self.log("🆕 詳細動画編集プロジェクトを新規作成しました。")
        self.detailed_autosave_label.configure(text="自動保存: 未設定")

    def save_detailed_project(self):
        if not self.detailed_project_path:
            raw = self.detailed_project_path_entry.get().strip()
            if raw:
                self.detailed_project_path = Path(raw)
            else:
                self.browse_detailed_project_path()
        if not self.detailed_project_path:
            return

        data = self._collect_detailed_project_data()
        try:
            self.detailed_project_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._mark_detailed_dirty(False)
            self.detailed_autosave_label.configure(text=f"自動保存: {self.detailed_project_path.name}")
            self.log("💾 詳細動画編集プロジェクトを保存しました。")
        except Exception as exc:
            messagebox.showerror("保存エラー", f"プロジェクトの保存に失敗しました:\n{exc}")

    def open_detailed_project(self):
        path = filedialog.askopenfilename(
            title="プロジェクトを開く",
            filetypes=[("Movie Maker Project", f"*{DETAILED_PROJECT_EXT}"), ("すべてのファイル", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"プロジェクトの読み込みに失敗しました:\n{exc}")
            return

        self.detailed_project_path = Path(path)
        self.detailed_project_path_entry.delete(0, "end")
        self.detailed_project_path_entry.insert(0, path)
        self._apply_detailed_project_data(data)
        self.detailed_autosave_label.configure(text=f"自動保存: {self.detailed_project_path.name}")
        self.log("📂 詳細動画編集プロジェクトを開きました。")

    def _schedule_detailed_autosave(self):
        if self._detailed_autosave_job:
            try:
                self.after_cancel(self._detailed_autosave_job)
            except Exception:
                pass

        def _auto():
            if self.detailed_project_path and self._detailed_dirty:
                self.save_detailed_project()
            self._schedule_detailed_autosave()

        self._detailed_autosave_job = self.after(DETAILED_AUTOSAVE_INTERVAL_MS, _auto)

    def _mark_detailed_dirty(self, dirty: bool = True):
        self._detailed_dirty = dirty

    # --------------------------
    # 詳細動画編集: 素材管理
    # --------------------------
    def add_detailed_assets(self):
        paths = filedialog.askopenfilenames(
            title="素材を追加",
            filetypes=[
                ("メディアファイル", "*.mp4;*.mov;*.mkv;*.mp3;*.wav;*.m4a;*.flac;*.png;*.jpg;*.jpeg;*.webp;*.bmp"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not paths:
            return
        for path in paths:
            asset = self._build_asset_metadata(path)
            self.detailed_assets.append(asset)
        self.refresh_detailed_assets()
        self._mark_detailed_dirty()

    def _build_asset_metadata(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        mime, _ = mimetypes.guess_type(str(p))
        kind = "other"
        if mime:
            if mime.startswith("video"):
                kind = "video"
            elif mime.startswith("image"):
                kind = "image"
            elif mime.startswith("audio"):
                kind = "audio"

        duration = None
        resolution = ""
        try:
            if kind == "video":
                clip = VideoFileClip(str(p))
                duration = clip.duration
                resolution = f"{clip.w}x{clip.h}"
                clip.close()
            elif kind == "audio":
                clip = AudioFileClip(str(p))
                duration = clip.duration
                clip.close()
            elif kind == "image":
                img = Image.open(str(p))
                resolution = f"{img.width}x{img.height}"
        except Exception:
            pass

        return {
            "path": str(p),
            "type": kind,
            "duration": duration,
            "resolution": resolution,
            "missing": not p.exists(),
        }

    def refresh_detailed_assets(self):
        if not hasattr(self, "detailed_asset_tree"):
            return
        for item in self.detailed_asset_tree.get_children():
            self.detailed_asset_tree.delete(item)
        for idx, asset in enumerate(self.detailed_assets):
            path = asset.get("path", "")
            exists = Path(path).exists()
            asset["missing"] = not exists
            duration = asset.get("duration")
            duration_txt = f"{duration:.2f}s" if duration else "-"
            resolution = asset.get("resolution") or "-"
            status = "OK" if exists else "不足"
            name = Path(path).name if path else ""
            tag = "even" if idx % 2 == 0 else "odd"
            self.detailed_asset_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(name, asset.get("type", "-"), duration_txt, resolution, status),
                tags=(tag,),
            )
        self._update_asset_preview(None)

    def _update_asset_preview(self, asset: Optional[Dict[str, Any]]):
        if not hasattr(self, "detailed_asset_thumb"):
            return
        if not asset:
            self.detailed_asset_thumb.configure(text="（素材を選択）", image="")
            self.detailed_asset_path_label.configure(text="")
            self._detailed_asset_imgtk = None
            return
        path = asset.get("path", "")
        self.detailed_asset_path_label.configure(text=path)
        if not path or not Path(path).exists():
            self.detailed_asset_thumb.configure(text="（ファイルが見つかりません）", image="")
            self._detailed_asset_imgtk = None
            return
        try:
            if asset.get("type") == "image":
                img = Image.open(path)
            elif asset.get("type") == "video":
                clip = VideoFileClip(path)
                frame = clip.get_frame(0)
                img = Image.fromarray(frame)
                clip.close()
            else:
                self.detailed_asset_thumb.configure(text="（プレビューなし）", image="")
                self._detailed_preview_imgtk = None
                return
            img.thumbnail((320, 180), Image.LANCZOS)
            self._detailed_asset_imgtk = ImageTk.PhotoImage(img)
            self.detailed_asset_thumb.configure(image=self._detailed_asset_imgtk, text="")
        except Exception:
            self.detailed_asset_thumb.configure(text="（プレビュー失敗）", image="")
            self._detailed_asset_imgtk = None

    def on_detailed_asset_select(self, _event=None):
        sel = self.detailed_asset_tree.selection()
        if not sel:
            self._update_asset_preview(None)
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.detailed_assets):
            return
        self._update_asset_preview(self.detailed_assets[idx])

    def remove_detailed_asset(self):
        sel = self.detailed_asset_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.detailed_assets):
            self.detailed_assets.pop(idx)
            self.refresh_detailed_assets()
            self._mark_detailed_dirty()

    def relink_detailed_asset(self):
        sel = self.detailed_asset_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.detailed_assets):
            return
        path = filedialog.askopenfilename(title="素材を再リンク")
        if not path:
            return
        self.detailed_assets[idx] = self._build_asset_metadata(path)
        self.refresh_detailed_assets()
        self._mark_detailed_dirty()

    # --------------------------
    # 詳細動画編集: タイムライン
    # --------------------------
    def browse_detailed_main_video(self):
        path = filedialog.askopenfilename(
            title="メイン動画を選択",
            filetypes=[("動画ファイル", "*.mp4;*.mov;*.mkv"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.detailed_main_video_entry.delete(0, "end")
            self.detailed_main_video_entry.insert(0, path)
            self._load_main_video_to_timeline(path)

    def _load_main_video_to_timeline(self, path: str):
        try:
            clip = VideoFileClip(path)
            duration = clip.duration
            clip.close()
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"動画の読み込みに失敗しました:\n{exc}")
            return
        self.detailed_timeline = [
            {"label": "clip-1", "start": 0.0, "end": float(duration)}
        ]
        self.refresh_detailed_timeline()
        self._mark_detailed_dirty()
        self._sync_preview_range(duration)

    def refresh_detailed_timeline(self):
        if not hasattr(self, "detailed_timeline_tree"):
            return
        for item in self.detailed_timeline_tree.get_children():
            self.detailed_timeline_tree.delete(item)
        for idx, clip in enumerate(self.detailed_timeline):
            start = float(clip.get("start", 0.0))
            end = float(clip.get("end", 0.0))
            duration = max(0.0, end - start)
            label = clip.get("label", f"clip-{idx + 1}")
            self.detailed_timeline_tree.insert(
                "", "end", iid=str(idx),
                values=(label, f"{start:.2f}", f"{end:.2f}", f"{duration:.2f}")
            )

    def on_timeline_select(self, _event=None):
        sel = self.detailed_timeline_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.detailed_timeline):
            return
        clip = self.detailed_timeline[idx]
        self.detailed_trim_in_entry.delete(0, "end")
        self.detailed_trim_in_entry.insert(0, f"{clip.get('start', 0.0):.2f}")
        self.detailed_trim_out_entry.delete(0, "end")
        self.detailed_trim_out_entry.insert(0, f"{clip.get('end', 0.0):.2f}")

    def apply_timeline_trim(self):
        sel = self.detailed_timeline_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.detailed_timeline):
            return
        try:
            start = parse_timecode_to_seconds(self.detailed_trim_in_entry.get().strip())
            end = parse_timecode_to_seconds(self.detailed_trim_out_entry.get().strip())
        except Exception as exc:
            messagebox.showerror("入力エラー", f"時間の形式が不正です:\n{exc}")
            return
        if end <= start:
            messagebox.showerror("入力エラー", "Out は In より大きくしてください。")
            return
        self.detailed_timeline[idx]["start"] = start
        self.detailed_timeline[idx]["end"] = end
        self.refresh_detailed_timeline()
        self._mark_detailed_dirty()

    def split_timeline_clip(self):
        sel = self.detailed_timeline_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.detailed_timeline):
            return
        clip = self.detailed_timeline[idx]
        try:
            split_time = parse_timecode_to_seconds(self.detailed_split_entry.get().strip())
        except Exception as exc:
            messagebox.showerror("入力エラー", f"Split 時間の形式が不正です:\n{exc}")
            return
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", 0.0))
        if split_time <= start or split_time >= end:
            messagebox.showerror("入力エラー", "Split 時間は In/Out の間に設定してください。")
            return
        first = {"label": clip.get("label", f"clip-{idx + 1}") + "a", "start": start, "end": split_time}
        second = {"label": clip.get("label", f"clip-{idx + 1}") + "b", "start": split_time, "end": end}
        self.detailed_timeline[idx:idx + 1] = [first, second]
        self.refresh_detailed_timeline()
        self._mark_detailed_dirty()

    def delete_timeline_clip(self):
        sel = self.detailed_timeline_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.detailed_timeline):
            self.detailed_timeline.pop(idx)
            self.refresh_detailed_timeline()
            self._mark_detailed_dirty()

    def move_timeline_clip(self, delta: int):
        sel = self.detailed_timeline_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(self.detailed_timeline):
            return
        self.detailed_timeline.insert(new_idx, self.detailed_timeline.pop(idx))
        self.refresh_detailed_timeline()
        self.detailed_timeline_tree.selection_set(str(new_idx))
        self._mark_detailed_dirty()

    def on_timeline_drag_start(self, event):
        row = self.detailed_timeline_tree.identify_row(event.y)
        if row:
            self._detailed_drag_timeline_iid = row

    def on_timeline_drag_motion(self, event):
        if not self._detailed_drag_timeline_iid:
            return
        row = self.detailed_timeline_tree.identify_row(event.y)
        if row and row != self._detailed_drag_timeline_iid:
            old_idx = int(self._detailed_drag_timeline_iid)
            new_idx = int(row)
            if 0 <= old_idx < len(self.detailed_timeline) and 0 <= new_idx < len(self.detailed_timeline):
                self.detailed_timeline.insert(new_idx, self.detailed_timeline.pop(old_idx))
                self.refresh_detailed_timeline()
                self.detailed_timeline_tree.selection_set(str(new_idx))
                self._detailed_drag_timeline_iid = str(new_idx)
                self._mark_detailed_dirty()

    # --------------------------
    # 詳細動画編集: オーバーレイ
    # --------------------------
    def _collect_overlay_from_form(self) -> Dict[str, Any]:
        return {
            "type": self.detailed_overlay_type_var.get(),
            "start": self._safe_float(self.detailed_overlay_start_entry.get().strip(), 0.0),
            "end": self._safe_float(self.detailed_overlay_end_entry.get().strip(), 0.0),
            "x": self._safe_int(self.detailed_overlay_x_entry.get().strip(), 0),
            "y": self._safe_int(self.detailed_overlay_y_entry.get().strip(), 0),
            "w": self._safe_int(self.detailed_overlay_w_entry.get().strip(), 0),
            "h": self._safe_int(self.detailed_overlay_h_entry.get().strip(), 0),
            "opacity": self._safe_float(self.detailed_overlay_opacity_entry.get().strip(), 1.0),
            "source": self.detailed_overlay_source_entry.get().strip(),
            "font_size": self._safe_int(self.detailed_overlay_font_entry.get().strip(), 32),
            "color": self.detailed_overlay_color_entry.get().strip() or "#ffffff",
            "outline": self._safe_int(self.detailed_overlay_outline_entry.get().strip(), 2),
        }

    def add_detailed_overlay(self):
        ov = self._collect_overlay_from_form()
        if ov["end"] <= ov["start"]:
            messagebox.showerror("入力エラー", "終了時間は開始時間より大きくしてください。")
            return
        self.detailed_overlays.append(ov)
        self.refresh_detailed_overlays()
        self._mark_detailed_dirty()

    def duplicate_detailed_overlay(self):
        sel = self.detailed_overlay_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.detailed_overlays):
            self.detailed_overlays.append(dict(self.detailed_overlays[idx]))
            self.refresh_detailed_overlays()
            self._mark_detailed_dirty()

    def delete_detailed_overlay(self):
        sel = self.detailed_overlay_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.detailed_overlays):
            self.detailed_overlays.pop(idx)
            self.refresh_detailed_overlays()
            self._mark_detailed_dirty()

    def on_detailed_overlay_select(self, _event=None):
        sel = self.detailed_overlay_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.detailed_overlays):
            return
        ov = self.detailed_overlays[idx]
        self.detailed_overlay_type_var.set(ov.get("type", "image"))
        self.detailed_overlay_start_entry.delete(0, "end")
        self.detailed_overlay_start_entry.insert(0, str(ov.get("start", 0.0)))
        self.detailed_overlay_end_entry.delete(0, "end")
        self.detailed_overlay_end_entry.insert(0, str(ov.get("end", 0.0)))
        self.detailed_overlay_x_entry.delete(0, "end")
        self.detailed_overlay_x_entry.insert(0, str(ov.get("x", 0)))
        self.detailed_overlay_y_entry.delete(0, "end")
        self.detailed_overlay_y_entry.insert(0, str(ov.get("y", 0)))
        self.detailed_overlay_w_entry.delete(0, "end")
        self.detailed_overlay_w_entry.insert(0, str(ov.get("w", 0)))
        self.detailed_overlay_h_entry.delete(0, "end")
        self.detailed_overlay_h_entry.insert(0, str(ov.get("h", 0)))
        self.detailed_overlay_opacity_entry.delete(0, "end")
        self.detailed_overlay_opacity_entry.insert(0, str(ov.get("opacity", 1.0)))
        self.detailed_overlay_source_entry.delete(0, "end")
        self.detailed_overlay_source_entry.insert(0, ov.get("source", ""))
        self.detailed_overlay_font_entry.delete(0, "end")
        self.detailed_overlay_font_entry.insert(0, str(ov.get("font_size", 32)))
        self.detailed_overlay_color_entry.delete(0, "end")
        self.detailed_overlay_color_entry.insert(0, str(ov.get("color", "#ffffff")))
        self.detailed_overlay_outline_entry.delete(0, "end")
        self.detailed_overlay_outline_entry.insert(0, str(ov.get("outline", 2)))

    def refresh_detailed_overlays(self):
        if not hasattr(self, "detailed_overlay_tree"):
            return
        for item in self.detailed_overlay_tree.get_children():
            self.detailed_overlay_tree.delete(item)
        for idx, ov in enumerate(self.detailed_overlays):
            self.detailed_overlay_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    ov.get("type", ""),
                    f"{ov.get('start', 0.0):.2f}",
                    f"{ov.get('end', 0.0):.2f}",
                    ov.get("x", 0),
                    ov.get("y", 0),
                    ov.get("w", 0),
                    ov.get("h", 0),
                    f"{ov.get('opacity', 1.0):.2f}",
                    ov.get("source", "")[:40],
                ),
            )

    # --------------------------
    # 詳細動画編集: プレビュー
    # --------------------------
    def _get_preview_clip(self) -> Optional[VideoFileClip]:
        path = self.detailed_main_video_entry.get().strip()
        if not path:
            return None
        if self._detailed_preview_clip and getattr(self._detailed_preview_clip, "filename", "") == path:
            return self._detailed_preview_clip
        if self._detailed_preview_clip:
            try:
                self._detailed_preview_clip.close()
            except Exception:
                pass
        try:
            self._detailed_preview_clip = VideoFileClip(path)
        except Exception:
            self._detailed_preview_clip = None
        return self._detailed_preview_clip

    def _sync_preview_range(self, duration: float):
        self.detailed_seek_slider.configure(from_=0, to=max(0.1, duration))
        self.detailed_seek_slider.set(0)
        self._detailed_preview_time = 0.0
        self.render_preview_frame(0.0)

    def render_preview_frame(self, t: float):
        clip = self._get_preview_clip()
        if not clip:
            return
        t = max(0.0, min(float(t), clip.duration))
        try:
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            img.thumbnail((640, 360), Image.LANCZOS)
            self._detailed_preview_imgtk = ImageTk.PhotoImage(img)
            self.detailed_preview_label.configure(image=self._detailed_preview_imgtk, text="")
        except Exception:
            self.detailed_preview_label.configure(text="プレビュー生成に失敗しました", image="")

    def on_preview_seek(self, value):
        try:
            t = float(value)
        except Exception:
            t = 0.0
        self._detailed_preview_time = t
        self.render_preview_frame(t)

    def start_detailed_preview(self):
        if self._detailed_preview_playing:
            return
        clip = self._get_preview_clip()
        if not clip:
            return
        self._detailed_preview_playing = True

        def _tick():
            if not self._detailed_preview_playing:
                return
            fps = clip.fps or 30
            self._detailed_preview_time += 1 / fps
            if self._detailed_preview_time >= clip.duration:
                self._detailed_preview_time = clip.duration
                self.stop_detailed_preview()
                return
            self.detailed_seek_slider.set(self._detailed_preview_time)
            self.render_preview_frame(self._detailed_preview_time)
            self._detailed_preview_job = self.after(int(1000 / fps), _tick)

        _tick()

    def stop_detailed_preview(self):
        self._detailed_preview_playing = False
        if self._detailed_preview_job:
            try:
                self.after_cancel(self._detailed_preview_job)
            except Exception:
                pass
            self._detailed_preview_job = None

    def step_preview(self, direction: int):
        clip = self._get_preview_clip()
        if not clip:
            return
        fps = clip.fps or 30
        self._detailed_preview_time = max(0.0, min(clip.duration, self._detailed_preview_time + direction / fps))
        self.detailed_seek_slider.set(self._detailed_preview_time)
        self.render_preview_frame(self._detailed_preview_time)

    # --------------------------
    # 詳細動画編集: 書き出し
    # --------------------------
    def browse_detailed_export_path(self):
        path = filedialog.asksaveasfilename(
            title="書き出し先",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.detailed_export_path_entry.delete(0, "end")
            self.detailed_export_path_entry.insert(0, path)

    def browse_detailed_bgm(self):
        path = filedialog.askopenfilename(
            title="BGMを選択",
            filetypes=[("音声ファイル", "*.mp3;*.wav;*.m4a;*.flac"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.detailed_bgm_entry.delete(0, "end")
            self.detailed_bgm_entry.insert(0, path)
            self._mark_detailed_dirty()

    def export_detailed_video(self):
        if not self.detailed_timeline:
            messagebox.showerror("書き出しエラー", "タイムラインが空です。")
            return
        output_path = self.detailed_export_path_entry.get().strip()
        if not output_path:
            messagebox.showerror("書き出しエラー", "出力先を指定してください。")
            return

        def _worker():
            try:
                self._export_detailed_video_worker(output_path)
                self.log("✅ 詳細動画編集の書き出しが完了しました。")
                messagebox.showinfo("完了", "書き出しが完了しました。")
            except Exception:
                tb = traceback.format_exc()
                self.log("❌ 詳細動画編集でエラー:\n" + tb)
                messagebox.showerror("エラー", f"書き出しに失敗しました:\n{tb}")

        threading.Thread(target=_worker, daemon=True).start()

    def _export_detailed_video_worker(self, output_path: str):
        base_path = self.detailed_main_video_entry.get().strip()
        if not base_path:
            raise RuntimeError("メイン動画が指定されていません。")
        base_clip = VideoFileClip(base_path)
        try:
            clips = []
            for clip in self.detailed_timeline:
                start = float(clip.get("start", 0.0))
                end = float(clip.get("end", 0.0))
                if end <= start:
                    continue
                clips.append(base_clip.subclip(start, end))
            if not clips:
                raise RuntimeError("有効なクリップがありません。")
            merged = concatenate_videoclips(clips, method="compose")

            overlay_clips = []
            for ov in self.detailed_overlays:
                start = float(ov.get("start", 0.0))
                end = float(ov.get("end", 0.0))
                if end <= start:
                    continue
                if ov.get("type") == "image":
                    src = ov.get("source", "")
                    if not src:
                        continue
                    img_clip = ImageClip(src)
                    w = int(ov.get("w", 0) or 0)
                    h = int(ov.get("h", 0) or 0)
                    if w > 0 or h > 0:
                        target_w = w if w > 0 else None
                        target_h = h if h > 0 else None
                        img_clip = img_clip.resize(newsize=(target_w, target_h))
                    img_clip = img_clip.with_start(start).with_end(end)
                    img_clip = img_clip.with_position((int(ov.get("x", 0)), int(ov.get("y", 0))))
                    img_clip = img_clip.with_opacity(float(ov.get("opacity", 1.0)))
                    overlay_clips.append(img_clip)
                else:
                    text_img = self._render_text_overlay(
                        ov.get("source", ""),
                        int(ov.get("font_size", 32)),
                        ov.get("color", "#ffffff"),
                        int(ov.get("outline", 2)),
                    )
                    img_clip = ImageClip(np.array(text_img))
                    img_clip = img_clip.with_start(start).with_end(end)
                    img_clip = img_clip.with_position((int(ov.get("x", 0)), int(ov.get("y", 0))))
                    img_clip = img_clip.with_opacity(float(ov.get("opacity", 1.0)))
                    overlay_clips.append(img_clip)

            final = CompositeVideoClip([merged] + overlay_clips, size=merged.size)

            # Audio
            audio_tracks = []
            if self.detailed_video_audio_var.get() and merged.audio:
                audio_tracks.append(merged.audio.volumex(float(self.detailed_video_volume_slider.get())))
            bgm_path = self.detailed_bgm_entry.get().strip()
            if bgm_path:
                bgm = AudioFileClip(bgm_path).volumex(float(self.detailed_bgm_volume_slider.get()))
                if bgm.duration < final.duration:
                    bgm = audio_loop(bgm, duration=final.duration)
                audio_tracks.append(bgm)
            if audio_tracks:
                mixed = CompositeAudioClip(audio_tracks)
                fade_in = self._safe_float(self.detailed_fade_in_entry.get().strip(), 0.0)
                fade_out = self._safe_float(self.detailed_fade_out_entry.get().strip(), 0.0)
                if fade_in > 0:
                    mixed = audio_fadein(mixed, fade_in)
                if fade_out > 0:
                    mixed = audio_fadeout(mixed, fade_out)
                final = final.with_audio(mixed)
            else:
                final = final.with_audio(None)

            # Export settings
            res = self.detailed_export_res_var.get()
            if res != "元サイズ" and "x" in res:
                w, h = res.split("x", 1)
                final = final.resize(newsize=(int(w), int(h)))
            fps = int(self._safe_int(self.detailed_export_fps_entry.get().strip(), 30))

            def _progress(value, _eta):
                self._update_detailed_export_progress(value)

            logger = TkMoviePyLogger(progress_fn=_progress, base=0, span=1.0)
            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                logger=logger,
            )
        finally:
            base_clip.close()

    def _update_detailed_export_progress(self, value: float):
        def _apply():
            v = max(0.0, min(1.0, float(value)))
            self.detailed_export_progress.set(v)
            self.detailed_export_label.configure(text=f"進捗: {int(v * 100)}%")

        self.after(0, _apply)

    def _render_text_overlay(self, text: str, font_size: int, color: str, outline: int) -> Image.Image:
        if not text:
            text = " "
        font = None
        for path in FONT_PATHS:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except Exception:
                    pass
        if not font:
            font = ImageFont.load_default()
        dummy = Image.new("RGBA", (10, 10))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=outline)
        width = bbox[2] - bbox[0] + 20
        height = bbox[3] - bbox[1] + 20
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        rgba = parse_hex_color(color, default=(255, 255, 255, 255))
        draw.text((10, 10), text, font=font, fill=rgba, stroke_width=outline, stroke_fill=(0, 0, 0, 255))
        return img

    def _safe_int(self, value: str, default: int) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    def _safe_float(self, value: str, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    # --------------------------
    # File dialogs
    # --------------------------
    def browse_script(self):
        path = filedialog.askopenfilename(
            title="原稿ファイルを選択",
            filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.script_entry.delete(0, "end")
            self.script_entry.insert(0, path)
            self.save_config()

    def browse_title_script(self):
        path = filedialog.askopenfilename(
            title="台本ファイルを選択",
            filetypes=[("SRT", "*.srt"), ("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")],
        )
        if path and hasattr(self, "title_script_entry"):
            self.title_script_entry.delete(0, "end")
            self.title_script_entry.insert(0, path)
            self.save_config()

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="画像ファイルを追加",
            filetypes=[("画像ファイル", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("すべてのファイル", "*.*")],
        )
        if paths:
            self.image_paths.extend(paths)
            self.image_paths = list(dict.fromkeys(self.image_paths))
            self.refresh_image_listbox()
            self.save_config()

    def clear_images(self):
        self.image_paths = []
        self.refresh_image_listbox()
        self.save_config()

    def refresh_image_listbox(self):
        self.img_listbox.delete("1.0", "end")
        for p in self.image_paths:
            self.img_listbox.insert("end", p + "\n")

    def browse_bgm(self):
        path = filedialog.askopenfilename(
            title="BGMファイルを選択",
            filetypes=[("音声ファイル", "*.mp3;*.wav;*.m4a;*.flac"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.bgm_entry.delete(0, "end")
            self.bgm_entry.insert(0, path)
            self.save_config()

    def browse_output(self):
        path = filedialog.askdirectory(title="出力フォルダを選択")
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)
            self.save_config()

    def browse_script_save_path(self):
        path = filedialog.asksaveasfilename(
            title="台本の保存先を選択",
            defaultextension=".txt",
            filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.script_save_path_entry.delete(0, "end")
            self.script_save_path_entry.insert(0, path)
            self.save_config()

    def browse_material_save_path(self):
        path = filedialog.askdirectory(title="画像の保存先フォルダを選択")
        if path:
            self.material_save_path_entry.delete(0, "end")
            self.material_save_path_entry.insert(0, path)
            self.save_config()

    # --------------------------
    # Script generation actions
    # --------------------------
    def insert_prompt_template(self):
        tpl = (
            "あなたはベテランのニュース番組の台本作家です。\n"
            "以下の条件で「キャスター」と「アナリスト」による対談形式の台本を日本語で作成してください。\n\n"
            "[話者と形式]\n"
            "- 行頭に必ず「キャスター：」「アナリスト：」のどちらかを付ける（全角コロン）\n"
            "- 顔文字・絵文字・記号多用は不可\n"
            "- 具体的な数値や根拠を交えて\n\n"
            "[トピック]\n"
            "（ここにテーマを記入）\n\n"
            "[追加要望]\n"
            "（必要なら）\n"
        )
        self._set_textbox(self.claude_prompt_text, tpl)

    def insert_material_prompt_template(self):
        tpl = (
            "主題/被写体:\n"
            "背景/場所:\n"
            "色味/雰囲気:\n"
            "構図/カメラアングル:\n"
            "必ず含めたい要素:\n"
            "避けたい要素:\n"
        )
        self._set_textbox(self.material_prompt_text, tpl)

    def _build_title_desc_prompt(self, script_text: str, count: int, extra: str) -> str:
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

    def copy_generated_material(self):
        path = self.material_save_path_entry.get().strip()
        if not path:
            messagebox.showinfo("コピー", "保存先パスが空です。")
            return
        self.clipboard_clear()
        self.clipboard_append(path)
        self.log("✅ 保存先パスをクリップボードにコピーしました。")

    def save_generated_material(self):
        if not getattr(self, "material_generated_image_bytes", None):
            messagebox.showerror("エラー", "生成画像がありません。")
            return
        save_dir = self._resolve_material_save_dir()
        if not save_dir:
            messagebox.showerror("エラー", "保存先フォルダが見つかりません。")
            return
        filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
        path = save_dir / filename
        try:
            image = Image.open(BytesIO(self.material_generated_image_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(path, format="JPEG", quality=95)
            self.material_save_path_entry.delete(0, "end")
            self.material_save_path_entry.insert(0, str(path))
            self.log(f"✅ 画像を保存しました: {path}")
            messagebox.showinfo("保存完了", f"保存しました:\n{path}")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")

    def copy_generated_script(self):
        txt = self._get_textbox(self.claude_output_text)
        if not txt.strip():
            messagebox.showinfo("コピー", "生成結果が空です。")
            return
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.log("✅ 台本をクリップボードにコピーしました。")

    def copy_generated_title_desc(self):
        txt = self._get_textbox(self.title_output_text)
        if not txt.strip():
            messagebox.showinfo("コピー", "生成結果が空です。")
            return
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.log("✅ タイトル・説明文をクリップボードにコピーしました。")

    def save_generated_script(self):
        txt = self._get_textbox(self.claude_output_text)
        if not txt.strip():
            messagebox.showerror("エラー", "生成結果が空です。")
            return
        path = self.script_save_path_entry.get().strip()
        if not path:
            messagebox.showerror("エラー", "保存先パスが空です。")
            return
        try:
            Path(path).write_text(txt, encoding="utf-8")
            self.log(f"✅ 台本を保存しました: {path}")
            messagebox.showinfo("保存完了", f"保存しました:\n{path}")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")

    def on_generate_script_clicked(self):
        engine = (self.script_engine_var.get() or "ClaudeCode").strip()
        prompt = self._get_textbox(self.claude_prompt_text)
        if not prompt.strip():
            messagebox.showerror("エラー", "プロンプトが空です。")
            return

        max_tokens = DEFAULT_CLAUDE_MAX_TOKENS
        if engine == "ClaudeCode":
            try:
                max_tokens = int(self.claude_max_tokens_entry.get().strip() or DEFAULT_CLAUDE_MAX_TOKENS)
            except ValueError:
                messagebox.showerror("エラー", "max_tokens は数値で入力してください。")
                return

        if engine == "Gemini" and not self._get_gemini_api_key():
            messagebox.showerror("エラー", "設定タブでGemini APIキーを入力してください。")
            return
        if engine == "ChatGPT" and not self._get_chatgpt_api_key():
            messagebox.showerror("エラー", "設定タブでChatGPT APIキーを入力してください。")
            return
        if engine == "ClaudeCode" and not self._get_claude_api_key():
            messagebox.showerror("エラー", "設定タブでClaudeCode APIキーを入力してください。")
            return

        self.save_config()
        self.btn_generate_script.configure(state="disabled", text="生成中...")
        self.set_status("Working", ok=True)
        self.log(f"=== {engine} 台本生成 開始 ===")
        self.update_progress(0.02)

        def worker():
            try:
                self.update_progress(0.08)
                if engine == "Gemini":
                    model = self.script_gemini_model_var.get().strip() or DEFAULT_SCRIPT_GEMINI_MODEL
                    out = generate_script_with_gemini(
                        api_key=self._get_gemini_api_key(),
                        prompt=prompt,
                        model=model,
                    )
                elif engine == "ChatGPT":
                    model = self.script_chatgpt_model_var.get().strip() or DEFAULT_SCRIPT_OPENAI_MODEL
                    out = generate_script_with_openai(
                        api_key=self._get_chatgpt_api_key(),
                        prompt=prompt,
                        model=model,
                    )
                else:
                    model = self.script_claude_model_var.get().strip() or DEFAULT_CLAUDE_MODEL
                    out = generate_script_with_claude(
                        api_key=self._get_claude_api_key(),
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                    )
                self.after(0, lambda: self._set_textbox(self.claude_output_text, out))
                self.log(f"✅ {engine} 台本生成 完了")
                self.update_progress(1.0)
                self.set_status("Ready", ok=True)
            except Exception as e:
                tb = traceback.format_exc()
                self.log(f"❌ {engine} 台本生成でエラー:\n" + tb)
                self.set_status("Error", ok=False)
                self.update_progress(0.0)
                self.after(0, lambda: messagebox.showerror("エラー", f"台本生成に失敗しました:\n{e}"))
            finally:
                self.after(
                    0,
                    lambda: self.btn_generate_script.configure(state="normal", text=f"▶ {engine}で台本生成"),
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_generate_title_desc_clicked(self):
        engine = (self.title_engine_var.get() or "Gemini").strip()
        script_path = self.title_script_entry.get().strip()
        if not script_path:
            messagebox.showerror("エラー", "台本ファイルを選択してください。")
            return

        try:
            count = int(self.title_count_entry.get().strip() or "5")
            if count <= 0 or count > 20:
                raise ValueError
        except ValueError:
            messagebox.showerror("エラー", "タイトル案の数は 1〜20 の整数で入力してください。")
            return

        if engine == "Gemini" and not self._get_gemini_api_key():
            messagebox.showerror("エラー", "設定タブでGemini APIキーを入力してください。")
            return
        if engine == "ChatGPT" and not self._get_chatgpt_api_key():
            messagebox.showerror("エラー", "設定タブでChatGPT APIキーを入力してください。")
            return
        if engine == "ClaudeCode" and not self._get_claude_api_key():
            messagebox.showerror("エラー", "設定タブでClaudeCode APIキーを入力してください。")
            return

        try:
            script_text = extract_script_text(script_path)
        except Exception as exc:
            messagebox.showerror("エラー", f"台本の読み込みに失敗しました:\n{exc}")
            return

        extra = self._get_textbox(self.title_extra_text)
        prompt = self._build_title_desc_prompt(script_text, count, extra)

        self.save_config()
        self.btn_generate_title_desc.configure(state="disabled", text="生成中...")
        self.set_status("Working", ok=True)
        self.log(f"=== {engine} タイトル・説明生成 開始 ===")
        self.update_progress(0.02)

        def worker():
            try:
                self.update_progress(0.08)
                if engine == "Gemini":
                    model = self.title_gemini_model_var.get().strip() or DEFAULT_SCRIPT_GEMINI_MODEL
                    out = generate_script_with_gemini(
                        api_key=self._get_gemini_api_key(),
                        prompt=prompt,
                        model=model,
                    )
                elif engine == "ChatGPT":
                    model = self.title_chatgpt_model_var.get().strip() or DEFAULT_SCRIPT_OPENAI_MODEL
                    out = generate_script_with_openai(
                        api_key=self._get_chatgpt_api_key(),
                        prompt=prompt,
                        model=model,
                        max_tokens=DEFAULT_TITLE_MAX_TOKENS,
                    )
                else:
                    model = self.title_claude_model_var.get().strip() or DEFAULT_CLAUDE_MODEL
                    out = generate_script_with_claude(
                        api_key=self._get_claude_api_key(),
                        prompt=prompt,
                        model=model,
                        max_tokens=DEFAULT_TITLE_MAX_TOKENS,
                    )
                self.after(0, lambda: self._set_textbox(self.title_output_text, out))
                self.log(f"✅ {engine} タイトル・説明生成 完了")
                self.update_progress(1.0)
                self.set_status("Ready", ok=True)
            except Exception as e:
                tb = traceback.format_exc()
                self.log(f"❌ {engine} タイトル・説明生成でエラー:\n" + tb)
                self.set_status("Error", ok=False)
                self.update_progress(0.0)
                self.after(0, lambda: messagebox.showerror("エラー", f"生成に失敗しました:\n{e}"))
            finally:
                self.after(
                    0,
                    lambda: self.btn_generate_title_desc.configure(state="normal", text=f"▶ {engine}で生成"),
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_generate_material_clicked(self):
        api_key = self._get_gemini_api_key()

        model = GEMINI_MATERIAL_DEFAULT_MODEL
        if hasattr(self, "material_model_entry"):
            model = self.material_model_entry.get().strip() or GEMINI_MATERIAL_DEFAULT_MODEL
        resolved_model, model_note = resolve_gemini_material_model(model)

        user_prompt = self._get_textbox(self.material_prompt_text)
        if not api_key:
            messagebox.showerror("エラー", "設定タブでGemini APIキーを入力してください。")
            return
        if not user_prompt.strip():
            messagebox.showerror("エラー", "プロンプトが空です。")
            return

        prompt = user_prompt.strip()

        self.save_config()
        self.btn_generate_material.configure(state="disabled", text="生成中...")
        self.set_status("Working", ok=True)
        self.log("=== Gemini 画像生成 開始 ===")
        if model_note:
            self.log(model_note)
        self.update_progress(0.02)
        self.material_generated_image_bytes = None
        self.material_generated_image_mime = None
        if hasattr(self, "material_output_label"):
            self.material_output_label.configure(text="生成中...", image=None)

        def worker():
            try:
                self.update_progress(0.08)
                image_bytes, mime_type = generate_materials_with_gemini(
                    api_key=api_key,
                    prompt=prompt,
                    model=resolved_model,
                )
                self.material_generated_image_bytes = image_bytes
                self.material_generated_image_mime = mime_type
                self.after(0, lambda: self._set_material_preview(image_bytes))
                self.log("✅ Gemini 画像生成 完了")
                self.update_progress(1.0)
                self.set_status("Ready", ok=True)
            except Exception as e:
                tb = traceback.format_exc()
                self.log("❌ Gemini 画像生成でエラー:\n" + tb)
                self.set_status("Error", ok=False)
                self.update_progress(0.0)
                self.after(0, lambda: messagebox.showerror("エラー", f"画像生成に失敗しました:\n{e}"))
            finally:
                self.after(0, lambda: self.btn_generate_material.configure(state="normal", text="▶ Geminiで画像生成"))

        threading.Thread(target=worker, daemon=True).start()

    def _resolve_ponchi_gemini_key(self) -> str:
        return self._get_gemini_api_key()

    def _ponchi_ideas_path(self, output_dir: str, srt_path: str) -> Optional[Path]:
        if not output_dir:
            return None
        srt_stem = Path(srt_path).stem
        return Path(output_dir) / srt_stem / f"{srt_stem}_ponchi_ideas.json"

    def _normalize_ponchi_suggestions(
        self,
        suggestions: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized = []
        for idx, item in enumerate(items, 1):
            suggestion = suggestions[idx - 1] if idx - 1 < len(suggestions) else {}
            start = suggestion.get("start") or format_seconds_to_timecode(item.get("start", 0))
            end = suggestion.get("end") or format_seconds_to_timecode(item.get("end", 0))
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

    def on_generate_ponchi_ideas_clicked(self):
        srt_path = self.ponchi_srt_entry.get().strip()
        engine = self.ponchi_suggestion_engine_var.get()

        openai_key = self._get_chatgpt_api_key()
        gemini_model = self.ponchi_gemini_model_entry.get().strip() or DEFAULT_PONCHI_GEMINI_MODEL
        openai_model = self.ponchi_openai_model_entry.get().strip() or DEFAULT_PONCHI_OPENAI_MODEL

        if not srt_path or not Path(srt_path).exists():
            messagebox.showerror("エラー", "有効なSRTファイルを選択してください。")
            return
        if engine == "Gemini" and not self._resolve_ponchi_gemini_key():
            messagebox.showerror("エラー", "設定タブでGemini APIキーを入力してください。")
            return
        if engine == "ChatGPT" and not openai_key:
            messagebox.showerror("エラー", "設定タブでChatGPT APIキーを入力してください。")
            return

        self.save_config()
        self.btn_generate_ponchi_ideas.configure(state="disabled", text="生成中...")
        self.btn_generate_ponchi_images.configure(state="disabled")
        self.set_status("Working", ok=True)
        self.log("=== ポンチ絵 案出し 開始 ===")
        self.update_progress(0.02)
        self._set_textbox(self.ponchi_output_text, "")

        def worker():
            try:
                items = parse_srt_file(srt_path)
                if not items:
                    raise RuntimeError("SRTから字幕が見つかりませんでした。")

                if engine == "Gemini":
                    suggestions = generate_ponchi_suggestions_with_gemini(
                        api_key=self._resolve_ponchi_gemini_key(),
                        items=items,
                        model=gemini_model,
                    )
                else:
                    suggestions = generate_ponchi_suggestions_with_openai(
                        api_key=openai_key,
                        items=items,
                        model=openai_model,
                    )

                if not isinstance(suggestions, list) or not suggestions:
                    raise RuntimeError("提案が空でした。")

                results = self._normalize_ponchi_suggestions(suggestions, items)
                self.ponchi_suggestions = results

                output_dir = self.ponchi_output_dir_entry.get().strip()
                ideas_path = self._ponchi_ideas_path(output_dir, srt_path)
                if ideas_path:
                    ideas_path.parent.mkdir(parents=True, exist_ok=True)
                    ideas_path.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                summary_lines = [
                    f"✅ {len(results)} 件のポンチ絵案を生成しました。",
                    f"JSON: {ideas_path}" if ideas_path else "JSON: (未保存)",
                    "",
                ]
                for item in results:
                    summary_lines.append(
                        f"{item['start']}〜{item['end']} | {item['visual_suggestion']} | {item['image_prompt']}"
                    )

                self.after(0, lambda: self._set_textbox(self.ponchi_output_text, "\n".join(summary_lines)))
                self.log("✅ ポンチ絵 案出し 完了")
                self.update_progress(1.0)
                self.set_status("Ready", ok=True)
                self.after(0, lambda: messagebox.showinfo("完了", "ポンチ絵の案出しが完了しました。"))
            except Exception as e:
                tb = traceback.format_exc()
                self.log("❌ ポンチ絵 案出しでエラー:\n" + tb)
                self.set_status("Error", ok=False)
                self.update_progress(0.0)
                self.after(0, lambda: messagebox.showerror("エラー", f"ポンチ絵の案出しに失敗しました:\n{e}"))
            finally:
                def _reset_buttons():
                    self.btn_generate_ponchi_ideas.configure(state="normal", text="▶ 案出し")
                    self.btn_generate_ponchi_images.configure(state="normal", text="▶ ポンチ絵作成")

                self.after(0, _reset_buttons)

        threading.Thread(target=worker, daemon=True).start()

    def on_generate_ponchi_images_clicked(self):
        srt_path = self.ponchi_srt_entry.get().strip()
        output_dir = self.ponchi_output_dir_entry.get().strip()

        gemini_key = self._resolve_ponchi_gemini_key()
        if not srt_path or not Path(srt_path).exists():
            messagebox.showerror("エラー", "有効なSRTファイルを選択してください。")
            return
        if not output_dir:
            messagebox.showerror("エラー", "出力フォルダを指定してください。")
            return
        if not gemini_key:
            messagebox.showerror("エラー", "設定タブでGemini APIキーを入力してください。")
            return

        self.save_config()
        self.btn_generate_ponchi_images.configure(state="disabled", text="生成中...")
        self.btn_generate_ponchi_ideas.configure(state="disabled")
        self.set_status("Working", ok=True)
        self.log("=== ポンチ絵作成 開始 ===")
        self.update_progress(0.02)

        def worker():
            try:
                items = parse_srt_file(srt_path)
                if not items:
                    raise RuntimeError("SRTから字幕が見つかりませんでした。")

                suggestions = self.ponchi_suggestions
                if not suggestions:
                    ideas_path = self._ponchi_ideas_path(output_dir, srt_path)
                    if ideas_path and ideas_path.exists():
                        suggestions = json.loads(ideas_path.read_text(encoding="utf-8"))
                if not isinstance(suggestions, list) or not suggestions:
                    raise RuntimeError("案が見つかりません。先に「案出し」を実行してください。")

                results = self._normalize_ponchi_suggestions(suggestions, items)
                srt_stem = Path(srt_path).stem
                output_dir_path = Path(output_dir) / srt_stem
                output_dir_path.mkdir(parents=True, exist_ok=True)
                total = len(results)
                images = []

                for idx, item in enumerate(results, 1):
                    prompt = item.get("image_prompt") or item.get("visual_suggestion") or item.get("text")
                    if not prompt:
                        prompt = "シンプルで分かりやすい図解のイラスト"

                    self.log(f"🎨 {idx}/{total}: {item['start']}〜{item['end']} の資料を生成中")
                    self.update_progress((idx - 1) / max(total, 1))
                    image_bytes, mime_type = generate_materials_with_gemini(
                        api_key=gemini_key,
                        prompt=prompt,
                        model=GEMINI_MATERIAL_DEFAULT_MODEL,
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
                        }
                    )
                    self.update_progress(idx / max(total, 1))

                json_path = output_dir_path / f"{srt_stem}_ponchi.json"
                json_path.write_text(
                    json.dumps(images, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                summary_lines = [
                    f"✅ {len(images)} 件のポンチ絵を生成しました。",
                    f"出力フォルダ: {output_dir_path}",
                    f"JSON: {json_path}",
                    "",
                ]
                for item in images:
                    summary_lines.append(
                        f"{item['start']}〜{item['end']} | {item['visual_suggestion']} | {item['image']}"
                    )

                self.after(0, lambda: self._set_textbox(self.ponchi_output_text, "\n".join(summary_lines)))
                self.log("✅ ポンチ絵作成 完了")
                self.update_progress(1.0)
                self.set_status("Ready", ok=True)
                self.after(0, lambda: messagebox.showinfo("完了", "ポンチ絵作成が完了しました。"))
            except Exception as e:
                tb = traceback.format_exc()
                self.log("❌ ポンチ絵作成でエラー:\n" + tb)
                self.set_status("Error", ok=False)
                self.update_progress(0.0)
                self.after(0, lambda: messagebox.showerror("エラー", f"ポンチ絵作成に失敗しました:\n{e}"))
            finally:
                def _reset_buttons():
                    self.btn_generate_ponchi_images.configure(state="normal", text="▶ ポンチ絵作成")
                    self.btn_generate_ponchi_ideas.configure(state="normal", text="▶ 案出し")

                self.after(0, _reset_buttons)

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------
    # Video generation actions
    # --------------------------
    def on_run_clicked(self):
        api_key = self._get_gemini_api_key()
        script_path = self.script_entry.get().strip()
        output_dir = self.output_entry.get().strip()

        tts_engine = self.tts_engine_var.get()

        if tts_engine == "Gemini" and not api_key:
            messagebox.showerror("エラー", "Gemini を使う場合は 設定タブでAPIキーを入力してください。")
            return
        if not script_path or not Path(script_path).exists():
            messagebox.showerror("エラー", "有効な原稿ファイルを選択してください。")
            return
        if not self.image_paths:
            messagebox.showerror("エラー", "画像を少なくとも1枚は追加してください。")
            return
        if not output_dir:
            messagebox.showerror("エラー", "出力フォルダを指定してください。")
            return

        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
            fps = int(self.fps_entry.get())

            caption_font_size = int(self.caption_font_entry.get() or FONT_SIZE)
            speaker_font_size = int(self.speaker_font_entry.get() or SPEAKER_FONT_SIZE)
            caption_max_chars = int(self.caption_width_entry.get() or CAPTION_MAX_CHARS_PER_LINE)
            caption_box_alpha = int(self.caption_alpha_entry.get() or CAPTION_BOX_ALPHA)

            caption_box_enabled = bool(self.caption_box_enabled_var.get())
            caption_box_height = int(self.caption_box_height_entry.get() or DEFAULT_CAPTION_BOX_HEIGHT)
        except ValueError:
            messagebox.showerror("エラー", "解像度・FPS・字幕設定は数値で入力してください。")
            return

        style_label = self.bg_off_style_var.get()
        bg_off_style = "shadow" if style_label == "影" else ("rounded_panel" if style_label == "角丸パネル" else "none")

        caption_text_color = (self.caption_text_color_entry.get().strip() or DEFAULT_CAPTION_TEXT_COLOR)
        tmp = caption_text_color[1:] if caption_text_color.startswith("#") else caption_text_color
        if len(tmp) != 6:
            messagebox.showerror("エラー", "字幕文字色は #RRGGBB（例: #FFFFFF）で指定してください。")
            return
        try:
            int(tmp, 16)
        except Exception:
            messagebox.showerror("エラー", "字幕文字色は #RRGGBB（例: #FFFFFF）で指定してください。")
            return
        if not caption_text_color.startswith("#"):
            caption_text_color = "#" + caption_text_color

        use_bgm = bool(self.use_bgm_var.get())
        bgm_path = self.bgm_entry.get().strip()
        if use_bgm and (not bgm_path or not Path(bgm_path).exists()):
            messagebox.showerror("エラー", "BGMを使用する場合は有効なファイルを選択してください。")
            return

        bgm_gain_db = float(self.bgm_gain_slider.get())
        voice_name = self.voice_entry.get().strip() or "Kore"

        vv_base_url = self.vv_baseurl_entry.get().strip() or DEFAULT_VOICEVOX_URL
        vv_mode_label = self.vv_mode_var.get()
        vv_mode_int = "rotation" if vv_mode_label == "ローテーション" else "two_person"

        rotation_raw = self.vv_rotation_entry.get().strip()
        vv_rotation_labels = []
        if rotation_raw:
            for tok in rotation_raw.split(","):
                tok = tok.strip()
                if tok:
                    vv_rotation_labels.append(tok)
        if not vv_rotation_labels:
            vv_rotation_labels = [str(x) for x in DEFAULT_VV_ROTATION]

        vv_caster_label = self.vv_caster_entry.get().strip() or DEFAULT_VV_CASTER_LABEL
        vv_analyst_label = self.vv_analyst_entry.get().strip() or DEFAULT_VV_ANALYST_LABEL

        try:
            vv_speed = float(self.vv_speed_slider.get() or DEFAULT_VV_SPEED)
        except ValueError:
            messagebox.showerror("エラー", "VOICEVOX の話速は数値で入力してください。")
            return

        self.save_config()
        self.log_text.delete("1.0", "end")
        self.update_progress(0.0)
        self.log("=== 動画生成開始 ===")
        self.set_status("Working", ok=True)

        self.run_button.configure(state="disabled", text="処理中...")

        def worker():
            try:
                generate_video(
                    api_key=api_key,
                    script_path=script_path,
                    image_paths=self.image_paths,
                    use_bgm=use_bgm,
                    bgm_path=bgm_path,
                    bgm_gain_db=bgm_gain_db,
                    output_dir=output_dir,
                    width=width,
                    height=height,
                    fps=fps,
                    voice_name=voice_name,
                    log_fn=self.log,
                    progress_fn=self.update_progress,
                    tts_engine=tts_engine,
                    vv_mode=vv_mode_int,
                    vv_rotation_labels=vv_rotation_labels,
                    vv_caster_label=vv_caster_label,
                    vv_analyst_label=vv_analyst_label,
                    vv_base_url=vv_base_url,
                    vv_speed_scale=vv_speed,
                    caption_font_size=caption_font_size,
                    speaker_font_size=speaker_font_size,
                    caption_max_chars=caption_max_chars,
                    caption_box_alpha=caption_box_alpha,
                    caption_box_enabled=caption_box_enabled,
                    caption_box_height=caption_box_height,
                    bg_off_style=bg_off_style,
                    caption_text_color=caption_text_color,
                )
                self.log("=== すべて完了しました ===")
                self.set_status("Ready", ok=True)
                messagebox.showinfo("完了", "動画とSRTの生成が完了しました。")
            except Exception as e:
                self.set_status("Error", ok=False)
                messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{e}")
            finally:
                self.after(0, lambda: self.run_button.configure(state="normal", text="▶ 動画を生成する"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = NewsShortGeneratorStudio()
    app.mainloop()
