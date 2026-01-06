"""
Video editing page for News Short Generator Studio.
Separated from the main window to keep the UI modular.
"""
from __future__ import annotations

from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox


class VideoEditPage:
    """Simple editing UI embedded into the main studio window."""

    def __init__(self, parent: ctk.CTkFrame, *, colors: dict[str, str], on_log):
        self.colors = colors
        self.on_log = on_log
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self._build_header()
        self._build_body()

    def _build_header(self):
        header = ctk.CTkFrame(self.frame, corner_radius=18, fg_color=self.colors["panel"])
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="動画編集",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=12)

    def _build_body(self):
        body = ctk.CTkScrollableFrame(self.frame, corner_radius=18, fg_color=self.colors["panel"])
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)

        row = 0

        source_card = self._section_frame(body)
        source_card.grid(row=row, column=0, sticky="ew", pady=(10, 12)); row += 1
        source_card.grid_columnconfigure(0, weight=1)
        self._section_label(source_card, "素材動画").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.source_entry = ctk.CTkEntry(source_card, height=34, corner_radius=12)
        self.source_entry.grid(row=1, column=0, sticky="ew")
        ctk.CTkButton(
            source_card,
            text="動画を選択",
            command=self._browse_source,
            height=36,
            corner_radius=12,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        trim_card = self._section_frame(body)
        trim_card.grid(row=row, column=0, sticky="ew", pady=(0, 12)); row += 1
        trim_card.grid_columnconfigure((0, 1), weight=1)
        self._section_label(trim_card, "トリミング").grid(row=0, column=0, sticky="w", columnspan=2, pady=(0, 6))
        self.start_entry = ctk.CTkEntry(trim_card, placeholder_text="開始 (秒)", height=34, corner_radius=12)
        self.end_entry = ctk.CTkEntry(trim_card, placeholder_text="終了 (秒)", height=34, corner_radius=12)
        self.start_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.end_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        export_card = self._section_frame(body)
        export_card.grid(row=row, column=0, sticky="ew", pady=(0, 12)); row += 1
        export_card.grid_columnconfigure(0, weight=1)
        self._section_label(export_card, "書き出し").grid(row=0, column=0, sticky="w", pady=(0, 6))
        export_row = ctk.CTkFrame(export_card, fg_color="transparent")
        export_row.grid(row=1, column=0, sticky="ew")
        export_row.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(export_row, height=34, corner_radius=12)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            export_row,
            text="保存先",
            command=self._browse_output,
            height=34,
            corner_radius=12,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            width=110,
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        ctk.CTkButton(
            body,
            text="プレビュー",
            command=self._preview,
            height=40,
            corner_radius=12,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
        ).grid(row=row, column=0, sticky="ew", pady=(0, 8)); row += 1

        ctk.CTkButton(
            body,
            text="書き出し",
            command=self._export,
            height=44,
            corner_radius=14,
            fg_color=self.colors["ok"],
            hover_color=self.colors["ok_hover"],
        ).grid(row=row, column=0, sticky="ew", pady=(0, 12)); row += 1

        hint = (
            "・開始/終了の秒数を指定すると該当区間だけを書き出します。\n"
            "・出力先を空にすると元動画と同じフォルダに保存します。\n"
            "・実際の編集ロジックは別途ワークフローに差し込めます。"
        )
        ctk.CTkLabel(
            body,
            text=hint,
            justify="left",
            text_color=self.colors["muted"],
        ).grid(row=row, column=0, sticky="w")

    def _section_label(self, parent, text: str):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors["text"],
            anchor="w",
        )

    def _section_frame(self, parent):
        return ctk.CTkFrame(
            parent,
            fg_color=self.colors["panel2"],
            corner_radius=12,
            border_width=1,
            border_color=self.colors["border"],
            padx=12,
            pady=10,
        )

    def _section_label(self, parent, text: str):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors["text"],
            anchor="w",
        )

    def _browse_source(self):
        path = filedialog.askopenfilename(title="素材動画を選択", filetypes=[("動画ファイル", "*.mp4;*.mov;*.mkv;*.avi"), ("すべて", "*.*")])
        if path:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, path)
            self.on_log(f"🎞️ 素材を選択: {path}")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="保存先を指定", defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("すべて", "*.*")])
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)
            self.on_log(f"💾 出力先を指定: {path}")

    def _preview(self):
        src = self.source_entry.get().strip()
        if not src:
            messagebox.showerror("エラー", "プレビューする動画を選んでください")
            return
        self.on_log(f"▶ プレビュー要求: {src}")
        messagebox.showinfo("プレビュー", "プレビュー機能はダミーです。別途編集処理を組み込んでください。")

    def _export(self):
        src = self.source_entry.get().strip()
        if not src:
            messagebox.showerror("エラー", "書き出す動画を選んでください")
            return
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()
        dest = self.output_entry.get().strip()
        if not dest:
            dest = str(Path(src).with_name(Path(src).stem + "_edited.mp4"))
            self.output_entry.insert(0, dest)
        info = f"開始: {start or '未指定'} / 終了: {end or '未指定'} / 保存先: {dest}"
        self.on_log(f"📤 書き出し要求: {info}")
        messagebox.showinfo("書き出し", "書き出し処理はスタブです。実際の編集処理を接続してください。")
