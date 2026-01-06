"""
Lightweight studio UI with video generation, script generation, and video editing pages.
The video editing UI is defined in a separate module for clarity.
"""
from __future__ import annotations

import customtkinter as ctk
from tkinter import messagebox

from video_edit_page import VideoEditPage


class StudioApp(ctk.CTk):
    COLORS = {
        "bg": "#0a0f1a",
        "panel": "#0b1220",
        "panel2": "#0d1626",
        "card": "#0f1a2e",
        "border": "#18253f",
        "text": "#e7eefc",
        "muted": "#9fb1d1",
        "accent": "#2d6cdf",
        "accent_hover": "#2a61c7",
        "button": "#172238",
        "button_hover": "#1b2a44",
        "ok": "#00b894",
        "ok_hover": "#019870",
    }

    def __init__(self):
        super().__init__()
        self.title("News Short Generator Studio")
        self.geometry("1200x760")
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=self.COLORS["bg"])

        self.active_page = "video"
        self.pages = {}

        self._build_layout()
        self._build_sidebar()
        self._build_center()
        self._build_log_panel()

        self.switch_page("video")

    # layout
    def _build_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)

        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=18, fg_color=self.COLORS["panel"])
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(14, 8), pady=14)
        self.sidebar.grid_propagate(False)

        self.center = ctk.CTkFrame(self, corner_radius=18, fg_color=self.COLORS["panel2"])
        self.center.grid(row=0, column=1, sticky="nsew", padx=8, pady=14)
        self.center.grid_rowconfigure(0, weight=1)
        self.center.grid_columnconfigure(0, weight=1)

        self.log_panel = ctk.CTkFrame(self, width=320, corner_radius=18, fg_color=self.COLORS["panel"])
        self.log_panel.grid(row=0, column=2, sticky="nse", padx=(8, 14), pady=14)
        self.log_panel.grid_propagate(False)

    def _build_sidebar(self):
        self.sidebar.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(self.sidebar, text="STUDIO", font=ctk.CTkFont(size=20, weight="bold"), text_color=self.COLORS["text"])
        title.grid(row=0, column=0, sticky="w", padx=14, pady=(14, 2))
        sub = ctk.CTkLabel(self.sidebar, text="News Short Generator", font=ctk.CTkFont(size=12), text_color=self.COLORS["muted"])
        sub.grid(row=1, column=0, sticky="w", padx=14)

        menu = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        menu.grid(row=2, column=0, sticky="nsew", padx=10, pady=(10, 10))
        menu.grid_columnconfigure(0, weight=1)

        self.btn_video = self._nav_button(menu, "🎬  動画生成", lambda: self.switch_page("video"))
        self.btn_script = self._nav_button(menu, "✍️  台本生成", lambda: self.switch_page("script"))
        self.btn_edit = self._nav_button(menu, "✂️  動画編集", lambda: self.switch_page("edit"))
        self.btn_about = self._nav_button(menu, "ℹ️  About", lambda: self.switch_page("about"))

        self.btn_video.grid(row=0, column=0, sticky="ew", pady=6)
        self.btn_script.grid(row=1, column=0, sticky="ew", pady=6)
        self.btn_edit.grid(row=2, column=0, sticky="ew", pady=6)
        self.btn_about.grid(row=3, column=0, sticky="ew", pady=6)

    def _nav_button(self, parent, text, cmd):
        return ctk.CTkButton(
            parent,
            text=text,
            command=cmd,
            height=44,
            corner_radius=14,
            fg_color=self.COLORS["card"],
            hover_color="#142545",
            text_color=self.COLORS["text"],
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        )

    def _build_center(self):
        self.page_container = ctk.CTkFrame(self.center, fg_color="transparent")
        self.page_container.grid(row=0, column=0, sticky="nsew")
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        for key in ("video", "script", "edit", "about"):
            frame = ctk.CTkFrame(self.page_container, fg_color="transparent")
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = frame

        self._build_video_page()
        self._build_script_page()
        self._build_edit_page()
        self._build_about_page()

    def _build_video_page(self):
        page = self.pages["video"]
        page.grid_rowconfigure(1, weight=1)
        self._page_header(page, "動画生成")
        placeholder = ctk.CTkLabel(page, text="動画生成フォームをここに配置", text_color=self.COLORS["muted"])
        placeholder.grid(row=1, column=0, padx=14, pady=14, sticky="nsew")

    def _build_script_page(self):
        page = self.pages["script"]
        page.grid_rowconfigure(1, weight=1)
        self._page_header(page, "台本生成")
        placeholder = ctk.CTkLabel(page, text="台本生成フォームをここに配置", text_color=self.COLORS["muted"])
        placeholder.grid(row=1, column=0, padx=14, pady=14, sticky="nsew")

    def _build_edit_page(self):
        page = self.pages["edit"]
        edit_page = VideoEditPage(page, colors=self.COLORS, on_log=self.log)
        edit_page.frame.grid(row=0, column=0, sticky="nsew")

    def _build_about_page(self):
        page = self.pages["about"]
        page.grid_rowconfigure(1, weight=1)
        self._page_header(page, "About")
        txt = ctk.CTkTextbox(page, corner_radius=14, fg_color=self.COLORS["bg"], border_width=1, border_color=self.COLORS["border"])
        txt.grid(row=1, column=0, sticky="nsew", padx=14, pady=14)
        txt.insert("end", "This is a lightweight shell for the studio UI.\n動画編集タブを追加しています。")
        txt.configure(state="disabled")

    def _page_header(self, page, title):
        header = ctk.CTkFrame(page, corner_radius=18, fg_color=self.COLORS["panel"])
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=self.COLORS["text"], anchor="w").grid(row=0, column=0, sticky="w", padx=14, pady=12)

    def _build_log_panel(self):
        self.log_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self.log_panel, text="ログ", font=ctk.CTkFont(size=14, weight="bold"), text_color=self.COLORS["text"], anchor="w").grid(row=0, column=0, sticky="w", padx=14, pady=(14, 4))
        self.log_text = ctk.CTkTextbox(self.log_panel, corner_radius=14, fg_color=self.COLORS["bg"], border_width=1, border_color=self.COLORS["border"])
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)

    def switch_page(self, key: str):
        for name, page in self.pages.items():
            if name == key:
                page.tkraise()
        self._set_active_nav(key)
        self.log(f"--- ページ切替: {key} ---")

    def _set_active_nav(self, key: str):
        def style(btn, active: bool):
            if active:
                btn.configure(fg_color="#14305f", hover_color="#17386f")
            else:
                btn.configure(fg_color=self.COLORS["card"], hover_color="#142545")

        style(self.btn_video, key == "video")
        style(self.btn_script, key == "script")
        style(self.btn_edit, key == "edit")
        style(self.btn_about, key == "about")

    def log(self, text: str):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")


if __name__ == "__main__":
    app = StudioApp()
    try:
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("アプリエラー", str(exc))