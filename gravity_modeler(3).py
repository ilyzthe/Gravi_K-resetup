#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gravity Modeler — рисование 2.5D-модели плотностного контраста поверх
картинки разреза + расчёт гравитационного поля от модели в реальном времени.

Возможности:
- Загрузка картинки разреза.
- Задание реальных координат разреза (километры по горизонтали, км глубины).
- Сетка nx x ny ячеек, каждая ячейка — призма с заданным контрастом плотности
  (delta rho в г/см³), вытянутая по простиранию на ±y_extent км для 2.5D.
- Рисование контрастов плотности кистью (как в программе-предшественнице).
- Расчёт g_z в произвольных точках наблюдения вдоль профиля по формуле
  Nagy 1966 / Plouff 1976 (см. Blakely 1996, p.187), векторизованной numpy.
- Загрузка наблюдённого поля из текстового файла (две колонки: x_км, g_мГал).
- Сравнение модельного и наблюдённого поля на общем графике + RMS-misfit.
- Совместимость по формату с .grd предшественника (Surfer DSAA).

Зависимости:
    pip install numpy pillow matplotlib
    (tkinter входит в стандартную поставку Python)

Запуск:
    python gravity_modeler.py
"""

from __future__ import annotations

import csv
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
#                       Физика: формула Nagy
# ============================================================

G_GRAV = 6.6743e-11      # м³/(кг·с²)
SI_TO_MGAL = 1.0e5       # м/с² -> мГал
KM = 1000.0


def prism_gz(x_obs: np.ndarray, y_obs: np.ndarray, z_obs: np.ndarray,
             x1: float, x2: float, y1: float, y2: float,
             z1: float, z2: float, density: float) -> np.ndarray:
    """
    Гравитационный эффект g_z (вертикальная составляющая) от прямоугольной
    3D-призмы с границами [x1,x2] x [y1,y2] x [z1,z2] и плотностью density
    (кг/м³) в массиве точек наблюдения. Ось z направлена ВНИЗ.

    Все координаты — в метрах. Возвращает массив g_z в мГал.
    Формула: Nagy 1966; Plouff 1976; Blakely 1996, p.187.
    """
    g = np.zeros_like(np.asarray(x_obs, dtype=float))
    for i, xc in enumerate((x1, x2)):
        for j, yc in enumerate((y1, y2)):
            for k, zc in enumerate((z1, z2)):
                mu = (-1.0) ** (i + j + k)
                x = xc - x_obs
                y = yc - y_obs
                z = zc - z_obs
                r = np.sqrt(x * x + y * y + z * z)
                F = (x * np.log(y + r + 1e-15)
                     + y * np.log(x + r + 1e-15)
                     - z * np.arctan2(x * y, z * r + 1e-15))
                g += mu * F
    return G_GRAV * density * g * SI_TO_MGAL


def compute_field_from_grid(
    grid_drho: np.ndarray,           # shape (ny, nx), значения в г/см³
    xmin_km: float, xmax_km: float,  # горизонтальные пределы разреза
    zmin_km: float, zmax_km: float,  # пределы глубин (zmin — кровля, zmax — подошва, оба >=0)
    x_obs_km: np.ndarray,            # точки наблюдения по профилю, км
    y_extent_km: float = 50.0,       # 2.5D: полудлина призмы по простиранию
    z_obs_km: float | np.ndarray = 0.0,   # глубина точек наблюдения (0 = поверхность,
                                          # >0 — ниже поверхности, <0 — над поверхностью)
) -> np.ndarray:
    """
    Возвращает массив g_z (мГал) в точках x_obs_km на глубине z_obs_km.
    z_obs_km может быть скаляром (одно значение для всех точек) или массивом
    той же длины, что x_obs_km. Ось z направлена вниз; «над поверхностью» —
    отрицательные z.
    Ячейки с нулевым контрастом плотности пропускаются.
    """
    ny, nx = grid_drho.shape

    x_edges = np.linspace(xmin_km, xmax_km, nx + 1) * KM   # м
    z_edges = np.linspace(zmin_km, zmax_km, ny + 1) * KM   # м (вниз положительно)
    y1 = -y_extent_km * KM
    y2 = +y_extent_km * KM

    x_obs_m = np.asarray(x_obs_km, dtype=float) * KM
    y_obs_m = np.zeros_like(x_obs_m)
    z_obs_m = np.broadcast_to(np.asarray(z_obs_km, dtype=float) * KM,
                                x_obs_m.shape).copy()

    g = np.zeros_like(x_obs_m)

    nonzero = np.argwhere(np.abs(grid_drho) > 1e-15)
    for iy, ix in nonzero:
        drho_si = float(grid_drho[iy, ix]) * 1000.0   # г/см³ -> кг/м³
        x1 = x_edges[ix]
        x2 = x_edges[ix + 1]
        z1 = z_edges[iy]
        z2 = z_edges[iy + 1]
        g += prism_gz(x_obs_m, y_obs_m, z_obs_m,
                      x1, x2, y1, y2, z1, z2, drho_si)
    return g


# ============================================================
#                  IO: формат .grd (Surfer DSAA)
# ============================================================

def read_dsaa(path: str) -> tuple[np.ndarray, float, float, float, float]:
    """Читает Surfer ASCII Grid (DSAA). Возвращает (grid, xmin, xmax, ymin, ymax)."""
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    if len(tokens) < 10 or tokens[0].upper() != "DSAA":
        raise ValueError("Не похоже на DSAA-файл (.grd).")
    nx = int(tokens[1])
    ny = int(tokens[2])
    xmin = float(tokens[3]); xmax = float(tokens[4])
    ymin = float(tokens[5]); ymax = float(tokens[6])
    values = [float(v) for v in tokens[9:9 + nx * ny]]
    if len(values) < nx * ny:
        raise ValueError(f"Недостаточно значений: {len(values)} < {nx*ny}.")
    return np.array(values, dtype=float).reshape(ny, nx), xmin, xmax, ymin, ymax


def write_dsaa(path: str, grid: np.ndarray,
               xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Пишет DSAA-grd, совместимый с программой-предшественником."""
    ny, nx = grid.shape
    vmin = float(np.min(grid)); vmax = float(np.max(grid))
    with open(path, "w", encoding="utf-8") as f:
        f.write("DSAA\n")
        f.write(f"{nx:10d}{ny:10d}\n")
        f.write(f"{xmin:12.3f}{xmax:11.3f}\n")
        f.write(f"{ymin:12.3f}{ymax:11.3f}\n")
        f.write(f"{vmin:12.3f}{vmax:10.3f}\n")
        flat = grid.reshape(-1)
        for i in range(0, len(flat), 4):
            chunk = flat[i:i + 4]
            f.write(" ".join(f"{v: .3f}" if v >= 0 else f"{v:.3f}" for v in chunk) + "\n")


def read_observed_field(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Читает наблюдённое поле из текстового файла.
    Принимает CSV/TSV/whitespace-separated, две колонки: x_km, g_mGal.
    Игнорирует строки-комментарии (#) и нечисловые заголовки.
    """
    xs, gs = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=None)  # whitespace
        for row in f:
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            # Пробуем разные разделители
            parts = row.replace(",", " ").replace(";", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); g = float(parts[1])
            except ValueError:
                continue
            xs.append(x); gs.append(g)
    if not xs:
        raise ValueError("Не нашёл числовых пар (x_км, g_мГал) в файле.")
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(gs)[order]


def _read_csv_with_header(path: str) -> tuple[list[str], list[list[str]]]:
    """
    Универсальное чтение CSV/TSV/whitespace-разделённых файлов.
    Возвращает (header, rows). Если первой строкой идёт нечисловой заголовок —
    он используется как имена колонок; иначе header = [].
    """
    rows: list[list[str]] = []
    header: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Пробуем разные разделители
            parts = [p.strip() for p in
                     line.replace(",", "\t").replace(";", "\t").split("\t")
                     if p.strip()]
            if len(parts) < 2:
                parts = line.split()
            if not header and rows == []:
                # Проверяем, числовая ли первая строка. Если хотя бы одна
                # ячейка не парсится — считаем её заголовком.
                try:
                    _ = [float(p.replace(",", ".")) for p in parts]
                    # это уже числа — заголовка нет
                    rows.append(parts)
                except ValueError:
                    header = parts
                    continue
            else:
                rows.append(parts)
    return header, rows


# ============================================================
#                          GUI
# ============================================================

DEFAULT_DRHO_PRESETS = [
    -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
]
DEFAULT_RHO_PRESETS = [
    2.30, 2.55, 2.60, 2.65, 2.66, 2.68, 2.70, 2.74, 2.78, 2.82, 2.85, 2.90,
]


class GravityModelerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gravity Modeler — рисование 2.5D-модели + расчёт g_z")
        self.root.geometry("1500x950")

        # ----- размеры рабочих областей -----
        self.canvas_width = 1050
        self.canvas_height = 520
        self.plot_height_px = 280

        # ----- состояние изображения -----
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.zoom = 1.0
        self.zoom_min = 0.25
        self.zoom_max = 6.0
        self.img_left = self.img_top = 0
        self.img_right = self.img_bottom = 0
        self.img_width = self.img_height = 0

        # ----- модель -----
        self.nx = 60
        self.ny = 20
        self.grid_values = np.zeros((self.ny, self.nx), dtype=float)

        # реальные координаты разреза (в км)
        self.xmin_km = 0.0
        self.xmax_km = 45.0
        self.zmin_km = 0.0    # кровля разреза (поверхность)
        self.zmax_km = 6.0    # подошва разреза

        # точки наблюдения
        self.n_obs = 200
        self.y_extent_km = 50.0

        # ----- наблюдённое поле -----
        self.obs_x_km: Optional[np.ndarray] = None
        self.obs_g_mgal: Optional[np.ndarray] = None
        # уровень фона у наблюдённого поля (для выравнивания)
        self.obs_baseline = 0.0

        # ----- последнее модельное поле -----
        self.model_x_km: Optional[np.ndarray] = None
        self.model_g_mgal: Optional[np.ndarray] = None

        # ----- состояние рисования -----
        self.drag_painting = False
        self.last_painted_cell = None
        self.allowed_values = list(DEFAULT_DRHO_PRESETS)

        # ----- рамка сетки в долях исходной картинки [0..1] -----
        # (frac_left, frac_top, frac_right, frac_bottom) — устойчиво к zoom.
        self.grid_frac = (0.0, 0.0, 1.0, 1.0)

        # ----- калибровка графика поля на картинке -----
        # Либо None, либо словарь с ключами:
        #   fx1, fy1 — фракции 1-го клика (соответствует x_left_km, g_low_mgal)
        #   fx2, fy2 — фракции 2-го клика (x_right_km, g_high_mgal)
        #   x_left_km, x_right_km, g_low_mgal, g_high_mgal
        self.plot_calib: Optional[dict] = None

        # цифрованные точки графика — в долях изображения
        self.digitized_pts_frac: list[tuple[float, float]] = []

        # режим клика по картинке
        # "paint" — обычное рисование Δρ
        # "grid_p1" / "grid_p2" — ждём кликов для рамки сетки
        # "cal_p1" / "cal_p2"   — ждём кликов для калибровки графика
        # "digitize"            — добавляем/удаляем точки цифровки
        self.click_mode = "paint"

        # перетаскивание рамки сетки целиком (Alt+ЛКМ)
        self.dragging_frame = False
        self.frame_drag_start = None  # (mouse_canvas_x, mouse_canvas_y, frac_left, frac_top)

        self._build_ui()
        self._draw_placeholder()
        self._update_plot()

    # ============== UI ==============
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # ----- Левая панель: Canvas + Scrollbar для прокрутки -----
        controls_outer = ttk.Frame(main)
        controls_outer.pack(side="left", fill="y", padx=(0, 6))

        controls_scroll = ttk.Scrollbar(controls_outer, orient="vertical")
        controls_scroll.pack(side="right", fill="y")

        self.controls_canvas = tk.Canvas(
            controls_outer, width=270, highlightthickness=0,
            yscrollcommand=controls_scroll.set, takefocus=0)
        self.controls_canvas.pack(side="left", fill="y")
        controls_scroll.config(command=self.controls_canvas.yview)

        controls = ttk.Frame(self.controls_canvas)
        self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        def _update_scroll_region(event=None):
            self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all"))
        controls.bind("<Configure>", _update_scroll_region)

        # Колёсико крутит левую панель только пока курсор над ней
        def _on_panel_wheel(event):
            self.controls_canvas.yview_scroll(int(-event.delta / 120), "units")
        self.controls_canvas.bind(
            "<Enter>",
            lambda e: self.controls_canvas.bind_all("<MouseWheel>", _on_panel_wheel))
        self.controls_canvas.bind(
            "<Leave>",
            lambda e: self.controls_canvas.unbind_all("<MouseWheel>"))

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        self._build_controls(controls)
        self._build_plot(right)
        self._build_canvas(right)

    def _build_controls(self, parent) -> None:
        # ===== Файлы =====
        box = ttk.LabelFrame(parent, text="Файлы")
        box.pack(fill="x", pady=3)
        ttk.Button(box, text="Сохранить сессию…",
                   command=self.save_session).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Открыть сессию…",
                   command=self.load_session).pack(fill="x", padx=4, pady=2)
        ttk.Separator(box).pack(fill="x", padx=4, pady=3)
        ttk.Button(box, text="Загрузить картинку разреза",
                   command=self.load_image).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Загрузить .grd модель",
                   command=self.load_grd).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Сохранить .grd модель",
                   command=self.save_grd).pack(fill="x", padx=4, pady=2)
        ttk.Separator(box).pack(fill="x", padx=4, pady=3)
        ttk.Button(box, text="Загрузить наблюдённое поле",
                   command=self.load_observed).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Импорт полевых данных…",
                   command=self.import_field_data).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Сохранить модельное поле",
                   command=self.save_model_field).pack(fill="x", padx=4, pady=2)

        zoom_row = ttk.Frame(box)
        zoom_row.pack(fill="x", padx=4, pady=3)
        ttk.Button(zoom_row, text="−", command=self.zoom_out, width=3).pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(zoom_row, textvariable=self.zoom_var, anchor="center")\
            .pack(side="left", fill="x", expand=True)
        ttk.Button(zoom_row, text="+", command=self.zoom_in, width=3).pack(side="right")

        # ===== Геометрия разреза =====
        box = ttk.LabelFrame(parent, text="Геометрия разреза, км")
        box.pack(fill="x", pady=3)
        self.xmin_var = tk.StringVar(value=str(self.xmin_km))
        self.xmax_var = tk.StringVar(value=str(self.xmax_km))
        self.zmin_var = tk.StringVar(value=str(self.zmin_km))
        self.zmax_var = tk.StringVar(value=str(self.zmax_km))
        self.yext_var = tk.StringVar(value=str(self.y_extent_km))
        self.nobs_var = tk.StringVar(value=str(self.n_obs))
        self._labeled_entry(box, "x min", self.xmin_var)
        self._labeled_entry(box, "x max", self.xmax_var)
        self._labeled_entry(box, "z кровля", self.zmin_var)
        self._labeled_entry(box, "z подошва", self.zmax_var)
        self._labeled_entry(box, "Y-extent (±, км)", self.yext_var)
        self._labeled_entry(box, "Точек наблюдения", self.nobs_var)

        # ===== Рамка сетки (положение на картинке) =====
        box = ttk.LabelFrame(parent, text="Рамка сетки на картинке")
        box.pack(fill="x", pady=3)
        ttk.Button(box, text="Задать рамку (2 клика)",
                   command=self.start_grid_frame).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="На всю картинку",
                   command=self.reset_grid_frame).pack(fill="x", padx=4, pady=2)
        ttk.Label(box,
                  text="Перетащить: Alt+ЛКМ\n"
                       "(рамка не обнуляется при\nсмене zoom/pan)",
                  justify="left", foreground="gray30")\
            .pack(fill="x", padx=4, pady=2)

        # ===== Сетка модели =====
        box = ttk.LabelFrame(parent, text="Сетка модели")
        box.pack(fill="x", pady=3)
        self.nx_var = tk.StringVar(value=str(self.nx))
        self.ny_var = tk.StringVar(value=str(self.ny))
        self._labeled_entry(box, "Nx (по X)", self.nx_var)
        self._labeled_entry(box, "Ny (по глубине)", self.ny_var)
        ttk.Button(box, text="Создать / обновить сетку",
                   command=self.rebuild_grid).pack(fill="x", padx=4, pady=3)
        ttk.Button(box, text="Очистить (все нули)",
                   command=self.clear_grid).pack(fill="x", padx=4, pady=2)

        # ===== Рисование =====
        box = ttk.LabelFrame(parent, text="Значения в ячейках, г/см³")
        box.pack(fill="x", pady=3)

        # Режим ввода: контраст vs абсолютная плотность
        self.input_mode_var = tk.StringVar(value="contrast")
        ttk.Radiobutton(box, text="Δρ (контраст)",
                        variable=self.input_mode_var, value="contrast",
                        command=self._on_input_mode_changed)\
            .pack(anchor="w", padx=4)
        ttk.Radiobutton(box, text="ρ (абсолютная плотность)",
                        variable=self.input_mode_var, value="absolute",
                        command=self._on_input_mode_changed)\
            .pack(anchor="w", padx=4)

        self.rho_bg_top_var = tk.StringVar(value="2.65")
        self.rho_bg_bot_var = tk.StringVar(value="2.65")
        self._labeled_entry(box, "ρ_верх (на кровле)", self.rho_bg_top_var)
        self._labeled_entry(box, "ρ_низ (на подошве)", self.rho_bg_bot_var)
        ttk.Label(box,
                  text="Если ρ_верх = ρ_низ — константа.\n"
                       "Иначе линейный градиент с глубиной.",
                  foreground="gray30", justify="left", wraplength=240)\
            .pack(fill="x", padx=4, pady=2)

        self.value_var = tk.StringVar(value="0.10")
        self.allowed_var = tk.StringVar(value="0.10")
        self.new_preset_var = tk.StringVar(value="")

        row = ttk.Frame(box)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Preset:").pack(side="left")
        self.preset_combo = ttk.Combobox(
            row, textvariable=self.allowed_var,
            values=self._preset_strings(), state="readonly", width=10)
        self.preset_combo.pack(side="right", fill="x", expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self._labeled_entry(box, "Текущее Δρ", self.value_var)

        add_row = ttk.Frame(box)
        add_row.pack(fill="x", padx=4, pady=2)
        ttk.Entry(add_row, textvariable=self.new_preset_var, width=10)\
            .pack(side="left", fill="x", expand=True)
        ttk.Button(add_row, text="+ preset",
                   command=self.add_preset).pack(side="right", padx=(4, 0))

        ttk.Button(box, text="Заполнить всё текущим Δρ",
                   command=self.fill_all).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Инвертировать знак",
                   command=self.invert_sign).pack(fill="x", padx=4, pady=2)

        ttk.Label(box, text=("ЛКМ: красить\n"
                             "ПКМ: ноль\n"
                             "Shift+ЛКМ: пипетка\n"
                             "Средняя кнопка: панорама"),
                  justify="left").pack(fill="x", padx=4, pady=4)

        # ===== Цифровка графика поля =====
        box = ttk.LabelFrame(parent, text="Цифровка графика на картинке")
        box.pack(fill="x", pady=3)

        self.cal_xleft_var  = tk.StringVar(value="0.0")
        self.cal_xright_var = tk.StringVar(value="45.0")
        self.cal_glow_var   = tk.StringVar(value="-10.0")
        self.cal_ghigh_var  = tk.StringVar(value="10.0")
        self._labeled_entry(box, "x лев. опоры, км",  self.cal_xleft_var)
        self._labeled_entry(box, "x прав. опоры, км", self.cal_xright_var)
        self._labeled_entry(box, "g низ опоры, мГал", self.cal_glow_var)
        self._labeled_entry(box, "g верх опоры, мГал", self.cal_ghigh_var)

        ttk.Button(box, text="Калибровать (2 клика)",
                   command=self.start_plot_calibration).pack(fill="x", padx=4, pady=2)

        self.digitize_btn = ttk.Button(box, text="Цифровать кривую",
                                       command=self.toggle_digitize)
        self.digitize_btn.pack(fill="x", padx=4, pady=2)

        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Удалить последнюю",
                   command=self.delete_last_digitized).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Очистить",
                   command=self.clear_digitization).pack(side="right", fill="x", expand=True, padx=(4, 0))

        ttk.Button(box, text="→ в наблюдённое поле",
                   command=self.apply_digitization_as_observed).pack(fill="x", padx=4, pady=2)

        self.digitize_status_var = tk.StringVar(value="Нет калибровки")
        ttk.Label(box, textvariable=self.digitize_status_var,
                  foreground="gray30", wraplength=240, justify="left")\
            .pack(fill="x", padx=4, pady=2)

        # ===== Расчёт поля =====
        box = ttk.LabelFrame(parent, text="Расчёт поля")
        box.pack(fill="x", pady=3)
        self.auto_recompute_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Авто-пересчёт при правке",
                        variable=self.auto_recompute_var).pack(anchor="w", padx=4, pady=2)
        self.use_obs_pickets_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Считать в точках наблюдений\n(если загружены)",
                        variable=self.use_obs_pickets_var).pack(anchor="w", padx=4, pady=2)
        ttk.Button(box, text="Пересчитать сейчас",
                   command=self.recompute_field).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Авто-подбор (инверсия)…",
                   command=self.run_inversion).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Карта поля над разрезом…",
                   command=self.show_field_map).pack(fill="x", padx=4, pady=2)

        self.misfit_var = tk.StringVar(value="RMS misfit: —")
        ttk.Label(box, textvariable=self.misfit_var).pack(anchor="w", padx=4, pady=2)

        # ===== Информация =====
        box = ttk.LabelFrame(parent, text="Курсор / статус")
        box.pack(fill="x", pady=3)
        self.info_var = tk.StringVar(value="Картинка не загружена")
        self.cursor_var = tk.StringVar(value="Ячейка: —, Δρ: —")
        ttk.Label(box, textvariable=self.info_var, wraplength=240,
                  justify="left").pack(fill="x", padx=4, pady=2)
        ttk.Label(box, textvariable=self.cursor_var, wraplength=240,
                  justify="left").pack(fill="x", padx=4, pady=2)

    def _build_plot(self, parent) -> None:
        plot_frame = ttk.LabelFrame(parent, text="Гравитационное поле g_z (мГал)")
        plot_frame.pack(side="top", fill="x", pady=(0, 4))

        self.figure = Figure(figsize=(10, 2.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("x, км")
        self.ax.set_ylabel("g_z, мГал")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill="x", expand=False)

    def _build_canvas(self, parent) -> None:
        cf = ttk.LabelFrame(parent, text="Разрез + модель Δρ (рисуйте кистью)")
        cf.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(cf, width=self.canvas_width,
                                 height=self.canvas_height, bg="white")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-2>", self.on_middle_press)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_release)

    def _labeled_entry(self, parent, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=var, width=10).pack(side="right")

    # ============== пресеты ==============
    def _preset_strings(self) -> list[str]:
        return [f"{v:.3f}" for v in self.allowed_values]

    def _refresh_preset_combo(self) -> None:
        self.allowed_values = sorted(set(round(float(v), 6) for v in self.allowed_values))
        self.preset_combo.configure(values=self._preset_strings())

    def add_preset(self) -> None:
        raw = self.new_preset_var.get().strip()
        if not raw:
            return
        try:
            v = float(raw.replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное Δρ.")
            return
        rounded = round(v, 6)
        if rounded not in [round(x, 6) for x in self.allowed_values]:
            self.allowed_values.append(rounded)
            self._refresh_preset_combo()
        self.allowed_var.set(f"{rounded:.3f}")
        self.value_var.set(f"{rounded:.3f}")
        self.new_preset_var.set("")

    def _on_preset_selected(self, event=None) -> None:
        self.value_var.set(self.allowed_var.get())

    def _on_input_mode_changed(self) -> None:
        """При переключении режима ввода — меняем пресеты и текущее значение."""
        mode = self.input_mode_var.get()
        if mode == "absolute":
            new_presets = list(DEFAULT_RHO_PRESETS)
            new_default = "2.65"
        else:
            new_presets = list(DEFAULT_DRHO_PRESETS)
            new_default = "0.10"
        self.allowed_values = new_presets
        self._refresh_preset_combo()
        self.value_var.set(new_default)
        self.allowed_var.set(new_default)
        self._maybe_recompute()

    # ============== работа с картинкой ==============
    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_width // 2, self.canvas_height // 2,
            text="Загрузите картинку разреза, затем создайте сетку и рисуйте Δρ",
            font=("Arial", 14), fill="gray40")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите картинку разреза",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self._image_path = path     # для save_session
            self.zoom = 1.0
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не открылась:\n{exc}")
            return
        self._prepare_display_image()
        self._redraw_scene()
        self.info_var.set(f"Картинка: {Path(path).name}\n"
                          f"{self.original_image.width}×{self.original_image.height} px")

    def _prepare_display_image(self) -> None:
        if self.original_image is None:
            return
        base = self.original_image.copy()
        base.thumbnail((self.canvas_width - 20, self.canvas_height - 20),
                       Image.Resampling.LANCZOS)
        w = max(1, int(base.width * self.zoom))
        h = max(1, int(base.height * self.zoom))
        img = base.resize((w, h), Image.Resampling.LANCZOS)
        self.display_image = img
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_width, self.img_height = img.size
        self.img_left = (self.canvas_width - self.img_width) // 2
        self.img_top = (self.canvas_height - self.img_height) // 2
        self.img_right = self.img_left + self.img_width
        self.img_bottom = self.img_top + self.img_height
        self.zoom_var.set(f"{int(round(self.zoom * 100))}%")

    # ============== zoom / pan ==============
    def zoom_in(self) -> None:
        if self.original_image is None:
            return
        self.zoom = min(self.zoom_max, self.zoom * 1.25)
        self._prepare_display_image(); self._redraw_scene()

    def zoom_out(self) -> None:
        if self.original_image is None:
            return
        self.zoom = max(self.zoom_min, self.zoom / 1.25)
        self._prepare_display_image(); self._redraw_scene()

    def on_mouse_wheel(self, event) -> None:
        if self.original_image is None:
            return
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_middle_press(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def on_middle_drag(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_middle_release(self, event) -> None:
        self.canvas.configure(cursor="")

    def _event_canvas_xy(self, event) -> tuple[float, float]:
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _update_scroll_region(self) -> None:
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        m = 200
        self.canvas.configure(scrollregion=(x0 - m, y0 - m, x1 + m, y1 + m))

    # ============== сетка ==============
    def _read_geometry(self) -> bool:
        """Перечитывает размеры разреза из полей формы. Возвращает True при успехе."""
        try:
            self.xmin_km = float(self.xmin_var.get().replace(",", "."))
            self.xmax_km = float(self.xmax_var.get().replace(",", "."))
            self.zmin_km = float(self.zmin_var.get().replace(",", "."))
            self.zmax_km = float(self.zmax_var.get().replace(",", "."))
            self.y_extent_km = float(self.yext_var.get().replace(",", "."))
            self.n_obs = max(2, int(self.nobs_var.get()))
            if self.xmax_km <= self.xmin_km or self.zmax_km <= self.zmin_km:
                raise ValueError("xmax<=xmin или zmax<=zmin")
            return True
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Геометрия разреза задана некорректно:\n{e}")
            return False

    def rebuild_grid(self) -> None:
        try:
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
            if nx <= 0 or ny <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Nx и Ny — положительные целые.")
            return
        self.nx = nx
        self.ny = ny
        self.grid_values = np.zeros((ny, nx), dtype=float)
        self._redraw_scene()
        self._maybe_recompute()

    def clear_grid(self) -> None:
        self.grid_values[:] = 0.0
        self._redraw_scene()
        self._maybe_recompute()

    def fill_all(self) -> None:
        try:
            v = float(self.value_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное Δρ.")
            return
        self.grid_values[:] = v
        self._redraw_scene()
        self._maybe_recompute()

    def invert_sign(self) -> None:
        self.grid_values *= -1.0
        self._redraw_scene()
        self._maybe_recompute()

    # ============== bbox сетки на canvas ==============
    def _grid_bbox_canvas(self) -> tuple[int, int, int, int]:
        """Возвращает (x0, y0, x1, y1) сетки на canvas с учётом grid_frac и zoom."""
        fl, ft, fr, fb = self.grid_frac
        x0 = self.img_left + fl * self.img_width
        y0 = self.img_top  + ft * self.img_height
        x1 = self.img_left + fr * self.img_width
        y1 = self.img_top  + fb * self.img_height
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

    def _grid_size_canvas(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self._grid_bbox_canvas()
        return max(1, x1 - x0), max(1, y1 - y0)

    # ============== отрисовка ==============
    def _redraw_scene(self) -> None:
        self.canvas.delete("all")
        if self.tk_image is None:
            self._draw_placeholder()
            return
        self.canvas.create_image(self.img_left, self.img_top,
                                  anchor="nw", image=self.tk_image)
        self._draw_colored_cells()
        self._draw_grid_lines()
        self._draw_plot_calibration()
        self._draw_digitized_points()
        self._update_scroll_region()

    def _draw_grid_lines(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            return
        x0, y0, x1, y1 = self._grid_bbox_canvas()
        gw, gh = self._grid_size_canvas()
        cw = gw / self.nx
        ch = gh / self.ny
        # Внешняя рамка
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", width=2)
        # Внутренние линии
        for ix in range(1, self.nx):
            x = x0 + ix * cw
            self.canvas.create_line(x, y0, x, y1, fill="black")
        for iy in range(1, self.ny):
            y = y0 + iy * ch
            self.canvas.create_line(x0, y, x1, y, fill="black")
        # Если задаём рамку — подсветим её красным
        if self.click_mode in ("grid_p1", "grid_p2"):
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                          outline="red", width=3, dash=(4, 3))

    def _draw_colored_cells(self) -> None:
        x0, y0, _x1, _y1 = self._grid_bbox_canvas()
        gw, gh = self._grid_size_canvas()
        cw = gw / self.nx
        ch = gh / self.ny
        absmax = max(abs(float(np.min(self.grid_values))),
                     abs(float(np.max(self.grid_values))), 1e-12)
        for iy in range(self.ny):
            for ix in range(self.nx):
                v = self.grid_values[iy, ix]
                if abs(v) < 1e-15:
                    continue
                cx0 = x0 + ix * cw
                cy0 = y0 + iy * ch
                cx1 = cx0 + cw
                cy1 = cy0 + ch
                self.canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                              fill=self._value_to_color(v, absmax),
                                              outline="", stipple="gray50")

    def _draw_plot_calibration(self) -> None:
        """Показывает 2 точки калибровки графика и соединяющую рамку."""
        if self.plot_calib is None:
            return
        c = self.plot_calib
        cx1 = self.img_left + c["fx1"] * self.img_width
        cy1 = self.img_top  + c["fy1"] * self.img_height
        cx2 = self.img_left + c["fx2"] * self.img_width
        cy2 = self.img_top  + c["fy2"] * self.img_height
        # Прямоугольник области графика (между двумя опорами)
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                      outline="green", width=1, dash=(3, 3))
        # Опорные точки
        for cx, cy, lbl in [(cx1, cy1, "1"), (cx2, cy2, "2")]:
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                     fill="green", outline="white")
            self.canvas.create_text(cx + 8, cy - 8, text=lbl,
                                     fill="green", font=("Arial", 9, "bold"))

    def _draw_digitized_points(self) -> None:
        for fx, fy in self.digitized_pts_frac:
            cx = self.img_left + fx * self.img_width
            cy = self.img_top  + fy * self.img_height
            self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                     fill="red", outline="white", width=1)

    @staticmethod
    def _value_to_color(value: float, absmax: float) -> str:
        t = max(-1.0, min(1.0, value / absmax))
        if t >= 0:
            r = 255
            g = int(255 * (1.0 - t))
            b = int(255 * (1.0 - t))
        else:
            t = abs(t)
            r = int(255 * (1.0 - t))
            g = int(255 * (1.0 - t))
            b = 255
        return f"#{r:02x}{g:02x}{b:02x}"

    # ============== обработка мыши ==============
    def _point_to_cell(self, x: float, y: float):
        x0, y0, x1, y1 = self._grid_bbox_canvas()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        gw, gh = self._grid_size_canvas()
        cw = gw / self.nx
        ch = gh / self.ny
        ix = min(self.nx - 1, max(0, int((x - x0) / cw)))
        iy = min(self.ny - 1, max(0, int((y - y0) / ch)))
        return ix, iy

    def _canvas_to_frac(self, cx: float, cy: float) -> Optional[tuple[float, float]]:
        """Переводит canvas-координаты в доли исходного изображения [0..1]."""
        if self.img_width <= 0 or self.img_height <= 0:
            return None
        fx = (cx - self.img_left) / self.img_width
        fy = (cy - self.img_top)  / self.img_height
        return fx, fy

    def on_left_click(self, event) -> None:
        x, y = self._event_canvas_xy(event)
        # Alt+ЛКМ — перетаскивание рамки сетки целиком
        if event.state & 0x20000:    # 0x20000 = Alt
            x0, y0, x1, y1 = self._grid_bbox_canvas()
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.dragging_frame = True
                fl, ft, fr, fb = self.grid_frac
                self.frame_drag_start = (x, y, fl, ft, fr, fb)
                self.canvas.configure(cursor="fleur")
                return

        # Режимы рамки сетки
        if self.click_mode == "grid_p1":
            frac = self._canvas_to_frac(x, y)
            if frac is None:
                return
            self._grid_p1_frac = frac
            self.click_mode = "grid_p2"
            self.info_var.set("Кликните 2-й угол рамки сетки")
            return
        if self.click_mode == "grid_p2":
            frac = self._canvas_to_frac(x, y)
            if frac is None:
                return
            fx1, fy1 = self._grid_p1_frac
            fx2, fy2 = frac
            fl, fr = sorted((fx1, fx2))
            ft, fb = sorted((fy1, fy2))
            # ограничим [0..1]
            fl = max(0.0, min(1.0, fl)); fr = max(0.0, min(1.0, fr))
            ft = max(0.0, min(1.0, ft)); fb = max(0.0, min(1.0, fb))
            if fr - fl < 0.02 or fb - ft < 0.02:
                messagebox.showwarning("Внимание",
                                        "Рамка слишком маленькая. Попробуйте ещё раз.")
                self.click_mode = "paint"
                self._redraw_scene()
                return
            self.grid_frac = (fl, ft, fr, fb)
            self.click_mode = "paint"
            self.info_var.set(
                f"Рамка сетки: x∈[{fl:.2f},{fr:.2f}], y∈[{ft:.2f},{fb:.2f}]")
            self._redraw_scene()
            self._maybe_recompute()
            return

        # Режимы калибровки графика
        if self.click_mode == "cal_p1":
            frac = self._canvas_to_frac(x, y)
            if frac is None:
                return
            self._cal_p1_frac = frac
            self.click_mode = "cal_p2"
            self.digitize_status_var.set(
                "1-й клик принят. Теперь — правый-верхний угол графика "
                "(где x=x_прав, g=g_верх).")
            return
        if self.click_mode == "cal_p2":
            frac = self._canvas_to_frac(x, y)
            if frac is None:
                return
            try:
                xL = float(self.cal_xleft_var.get().replace(",", "."))
                xR = float(self.cal_xright_var.get().replace(",", "."))
                gL = float(self.cal_glow_var.get().replace(",", "."))
                gH = float(self.cal_ghigh_var.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректные значения опор калибровки.")
                self.click_mode = "paint"
                return
            fx1, fy1 = self._cal_p1_frac
            fx2, fy2 = frac
            self.plot_calib = dict(
                fx1=fx1, fy1=fy1, fx2=fx2, fy2=fy2,
                x_left_km=xL, x_right_km=xR,
                g_low_mgal=gL, g_high_mgal=gH,
            )
            self.click_mode = "paint"
            self.digitize_status_var.set(
                f"Откалибровано: x∈[{xL},{xR}] км, g∈[{gL},{gH}] мГал.\n"
                "Теперь нажмите «Цифровать кривую».")
            self._redraw_scene()
            return

        # Режим цифровки графика
        if self.click_mode == "digitize":
            frac = self._canvas_to_frac(x, y)
            if frac is None:
                return
            self.digitized_pts_frac.append(frac)
            self._redraw_scene()
            self.digitize_status_var.set(
                f"Точек цифровки: {len(self.digitized_pts_frac)}. "
                "ПКМ — удалить ближайшую, кнопка снова — выйти.")
            return

        # Обычный режим — рисование Δρ
        cell = self._point_to_cell(x, y)
        if cell is None:
            return
        ix, iy = cell
        if event.state & 0x0001:        # Shift -> пипетка
            v = self.grid_values[iy, ix]
            self.value_var.set(f"{v:.3f}")
            self.allowed_var.set(f"{v:.3f}")
            return
        self.drag_painting = True
        self.last_painted_cell = None
        self._paint_cell(ix, iy)

    def on_left_drag(self, event) -> None:
        x, y = self._event_canvas_xy(event)
        # перетаскивание рамки сетки
        if self.dragging_frame and self.frame_drag_start is not None:
            x_start, y_start, fl0, ft0, fr0, fb0 = self.frame_drag_start
            dx_frac = (x - x_start) / max(1, self.img_width)
            dy_frac = (y - y_start) / max(1, self.img_height)
            new_fl = fl0 + dx_frac
            new_fr = fr0 + dx_frac
            new_ft = ft0 + dy_frac
            new_fb = fb0 + dy_frac
            # удерживаем рамку в пределах [0..1]
            if new_fl < 0:
                new_fr -= new_fl; new_fl = 0.0
            if new_ft < 0:
                new_fb -= new_ft; new_ft = 0.0
            if new_fr > 1:
                new_fl -= (new_fr - 1); new_fr = 1.0
            if new_fb > 1:
                new_ft -= (new_fb - 1); new_fb = 1.0
            self.grid_frac = (new_fl, new_ft, new_fr, new_fb)
            self._redraw_scene()
            return

        if not self.drag_painting:
            return
        cell = self._point_to_cell(x, y)
        if cell is None:
            return
        ix, iy = cell
        if self.last_painted_cell == (ix, iy):
            return
        self._paint_cell(ix, iy)

    def on_left_release(self, event) -> None:
        if self.dragging_frame:
            self.dragging_frame = False
            self.frame_drag_start = None
            self.canvas.configure(cursor="")
            self._maybe_recompute()
            return
        self.drag_painting = False
        self.last_painted_cell = None
        self._maybe_recompute()

    def on_right_click(self, event) -> None:
        x, y = self._event_canvas_xy(event)

        # В режиме цифровки ПКМ удаляет ближайшую точку
        if self.click_mode == "digitize" and self.digitized_pts_frac:
            frac = self._canvas_to_frac(x, y)
            if frac is None:
                return
            fx, fy = frac
            # ближайшая точка по канвас-расстоянию
            best_i = -1; best_d = float("inf")
            for i, (px, py) in enumerate(self.digitized_pts_frac):
                d = (px - fx) ** 2 + (py - fy) ** 2
                if d < best_d:
                    best_d = d; best_i = i
            if best_i >= 0:
                del self.digitized_pts_frac[best_i]
                self._redraw_scene()
                self.digitize_status_var.set(
                    f"Точек цифровки: {len(self.digitized_pts_frac)}.")
            return

        # Иначе — обнуляем ячейку модели
        cell = self._point_to_cell(x, y)
        if cell is None:
            return
        ix, iy = cell
        self.grid_values[iy, ix] = 0.0
        self._redraw_scene()
        self.cursor_var.set(f"Ячейка: ({ix},{iy}), Δρ=0")
        self._maybe_recompute()

    def on_mouse_move(self, event) -> None:
        x, y = self._event_canvas_xy(event)
        cell = self._point_to_cell(x, y)
        if cell is None:
            self.cursor_var.set("Ячейка: —, Δρ: —")
            return
        ix, iy = cell
        v = self.grid_values[iy, ix]
        # реальные координаты ячейки
        if self._read_geometry_silent():
            cx = self.xmin_km + (ix + 0.5) * (self.xmax_km - self.xmin_km) / self.nx
            cz = self.zmin_km + (iy + 0.5) * (self.zmax_km - self.zmin_km) / self.ny
            self.cursor_var.set(
                f"Ячейка ({ix},{iy})\n"
                f"x≈{cx:.2f} км, z≈{cz:.2f} км\n"
                f"Δρ = {v:.3f} г/см³")
        else:
            self.cursor_var.set(f"Ячейка ({ix},{iy}), Δρ={v:.3f}")

    def _read_geometry_silent(self) -> bool:
        try:
            self.xmin_km = float(self.xmin_var.get().replace(",", "."))
            self.xmax_km = float(self.xmax_var.get().replace(",", "."))
            self.zmin_km = float(self.zmin_var.get().replace(",", "."))
            self.zmax_km = float(self.zmax_var.get().replace(",", "."))
            return True
        except ValueError:
            return False

    def _paint_cell(self, ix: int, iy: int) -> None:
        try:
            v = float(self.value_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное Δρ.")
            self.drag_painting = False
            return
        self.grid_values[iy, ix] = v
        self.last_painted_cell = (ix, iy)
        self._redraw_scene()
        self.cursor_var.set(f"({ix},{iy}) ← Δρ={v:.3f}")
        # автопересчёт во время рисования делаем не на каждой ячейке,
        # а в release (см. on_left_release), чтобы не тормозить

    # ============== загрузка / сохранение .grd ==============
    def load_grd(self) -> None:
        path = filedialog.askopenfilename(
            title="Загрузить .grd",
            filetypes=[("GRD", "*.grd"), ("All files", "*.*")])
        if not path:
            return
        try:
            grid, xmin, xmax, ymin, ymax = read_dsaa(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не загрузилось:\n{exc}")
            return
        self.grid_values = grid.astype(float)
        self.ny, self.nx = self.grid_values.shape
        self.nx_var.set(str(self.nx))
        self.ny_var.set(str(self.ny))
        self.xmin_var.set(f"{xmin:.3f}")
        self.xmax_var.set(f"{xmax:.3f}")
        self.zmin_var.set(f"{ymin:.3f}")
        self.zmax_var.set(f"{ymax:.3f}")
        # Пополняем пресеты ненулевыми значениями из загруженной модели
        unique = np.unique(np.round(
            self.grid_values[np.abs(self.grid_values) > 1e-15], 6))
        for v in unique:
            if round(float(v), 6) not in [round(x, 6) for x in self.allowed_values]:
                self.allowed_values.append(float(v))
        self._refresh_preset_combo()
        self._read_geometry_silent()
        self._redraw_scene()
        self.info_var.set(f"Сетка: {Path(path).name}\n{self.nx}×{self.ny}")
        self._maybe_recompute()

    def save_grd(self) -> None:
        if not self._read_geometry():
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить .grd", defaultextension=".grd",
            filetypes=[("GRD", "*.grd"), ("All files", "*.*")])
        if not path:
            return
        try:
            write_dsaa(path, self.grid_values,
                       self.xmin_km, self.xmax_km,
                       self.zmin_km, self.zmax_km)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")
            return
        messagebox.showinfo("Готово", f"Сохранено:\n{path}")

    def load_observed(self) -> None:
        path = filedialog.askopenfilename(
            title="Файл наблюдённого поля (две колонки: x_км g_мГал)",
            filetypes=[("Текст/CSV", "*.txt *.csv *.dat *.tsv"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            xs, gs = read_observed_field(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не загрузилось:\n{exc}")
            return
        self.obs_x_km = xs
        self.obs_g_mgal = gs
        # Сместим наблюдённое в средний ноль (Δg) — у моделирования по контрасту
        # плотности базис всё равно нулевой
        self.obs_baseline = float(np.mean(gs))
        self.info_var.set(f"Наблюдённое: {Path(path).name}\n"
                          f"{len(xs)} точек, x∈[{xs.min():.1f},{xs.max():.1f}] км")
        self._update_plot()
        self._update_misfit()

    def save_model_field(self) -> None:
        if self.model_x_km is None or self.model_g_mgal is None:
            messagebox.showwarning("Внимание", "Сначала пересчитайте поле.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить модельное поле", defaultextension=".txt",
            filetypes=[("TXT", "*.txt"), ("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("# x_km  g_z_mGal\n")
            for x, g in zip(self.model_x_km, self.model_g_mgal):
                f.write(f"{x:10.4f}  {g:+10.5f}\n")
        messagebox.showinfo("Готово", f"Сохранено:\n{path}")

    # ============== расчёт поля ==============
    def _maybe_recompute(self) -> None:
        if self.auto_recompute_var.get():
            self.recompute_field()

    # ============== режимы клика ==============
    def start_grid_frame(self) -> None:
        if self.tk_image is None:
            messagebox.showinfo("Сначала картинку",
                                 "Загрузите картинку разреза.")
            return
        self.click_mode = "grid_p1"
        self.info_var.set("Кликните 1-й угол рамки сетки на картинке")
        self._redraw_scene()

    def reset_grid_frame(self) -> None:
        self.grid_frac = (0.0, 0.0, 1.0, 1.0)
        self.click_mode = "paint"
        self._redraw_scene()
        self._maybe_recompute()

    def start_plot_calibration(self) -> None:
        if self.tk_image is None:
            messagebox.showinfo("Сначала картинку",
                                 "Загрузите картинку разреза.")
            return
        # Проверим, что введённые значения парсятся
        try:
            float(self.cal_xleft_var.get().replace(",", "."))
            float(self.cal_xright_var.get().replace(",", "."))
            float(self.cal_glow_var.get().replace(",", "."))
            float(self.cal_ghigh_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка",
                                  "Заполните 4 поля опор калибровки числами.")
            return
        self.click_mode = "cal_p1"
        self.digitize_status_var.set(
            "Кликните на левый-нижний угол области графика "
            "(где x=x_лев, g=g_низ).")

    def toggle_digitize(self) -> None:
        if self.plot_calib is None:
            messagebox.showwarning(
                "Сначала калибровка",
                "Сначала откалибруйте оси графика "
                "(кнопка «Калибровать (2 клика)»).")
            return
        if self.click_mode == "digitize":
            self.click_mode = "paint"
            self.digitize_btn.configure(text="Цифровать кривую")
            self.digitize_status_var.set(
                f"Цифровка завершена. Точек: {len(self.digitized_pts_frac)}.")
        else:
            self.click_mode = "digitize"
            self.digitize_btn.configure(text="Завершить цифровку")
            self.digitize_status_var.set(
                "Кликайте по кривой ЛКМ. ПКМ — удалить ближайшую точку.")

    def delete_last_digitized(self) -> None:
        if self.digitized_pts_frac:
            self.digitized_pts_frac.pop()
            self._redraw_scene()
            self.digitize_status_var.set(
                f"Точек цифровки: {len(self.digitized_pts_frac)}.")

    def clear_digitization(self) -> None:
        self.digitized_pts_frac.clear()
        self._redraw_scene()
        self.digitize_status_var.set("Цифровка очищена.")

    def apply_digitization_as_observed(self) -> None:
        """Преобразует цифрованные точки в (x_km, g_mGal) через калибровку
        и сохраняет в obs_x_km / obs_g_mgal."""
        if self.plot_calib is None:
            messagebox.showwarning("Нет калибровки",
                                    "Сначала откалибруйте оси графика.")
            return
        if not self.digitized_pts_frac:
            messagebox.showwarning("Нет точек",
                                    "Сначала отцифруйте кривую.")
            return
        c = self.plot_calib
        # Аффинное преобразование: x = xL + (fx - fx1)/(fx2 - fx1) * (xR - xL)
        dfx = c["fx2"] - c["fx1"]
        dfy = c["fy2"] - c["fy1"]
        dx  = c["x_right_km"] - c["x_left_km"]
        dg  = c["g_high_mgal"] - c["g_low_mgal"]
        if abs(dfx) < 1e-9 or abs(dfy) < 1e-9:
            messagebox.showerror("Ошибка",
                                  "Точки калибровки совпадают по одной из осей.")
            return
        xs, gs = [], []
        for fx, fy in self.digitized_pts_frac:
            x_km = c["x_left_km"] + (fx - c["fx1"]) / dfx * dx
            g_mg = c["g_low_mgal"] + (fy - c["fy1"]) / dfy * dg
            xs.append(x_km); gs.append(g_mg)
        order = np.argsort(xs)
        self.obs_x_km = np.array(xs)[order]
        self.obs_g_mgal = np.array(gs)[order]
        self.obs_baseline = float(np.mean(self.obs_g_mgal))
        self.info_var.set(f"Цифровка → наблюдённое поле\n"
                           f"{len(xs)} точек, x∈[{self.obs_x_km.min():.1f},"
                           f"{self.obs_x_km.max():.1f}] км")
        self._update_plot()
        self._update_misfit()
        self.digitize_status_var.set(
            f"{len(xs)} точек переданы как наблюдённое поле.")

    def recompute_field(self) -> None:
        if not self._read_geometry():
            return
        # Точки наблюдения: либо равномерные, либо реальные пикеты
        if (self.use_obs_pickets_var.get()
                and self.obs_x_km is not None and len(self.obs_x_km) >= 2):
            x_obs = self.obs_x_km
        else:
            x_obs = np.linspace(self.xmin_km, self.xmax_km, self.n_obs)

        # Готовим сетку контрастов: либо берём как есть, либо вычитаем ρ₀
        try:
            mode = self.input_mode_var.get()
        except AttributeError:
            mode = "contrast"
        if mode == "absolute":
            try:
                rho_top = float(self.rho_bg_top_var.get().replace(",", "."))
                rho_bot = float(self.rho_bg_bot_var.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректные значения фона.")
                return
            # Линейный градиент фона: ρ_фон(z) = ρ_верх + (ρ_низ − ρ_верх)·(z − zmin)/(zmax − zmin).
            # Если ρ_верх = ρ_низ — получится константа.
            ny, nx = self.grid_values.shape
            z_centers = self.zmin_km + (np.arange(ny) + 0.5) \
                * (self.zmax_km - self.zmin_km) / ny
            if abs(self.zmax_km - self.zmin_km) < 1e-9:
                rho_bg_col = np.full(ny, rho_top)
            else:
                t = (z_centers - self.zmin_km) / (self.zmax_km - self.zmin_km)
                rho_bg_col = rho_top + t * (rho_bot - rho_top)
            rho_bg_grid = np.broadcast_to(rho_bg_col[:, None], (ny, nx))
            # Пустые ячейки заполняем фоновой породой → их контраст = 0.
            filled = np.where(np.abs(self.grid_values) > 1e-15,
                                self.grid_values, rho_bg_grid)
            grid_for_calc = filled - rho_bg_grid
        else:
            grid_for_calc = self.grid_values

        try:
            g = compute_field_from_grid(
                grid_for_calc,
                self.xmin_km, self.xmax_km,
                self.zmin_km, self.zmax_km,
                x_obs, y_extent_km=self.y_extent_km)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не посчиталось:\n{exc}")
            return
        self.model_x_km = x_obs
        self.model_g_mgal = g
        self._update_plot()
        self._update_misfit()

    def _update_plot(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("x, км")
        self.ax.set_ylabel("g_z, мГал (центрировано)")
        self.ax.grid(True, alpha=0.3)

        plotted = False
        if self.obs_x_km is not None and self.obs_g_mgal is not None:
            obs_centered = self.obs_g_mgal - float(np.mean(self.obs_g_mgal))
            self.ax.plot(self.obs_x_km, obs_centered, "k.-",
                          label="наблюдённое",
                          markersize=4, linewidth=1)
            plotted = True
        if self.model_x_km is not None and self.model_g_mgal is not None:
            model_centered = self.model_g_mgal - float(np.mean(self.model_g_mgal))
            self.ax.plot(self.model_x_km, model_centered, "r-",
                          label="модельное", linewidth=1.6)
            plotted = True
        if plotted:
            self.ax.legend(loc="best", fontsize=8)
            self.ax.set_xlim(self.xmin_km, self.xmax_km)
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _update_misfit(self) -> None:
        if (self.obs_x_km is None or self.model_x_km is None
                or self.obs_g_mgal is None or self.model_g_mgal is None):
            self.misfit_var.set("RMS misfit: —")
            return
        # Центрируем оба ряда — общий сдвиг уровня не должен мешать
        obs_centered = self.obs_g_mgal - float(np.mean(self.obs_g_mgal))
        model_centered = self.model_g_mgal - float(np.mean(self.model_g_mgal))
        mask = ((self.model_x_km >= self.obs_x_km.min())
                & (self.model_x_km <= self.obs_x_km.max()))
        if not mask.any():
            self.misfit_var.set("RMS misfit: нет пересечения по x")
            return
        obs_at_model = np.interp(self.model_x_km[mask],
                                  self.obs_x_km, obs_centered)
        diff = model_centered[mask] - obs_at_model
        rms = float(np.sqrt(np.mean(diff * diff)))
        self.misfit_var.set(f"RMS misfit: {rms:.3f} мГал")


    # ============== импорт полевых данных ==============
    def import_field_data(self) -> None:
        """Открывает диалог выбора колонок CSV-файла + опции поправок Бугера."""
        path = filedialog.askopenfilename(
            title="CSV/TXT с полевыми данными",
            filetypes=[("Текст/CSV", "*.txt *.csv *.dat *.tsv"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            header, rows = _read_csv_with_header(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не прочитался файл:\n{exc}")
            return
        if not rows:
            messagebox.showerror("Ошибка", "В файле нет числовых строк.")
            return
        if not header:
            # Если заголовка не обнаружено — генерируем имена col1..colN
            header = [f"col{i+1}" for i in range(len(rows[0]))]

        # ---- Диалог выбора колонок ----
        dlg = tk.Toplevel(self.root)
        dlg.title("Импорт полевых данных")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text=f"Файл: {Path(path).name}",
                  foreground="gray30").grid(row=0, column=0, columnspan=2,
                                              sticky="w", pady=(0, 6))
        ttk.Label(frm, text=f"Найдено строк данных: {len(rows)}",
                  foreground="gray30").grid(row=1, column=0, columnspan=2,
                                              sticky="w", pady=(0, 10))

        # Выбор колонок
        col_options = ["—"] + list(header)
        x_var = tk.StringVar(value=header[0] if header else "—")
        g_var = tk.StringVar(value=header[1] if len(header) > 1 else "—")
        h_var = tk.StringVar(value="—")

        ttk.Label(frm, text="Колонка x, км:").grid(row=2, column=0,
                                                     sticky="w", pady=2)
        ttk.Combobox(frm, textvariable=x_var, values=col_options,
                      state="readonly", width=20)\
            .grid(row=2, column=1, sticky="ew", pady=2, padx=(6, 0))

        ttk.Label(frm, text="Колонка g_наблюд, мГал:").grid(row=3, column=0,
                                                              sticky="w", pady=2)
        ttk.Combobox(frm, textvariable=g_var, values=col_options,
                      state="readonly", width=20)\
            .grid(row=3, column=1, sticky="ew", pady=2, padx=(6, 0))

        ttk.Label(frm, text="Колонка h, м (опц.):").grid(row=4, column=0,
                                                          sticky="w", pady=2)
        ttk.Combobox(frm, textvariable=h_var, values=col_options,
                      state="readonly", width=20)\
            .grid(row=4, column=1, sticky="ew", pady=2, padx=(6, 0))

        # Поправки
        ttk.Separator(frm, orient="horizontal").grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(frm, text="Поправки (если выбрана колонка h):",
                  font=("Arial", 9, "bold")).grid(row=6, column=0,
                                                    columnspan=2, sticky="w")
        apply_fa_var = tk.BooleanVar(value=False)
        apply_b_var = tk.BooleanVar(value=False)
        sigma_var = tk.StringVar(value="2.30")
        ttk.Checkbutton(frm, text="Поправка за свободный воздух (+0.3086·h)",
                          variable=apply_fa_var)\
            .grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Поправка Бугера (−0.0419·σ·h)",
                          variable=apply_b_var)\
            .grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, text="σ слоя Бугера, г/см³:").grid(row=9, column=0,
                                                            sticky="w", pady=2)
        ttk.Entry(frm, textvariable=sigma_var, width=10)\
            .grid(row=9, column=1, sticky="w", pady=2, padx=(6, 0))

        # Превью первой строки данных
        ttk.Separator(frm, orient="horizontal").grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=8)
        preview_text = "Первая строка: " + "  ".join(
            f"{h}={v}" for h, v in zip(header, rows[0]))
        ttk.Label(frm, text=preview_text, foreground="gray30",
                  wraplength=420, justify="left").grid(row=11, column=0,
                                                         columnspan=2, sticky="w")

        # OK / Cancel
        btn_row = ttk.Frame(frm)
        btn_row.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        result = {}

        def on_ok():
            if x_var.get() not in header or g_var.get() not in header:
                messagebox.showerror("Ошибка",
                                      "Укажите колонки x и g.")
                return
            try:
                ix = header.index(x_var.get())
                ig = header.index(g_var.get())
                ih = header.index(h_var.get()) if h_var.get() in header else -1
                sigma = float(sigma_var.get().replace(",", "."))
            except (ValueError, IndexError):
                messagebox.showerror("Ошибка", "Некорректные параметры.")
                return
            xs, gs, hs = [], [], []
            for r in rows:
                try:
                    x = float(r[ix].replace(",", "."))
                    g = float(r[ig].replace(",", "."))
                except (ValueError, IndexError):
                    continue
                h = None
                if ih >= 0:
                    try:
                        h = float(r[ih].replace(",", "."))
                    except (ValueError, IndexError):
                        h = None
                xs.append(x); gs.append(g); hs.append(h)
            if not xs:
                messagebox.showerror("Ошибка", "Не нашлось числовых строк.")
                return
            xs = np.array(xs); gs = np.array(gs)
            corrections_log = []
            if (apply_fa_var.get() or apply_b_var.get()):
                if any(h is None for h in hs):
                    messagebox.showerror(
                        "Ошибка",
                        "Не во всех строках есть высота — поправки невозможны.")
                    return
                hs_arr = np.array(hs, dtype=float)
                if apply_fa_var.get():
                    gs = gs + 0.3086 * hs_arr
                    corrections_log.append("Δg_FA = +0.3086·h")
                if apply_b_var.get():
                    gs = gs - 0.0419 * sigma * hs_arr
                    corrections_log.append(f"Δg_B  = −0.0419·{sigma}·h")
            order = np.argsort(xs)
            self.obs_x_km = xs[order]
            self.obs_g_mgal = gs[order]
            self.obs_baseline = float(np.mean(self.obs_g_mgal))
            self.info_var.set(
                f"Полевые данные: {Path(path).name}\n"
                f"{len(self.obs_x_km)} пикетов; "
                + ("поправки: " + ", ".join(corrections_log)
                   if corrections_log else "без поправок"))
            self._update_plot()
            self._update_misfit()
            dlg.destroy()

        ttk.Button(btn_row, text="Импортировать", command=on_ok)\
            .pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="Отмена", command=dlg.destroy)\
            .pack(side="right")

        dlg.wait_window()

    # ============== линейная инверсия ==============
    def run_inversion(self) -> None:
        """Регуляризованная линейная инверсия плотностных контрастов.

        Решает задачу:  min ||G·m − d_obs||² + λ²·||m − m₀||²
        где m — вектор контрастов в ячейках, m₀ — текущая модель,
        G — матрица откликов g_z в точках наблюдений на единичные контрасты,
        d_obs — наблюдённое поле (центрированное).
        """
        if self.obs_x_km is None or self.obs_g_mgal is None:
            messagebox.showinfo("Нет данных",
                                  "Сначала загрузите наблюдённое поле "
                                  "(или импортируйте полевые данные).")
            return
        if not self._read_geometry():
            return

        ny, nx = self.grid_values.shape

        # ---- Диалог настроек ----
        dlg = tk.Toplevel(self.root)
        dlg.title("Авто-подбор плотностей (линейная инверсия)")
        dlg.transient(self.root)
        dlg.grab_set()
        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm,
                  text=f"Сетка: {nx}×{ny} = {nx*ny} ячеек\n"
                       f"Точек наблюдения: {len(self.obs_x_km)}",
                  foreground="gray30")\
            .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        lam_var = tk.StringVar(value="0.1")
        dmin_var = tk.StringVar(value="-0.30")
        dmax_var = tk.StringVar(value="+0.30")
        keep_zero_var = tk.BooleanVar(value=True)

        ttk.Label(frm, text="λ (регуляризация, чем больше — сильнее держит модель):")\
            .grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=lam_var, width=10)\
            .grid(row=1, column=1, sticky="w", padx=(6, 0))
        ttk.Label(frm, text="Δρ_min, г/см³:")\
            .grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=dmin_var, width=10)\
            .grid(row=2, column=1, sticky="w", padx=(6, 0))
        ttk.Label(frm, text="Δρ_max, г/см³:")\
            .grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=dmax_var, width=10)\
            .grid(row=3, column=1, sticky="w", padx=(6, 0))
        ttk.Checkbutton(frm,
                          text="Не трогать пустые ячейки (=0)",
                          variable=keep_zero_var)\
            .grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(frm,
                  text="Внимание: формирование матрицы — N_obs × N_cells\n"
                       "вычислений призмы. При большой сетке может занять\n"
                       "до минуты.",
                  foreground="gray30", justify="left")\
            .grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 4))

        status = tk.StringVar(value="")
        ttk.Label(frm, textvariable=status, foreground="gray20")\
            .grid(row=6, column=0, columnspan=2, sticky="w")

        progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        progress.grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        cancelled = {"flag": False}

        def on_run():
            try:
                lam = float(lam_var.get().replace(",", "."))
                dmin = float(dmin_var.get().replace(",", "."))
                dmax = float(dmax_var.get().replace(",", "."))
                if dmax <= dmin:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректные параметры.")
                return

            obs_x = self.obs_x_km
            obs_g = self.obs_g_mgal.copy()       # без центрирования

            # Маска ячеек: какие подбираем
            if keep_zero_var.get():
                mask = (np.abs(self.grid_values) > 1e-15).reshape(-1)
            else:
                mask = np.ones(ny * nx, dtype=bool)

            n_active = int(mask.sum())
            if n_active == 0:
                messagebox.showinfo("Нет ячеек",
                                      "Нет ячеек для подбора. "
                                      "Сначала задайте стартовую модель "
                                      "(нарисуйте контуры).")
                return

            # Формируем матрицу G: для каждой активной ячейки — отклик 1 г/см³
            x_edges = np.linspace(self.xmin_km, self.xmax_km, nx + 1) * KM
            z_edges = np.linspace(self.zmin_km, self.zmax_km, ny + 1) * KM
            y1 = -self.y_extent_km * KM
            y2 = +self.y_extent_km * KM
            x_obs_m = obs_x * KM
            y_obs_m = np.zeros_like(x_obs_m)
            z_obs_m = np.zeros_like(x_obs_m)

            n_obs = len(x_obs_m)
            G = np.zeros((n_obs, n_active))
            active_indices = np.where(mask)[0]

            status.set("Считаю матрицу откликов…")
            progress.configure(maximum=n_active, value=0)
            dlg.update_idletasks()

            for k, jflat in enumerate(active_indices):
                if cancelled["flag"]:
                    break
                iy, ix = divmod(int(jflat), nx)
                # вклад одной ячейки с Δρ = 1 г/см³ → плотность 1000 кг/м³
                g_unit = prism_gz(x_obs_m, y_obs_m, z_obs_m,
                                   x_edges[ix], x_edges[ix + 1],
                                   y1, y2,
                                   z_edges[iy], z_edges[iy + 1],
                                   density=1000.0)
                G[:, k] = g_unit
                if k % max(1, n_active // 50) == 0:
                    progress.configure(value=k + 1)
                    status.set(f"Матрица откликов: {k+1}/{n_active}…")
                    dlg.update_idletasks()

            if cancelled["flag"]:
                status.set("Отменено.")
                return

            status.set("Решаю систему…")
            dlg.update_idletasks()

            # Стартовая модель
            m0 = self.grid_values.reshape(-1)[active_indices].copy()

            # Аугментируем матрицу свободным столбцом «1» — это учитывает
            # неизвестный уровень нулевого поля (любая аномалия Буге
            # определена с точностью до постоянной).
            G_aug = np.hstack([G, np.ones((n_obs, 1))])
            n_var = n_active + 1

            # Регуляризуем только m (тянем к m₀), сдвиг — нет.
            # Минимизируется ||G_aug·v − obs||² + λ²·||v_m − m₀||²
            reg = np.zeros((n_var, n_var))
            reg[:n_active, :n_active] = (lam ** 2) * np.eye(n_active)
            target = np.concatenate([m0, [0.0]])     # сдвиг тянем к 0

            A = G_aug.T @ G_aug + reg
            b = G_aug.T @ obs_g + reg @ target
            try:
                v_sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as exc:
                messagebox.showerror("Ошибка", f"Система не решилась:\n{exc}")
                return
            m_sol = v_sol[:n_active]
            shift_sol = float(v_sol[-1])

            # Ограничения min/max
            m_sol = np.clip(m_sol, dmin, dmax)

            # Вернуть в сетку
            new_grid = self.grid_values.reshape(-1).copy()
            new_grid[active_indices] = m_sol
            self.grid_values = new_grid.reshape(ny, nx)

            # Обновляем baseline наблюдений в соответствии с подобранным сдвигом —
            # тогда модельное и наблюдённое сядут на один уровень.
            self.obs_baseline = float(np.mean(self.obs_g_mgal)) - shift_sol

            self._redraw_scene()
            self.recompute_field()

            status.set(
                f"Готово. Подобран сдвиг уровня: {shift_sol:+.2f} мГал.\n"
                f"Контрасты в пределах [{m_sol.min():+.3f}, {m_sol.max():+.3f}].")
            dlg.update_idletasks()
            dlg.after(1200, dlg.destroy)

        def on_cancel():
            cancelled["flag"] = True
            dlg.destroy()

        ttk.Button(btn_row, text="Запустить", command=on_run)\
            .pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="Отмена", command=on_cancel)\
            .pack(side="right")

        dlg.wait_window()


    # ============== сохранение и загрузка сессии ==============
    def save_session(self) -> None:
        """Сохраняет всё текущее состояние программы в JSON-файл (.gms)."""
        path = filedialog.asksaveasfilename(
            title="Сохранить сессию",
            defaultextension=".gms",
            filetypes=[("Gravity Modeler Session", "*.gms"),
                       ("JSON", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return

        if not self._read_geometry_silent():
            messagebox.showerror("Ошибка", "Некорректная геометрия разреза.")
            return

        state = {
            "version": 3,
            "image_path": getattr(self, "_image_path", None),
            "geometry": {
                "xmin_km": self.xmin_km, "xmax_km": self.xmax_km,
                "zmin_km": self.zmin_km, "zmax_km": self.zmax_km,
                "y_extent_km": self.y_extent_km, "n_obs": self.n_obs,
            },
            "grid": {
                "nx": self.nx, "ny": self.ny,
                "frac": list(self.grid_frac),
                "values": self.grid_values.tolist(),
            },
            "input_mode": self.input_mode_var.get(),
            "rho_bg_top": self.rho_bg_top_var.get(),
            "rho_bg_bot": self.rho_bg_bot_var.get(),
            "current_value": self.value_var.get(),
            "presets": list(self.allowed_values),
            "plot_calib": self.plot_calib,
            "digitized_pts_frac": list(self.digitized_pts_frac),
            "obs": (None if self.obs_x_km is None else {
                "x_km": self.obs_x_km.tolist(),
                "g_mgal": self.obs_g_mgal.tolist(),
            }),
            "model": (None if self.model_x_km is None else {
                "x_km": self.model_x_km.tolist(),
                "g_mgal": self.model_g_mgal.tolist(),
            }),
            "calib_panel": {
                "x_left": self.cal_xleft_var.get(),
                "x_right": self.cal_xright_var.get(),
                "g_low": self.cal_glow_var.get(),
                "g_high": self.cal_ghigh_var.get(),
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")
            return
        messagebox.showinfo("Готово",
                              f"Сессия сохранена:\n{path}\n\n"
                              f"Картинка хранится по ссылке. Если её "
                              f"переместить, при открытии сессии нужно будет "
                              f"подгрузить заново.")

    def load_session(self) -> None:
        path = filedialog.askopenfilename(
            title="Открыть сессию",
            filetypes=[("Gravity Modeler Session", "*.gms"),
                       ("JSON", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не загрузилось:\n{exc}")
            return

        # Картинка — если путь жив, восстанавливаем
        img_path = state.get("image_path")
        if img_path and Path(img_path).exists():
            try:
                self.original_image = Image.open(img_path).convert("RGB")
                self._image_path = img_path
                self.zoom = 1.0
                self._prepare_display_image()
            except Exception as exc:
                messagebox.showwarning("Внимание",
                                        f"Не открылась картинка:\n{exc}\n"
                                        f"Загрузите её вручную.")
        else:
            if img_path:
                messagebox.showwarning(
                    "Внимание",
                    f"Картинка не найдена:\n{img_path}\n"
                    f"Загрузите её вручную после восстановления сессии.")

        geom = state.get("geometry", {})
        self.xmin_km = geom.get("xmin_km", 0.0)
        self.xmax_km = geom.get("xmax_km", 45.0)
        self.zmin_km = geom.get("zmin_km", 0.0)
        self.zmax_km = geom.get("zmax_km", 6.0)
        self.y_extent_km = geom.get("y_extent_km", 50.0)
        self.n_obs = geom.get("n_obs", 200)
        self.xmin_var.set(str(self.xmin_km))
        self.xmax_var.set(str(self.xmax_km))
        self.zmin_var.set(str(self.zmin_km))
        self.zmax_var.set(str(self.zmax_km))
        self.yext_var.set(str(self.y_extent_km))
        self.nobs_var.set(str(self.n_obs))

        grid = state.get("grid", {})
        self.nx = int(grid.get("nx", 60))
        self.ny = int(grid.get("ny", 20))
        self.nx_var.set(str(self.nx))
        self.ny_var.set(str(self.ny))
        self.grid_frac = tuple(grid.get("frac", [0.0, 0.0, 1.0, 1.0]))
        vals = grid.get("values")
        if vals is not None:
            self.grid_values = np.array(vals, dtype=float)
        else:
            self.grid_values = np.zeros((self.ny, self.nx), dtype=float)

        self.input_mode_var.set(state.get("input_mode", "contrast"))
        self.rho_bg_top_var.set(state.get("rho_bg_top", "2.65"))
        self.rho_bg_bot_var.set(state.get("rho_bg_bot", "2.65"))
        self.value_var.set(state.get("current_value", "0.10"))

        self.allowed_values = list(state.get("presets", DEFAULT_DRHO_PRESETS))
        self._refresh_preset_combo()
        self.allowed_var.set(self.value_var.get())

        self.plot_calib = state.get("plot_calib")
        self.digitized_pts_frac = [tuple(p) for p
                                     in state.get("digitized_pts_frac", [])]

        cp = state.get("calib_panel", {})
        self.cal_xleft_var.set(cp.get("x_left", "0.0"))
        self.cal_xright_var.set(cp.get("x_right", "45.0"))
        self.cal_glow_var.set(cp.get("g_low", "-10.0"))
        self.cal_ghigh_var.set(cp.get("g_high", "10.0"))

        obs = state.get("obs")
        if obs is not None:
            self.obs_x_km = np.array(obs["x_km"], dtype=float)
            self.obs_g_mgal = np.array(obs["g_mgal"], dtype=float)
            self.obs_baseline = float(np.mean(self.obs_g_mgal))
        else:
            self.obs_x_km = None
            self.obs_g_mgal = None

        model = state.get("model")
        if model is not None:
            self.model_x_km = np.array(model["x_km"], dtype=float)
            self.model_g_mgal = np.array(model["g_mgal"], dtype=float)
        else:
            self.model_x_km = None
            self.model_g_mgal = None

        self.click_mode = "paint"
        self._redraw_scene()
        self._update_plot()
        self._update_misfit()
        self.info_var.set(f"Сессия восстановлена:\n{Path(path).name}")

    # ============== карта поля над разрезом ==============
    def show_field_map(self) -> None:
        """Открывает окно с heatmap g_z(x, z) в полупространстве z >= z_top.

        Показывает, как поле затухает с высотой над разрезом и как оно
        распределено в нижней (приповерхностной) части полупространства.
        """
        if not self._read_geometry():
            return
        if not np.any(np.abs(self.grid_values) > 1e-15):
            messagebox.showinfo("Пусто",
                                  "Сначала нарисуйте модель.")
            return

        # Готовим сетку для heatmap
        nx_map = 120
        nz_map = 80
        # Карту строим в координатах разреза + до 30% выше его кровли
        z_above = self.zmin_km - 0.3 * (self.zmax_km - self.zmin_km)
        z_below = self.zmax_km
        x_grid = np.linspace(self.xmin_km, self.xmax_km, nx_map)
        z_grid = np.linspace(z_above, z_below, nz_map)
        X, Z = np.meshgrid(x_grid, z_grid)
        x_flat = X.ravel()
        z_flat = Z.ravel()

        # Нужно учесть режим (contrast / absolute с фоном)
        mode = self.input_mode_var.get()
        if mode == "absolute":
            try:
                rho_top = float(self.rho_bg_top_var.get().replace(",", "."))
                rho_bot = float(self.rho_bg_bot_var.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректные значения фона.")
                return
            ny_, nx_ = self.grid_values.shape
            t = (np.arange(ny_) + 0.5) / max(1, ny_)
            rho_bg_col = rho_top + t * (rho_bot - rho_top)
            rho_bg_grid = np.broadcast_to(rho_bg_col[:, None], (ny_, nx_))
            filled = np.where(np.abs(self.grid_values) > 1e-15,
                                self.grid_values, rho_bg_grid)
            grid_for_calc = filled - rho_bg_grid
        else:
            grid_for_calc = self.grid_values

        # Расчёт. Чтобы не висеть UI — простой статус-диалог.
        progress = tk.Toplevel(self.root)
        progress.title("Расчёт карты поля…")
        progress.transient(self.root)
        ttk.Label(progress, text="Считаю g_z(x, z) на сетке…",
                  padding=20).pack()
        bar = ttk.Progressbar(progress, mode="indeterminate")
        bar.pack(fill="x", padx=20, pady=(0, 20))
        bar.start(20)
        progress.update_idletasks()

        try:
            g_flat = compute_field_from_grid(
                grid_for_calc,
                self.xmin_km, self.xmax_km,
                self.zmin_km, self.zmax_km,
                x_flat, y_extent_km=self.y_extent_km,
                z_obs_km=z_flat)
            g_map = g_flat.reshape(Z.shape)
        finally:
            bar.stop()
            progress.destroy()

        # Окно с картой
        win = tk.Toplevel(self.root)
        win.title("Карта g_z(x, z) над разрезом")
        win.geometry("900x600")

        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        # z вниз положительная — инвертируем ось
        absmax = float(np.nanmax(np.abs(g_map))) or 1e-9
        im = ax.imshow(g_map, origin="upper",
                        extent=(self.xmin_km, self.xmax_km, z_below, z_above),
                        aspect="auto", cmap="RdBu_r",
                        vmin=-absmax, vmax=+absmax)
        ax.axhline(self.zmin_km, color="black", linewidth=1, linestyle="--",
                    alpha=0.6, label="дневная поверхность")
        ax.set_xlabel("x, км")
        ax.set_ylabel("z, км (вниз положительная)")
        ax.set_title("g_z, мГал — поле над и внутри разреза")
        cbar = fig.colorbar(im, ax=ax, label="g_z, мГал")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Кнопка сохранения
        bottom = ttk.Frame(win); bottom.pack(fill="x", pady=4)
        def save_png():
            p = filedialog.asksaveasfilename(
                title="Сохранить карту", defaultextension=".png",
                filetypes=[("PNG", "*.png")])
            if p:
                fig.savefig(p, dpi=200, bbox_inches="tight")
        ttk.Button(bottom, text="Сохранить PNG", command=save_png)\
            .pack(side="right", padx=8)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    GravityModelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()