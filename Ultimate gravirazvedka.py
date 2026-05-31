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
#                  Физика: Nagy 2.5D + Talwani 2D
# ============================================================

G_GRAV = 6.6743e-11      # м³/(кг·с²)
SI_TO_MGAL = 1.0e5       # м/с² -> мГал
KM = 1000.0

# Константа для Talwani в смешанных единицах: координаты в км, плотность в г/см³,
# результат в мГал. 2·G·(перевод г/см³→кг/м³)·(перевод км→м для длины)·(м/с²→мГал)
# = 2·G·1000·1000·1e5 = 2·G·1e11
GRAV_TALWANI = 2.0 * G_GRAV * 1.0e11   # ≈ 13.3486 мГал·км/(г/см³)


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


def gz_polygon_2d_vec(x_obs_arr: np.ndarray, z_obs: float,
                       poly_x: list, poly_z: list) -> np.ndarray:
    """
    Чисто 2D-формула Talwani 1959 / Blakely 1995 eq.6.18 для полигонального
    тела бесконечного по простиранию. Координаты в км, ось z вниз.

    Возвращает безразмерный интеграл (без множителя 2G·Δρ).
    Векторизована по точкам наблюдения.

    Полигон обходится против часовой стрелки в системе z-вниз: для
    прямоугольника это (x1,z1)→(x2,z1)→(x2,z2)→(x1,z2), то есть top L→R,
    bottom R→L.
    """
    x_obs = np.asarray(x_obs_arr, dtype=float)
    total = np.zeros_like(x_obs)
    n = len(poly_x)
    for i in range(n):
        j = (i + 1) % n
        X1 = poly_x[i] - x_obs
        Z1 = poly_z[i] - z_obs
        X2 = poly_x[j] - x_obs
        Z2 = poly_z[j] - z_obs
        R1sq = np.maximum(X1 * X1 + Z1 * Z1, 1e-20)
        R2sq = np.maximum(X2 * X2 + Z2 * Z2, 1e-20)
        dX = poly_x[j] - poly_x[i]
        dZ = poly_z[j] - poly_z[i]
        Lsq = dX * dX + dZ * dZ
        if Lsq < 1e-20:
            continue
        # φ = arctan2(X, Z) — угол от ВЕРТИКАЛИ (важная деталь Blakely)
        phi1 = np.arctan2(X1, Z1)
        phi2 = np.arctan2(X2, Z2)
        dphi = phi2 - phi1
        dphi = np.where(dphi > np.pi, dphi - 2.0 * np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)
        log_R = 0.5 * np.log(R2sq / R1sq)
        q_prime = (X1 * dZ - Z1 * dX) / Lsq
        total += q_prime * (dZ * log_R + dX * dphi)
    return total


def compute_field_talwani_2d(
    grid_drho: np.ndarray,
    xmin_km: float, xmax_km: float,
    zmin_km: float, zmax_km: float,
    x_obs_km: np.ndarray,
    z_obs_km: float = 0.0,
) -> np.ndarray:
    """
    Поле g_z (мГал) в точках x_obs_km для сетки контрастов плотности (г/см³),
    интерпретируемой как набор прямоугольных 2D-призм бесконечной длины
    по простиранию. Координаты в км.

    Чисто 2D — без параметра Y-extent, что часто физически корректнее для
    протяжённых геологических структур (авлакоген, разлом, длинная складка)
    и быстрее формулы Nagy.
    """
    ny, nx = grid_drho.shape
    x_edges = np.linspace(xmin_km, xmax_km, nx + 1)
    z_edges = np.linspace(zmin_km, zmax_km, ny + 1)
    x_obs = np.asarray(x_obs_km, dtype=float)
    g = np.zeros(len(x_obs))
    nonzero = np.argwhere(np.abs(grid_drho) > 1e-15)
    for iy, ix in nonzero:
        drho = float(grid_drho[iy, ix])
        x1, x2 = float(x_edges[ix]), float(x_edges[ix + 1])
        z1, z2 = float(z_edges[iy]), float(z_edges[iy + 1])
        # CCW в системе z-вниз: top L→R, then bottom R→L
        poly_x = [x1, x2, x2, x1]
        poly_z = [z1, z1, z2, z2]
        g += GRAV_TALWANI * drho * gz_polygon_2d_vec(x_obs, z_obs_km, poly_x, poly_z)
    return g


def compute_field_from_grid(
    grid_drho: np.ndarray,           # shape (ny, nx), значения в г/см³
    xmin_km: float, xmax_km: float,  # горизонтальные пределы разреза
    zmin_km: float, zmax_km: float,  # пределы глубин (zmin — кровля, zmax — подошва, оба >=0)
    x_obs_km: np.ndarray,            # точки наблюдения по профилю, км
    y_extent_km: float = 50.0,       # 2.5D: полудлина призмы по простиранию
) -> np.ndarray:
    """
    Возвращает массив g_z (мГал) в точках x_obs_km. Наблюдатель на z=0, y=0.
    Ячейки с нулевым контрастом плотности пропускаются (без вклада).
    """
    ny, nx = grid_drho.shape

    x_edges = np.linspace(xmin_km, xmax_km, nx + 1) * KM   # м
    z_edges = np.linspace(zmin_km, zmax_km, ny + 1) * KM   # м (вниз положительно)
    y1 = -y_extent_km * KM
    y2 = +y_extent_km * KM

    x_obs_m = np.asarray(x_obs_km, dtype=float) * KM
    y_obs_m = np.zeros_like(x_obs_m)
    z_obs_m = np.zeros_like(x_obs_m)

    g = np.zeros_like(x_obs_m)

    # Перебираем ячейки. На разумных размерах сетки (e.g. 100x30 = 3000 ячеек)
    # и 200 точках наблюдения это занимает доли секунды.
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

    Поддерживаемые форматы:
      • 2 колонки: x_км, g_z (мГал)
      • 4 колонки: x_км, y, z, g_z — формат типа Grav_air.dat
                   (берётся 1-я колонка как X и 4-я как g_z; y и z игнорируются)

    Разделитель — пробелы, табуляция, запятая или точка с запятой.
    Игнорирует строки-комментарии (#) и нечисловые заголовки.
    """
    xs, gs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            # Пробуем разные разделители
            parts = row.replace(",", " ").replace(";", " ").split()
            # Все ли «как число»? Парсим первую и нужный по числу колонок столбец
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            if len(nums) == 2:
                x, g = nums[0], nums[1]
            elif len(nums) >= 4:
                # формат x, y, z, g — берём 1-ю и 4-ю
                x, g = nums[0], nums[3]
            else:
                continue
            xs.append(x); gs.append(g)
    if not xs:
        raise ValueError("Не нашёл числовых пар (x_км, g_мГал) в файле.")
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(gs)[order]


# ============================================================
#                          GUI
# ============================================================

DEFAULT_DRHO_PRESETS = [
    -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
]
DEFAULT_RHO_PRESETS = [
    2.30, 2.55, 2.60, 2.65, 2.66, 2.68, 2.70, 2.74, 2.78, 2.82, 2.85, 2.90,
]

# Палитра в стиле геологических разрезов (как на рисунках Ситчихина и др.)
# 12 различимых цветов — для типичного числа плотностных категорий.
GEO_PALETTE = [
    "#fff2cc",  # светло-жёлтый
    "#fde6a8",  # песочный
    "#e6dc8a",  # песочно-зелёный
    "#b8d090",  # светло-зелёный
    "#7eb87a",  # средне-зелёный
    "#80b8c0",  # серо-голубой
    "#9cb4d8",  # голубой
    "#a880c8",  # светло-сиреневый
    "#d49cc4",  # розово-сиреневый
    "#e89a8a",  # светло-красный
    "#c87060",  # терракотовый
    "#9c5050",  # тёмно-розовый
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

        # ----- продление модели за рамку сетки -----
        # Доп. колонки слева и справа с значениями крайних колонок основной сетки.
        # Это устраняет краевые эффекты на флангах профиля. Рисуются полупрозрачно
        # рядом с основной сеткой (вне рамки grid_frac).
        self.pad_cells = 5

        # ----- режим ввода значений в ячейки -----
        # "contrast" — значения = Δρ (контрасты, как раньше)
        # "absolute" — значения = ρ (абсолютные плотности), фон задаётся отдельно
        self.input_mode = "contrast"
        # фон в режиме absolute: линейный градиент от ρ_верх (z=zmin) до ρ_низ (z=zmax)
        self.rho_bg_top = 2.65
        self.rho_bg_bot = 2.65

        # ----- сессия -----
        self.session_path: Optional[str] = None
        self.image_path: Optional[str] = None

        # ----- кэш цветов: значение → строка hex-цвета -----
        # Стабильное назначение цветов: при добавлении новых значений старые
        # не меняют цвет. Значения округляются до 6 знаков для устойчивости float-сравнений.
        self.color_map_cache: dict[float, str] = {}

        # ----- метод расчёта поля -----
        # "talwani2d" — чистое 2D (бесконечное простирание), быстрее
        # "nagy25d"   — 2.5D с конечной длиной по Y (Y-extent), для компактных тел
        self.calc_mode = "talwani2d"

        self._build_menu()
        self._build_ui()
        self._draw_placeholder()
        self._update_plot()

    # ============== Меню сверху ==============
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Новая сессия",
                                command=self.new_session)
        file_menu.add_separator()
        file_menu.add_command(label="Открыть сессию…",
                                command=self.load_session,
                                accelerator="Ctrl+O")
        file_menu.add_command(label="Сохранить сессию",
                                command=self.save_session,
                                accelerator="Ctrl+S")
        file_menu.add_command(label="Сохранить сессию как…",
                                command=self.save_session_as,
                                accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Загрузить картинку разреза…",
                                command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Загрузить .grd модель…",
                                command=self.load_grd)
        file_menu.add_command(label="Сохранить .grd модель…",
                                command=self.save_grd)
        file_menu.add_separator()
        file_menu.add_command(label="Экспорт в DOS-формат (.GRD + .PRO)…",
                                command=self.export_dos_package)
        file_menu.add_separator()
        file_menu.add_command(label="Загрузить наблюдённое поле…",
                                command=self.load_observed)
        file_menu.add_command(label="Сохранить наблюдённое поле…",
                                command=self.save_observed_field)
        file_menu.add_command(label="Сохранить модельное поле…",
                                command=self.save_model_field)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        self.root.config(menu=menubar)

        # Горячие клавиши
        self.root.bind("<Control-s>", lambda e: self.save_session())
        self.root.bind("<Control-S>", lambda e: self.save_session_as())
        self.root.bind("<Control-o>", lambda e: self.load_session())

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
        ttk.Button(box, text="Загрузить картинку разреза",
                   command=self.load_image).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Загрузить .grd модель",
                   command=self.load_grd).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Сохранить .grd модель",
                   command=self.save_grd).pack(fill="x", padx=4, pady=2)
        ttk.Separator(box).pack(fill="x", padx=4, pady=3)
        ttk.Button(box, text="Загрузить наблюдённое поле",
                   command=self.load_observed).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Сохранить наблюдённое поле",
                   command=self.save_observed_field).pack(fill="x", padx=4, pady=2)
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
        self.pad_var  = tk.StringVar(value=str(self.pad_cells))
        self._labeled_entry(box, "x min", self.xmin_var)
        self._labeled_entry(box, "x max", self.xmax_var)
        self._labeled_entry(box, "z кровля", self.zmin_var)
        self._labeled_entry(box, "z подошва", self.zmax_var)
        self._labeled_entry(box, "Y-extent (±, км)", self.yext_var)
        self._labeled_entry(box, "Точек наблюдения", self.nobs_var)
        self._labeled_entry(box, "Расширение, ячеек", self.pad_var)
        ttk.Label(box,
                  text="Расширение продляет крайние\n"
                       "столбцы по бокам сетки.\n"
                       "Убирает краевые эффекты.",
                  foreground="gray30", justify="left", wraplength=240)\
            .pack(fill="x", padx=4, pady=2)

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

        # Радио-кнопки режима ввода
        self.input_mode_var = tk.StringVar(value=self.input_mode)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Radiobutton(row, text="Δρ (контраст)",
                          variable=self.input_mode_var, value="contrast",
                          command=self._on_input_mode_changed)\
            .pack(side="left")
        ttk.Radiobutton(row, text="ρ (абсолютная)",
                          variable=self.input_mode_var, value="absolute",
                          command=self._on_input_mode_changed)\
            .pack(side="left", padx=(8, 0))

        # Поля фона (актуальны в режиме absolute)
        self.rho_top_var = tk.StringVar(value=str(self.rho_bg_top))
        self.rho_bot_var = tk.StringVar(value=str(self.rho_bg_bot))
        self._labeled_entry(box, "Фон ρ верх", self.rho_top_var)
        self._labeled_entry(box, "Фон ρ низ", self.rho_bot_var)
        ttk.Label(box,
                  text="Линейный градиент по глубине.\n"
                       "Если ρ верх = ρ низ — константа.\n"
                       "Только в режиме «ρ».",
                  foreground="gray30", justify="left", wraplength=240)\
            .pack(fill="x", padx=4, pady=2)

        ttk.Separator(box).pack(fill="x", padx=4, pady=3)

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

        # Метод расчёта
        self.calc_mode_var = tk.StringVar(value=self.calc_mode)
        ttk.Label(box, text="Метод:").pack(anchor="w", padx=4, pady=(2, 0))
        ttk.Radiobutton(box, text="Talwani 2D (быстрее)",
                          variable=self.calc_mode_var, value="talwani2d",
                          command=self._on_calc_mode_changed)\
            .pack(anchor="w", padx=12)
        ttk.Radiobutton(box, text="Nagy 2.5D (учёт Y-extent)",
                          variable=self.calc_mode_var, value="nagy25d",
                          command=self._on_calc_mode_changed)\
            .pack(anchor="w", padx=12)
        ttk.Separator(box).pack(fill="x", padx=4, pady=3)

        self.auto_recompute_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Авто-пересчёт при правке",
                        variable=self.auto_recompute_var).pack(anchor="w", padx=4, pady=2)
        self.visual_align_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box,
                          text="Совместить уровни (вычесть рег. сдвиг)",
                          variable=self.visual_align_var,
                          command=self._update_plot)\
            .pack(anchor="w", padx=4, pady=2)
        ttk.Button(box, text="Пересчитать сейчас",
                   command=self.recompute_field).pack(fill="x", padx=4, pady=2)

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
        """При переключении режима меняем пресеты и значение по умолчанию."""
        self.input_mode = self.input_mode_var.get()
        if self.input_mode == "absolute":
            self.allowed_values = list(DEFAULT_RHO_PRESETS)
            default = "2.65"
        else:
            self.allowed_values = list(DEFAULT_DRHO_PRESETS)
            default = "0.10"
        self._refresh_preset_combo()
        self.value_var.set(default)
        self.allowed_var.set(default)
        self._maybe_recompute()

    def _background_grid(self) -> np.ndarray:
        """Возвращает 1D массив длины ny с фоновой плотностью на глубине каждой строки.
        Линейный градиент: ρ_верх → ρ_низ."""
        try:
            rho_top = float(self.rho_top_var.get().replace(",", "."))
            rho_bot = float(self.rho_bot_var.get().replace(",", "."))
        except ValueError:
            rho_top = rho_bot = self.rho_bg_top
        self.rho_bg_top = rho_top
        self.rho_bg_bot = rho_bot
        if self.ny == 0 or self.zmax_km == self.zmin_km:
            return np.full(max(self.ny, 1), rho_top)
        if abs(rho_top - rho_bot) < 1e-9:
            return np.full(self.ny, rho_top)
        t = (np.arange(self.ny) + 0.5) / self.ny
        return rho_top + t * (rho_bot - rho_top)

    def _compute_contrast_grid(self) -> np.ndarray:
        """Возвращает сетку контрастов Δρ для прямой задачи.

        В режиме 'contrast' — значения сетки используются как есть.
        В режиме 'absolute' — из ненулевых ячеек вычитается фон ρ(z);
        пустые ячейки трактуем как фоновую породу (контраст = 0).
        """
        if self.input_mode != "absolute":
            return self.grid_values
        bg_col = self._background_grid()       # shape (ny,)
        bg_2d = bg_col[:, None]                # (ny, 1) — broadcast по x
        filled = np.where(np.abs(self.grid_values) > 1e-15,
                            self.grid_values - bg_2d, 0.0)
        return filled

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
            self.image_path = path
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
            self.pad_cells = max(0, int(self.pad_var.get()))
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
        self._draw_extended_cells()
        self._draw_colored_cells()
        self._draw_grid_lines()
        self._draw_extension_outline()
        self._draw_plot_calibration()
        self._draw_digitized_points()
        self._draw_legend()
        self._update_scroll_region()

    def _draw_legend(self) -> None:
        """Рисует справа сверху легенду цвет → значение."""
        # Уникальные ненулевые значения
        unique_raw = np.unique(self.grid_values)
        unique = sorted([self._round_value(v) for v in unique_raw
                          if abs(v) > 1e-15])
        if not unique:
            return
        # Обновляем кэш цветов
        self._build_color_map()

        # Размеры
        box_w, box_h = 18, 14
        row_h = 19
        legend_w = 130
        header_h = 28
        legend_h = header_h + len(unique) * row_h + 6

        # Положение: правый верхний угол canvas
        cw = int(self.canvas.cget("width"))
        pad = 8
        x0 = cw - legend_w - pad
        y0 = pad
        x1 = x0 + legend_w
        y1 = y0 + legend_h

        # Фон легенды
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                       fill="white", outline="gray50", width=1)
        # Заголовок
        title = ("Плотности, г/см³" if self.input_mode == "absolute"
                 else "Контрасты Δρ")
        self.canvas.create_text((x0 + x1) / 2, y0 + 11,
                                  text=title,
                                  font=("Arial", 9, "bold"), fill="black")
        # Разделитель под заголовком
        self.canvas.create_line(x0 + 4, y0 + 22, x1 - 4, y0 + 22,
                                  fill="gray60")
        # Строки
        for i, v in enumerate(unique):
            y = y0 + header_h + i * row_h
            color = self.color_map_cache.get(v, "#cccccc")
            self.canvas.create_rectangle(x0 + 8, y, x0 + 8 + box_w, y + box_h,
                                           fill=color, outline="black")
            if self.input_mode == "absolute":
                label = f"{v:.3f}"
            else:
                label = f"{v:+.3f}"
            self.canvas.create_text(x0 + 8 + box_w + 6, y + box_h / 2,
                                      text=label, anchor="w",
                                      font=("Arial", 9), fill="black")

    def _draw_extended_cells(self) -> None:
        """Рисует доп. ячейки слева и справа от сетки, продляющие крайние столбцы.

        Эти ячейки не редактируются и показывают пользователю, какие данные
        участвуют в расчёте за пределами видимой сетки.
        Отрисовываются полупрозрачно (stipple) и без рамки.
        """
        if self.pad_cells <= 0 or self.nx <= 0 or self.ny <= 0:
            return
        x0, y0, x1, y1 = self._grid_bbox_canvas()
        gw, gh = self._grid_size_canvas()
        cw = gw / self.nx
        ch = gh / self.ny

        for iy in range(self.ny):
            # значения крайних столбцов
            v_left = self.grid_values[iy, 0]
            v_right = self.grid_values[iy, -1]
            cy0 = y0 + iy * ch
            cy1 = cy0 + ch
            # слева
            if abs(v_left) > 1e-15:
                for k in range(self.pad_cells):
                    cx0 = x0 - (k + 1) * cw
                    cx1 = cx0 + cw
                    self.canvas.create_rectangle(
                        cx0, cy0, cx1, cy1,
                        fill=self._color_for(v_left),
                        outline="", stipple="gray25")
            # справа
            if abs(v_right) > 1e-15:
                for k in range(self.pad_cells):
                    cx0 = x1 + k * cw
                    cx1 = cx0 + cw
                    self.canvas.create_rectangle(
                        cx0, cy0, cx1, cy1,
                        fill=self._color_for(v_right),
                        outline="", stipple="gray25")

    def _draw_extension_outline(self) -> None:
        """Тонкая рамка вокруг расширения, чтобы пользователь видел его границы."""
        if self.pad_cells <= 0 or self.nx <= 0:
            return
        x0, y0, x1, y1 = self._grid_bbox_canvas()
        gw, _ = self._grid_size_canvas()
        cw = gw / self.nx
        ext_left = x0 - self.pad_cells * cw
        ext_right = x1 + self.pad_cells * cw
        # пунктирная рамка вокруг расширенной области
        self.canvas.create_rectangle(ext_left, y0, ext_right, y1,
                                       outline="gray50", width=1, dash=(2, 3))
        # пометки «продление»
        self.canvas.create_text(
            (ext_left + x0) / 2, y0 - 8,
            text="← продление", fill="gray50", font=("Arial", 8))
        self.canvas.create_text(
            (x1 + ext_right) / 2, y0 - 8,
            text="продление →", fill="gray50", font=("Arial", 8))

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
        # обновляем кэш цветов
        self._build_color_map()
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
                                              fill=self._color_for(v),
                                              outline="")

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
    def _round_value(v: float) -> float:
        """Округление для устойчивого сравнения float-значений в кэше цветов."""
        return round(float(v), 6)

    def _build_color_map(self) -> dict[float, str]:
        """Возвращает map значение → hex-цвет для всех уникальных ненулевых
        значений в основной сетке. Старые назначения сохраняются (стабильность
        цветов при добавлении новых значений)."""
        # уникальные значения сетки
        unique_raw = np.unique(self.grid_values)
        unique = [self._round_value(v) for v in unique_raw if abs(v) > 1e-15]

        # Удалить из кэша значения, которых больше нет
        self.color_map_cache = {v: c for v, c in self.color_map_cache.items()
                                  if v in unique}

        # Назначить цвета новым значениям, используя свободные слоты палитры
        used = set(self.color_map_cache.values())
        for v in sorted(unique):
            if v in self.color_map_cache:
                continue
            chosen = None
            for color in GEO_PALETTE:
                if color not in used:
                    chosen = color; used.add(color); break
            if chosen is None:
                # палитра исчерпана — циклично, но это редкий случай
                idx = len(self.color_map_cache) % len(GEO_PALETTE)
                chosen = GEO_PALETTE[idx]
            self.color_map_cache[v] = chosen

        return self.color_map_cache

    def _color_for(self, value: float) -> str:
        """Цвет для конкретного значения ячейки (использует кэш)."""
        v = self._round_value(value)
        if v in self.color_map_cache:
            return self.color_map_cache[v]
        # Если кэш ещё не построен — построим
        self._build_color_map()
        return self.color_map_cache.get(v, "#cccccc")

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

    def save_observed_field(self) -> None:
        """Экспорт оцифрованного/загруженного наблюдённого поля.

        Формат: CSV без заголовка, 4 колонки через запятую:
            x_км, y, z, g_z_мГал
        где y и z для 2D-профиля = 0. Окончания строк CRLF (для совместимости
        с Windows-программами). Совместимо с форматом DOS-инструментов
        (например, Grav_air.dat).
        """
        if self.obs_x_km is None or self.obs_g_mgal is None:
            messagebox.showwarning(
                "Внимание",
                "Наблюдённое поле не задано.\n"
                "Сначала загрузите его или цифруйте кривую с картинки\n"
                "и нажмите «→ в наблюдённое поле».")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить наблюдённое поле",
            defaultextension=".dat",
            filetypes=[("DAT (CSV)", "*.dat"), ("CSV", "*.csv"),
                       ("Текст", "*.txt"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                for x, g in zip(self.obs_x_km, self.obs_g_mgal):
                    # Формат как в Grav_air.dat: x,0,0,g — CRLF
                    f.write(f"{x:g},0,0,{g:g}\r\n")
            messagebox.showinfo("Готово",
                                  f"Сохранено:\n{path}\n"
                                  f"Точек: {len(self.obs_x_km)}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")

    def export_dos_package(self) -> None:
        """Экспортирует модель и наблюдения в формате, понятном DOS-программе.

        Создаёт пару файлов с общим префиксом:
        - <префикс>.GRD — сетка плотностей (DSAA, уже совместимый формат)
        - <префикс>.PRO — параметры профиля + наблюдённое/модельное поле + сдвиг

        Этот пакет позволяет открыть модель в старой DOS-программе для проверки
        и сравнения. Если в наблюдении нет данных — записывается нулями.
        """
        if not self._read_geometry_silent():
            messagebox.showerror("Ошибка", "Некорректная геометрия разреза.")
            return
        # Сначала пересчитаем поле — нам нужно модельное для записи в .PRO
        if self.model_x_km is None or self.model_g_mgal is None:
            self.recompute_field()
            if self.model_x_km is None:
                messagebox.showwarning("Внимание",
                                          "Сначала пересчитайте поле.")
                return

        path = filedialog.asksaveasfilename(
            title="Префикс для пакета DOS-файлов (.GRD + .PRO)",
            defaultextension=".PRO",
            filetypes=[("DOS PRO", "*.PRO"), ("Все файлы", "*.*")])
        if not path:
            return
        # Убираем расширение если есть и формируем оба пути
        prefix = path
        for ext in (".PRO", ".pro", ".GRD", ".grd"):
            if prefix.endswith(ext):
                prefix = prefix[:-len(ext)]
                break
        grd_path = prefix + ".GRD"
        pro_path = prefix + ".PRO"

        try:
            # === GRD ===
            write_dsaa(grd_path, self.grid_values,
                       self.xmin_km, self.xmax_km,
                       self.zmin_km, self.zmax_km)

            # === PRO ===
            self._write_pro_file(pro_path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")
            return

        messagebox.showinfo(
            "Готово",
            f"Записаны DOS-файлы:\n"
            f"  {Path(grd_path).name}\n"
            f"  {Path(pro_path).name}\n\n"
            f"Префикс: {Path(prefix).name}\n\n"
            f"Перенесите оба в папку с DOS-программой и попробуйте открыть.\n"
            f"Если потребуется ещё .1/.2 — сообщите, какие именно.")

    def _write_pro_file(self, path: str) -> None:
        """Запись .PRO в формате, который воспроизводит структуру файлов
        старой DOS-программы (CP866, CRLF, Fortran-подобное форматирование)."""
        n = len(self.model_x_km)
        x_obs = self.model_x_km
        g_calc = self.model_g_mgal

        if self.obs_x_km is not None and self.obs_g_mgal is not None:
            g_nabl = np.interp(x_obs, self.obs_x_km, self.obs_g_mgal,
                                left=self.obs_g_mgal[0],
                                right=self.obs_g_mgal[-1])
            # Сдвиг с тем же знаком, что в DOS-файлах:
            #   G_наблюд[i] = G_вычислен[i] + сдвиг  (проверено по 01DIR1.PRO)
            #   ⇒ сдвиг = mean(G_наблюд) − mean(G_вычислен)
            sdvig = float(np.mean(g_nabl) - np.mean(g_calc))
        else:
            g_nabl = np.zeros(n)
            sdvig = 0.0

        y_obs = np.zeros(n)
        h_obs = np.zeros(n)

        def fmt_value(v: float) -> str:
            """Имитирует Fortran G16.6: ширина ровно 16 символов."""
            av = abs(v)
            if av == 0.0:
                return "    0.000000E+00"
            if 0.1 <= av < 100000.0:
                return f"{v:16.6f}"
            return f"{v:16.6E}"

        with open(path, "w", encoding="cp866", newline="") as f:
            f.write("           0           0\r\n")
            f.write(" Число точек \r\n")
            f.write(f"{n:12d}\r\n")
            f.write(" Масштаб x,h \r\n")
            f.write(f"{fmt_value(1.0)}{fmt_value(1.0)}\r\n")
            f.write(" X1, X2  пр. \r\n")
            f.write(f"{fmt_value(float(x_obs[0]))}{fmt_value(float(x_obs[-1]))}\r\n")
            f.write(" Y1, Y2  пр. \r\n")
            f.write(f"{fmt_value(0.0)}{fmt_value(0.0)}\r\n")
            f.write(" eps, сдвиг  \r\n")
            # eps — допуск подбора (некритично для импорта в DOS-программу)
            f.write(f"{fmt_value(0.001)}{fmt_value(sdvig)}\r\n")

            def write_block(name: str, arr):
                f.write(f" {name}\r\n")
                for i in range(0, len(arr), 4):
                    line = "".join(fmt_value(float(v)) for v in arr[i:i + 4])
                    f.write(line + "\r\n")

            write_block("G вычислен. ", g_calc)
            write_block("G наблюден. ", g_nabl)
            write_block("Высоты т. н.", h_obs)
            write_block("X-коорд т.н.", x_obs)
            write_block("Y-коорд т.н.", y_obs)
            write_block("Давл.рельефа", np.zeros(n))
            f.write(" Глубина давл\r\n")
            f.write(f"{fmt_value(0.0)}\r\n")
            write_block("Давление    ", np.zeros(n))

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

    # ============== сессии (.gms) ==============
    def _collect_session_state(self) -> dict:
        """Собирает всё состояние программы в словарь для JSON-сериализации."""
        self._read_geometry_silent()
        return {
            "version": 1,
            "image_path": self.image_path,
            "geometry": {
                "xmin_km": self.xmin_km, "xmax_km": self.xmax_km,
                "zmin_km": self.zmin_km, "zmax_km": self.zmax_km,
                "y_extent_km": self.y_extent_km,
                "n_obs": self.n_obs,
                "pad_cells": self.pad_cells,
            },
            "grid": {
                "nx": self.nx, "ny": self.ny,
                "frac": list(self.grid_frac),
                "values": self.grid_values.tolist(),
            },
            "presets": list(self.allowed_values),
            "current_value": self.value_var.get(),
            "input_mode": self.input_mode,
            "rho_bg_top": self.rho_bg_top,
            "rho_bg_bot": self.rho_bg_bot,
            "plot_calib": self.plot_calib,
            "digitized_pts_frac": [list(p) for p in self.digitized_pts_frac],
            "calib_panel": {
                "x_left": self.cal_xleft_var.get(),
                "x_right": self.cal_xright_var.get(),
                "g_low": self.cal_glow_var.get(),
                "g_high": self.cal_ghigh_var.get(),
            },
            "obs": (None if self.obs_x_km is None else {
                "x_km": self.obs_x_km.tolist(),
                "g_mgal": self.obs_g_mgal.tolist(),
            }),
            "color_map": {str(k): v for k, v in self.color_map_cache.items()},
            "calc_mode": self.calc_mode,
            "visual_align": (self.visual_align_var.get()
                              if hasattr(self, "visual_align_var") else False),
        }

    def _apply_session_state(self, state: dict) -> None:
        """Применяет загруженное состояние ко всей программе."""
        # Картинка
        img_path = state.get("image_path")
        if img_path and Path(img_path).exists():
            try:
                self.original_image = Image.open(img_path).convert("RGB")
                self.image_path = img_path
                self.zoom = 1.0
                self._prepare_display_image()
            except Exception as exc:
                messagebox.showwarning(
                    "Внимание",
                    f"Не открылась картинка ({exc}). Загрузите её вручную.")
        elif img_path:
            messagebox.showwarning(
                "Внимание",
                f"Картинка не найдена:\n{img_path}\n"
                f"Загрузите её вручную после восстановления сессии.")

        # Геометрия
        geom = state.get("geometry", {})
        self.xmin_km = geom.get("xmin_km", 0.0)
        self.xmax_km = geom.get("xmax_km", 45.0)
        self.zmin_km = geom.get("zmin_km", 0.0)
        self.zmax_km = geom.get("zmax_km", 6.0)
        self.y_extent_km = geom.get("y_extent_km", 50.0)
        self.n_obs = geom.get("n_obs", 200)
        self.pad_cells = geom.get("pad_cells", 5)
        self.xmin_var.set(str(self.xmin_km))
        self.xmax_var.set(str(self.xmax_km))
        self.zmin_var.set(str(self.zmin_km))
        self.zmax_var.set(str(self.zmax_km))
        self.yext_var.set(str(self.y_extent_km))
        self.nobs_var.set(str(self.n_obs))
        self.pad_var.set(str(self.pad_cells))

        # Сетка
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

        # Пресеты
        self.allowed_values = list(state.get("presets", DEFAULT_DRHO_PRESETS))
        self._refresh_preset_combo()
        cur = state.get("current_value", "0.10")
        self.value_var.set(cur)
        self.allowed_var.set(cur)

        # Режим ввода и фон
        self.input_mode = state.get("input_mode", "contrast")
        self.input_mode_var.set(self.input_mode)
        self.rho_bg_top = state.get("rho_bg_top", 2.65)
        self.rho_bg_bot = state.get("rho_bg_bot", 2.65)
        self.rho_top_var.set(str(self.rho_bg_top))
        self.rho_bot_var.set(str(self.rho_bg_bot))

        # Калибровка и цифровка
        self.plot_calib = state.get("plot_calib")
        self.digitized_pts_frac = [tuple(p) for p
                                     in state.get("digitized_pts_frac", [])]
        cp = state.get("calib_panel", {})
        self.cal_xleft_var.set(cp.get("x_left", "0.0"))
        self.cal_xright_var.set(cp.get("x_right", "45.0"))
        self.cal_glow_var.set(cp.get("g_low", "-10.0"))
        self.cal_ghigh_var.set(cp.get("g_high", "10.0"))

        # Наблюдения
        obs = state.get("obs")
        if obs is not None:
            self.obs_x_km = np.array(obs["x_km"], dtype=float)
            self.obs_g_mgal = np.array(obs["g_mgal"], dtype=float)
            self.obs_baseline = float(np.mean(self.obs_g_mgal))
        else:
            self.obs_x_km = None
            self.obs_g_mgal = None
            self.obs_baseline = 0.0

        # Кэш цветов
        cmap = state.get("color_map", {})
        self.color_map_cache = {float(k): v for k, v in cmap.items()}

        # Метод расчёта
        self.calc_mode = state.get("calc_mode", "talwani2d")
        self.calc_mode_var.set(self.calc_mode)

        # Опция совмещения уровней
        if hasattr(self, "visual_align_var"):
            self.visual_align_var.set(state.get("visual_align", False))

        self.click_mode = "paint"
        self._redraw_scene()
        self._update_plot()
        self._update_misfit()

    def new_session(self) -> None:
        if not messagebox.askyesno(
                "Новая сессия",
                "Начать новую сессию? Текущие несохранённые правки будут потеряны."):
            return
        self.original_image = None
        self.tk_image = None
        self.image_path = None
        self.session_path = None
        self.grid_values = np.zeros((self.ny, self.nx), dtype=float)
        self.grid_frac = (0.0, 0.0, 1.0, 1.0)
        self.plot_calib = None
        self.digitized_pts_frac = []
        self.obs_x_km = None
        self.obs_g_mgal = None
        self.model_x_km = None
        self.model_g_mgal = None
        self.color_map_cache = {}
        self.click_mode = "paint"
        self._update_window_title()
        self._redraw_scene()
        self._update_plot()
        self._update_misfit()
        self.info_var.set("Новая сессия.")

    def save_session(self) -> None:
        if self.session_path is None:
            self.save_session_as()
            return
        self._save_session_to(self.session_path)

    def save_session_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Сохранить сессию",
            defaultextension=".gms",
            filetypes=[("Gravity Modeler Session", "*.gms"),
                       ("JSON", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        self._save_session_to(path)
        self.session_path = path
        self._update_window_title()

    def _save_session_to(self, path: str) -> None:
        try:
            state = self._collect_session_state()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            self.info_var.set(f"Сохранено: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")

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
            messagebox.showerror("Ошибка", f"Не открылось:\n{exc}")
            return
        self._apply_session_state(state)
        self.session_path = path
        self._update_window_title()
        self.info_var.set(f"Сессия открыта: {Path(path).name}")

    def _update_window_title(self) -> None:
        base = "Gravity Modeler — рисование 2.5D-модели + расчёт g_z"
        if self.session_path:
            self.root.title(f"{base}  —  {Path(self.session_path).name}")
        else:
            self.root.title(base)

    def _on_calc_mode_changed(self) -> None:
        self.calc_mode = self.calc_mode_var.get()
        self._maybe_recompute()

    def recompute_field(self) -> None:
        if not self._read_geometry():
            return
        x_obs = np.linspace(self.xmin_km, self.xmax_km, self.n_obs)
        try:
            grid_for_calc, xmin_eff, xmax_eff = self._extended_grid()
            if self.calc_mode == "talwani2d":
                # Чистое 2D: координаты в км, плотность в г/см³
                g = compute_field_talwani_2d(
                    grid_for_calc,
                    xmin_eff, xmax_eff,
                    self.zmin_km, self.zmax_km,
                    x_obs)
            else:
                # 2.5D Nagy с конечной длиной Y
                g = compute_field_from_grid(
                    grid_for_calc,
                    xmin_eff, xmax_eff,
                    self.zmin_km, self.zmax_km,
                    x_obs, y_extent_km=self.y_extent_km)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не посчиталось:\n{exc}")
            return
        self.model_x_km = x_obs
        self.model_g_mgal = g
        self._update_plot()
        self._update_misfit()

    def _extended_grid(self) -> tuple[np.ndarray, float, float]:
        """Возвращает сетку контрастов Δρ, расширенную на pad_cells колонок
        слева и справа.

        Доп. колонки получают значения крайних столбцов основной сетки —
        это эквивалентно «продолжению» геологии на pad_cells * ширина_ячейки
        километров за пределы профиля. На полях расчёта эффект — устранение
        краевых артефактов на флангах.
        """
        contrast = self._compute_contrast_grid()
        if self.pad_cells <= 0 or self.nx == 0:
            return contrast, self.xmin_km, self.xmax_km
        dx_per_cell = (self.xmax_km - self.xmin_km) / self.nx
        left_col = contrast[:, 0:1]
        right_col = contrast[:, -1:]
        left_ext = np.tile(left_col, (1, self.pad_cells))
        right_ext = np.tile(right_col, (1, self.pad_cells))
        extended = np.hstack([left_ext, contrast, right_ext])
        xmin_eff = self.xmin_km - self.pad_cells * dx_per_cell
        xmax_eff = self.xmax_km + self.pad_cells * dx_per_cell
        return extended, xmin_eff, xmax_eff

    def _update_plot(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("x, км")
        self.ax.set_ylabel("g_z, мГал")
        self.ax.grid(True, alpha=0.3)

        # Считаем регион. сдвиг (используется и в misfit, и для опции «совместить»)
        shift_for_plot = 0.0
        align = (hasattr(self, "visual_align_var")
                  and self.visual_align_var.get()
                  and self.obs_x_km is not None and self.obs_g_mgal is not None
                  and self.model_x_km is not None and self.model_g_mgal is not None)
        if align:
            mask = ((self.model_x_km >= self.obs_x_km.min())
                    & (self.model_x_km <= self.obs_x_km.max()))
            if mask.any():
                obs_at_model = np.interp(self.model_x_km[mask],
                                          self.obs_x_km, self.obs_g_mgal)
                shift_for_plot = float(np.mean(obs_at_model
                                                 - self.model_g_mgal[mask]))

        plotted = False
        if self.obs_x_km is not None and self.obs_g_mgal is not None:
            self.ax.plot(self.obs_x_km, self.obs_g_mgal, "k.-",
                          label="наблюдённое",
                          markersize=4, linewidth=1)
            plotted = True
        if self.model_x_km is not None and self.model_g_mgal is not None:
            label = "модельное"
            if align and abs(shift_for_plot) > 1e-9:
                label = f"модельное (сдвинуто на {shift_for_plot:+.2f} мГал)"
            self.ax.plot(self.model_x_km,
                          self.model_g_mgal + shift_for_plot, "r-",
                          label=label, linewidth=1.6)
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
        # Перекрытие по x
        mask = ((self.model_x_km >= self.obs_x_km.min())
                & (self.model_x_km <= self.obs_x_km.max()))
        if not mask.any():
            self.misfit_var.set("RMS misfit: нет пересечения по x")
            return
        obs_at_model = np.interp(self.model_x_km[mask],
                                  self.obs_x_km, self.obs_g_mgal)
        model_in = self.model_g_mgal[mask]
        # Автоматически снимаем константный сдвиг между моделью и наблюдением
        # (это региональная составляющая поля Буге, которая определена до константы)
        shift = float(np.mean(obs_at_model - model_in))
        diff = (model_in + shift) - obs_at_model
        rms = float(np.sqrt(np.mean(diff * diff)))
        self.misfit_var.set(
            f"RMS misfit: {rms:.3f} мГал\n"
            f"Регион. сдвиг: {shift:+.2f} мГал")


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