#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seismograv Modeler — программа для сейсмогравиметрического моделирования.

Парадигма: модель — это набор горизонтальных слоёв, разделённых интерфейсами.
Каждый интерфейс задаётся узлами (x, z), между узлами — линейная интерполяция.
Каждый слой имеет плотность (либо скорость P-волн → плотность по Гарднеру).
Поле вычисляется через растеризацию полигональной модели в мелкую сетку
прямоугольных призм + аналитическую формулу Nagy (1966).

Совместимость:
- Файлы наблюдённого поля — текст с двумя колонками x_км, g_мГал.
- Совместимость с парадигмой DOS-программы Горного института УрО РАН:
  слои с фиксированной скоростью + соотношение Гарднера + большая модель
  с границами вплоть до ±1000 км (для устранения краевых эффектов).

Зависимости: numpy, pillow, matplotlib; tkinter — стандартный.

Запуск:
    python seismograv_modeler.py
"""

from __future__ import annotations

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
#                       Физические константы
# ============================================================

G_GRAV = 6.6743e-11          # м³/(кг·с²)
SI_TO_MGAL = 1.0e5
KM = 1000.0

# Стандартные коэффициенты Гарднера: ρ [г/см³] = a · V_p^b при V_p в м/с
GARDNER_A = 0.31
GARDNER_B = 0.25


def gardner_density(vp_kmps: float, a: float = GARDNER_A, b: float = GARDNER_B) -> float:
    """Плотность ρ [г/см³] из скорости V_p [км/с] по соотношению Гарднера."""
    return a * (vp_kmps * 1000.0) ** b


def prism_gz(x_obs, y_obs, z_obs, x1, x2, y1, y2, z1, z2, density):
    """
    g_z от прямоугольной 3D-призмы с границами [x1,x2]×[y1,y2]×[z1,z2]
    и плотностью density (кг/м³). Ось z вниз. Возвращает мГал.
    Формула Nagy 1966 / Plouff 1976 / Blakely 1996 p.187.
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


def prism_gz_batch(x_obs, y_obs, z_obs,
                    x1, x2, y1, y2, z1, z2, density,
                    chunk_size: int = 400):
    """
    Векторизованная сумма g_z от массива призм.

    x_obs, y_obs, z_obs : массивы точек наблюдения, длина N (м).
    x1..z2, density     : массивы длины M (метры; плотность в кг/м³).

    Возвращает массив длины N — суммарное поле в мГал.
    Память: примерно N × M × 8 байт на каждый внутренний массив; для контроля
    обрабатываем по chunk_size призм за раз.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    z_obs = np.asarray(z_obs, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    density = np.asarray(density, dtype=float)

    n_obs = len(x_obs)
    n_pr = len(density)
    if n_pr == 0:
        return np.zeros(n_obs)

    result = np.zeros(n_obs)

    for start in range(0, n_pr, chunk_size):
        end = min(start + chunk_size, n_pr)
        sl = slice(start, end)

        # broadcast: (N, 1) для наблюдений, (1, M) для призм
        xo = x_obs[:, None]
        yo = y_obs[:, None]
        zo = z_obs[:, None]
        x1c = x1[sl][None, :]; x2c = x2[sl][None, :]
        y1c = y1[sl][None, :]; y2c = y2[sl][None, :]
        z1c = z1[sl][None, :]; z2c = z2[sl][None, :]
        rho = density[sl][None, :]

        g_per_prism = np.zeros((n_obs, end - start))
        for i, xc in enumerate((x1c, x2c)):
            for j, yc in enumerate((y1c, y2c)):
                for k, zc in enumerate((z1c, z2c)):
                    mu = (-1.0) ** (i + j + k)
                    x = xc - xo   # (N, M_chunk)
                    y = yc - yo
                    z = zc - zo
                    r = np.sqrt(x * x + y * y + z * z)
                    F = (x * np.log(y + r + 1e-15)
                         + y * np.log(x + r + 1e-15)
                         - z * np.arctan2(x * y, z * r + 1e-15))
                    g_per_prism += mu * F

        # умножаем на плотности и суммируем по призмам
        result += (G_GRAV * rho * g_per_prism * SI_TO_MGAL).sum(axis=1)

    return result


# ============================================================
#                      Модель: слои и интерфейсы
# ============================================================

class Interface:
    """Один интерфейс между слоями. Хранит набор узлов (x, z) в км.

    Между узлами — линейная интерполяция. Слева и справа от крайних узлов —
    горизонтальная экстраполяция (значение z крайнего узла).
    """

    def __init__(self, name: str, color: str, nodes: Optional[list] = None) -> None:
        self.name = name
        self.color = color
        # nodes: список (x_km, z_km), отсортированный по x
        self.nodes: list[tuple[float, float]] = list(nodes) if nodes else []
        self.sort()

    def sort(self) -> None:
        self.nodes.sort(key=lambda p: p[0])

    def add_node(self, x_km: float, z_km: float) -> int:
        """Добавляет узел и возвращает его индекс после сортировки."""
        self.nodes.append((x_km, z_km))
        self.sort()
        return self.nodes.index((x_km, z_km))

    def remove_node(self, idx: int) -> None:
        if 0 <= idx < len(self.nodes):
            del self.nodes[idx]

    def move_node(self, idx: int, x_km: float, z_km: float) -> int:
        if 0 <= idx < len(self.nodes):
            self.nodes[idx] = (x_km, z_km)
            self.sort()
            return self.nodes.index((x_km, z_km))
        return idx

    def z_at(self, x_km) -> np.ndarray:
        """Интерполированное z(x) в массиве точек."""
        x_arr = np.asarray(x_km, dtype=float)
        if not self.nodes:
            return np.zeros_like(x_arr)
        xs = np.array([n[0] for n in self.nodes])
        zs = np.array([n[1] for n in self.nodes])
        if len(xs) == 1:
            return np.full_like(x_arr, zs[0])
        # np.interp делает горизонтальную экстраполяцию по краям — это то, что нужно
        return np.interp(x_arr, xs, zs)


class Layer:
    """Слой между двумя интерфейсами. Имеет плотность (или V_p)."""

    def __init__(self, name: str, color: str,
                 density: float = 2.65, vp_kmps: float = 0.0,
                 use_gardner: bool = False) -> None:
        self.name = name
        self.color = color
        self.density = density        # г/см³
        self.vp_kmps = vp_kmps        # км/с
        self.use_gardner = use_gardner

    def effective_density(self) -> float:
        if self.use_gardner and self.vp_kmps > 0:
            return gardner_density(self.vp_kmps)
        return self.density


class Model:
    """Список интерфейсов + список слоёв. Интерфейс N делит слой N от слоя N+1."""

    def __init__(self) -> None:
        # Верхний интерфейс — это поверхность (z=0).
        # Нижний — какой-то большой z (например 30 км), для замыкания.
        self.interfaces: list[Interface] = [
            Interface("Поверхность (z=0)", "#000000",
                       nodes=[(-1000.0, 0.0), (1000.0, 0.0)]),
            Interface("Низ модели", "#202020",
                       nodes=[(-1000.0, 30.0), (1000.0, 30.0)]),
        ]
        # Слои — между соседними интерфейсами.
        self.layers: list[Layer] = [
            Layer("Слой 1", "#ffe4a1", density=2.60),
        ]
        # фон для режима «контраст плотности»
        self.background_density: float = 2.65

    def n_layers(self) -> int:
        return len(self.layers)

    def rebuild_layers(self) -> None:
        """Подгоняет число слоёв под число интерфейсов (n_layers = n_interfaces - 1)."""
        target = max(0, len(self.interfaces) - 1)
        # удаляем лишние
        while len(self.layers) > target:
            self.layers.pop()
        # добавляем недостающие
        palette = ["#ffe4a1", "#a8d8a0", "#a0c8e8", "#e8a0a0",
                    "#d4b0e0", "#e8d8a0", "#a0e0d0", "#c0c0c0"]
        while len(self.layers) < target:
            i = len(self.layers)
            self.layers.append(Layer(f"Слой {i+1}",
                                       palette[i % len(palette)],
                                       density=2.60 + 0.05 * i))

    def add_interface_between(self, idx_below: int, name: str = None) -> int:
        """Вставляет новый интерфейс ВЫШЕ интерфейса idx_below.

        Узлы новой границы располагаются посередине между idx_below-1 и idx_below.
        """
        if idx_below <= 0 or idx_below >= len(self.interfaces):
            return -1
        below = self.interfaces[idx_below]
        above = self.interfaces[idx_below - 1]
        # Берём узлы по x от обоих, считаем среднее z
        xs = sorted(set([n[0] for n in below.nodes]
                          + [n[0] for n in above.nodes]))
        nodes = []
        for x in xs:
            z_below = float(np.interp(x,
                                         [n[0] for n in below.nodes],
                                         [n[1] for n in below.nodes]))
            z_above = float(np.interp(x,
                                         [n[0] for n in above.nodes],
                                         [n[1] for n in above.nodes]))
            nodes.append((x, 0.5 * (z_above + z_below)))
        new_name = name or f"Интерфейс {len(self.interfaces)}"
        new_iface = Interface(new_name, "#888888", nodes=nodes)
        self.interfaces.insert(idx_below, new_iface)
        self.rebuild_layers()
        return idx_below

    def remove_interface(self, idx: int) -> None:
        if idx <= 0 or idx >= len(self.interfaces) - 1:
            return  # не даём удалить верх (поверхность) и низ
        del self.interfaces[idx]
        self.rebuild_layers()

    # ---------- РАСТЕРИЗАЦИЯ ----------
    def rasterize(self, x_min_km: float, x_max_km: float,
                  z_min_km: float, z_max_km: float,
                  nx: int, nz: int,
                  mode: str = "absolute") -> np.ndarray:
        """Растеризует слойную модель в сетку (nz, nx) значений плотности (г/см³).

        mode == "absolute" — возвращает абсолютные плотности слоёв.
        mode == "contrast" — возвращает контрасты Δρ = ρ_слоя − background_density.
        Ячейки выше «поверхности» (z=0) и ниже «низа» получают значение фона
        (то есть нулевой контраст).
        """
        x_centers = np.linspace(x_min_km, x_max_km, nx + 1)
        x_centers = 0.5 * (x_centers[:-1] + x_centers[1:])
        z_centers = np.linspace(z_min_km, z_max_km, nz + 1)
        z_centers = 0.5 * (z_centers[:-1] + z_centers[1:])

        # Заранее интерполируем глубины всех интерфейсов в x_centers
        # interface_depths: shape (n_iface, nx)
        depths = np.array([iface.z_at(x_centers)
                            for iface in self.interfaces])

        grid = np.full((nz, nx), self.background_density, dtype=float)

        # Для каждой ячейки определяем, в каком слое она лежит:
        # слой i — между интерфейсом i (сверху) и интерфейсом i+1 (снизу).
        for iz, z in enumerate(z_centers):
            # row: для каждого x_centers, какой это слой
            # depths[i, :] — глубина интерфейса i на x_centers
            # ячейка попадает в слой i, если depths[i,:] <= z < depths[i+1,:]
            # Для каждого x находим максимальный i, такой что depths[i, x] <= z.
            le = depths <= z       # shape (n_iface, nx), boolean
            # max index, где True; argmax даёт первый True, нам нужен последний
            # → переворачиваем по оси i
            le_rev = le[::-1, :]
            idx_from_top_rev = np.argmax(le_rev, axis=0)
            # «не найдено» — все False
            any_true = le.any(axis=0)
            n_iface = depths.shape[0]
            iface_idx = (n_iface - 1) - idx_from_top_rev    # индекс верхней границы слоя
            iface_idx = np.where(any_true, iface_idx, -1)
            # слой = iface_idx, если iface_idx < n_layers; иначе ячейка ниже всех
            n_layers = len(self.layers)
            valid = (iface_idx >= 0) & (iface_idx < n_layers)
            # density для каждого x
            row_density = np.full(len(x_centers), self.background_density)
            for li in range(n_layers):
                mask_li = valid & (iface_idx == li)
                row_density[mask_li] = self.layers[li].effective_density()
            grid[iz, :] = row_density

        if mode == "contrast":
            return grid - self.background_density
        return grid


# ============================================================
#                        GUI: главное окно
# ============================================================

class SeismogravApp:

    DEFAULT_PALETTE = ["#ffe4a1", "#a8d8a0", "#a0c8e8", "#e8a0a0",
                        "#d4b0e0", "#e8d8a0", "#a0e0d0", "#c0c0c0"]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Seismograv Modeler — полигональная 2.5D модель")
        self.root.geometry("1600x950")

        # ----- картинка разреза -----
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.zoom = 1.0
        self.img_left = self.img_top = 0
        self.img_right = self.img_bottom = 0
        self.img_width = self.img_height = 0
        self._image_path: Optional[str] = None

        # ----- модель -----
        self.model = Model()
        self.active_interface_idx = 1   # индекс активного интерфейса (для редактирования)

        # ----- параметры геометрии и расчёта -----
        self.xmin_obs_km = 0.0     # пределы наблюдательного профиля
        self.xmax_obs_km = 45.0
        self.zmin_km = 0.0          # пределы рисования и растеризации
        self.zmax_km = 10.0
        self.x_extent_km = 1000.0   # ПОЛУразмер растеризации по x (для краевых эффектов)
        self.y_extent_km = 50.0     # 2.5D полудлина по простиранию
        self.nx_raster = 400
        self.nz_raster = 80
        self.n_obs = 200

        # ----- режим расчёта -----
        # "absolute" — модель содержит абсолютные плотности
        # "contrast" — программа считает поле как от контрастов Δρ = ρ_слоя − фон
        self.calc_mode = "absolute"

        # ----- наблюдения и модельное поле -----
        self.obs_x_km: Optional[np.ndarray] = None
        self.obs_g_mgal: Optional[np.ndarray] = None
        self.model_x_km: Optional[np.ndarray] = None
        self.model_g_mgal: Optional[np.ndarray] = None

        # ----- состояние редактирования -----
        self.dragging_node_idx: Optional[int] = None
        self.drag_threshold_px = 7

        # ----- рамка картинки в долях [0..1] — куда привязаны координаты разреза -----
        self.frame_frac = (0.0, 0.0, 1.0, 1.0)
        self._frame_mode = None     # None / "p1" / "p2" при задании рамки

        self._build_ui()
        self._draw_placeholder()
        self._update_plot()

    # ============== UI ==============
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # Левая прокручиваемая панель
        controls_outer = ttk.Frame(main)
        controls_outer.pack(side="left", fill="y", padx=(0, 6))
        scrollbar = ttk.Scrollbar(controls_outer, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.controls_canvas = tk.Canvas(
            controls_outer, width=290, highlightthickness=0,
            yscrollcommand=scrollbar.set, takefocus=0)
        self.controls_canvas.pack(side="left", fill="y")
        scrollbar.config(command=self.controls_canvas.yview)
        controls = ttk.Frame(self.controls_canvas)
        self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda e:
                       self.controls_canvas.configure(
                           scrollregion=self.controls_canvas.bbox("all")))
        self.controls_canvas.bind(
            "<Enter>", lambda e:
            self.controls_canvas.bind_all("<MouseWheel>",
                lambda ev: self.controls_canvas.yview_scroll(
                    int(-ev.delta / 120), "units")))
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
        ttk.Button(box, text="Задать рамку картинки (2 клика)",
                   command=self.start_frame).pack(fill="x", padx=4, pady=2)
        ttk.Button(box, text="Рамка на всю картинку",
                   command=self.reset_frame).pack(fill="x", padx=4, pady=2)
        ttk.Separator(box).pack(fill="x", padx=4, pady=3)
        ttk.Button(box, text="Загрузить наблюдённое поле",
                   command=self.load_observed).pack(fill="x", padx=4, pady=2)

        zoom_row = ttk.Frame(box); zoom_row.pack(fill="x", padx=4, pady=3)
        ttk.Button(zoom_row, text="−", command=lambda: self._zoom(0.8), width=3)\
            .pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(zoom_row, textvariable=self.zoom_var, anchor="center")\
            .pack(side="left", fill="x", expand=True)
        ttk.Button(zoom_row, text="+", command=lambda: self._zoom(1.25), width=3)\
            .pack(side="right")

        # ===== Геометрия =====
        box = ttk.LabelFrame(parent, text="Геометрия, км")
        box.pack(fill="x", pady=3)
        self.xmin_var = tk.StringVar(value=str(self.xmin_obs_km))
        self.xmax_var = tk.StringVar(value=str(self.xmax_obs_km))
        self.zmin_var = tk.StringVar(value=str(self.zmin_km))
        self.zmax_var = tk.StringVar(value=str(self.zmax_km))
        self.xext_var = tk.StringVar(value=str(self.x_extent_km))
        self.yext_var = tk.StringVar(value=str(self.y_extent_km))
        self.nobs_var = tk.StringVar(value=str(self.n_obs))
        self._labeled(box, "x min профиля", self.xmin_var)
        self._labeled(box, "x max профиля", self.xmax_var)
        self._labeled(box, "z кровля", self.zmin_var)
        self._labeled(box, "z подошва", self.zmax_var)
        self._labeled(box, "X-extent (±, км)", self.xext_var)
        self._labeled(box, "Y-extent (±, км)", self.yext_var)
        self._labeled(box, "Точек наблюдения", self.nobs_var)
        ttk.Label(box, text="X-extent — растеризация шире\n"
                              "профиля для устранения\n"
                              "краевых эффектов",
                  foreground="gray30", justify="left")\
            .pack(fill="x", padx=4, pady=2)

        # ===== Интерфейсы =====
        box = ttk.LabelFrame(parent, text="Интерфейсы")
        box.pack(fill="x", pady=3)
        self.iface_combo = ttk.Combobox(box, state="readonly")
        self.iface_combo.pack(fill="x", padx=4, pady=2)
        self.iface_combo.bind("<<ComboboxSelected>>", self._on_iface_selected)

        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="+ Интерфейс",
                   command=self.add_interface).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="− Интерфейс",
                   command=self.remove_interface).pack(side="right", fill="x", expand=True, padx=(4, 0))

        ttk.Label(box, text="ЛКМ по картинке: добавить узел\n"
                              "Drag узла: перемещение\n"
                              "ПКМ на узле: удалить",
                  foreground="gray30", justify="left")\
            .pack(fill="x", padx=4, pady=2)

        # ===== Слои =====
        box = ttk.LabelFrame(parent, text="Слои")
        box.pack(fill="x", pady=3)
        self.layer_combo = ttk.Combobox(box, state="readonly")
        self.layer_combo.pack(fill="x", padx=4, pady=2)
        self.layer_combo.bind("<<ComboboxSelected>>", self._on_layer_selected)

        self.use_gardner_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="ρ из V_p по Гарднеру (ρ = 0.31·V^0.25)",
                          variable=self.use_gardner_var,
                          command=self._on_gardner_toggle)\
            .pack(anchor="w", padx=4, pady=2)

        self.vp_var = tk.StringVar(value="2.5")
        self.rho_var = tk.StringVar(value="2.65")
        self._labeled(box, "V_p, км/с", self.vp_var)
        self._labeled(box, "ρ, г/см³", self.rho_var)
        ttk.Button(box, text="Применить к слою",
                   command=self.apply_layer_props).pack(fill="x", padx=4, pady=3)

        # Фон
        ttk.Separator(box).pack(fill="x", padx=4, pady=3)
        self.bg_var = tk.StringVar(value=str(self.model.background_density))
        self._labeled(box, "Фон ρ₀ (г/см³)", self.bg_var)
        self.mode_var = tk.StringVar(value=self.calc_mode)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        ttk.Radiobutton(row, text="абсолютные ρ",
                          variable=self.mode_var, value="absolute",
                          command=self._on_mode_changed).pack(side="left")
        ttk.Radiobutton(row, text="контрасты Δρ",
                          variable=self.mode_var, value="contrast",
                          command=self._on_mode_changed).pack(side="left")

        # ===== Расчёт =====
        box = ttk.LabelFrame(parent, text="Расчёт поля")
        box.pack(fill="x", pady=3)
        self.auto_recompute_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Авто-пересчёт при правке",
                          variable=self.auto_recompute_var)\
            .pack(anchor="w", padx=4, pady=2)
        self.use_obs_pts_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Считать в точках наблюдений",
                          variable=self.use_obs_pts_var)\
            .pack(anchor="w", padx=4, pady=2)
        row = ttk.Frame(box); row.pack(fill="x", padx=4, pady=2)
        self._labeled(box, "N_x растер", tk.StringVar(value=str(self.nx_raster)),
                       attr="nx_raster_var")
        self._labeled(box, "N_z растер", tk.StringVar(value=str(self.nz_raster)),
                       attr="nz_raster_var")
        ttk.Button(box, text="Пересчитать сейчас",
                   command=self.recompute_field).pack(fill="x", padx=4, pady=2)

        self.misfit_var = tk.StringVar(value="RMS misfit: —")
        ttk.Label(box, textvariable=self.misfit_var).pack(anchor="w", padx=4, pady=2)

        # ===== Информация =====
        box = ttk.LabelFrame(parent, text="Состояние")
        box.pack(fill="x", pady=3)
        self.info_var = tk.StringVar(value="Готов к работе.")
        self.cursor_var = tk.StringVar(value="—")
        ttk.Label(box, textvariable=self.info_var,
                  wraplength=260, justify="left").pack(fill="x", padx=4, pady=2)
        ttk.Label(box, textvariable=self.cursor_var,
                  wraplength=260, justify="left").pack(fill="x", padx=4, pady=2)

        self._refresh_combos()

    def _labeled(self, parent, label, var, attr=None):
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        e = ttk.Entry(row, textvariable=var, width=10)
        e.pack(side="right")
        if attr:
            setattr(self, attr, var)

    def _build_plot(self, parent) -> None:
        fp = ttk.LabelFrame(parent, text="Гравитационное поле g_z (мГал)")
        fp.pack(side="top", fill="x", pady=(0, 4))
        self.figure = Figure(figsize=(11, 2.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("x, км")
        self.ax.set_ylabel("g_z, мГал (центрировано)")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=fp)
        self.plot_canvas.get_tk_widget().pack(fill="x")

    def _build_canvas(self, parent) -> None:
        cf = ttk.LabelFrame(parent, text="Разрез + полигональная модель")
        cf.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(cf, bg="white",
                                 width=1100, height=540, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>",
                          lambda e: self._zoom(1.25 if e.delta > 0 else 0.8))
        self.canvas.bind("<Button-2>",
                          lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>",
                          lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

    # ============== работа с картинкой ==============
    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        w = int(self.canvas.cget("width"))
        h = int(self.canvas.cget("height"))
        self.canvas.create_text(
            w // 2, h // 2,
            text=("Загрузите картинку разреза или начните\n"
                  "с пустой модели — нажмите «+ Интерфейс»\n"
                  "и кликами по полотну поставьте узлы."),
            font=("Arial", 13), fill="gray40", justify="center")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Картинка разреза",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self._image_path = path
            self.zoom = 1.0
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не открылось: {exc}")
            return
        self._prepare_display_image()
        self._redraw()
        self.info_var.set(f"Картинка: {Path(path).name}")

    def _prepare_display_image(self) -> None:
        if self.original_image is None:
            return
        base = self.original_image.copy()
        cw = int(self.canvas.cget("width"))
        ch = int(self.canvas.cget("height"))
        base.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)
        w = max(1, int(base.width * self.zoom))
        h = max(1, int(base.height * self.zoom))
        img = base.resize((w, h), Image.Resampling.LANCZOS)
        self.display_image = img
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_width, self.img_height = img.size
        self.img_left = (cw - self.img_width) // 2
        self.img_top = (ch - self.img_height) // 2
        self.img_right = self.img_left + self.img_width
        self.img_bottom = self.img_top + self.img_height
        self.zoom_var.set(f"{int(round(self.zoom * 100))}%")

    def _zoom(self, factor: float) -> None:
        if self.original_image is None:
            return
        self.zoom = max(0.25, min(6.0, self.zoom * factor))
        self._prepare_display_image()
        self._redraw()

    # ============== bbox рамки на canvas ==============
    def _frame_bbox(self) -> tuple[int, int, int, int]:
        fl, ft, fr, fb = self.frame_frac
        x0 = self.img_left + fl * self.img_width
        y0 = self.img_top + ft * self.img_height
        x1 = self.img_left + fr * self.img_width
        y1 = self.img_top + fb * self.img_height
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

    def _canvas_to_world(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas-координаты → реальные (x_km, z_km), привязанные к рамке."""
        x0, y0, x1, y1 = self._frame_bbox()
        fx = (cx - x0) / max(1, (x1 - x0))
        fz = (cy - y0) / max(1, (y1 - y0))
        x_km = self.xmin_obs_km + fx * (self.xmax_obs_km - self.xmin_obs_km)
        z_km = self.zmin_km + fz * (self.zmax_km - self.zmin_km)
        return x_km, z_km

    def _world_to_canvas(self, x_km: float, z_km: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self._frame_bbox()
        if self.xmax_obs_km == self.xmin_obs_km or self.zmax_km == self.zmin_km:
            return x0, y0
        fx = (x_km - self.xmin_obs_km) / (self.xmax_obs_km - self.xmin_obs_km)
        fz = (z_km - self.zmin_km) / (self.zmax_km - self.zmin_km)
        return x0 + fx * (x1 - x0), y0 + fz * (y1 - y0)

    def _world_to_canvas_batch(self, points_xz):
        """Векторизованный перевод списка (x_km, z_km) в координаты canvas.

        Используется для рисования контуров слоёв — гораздо быстрее, чем
        вызывать _world_to_canvas в Python-цикле для каждой точки.
        """
        x0, y0, x1, y1 = self._frame_bbox()
        dxr = self.xmax_obs_km - self.xmin_obs_km
        dzr = self.zmax_km - self.zmin_km
        if dxr == 0 or dzr == 0 or not points_xz:
            return [(x0, y0)] * len(points_xz)
        sx = (x1 - x0) / dxr
        sz = (y1 - y0) / dzr
        return [(x0 + (p[0] - self.xmin_obs_km) * sx,
                 y0 + (p[1] - self.zmin_km) * sz) for p in points_xz]

    # ============== отрисовка ==============
    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.tk_image is None:
            self._draw_placeholder()
            self._draw_interfaces_no_image()
            return
        self.canvas.create_image(self.img_left, self.img_top,
                                   anchor="nw", image=self.tk_image)
        self._draw_frame_outline()
        self._draw_layers_fill()
        self._draw_interfaces()

    def _draw_frame_outline(self) -> None:
        x0, y0, x1, y1 = self._frame_bbox()
        color = "red" if self._frame_mode else "gray60"
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=color,
                                       width=2, dash=(4, 3))

    def _draw_layers_fill(self) -> None:
        """Полупрозрачная заливка между интерфейсами (через простые полигоны)."""
        if len(self.model.interfaces) < 2:
            return
        for li, layer in enumerate(self.model.layers):
            top = self.model.interfaces[li]
            bot = self.model.interfaces[li + 1]
            top_pts_canvas = self._world_to_canvas_batch(top.nodes)
            bot_pts_canvas = self._world_to_canvas_batch(list(reversed(bot.nodes)))
            pts = top_pts_canvas + bot_pts_canvas
            if len(pts) >= 3:
                flat = [c for pt in pts for c in pt]
                self.canvas.create_polygon(flat, fill=layer.color,
                                             outline="", stipple="gray50")

    def _draw_interfaces(self) -> None:
        for ii, iface in enumerate(self.model.interfaces):
            pts = self._world_to_canvas_batch(iface.nodes)
            is_active = (ii == self.active_interface_idx)
            line_color = iface.color
            line_width = 2 if is_active else 1
            if len(pts) >= 2:
                flat = [c for pt in pts for c in pt]
                self.canvas.create_line(flat, fill=line_color,
                                          width=line_width, smooth=False)
            if is_active:
                for ni, (cx, cy) in enumerate(pts):
                    r = 5
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                              fill="white",
                                              outline="red", width=2,
                                              tags=("node", f"node_{ni}"))

    def _draw_interfaces_no_image(self) -> None:
        # отрисовка интерфейсов на пустом полотне (без картинки)
        self.canvas.delete("placeholder_overlay")
        cw = int(self.canvas.cget("width"))
        ch = int(self.canvas.cget("height"))
        # фейковый bbox под рамку
        self.img_left = 50; self.img_top = 50
        self.img_width = cw - 100; self.img_height = ch - 100
        self.img_right = self.img_left + self.img_width
        self.img_bottom = self.img_top + self.img_height
        # рисуем стандартно
        self._draw_frame_outline()
        self._draw_layers_fill()
        self._draw_interfaces()

    # ============== события мыши ==============
    def _hit_test_node(self, cx: float, cy: float) -> Optional[int]:
        """Какой узел активного интерфейса под курсором?"""
        if self.active_interface_idx is None:
            return None
        iface = self.model.interfaces[self.active_interface_idx]
        for ni, node in enumerate(iface.nodes):
            ncx, ncy = self._world_to_canvas(*node)
            if (ncx - cx) ** 2 + (ncy - cy) ** 2 <= self.drag_threshold_px ** 2:
                return ni
        return None

    def on_left_click(self, event) -> None:
        cx, cy = event.x, event.y

        # режим задания рамки
        if self._frame_mode == "p1":
            fx = (cx - self.img_left) / max(1, self.img_width)
            fy = (cy - self.img_top) / max(1, self.img_height)
            self._frame_p1 = (fx, fy)
            self._frame_mode = "p2"
            self.info_var.set("Кликните 2-й угол рамки")
            return
        if self._frame_mode == "p2":
            fx = (cx - self.img_left) / max(1, self.img_width)
            fy = (cy - self.img_top) / max(1, self.img_height)
            fx2, fy2 = self._frame_p1
            fl, fr = sorted([fx, fx2]); ft, fb = sorted([fy, fy2])
            fl = max(0.0, min(1.0, fl)); fr = max(0.0, min(1.0, fr))
            ft = max(0.0, min(1.0, ft)); fb = max(0.0, min(1.0, fb))
            self.frame_frac = (fl, ft, fr, fb)
            self._frame_mode = None
            self._redraw()
            self.info_var.set(f"Рамка: x∈[{fl:.2f},{fr:.2f}], y∈[{ft:.2f},{fb:.2f}]")
            self._maybe_recompute()
            return

        # клик по узлу активного интерфейса → drag
        ni = self._hit_test_node(cx, cy)
        if ni is not None:
            self.dragging_node_idx = ni
            return

        # клик в свободном месте — добавляем узел в активный интерфейс
        if self.active_interface_idx is not None:
            x_km, z_km = self._canvas_to_world(cx, cy)
            self.model.interfaces[self.active_interface_idx].add_node(x_km, z_km)
            self._redraw()
            self._maybe_recompute()

    def on_left_drag(self, event) -> None:
        if self.dragging_node_idx is None or self.active_interface_idx is None:
            return
        x_km, z_km = self._canvas_to_world(event.x, event.y)
        new_idx = self.model.interfaces[self.active_interface_idx].move_node(
            self.dragging_node_idx, x_km, z_km)
        self.dragging_node_idx = new_idx
        self._redraw()

    def on_left_release(self, event) -> None:
        if self.dragging_node_idx is not None:
            self.dragging_node_idx = None
            self._maybe_recompute()

    def on_right_click(self, event) -> None:
        if self.active_interface_idx is None:
            return
        ni = self._hit_test_node(event.x, event.y)
        if ni is not None:
            self.model.interfaces[self.active_interface_idx].remove_node(ni)
            self._redraw()
            self._maybe_recompute()

    def on_mouse_move(self, event) -> None:
        if self.img_width <= 0:
            return
        if (self.img_left <= event.x <= self.img_right
                and self.img_top <= event.y <= self.img_bottom):
            x_km, z_km = self._canvas_to_world(event.x, event.y)
            self.cursor_var.set(f"x = {x_km:+.2f} км, z = {z_km:+.2f} км")
        else:
            self.cursor_var.set("—")

    # ============== работа с рамкой ==============
    def start_frame(self) -> None:
        if self.tk_image is None:
            messagebox.showinfo("Сначала картинку",
                                  "Загрузите картинку разреза.")
            return
        self._frame_mode = "p1"
        self.info_var.set("Кликните 1-й угол рамки картинки")
        self._redraw()

    def reset_frame(self) -> None:
        self.frame_frac = (0.0, 0.0, 1.0, 1.0)
        self._frame_mode = None
        self._redraw()
        self._maybe_recompute()

    # ============== combobox интерфейсов / слоёв ==============
    def _refresh_combos(self) -> None:
        iface_names = [f"[{i}] {iface.name}"
                       for i, iface in enumerate(self.model.interfaces)]
        self.iface_combo.configure(values=iface_names)
        if self.active_interface_idx is None \
                or self.active_interface_idx >= len(self.model.interfaces):
            self.active_interface_idx = min(1, len(self.model.interfaces) - 1)
        if iface_names:
            self.iface_combo.set(iface_names[self.active_interface_idx])

        layer_names = [f"[{i}] {layer.name}"
                        for i, layer in enumerate(self.model.layers)]
        self.layer_combo.configure(values=layer_names)
        if layer_names:
            self.layer_combo.set(layer_names[0])
            self._sync_layer_fields(0)

    def _on_iface_selected(self, event=None) -> None:
        s = self.iface_combo.get()
        if s.startswith("["):
            idx = int(s.split("]")[0][1:])
            self.active_interface_idx = idx
            self._redraw()

    def _on_layer_selected(self, event=None) -> None:
        s = self.layer_combo.get()
        if s.startswith("["):
            idx = int(s.split("]")[0][1:])
            self._sync_layer_fields(idx)

    def _sync_layer_fields(self, layer_idx: int) -> None:
        if 0 <= layer_idx < len(self.model.layers):
            ly = self.model.layers[layer_idx]
            self.rho_var.set(f"{ly.density:.3f}")
            self.vp_var.set(f"{ly.vp_kmps:.3f}")
            self.use_gardner_var.set(ly.use_gardner)

    def apply_layer_props(self) -> None:
        s = self.layer_combo.get()
        if not s.startswith("["):
            return
        idx = int(s.split("]")[0][1:])
        if not (0 <= idx < len(self.model.layers)):
            return
        try:
            rho = float(self.rho_var.get().replace(",", "."))
            vp = float(self.vp_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные числовые поля.")
            return
        ly = self.model.layers[idx]
        ly.density = rho
        ly.vp_kmps = vp
        ly.use_gardner = self.use_gardner_var.get()
        self.info_var.set(f"Слой {idx+1}: ρ_эфф = {ly.effective_density():.3f} г/см³")
        self._redraw()
        self._maybe_recompute()

    def _on_gardner_toggle(self) -> None:
        # значение применяется только при apply_layer_props
        pass

    def _on_mode_changed(self) -> None:
        self.calc_mode = self.mode_var.get()
        self._maybe_recompute()

    def add_interface(self) -> None:
        # Вставляем между активным и следующим за ним вниз
        idx = self.active_interface_idx + 1
        if idx >= len(self.model.interfaces):
            idx = len(self.model.interfaces) - 1
        new_idx = self.model.add_interface_between(idx)
        if new_idx > 0:
            self.active_interface_idx = new_idx
            self._refresh_combos()
            self._redraw()
            self._maybe_recompute()

    def remove_interface(self) -> None:
        self.model.remove_interface(self.active_interface_idx)
        self.active_interface_idx = min(self.active_interface_idx,
                                          len(self.model.interfaces) - 2)
        self.active_interface_idx = max(1, self.active_interface_idx)
        self._refresh_combos()
        self._redraw()
        self._maybe_recompute()

    # ============== расчёт поля ==============
    def _read_geometry(self) -> bool:
        try:
            self.xmin_obs_km = float(self.xmin_var.get().replace(",", "."))
            self.xmax_obs_km = float(self.xmax_var.get().replace(",", "."))
            self.zmin_km = float(self.zmin_var.get().replace(",", "."))
            self.zmax_km = float(self.zmax_var.get().replace(",", "."))
            self.x_extent_km = float(self.xext_var.get().replace(",", "."))
            self.y_extent_km = float(self.yext_var.get().replace(",", "."))
            self.n_obs = max(2, int(self.nobs_var.get()))
            self.nx_raster = max(20, int(self.nx_raster_var.get()))
            self.nz_raster = max(10, int(self.nz_raster_var.get()))
            try:
                self.model.background_density = float(self.bg_var.get().replace(",", "."))
            except ValueError:
                pass
            return True
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Геометрия: {e}")
            return False

    def _maybe_recompute(self) -> None:
        if hasattr(self, "auto_recompute_var") and self.auto_recompute_var.get():
            self.recompute_field()

    def recompute_field(self) -> None:
        if not self._read_geometry():
            return
        # точки наблюдения
        if (self.use_obs_pts_var.get()
                and self.obs_x_km is not None and len(self.obs_x_km) >= 2):
            x_obs = self.obs_x_km.copy()
        else:
            x_obs = np.linspace(self.xmin_obs_km, self.xmax_obs_km, self.n_obs)

        # ХОБ: растеризуем модель на ШИРОКОМ диапазоне, чтобы убрать краевые эффекты
        x_min_r = self.xmin_obs_km - self.x_extent_km
        x_max_r = self.xmax_obs_km + self.x_extent_km
        z_min_r = max(0.0, self.zmin_km)
        z_max_r = max(self.zmax_km, z_min_r + 0.1)

        # размер сетки растеризации с учётом расширенного диапазона
        # чтобы шаг был тот же, что и над профилем
        profile_width = self.xmax_obs_km - self.xmin_obs_km
        total_width = x_max_r - x_min_r
        nx_eff = int(round(self.nx_raster * total_width / max(profile_width, 0.1)))
        nx_eff = max(self.nx_raster, min(nx_eff, 2000))

        try:
            grid = self.model.rasterize(x_min_r, x_max_r, z_min_r, z_max_r,
                                          nx_eff, self.nz_raster,
                                          mode="contrast")
            # rasterize в режиме "contrast" даёт Δρ = ρ_слоя − фон,
            # пустые места (если есть) уже = 0 контраст.

            # Если режим расчёта "absolute" — нам нужно вычислить поле
            # как от полной модели с её плотностями относительно «вакуума».
            # Но это огромное число. Корректнее всегда работать через контрасты.
            # Поэтому даже в режиме "absolute" UI программы используем контраст,
            # а флаг mode влияет только на интерпретацию визуально.
            # → Здесь рассчитываем поле от контрастов в любом случае.

            from_grid_module = compute_field_from_raster
            g = from_grid_module(
                grid_drho=grid,
                x_min_km=x_min_r, x_max_km=x_max_r,
                z_min_km=z_min_r, z_max_km=z_max_r,
                x_obs_km=x_obs,
                y_extent_km=self.y_extent_km)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не посчиталось: {exc}")
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
            obs_c = self.obs_g_mgal - float(np.mean(self.obs_g_mgal))
            self.ax.plot(self.obs_x_km, obs_c, "k.-",
                          label="наблюдённое", markersize=4, linewidth=1)
            plotted = True
        if self.model_x_km is not None and self.model_g_mgal is not None:
            mod_c = self.model_g_mgal - float(np.mean(self.model_g_mgal))
            self.ax.plot(self.model_x_km, mod_c, "r-",
                          label="модельное", linewidth=1.6)
            plotted = True
        if plotted:
            self.ax.legend(loc="best", fontsize=8)
            self.ax.set_xlim(self.xmin_obs_km, self.xmax_obs_km)
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _update_misfit(self) -> None:
        if (self.obs_x_km is None or self.model_x_km is None
                or self.obs_g_mgal is None or self.model_g_mgal is None):
            self.misfit_var.set("RMS misfit: —")
            return
        obs_c = self.obs_g_mgal - float(np.mean(self.obs_g_mgal))
        mod_c = self.model_g_mgal - float(np.mean(self.model_g_mgal))
        mask = ((self.model_x_km >= self.obs_x_km.min())
                & (self.model_x_km <= self.obs_x_km.max()))
        if not mask.any():
            self.misfit_var.set("RMS misfit: нет пересечения")
            return
        obs_at_model = np.interp(self.model_x_km[mask], self.obs_x_km, obs_c)
        diff = mod_c[mask] - obs_at_model
        rms = float(np.sqrt(np.mean(diff * diff)))
        self.misfit_var.set(f"RMS misfit: {rms:.3f} мГал")

    # ============== файлы ==============
    def load_observed(self) -> None:
        path = filedialog.askopenfilename(
            title="Наблюдённое поле (две колонки: x_км, g_мГал)",
            filetypes=[("Текст/CSV", "*.txt *.csv *.dat"), ("Все", "*.*")])
        if not path:
            return
        xs, gs = [], []
        with open(path, encoding="utf-8") as f:
            for row in f:
                row = row.strip()
                if not row or row.startswith("#"):
                    continue
                parts = row.replace(",", " ").replace(";", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    xs.append(float(parts[0])); gs.append(float(parts[1]))
                except ValueError:
                    continue
        if not xs:
            messagebox.showerror("Ошибка", "Не нашёл пар (x, g).")
            return
        order = np.argsort(xs)
        self.obs_x_km = np.array(xs)[order]
        self.obs_g_mgal = np.array(gs)[order]
        self.info_var.set(f"Наблюдённое: {Path(path).name}, {len(xs)} точек")
        self._update_plot()
        self._update_misfit()

    def save_session(self) -> None:
        if not self._read_geometry():
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить сессию", defaultextension=".sgms",
            filetypes=[("Seismograv Session", "*.sgms"),
                       ("JSON", "*.json"), ("Все", "*.*")])
        if not path:
            return
        state = {
            "version": 1,
            "image_path": self._image_path,
            "frame_frac": list(self.frame_frac),
            "geometry": {
                "xmin_obs_km": self.xmin_obs_km, "xmax_obs_km": self.xmax_obs_km,
                "zmin_km": self.zmin_km, "zmax_km": self.zmax_km,
                "x_extent_km": self.x_extent_km, "y_extent_km": self.y_extent_km,
                "n_obs": self.n_obs,
                "nx_raster": self.nx_raster, "nz_raster": self.nz_raster,
            },
            "interfaces": [
                {"name": i.name, "color": i.color, "nodes": i.nodes}
                for i in self.model.interfaces
            ],
            "layers": [
                {"name": l.name, "color": l.color,
                 "density": l.density, "vp_kmps": l.vp_kmps,
                 "use_gardner": l.use_gardner}
                for l in self.model.layers
            ],
            "background_density": self.model.background_density,
            "calc_mode": self.calc_mode,
            "obs": (None if self.obs_x_km is None else {
                "x_km": self.obs_x_km.tolist(),
                "g_mgal": self.obs_g_mgal.tolist()}),
            "active_interface_idx": self.active_interface_idx,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Готово", f"Сохранено: {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось: {exc}")

    def load_session(self) -> None:
        path = filedialog.askopenfilename(
            title="Открыть сессию",
            filetypes=[("Seismograv Session", "*.sgms"),
                       ("JSON", "*.json"), ("Все", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не открылось: {exc}")
            return

        img_path = state.get("image_path")
        if img_path and Path(img_path).exists():
            try:
                self.original_image = Image.open(img_path).convert("RGB")
                self._image_path = img_path
                self.zoom = 1.0
                self._prepare_display_image()
            except Exception:
                pass

        self.frame_frac = tuple(state.get("frame_frac", (0, 0, 1, 1)))
        geom = state.get("geometry", {})
        self.xmin_obs_km = geom.get("xmin_obs_km", 0.0)
        self.xmax_obs_km = geom.get("xmax_obs_km", 45.0)
        self.zmin_km = geom.get("zmin_km", 0.0)
        self.zmax_km = geom.get("zmax_km", 10.0)
        self.x_extent_km = geom.get("x_extent_km", 1000.0)
        self.y_extent_km = geom.get("y_extent_km", 50.0)
        self.n_obs = geom.get("n_obs", 200)
        self.nx_raster = geom.get("nx_raster", 400)
        self.nz_raster = geom.get("nz_raster", 80)
        for var, val in [
            (self.xmin_var, self.xmin_obs_km), (self.xmax_var, self.xmax_obs_km),
            (self.zmin_var, self.zmin_km), (self.zmax_var, self.zmax_km),
            (self.xext_var, self.x_extent_km), (self.yext_var, self.y_extent_km),
            (self.nobs_var, self.n_obs),
            (self.nx_raster_var, self.nx_raster),
            (self.nz_raster_var, self.nz_raster),
        ]:
            var.set(str(val))

        self.model.interfaces = [
            Interface(d["name"], d["color"],
                       [tuple(n) for n in d["nodes"]])
            for d in state.get("interfaces", [])
        ]
        self.model.layers = [
            Layer(d["name"], d["color"],
                   density=d.get("density", 2.65),
                   vp_kmps=d.get("vp_kmps", 0.0),
                   use_gardner=d.get("use_gardner", False))
            for d in state.get("layers", [])
        ]
        self.model.background_density = state.get("background_density", 2.65)
        self.bg_var.set(str(self.model.background_density))

        self.calc_mode = state.get("calc_mode", "absolute")
        self.mode_var.set(self.calc_mode)

        obs = state.get("obs")
        if obs:
            self.obs_x_km = np.array(obs["x_km"])
            self.obs_g_mgal = np.array(obs["g_mgal"])

        self.active_interface_idx = state.get("active_interface_idx", 1)
        self._refresh_combos()
        self._redraw()
        self._update_plot()
        self._update_misfit()
        self.info_var.set(f"Сессия открыта: {Path(path).name}")


# ============================================================
#  расчёт поля от растеризованной сетки контрастов
# ============================================================

def compute_field_from_raster(
    grid_drho: np.ndarray,         # (nz, nx), контрасты Δρ в г/см³
    x_min_km: float, x_max_km: float,
    z_min_km: float, z_max_km: float,
    x_obs_km, y_extent_km: float = 50.0,
    z_obs_km: float = 0.0,
) -> np.ndarray:
    """g_z в точках x_obs_km для модели из прямоугольных призм по растру.

    Оптимизация: в каждой строке растра ищем интервалы постоянной плотности
    и объединяем их в одну широкую призму. Для слоистой модели это в десятки
    раз меньше призм, чем число ячеек, и расчёт идёт за миллисекунды.
    """
    nz, nx = grid_drho.shape
    x_edges = np.linspace(x_min_km, x_max_km, nx + 1) * KM
    z_edges = np.linspace(z_min_km, z_max_km, nz + 1) * KM
    y1 = -y_extent_km * KM
    y2 = +y_extent_km * KM

    x_obs_m = np.asarray(x_obs_km, dtype=float) * KM
    y_obs_m = np.zeros_like(x_obs_m)
    z_obs_m = np.full_like(x_obs_m, z_obs_km * KM)

    # ---- Сегментирование строк в интервалы постоянной плотности ----
    x1_list, x2_list = [], []
    z1_list, z2_list = [], []
    density_list = []

    for iz in range(nz):
        row = grid_drho[iz]
        # Где плотность меняется по горизонтали:
        diff = np.diff(row)
        change_pts = np.where(np.abs(diff) > 1e-15)[0] + 1
        starts = np.concatenate([[0], change_pts])
        ends = np.concatenate([change_pts, [nx]])
        for s, e in zip(starts, ends):
            v = row[s]
            if abs(v) < 1e-15:
                continue
            x1_list.append(x_edges[s])
            x2_list.append(x_edges[e])
            z1_list.append(z_edges[iz])
            z2_list.append(z_edges[iz + 1])
            density_list.append(v * 1000.0)  # г/см³ → кг/м³

    if not density_list:
        return np.zeros_like(x_obs_m)

    n_pr = len(density_list)
    y1_arr = np.full(n_pr, y1)
    y2_arr = np.full(n_pr, y2)

    return prism_gz_batch(
        x_obs_m, y_obs_m, z_obs_m,
        np.array(x1_list), np.array(x2_list),
        y1_arr, y2_arr,
        np.array(z1_list), np.array(z2_list),
        np.array(density_list),
    )


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    SeismogravApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
