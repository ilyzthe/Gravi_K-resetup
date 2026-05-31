#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magnetic Modeler — интерактивный калькулятор магнитных аномалий
от типовых тел (сфера, горизонтальный цилиндр, точечный полюс,
тонкий вертикальный пласт) с возможностью перетаскивания тела
прямо на разрезе и независимого управления вектором намагниченности
тела и вектором поля Земли.

Компоненты поля:
  Z  — вертикальная составляющая аномального поля (положительная вниз)
  H  — горизонтальная составляющая вдоль профиля
  ΔT — полная аномалия, проекция аномального вектора B на направление T₀

Все расчёты в относительных единицах (J безразмерное).
Координаты в метрах. Глубина положительна вниз.

Зависимости:
    pip install numpy matplotlib
    (tkinter входит в стандартную поставку Python)

Запуск:
    python magnetic_modeler.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches


BODY_TYPES = [
    "Сфера (3D)",
    "Горизонтальный цилиндр (2D)",
    "Точечный полюс",
    "Вертикальный пласт (тонкий)",
    "Куб (2D квадратный брус)",
    "Горизонтальный пласт",
    "Полубесконечная ступенька",
    "Наклонный пласт (ограниченный)",
    "Наклонный пласт (бесконечный)",
]


# ============================================================
#                Физика магнитных аномалий
# ============================================================
#
# Система координат: ось x — вдоль профиля, ось y — поперёк профиля,
# ось z — вертикально вниз. Точки наблюдения на z=0.
# Тело с центром в (x0, 0, h), h > 0.
#
# Единичный вектор намагниченности тела:
#   m̂ = (cos I_m · cos D_m,  cos I_m · sin D_m,  sin I_m)
# Аналогично для вектора поля Земли t̂ через (I_T, D_T).
#
# D_m, D_T — углы между проекцией соответствующего вектора на
# горизонтальную плоскость и осью профиля (положительные вправо).
#
# Формулы:
#   Сфера → 3D точечный диполь
#       B = (m · V) · [3(m̂·r̂)r̂ - m̂] / r³,   V = (4/3)π·a³
#   Горизонтальный цилиндр → 2D линейный диполь
#       B = (m · A) · [2(m̂·r̂)r̂ - m̂] / r²,   A = π·a²
#   Точечный полюс → монополь
#       B = m · r̂ / r²
#   Вертикальный пласт (тонкий, полубесконечный вниз) →
#   2D «полюс» на верхней кромке, его интенсивность
#   пропорциональна вертикальной компоненте намагниченности
#
#   Куб, горизонтальный пласт, полубесконечная ступенька →
#   все три — 2D прямоугольные брусья (бесконечные по простиранию y)
#   с однородной намагниченностью. Поле получено аналитическим
#   интегрированием 2D-диполя по прямоугольному сечению:
#       I_A = ∫∫ (u² - v²)/r⁴ du dv = арктангенсы по углам
#       I_B = ∫∫ 2uv/r⁴ du dv = логарифм отношения квадратов r
#   где u = x' − x_obs, v = z' (глубина), r² = u² + v².
#   Тогда Bx = J·(m_x·I_A + m_z·I_B), Bz = J·(m_x·I_B − m_z·I_A).
#   Ступенька — предел общей формулы при правой границе → +∞.
# ============================================================


def _rect_2d_field(x_obs, x_L, x_R, z_T, z_B, J, mx, mz):
    """
    Поле 2D-прямоугольного бруса с однородной намагниченностью.

    Брус занимает x' ∈ [x_L, x_R], z' ∈ [z_T, z_B] (z вниз),
    бесконечный по y. Намагниченность J·(mx, ·, mz) (y-компонента
    не вносит вклада на профиле y=0). Возвращает (Bx, Bz).
    """
    u_L = x_L - x_obs
    u_R = x_R - x_obs

    # I_A: интеграл от (u²−v²)/r⁴
    I_A = (np.arctan2(z_T, u_R) - np.arctan2(z_B, u_R)
           + np.arctan2(z_B, u_L) - np.arctan2(z_T, u_L))

    # I_B: интеграл от 2uv/r⁴ — логарифм отношения квадратов r.
    eps = 1e-12
    num = (u_L * u_L + z_B * z_B) * (u_R * u_R + z_T * z_T)
    den = (u_R * u_R + z_B * z_B) * (u_L * u_L + z_T * z_T)
    I_B = 0.5 * np.log(np.maximum(num, eps) / np.maximum(den, eps))

    Bx = J * (mx * I_A + mz * I_B)
    Bz = J * (mx * I_B - mz * I_A)
    return Bx, Bz


def _rect_2d_step_field(x_obs, x_L, z_T, z_B, J, mx, mz):
    """
    Поле 2D-полубесконечной ступеньки: брус x' ∈ [x_L, +∞),
    z' ∈ [z_T, z_B]. Получается из _rect_2d_field в пределе u_R → +∞.
    """
    u_L = x_L - x_obs

    I_A = np.arctan2(z_B, u_L) - np.arctan2(z_T, u_L)

    eps = 1e-12
    num = u_L * u_L + z_B * z_B
    den = u_L * u_L + z_T * z_T
    I_B = 0.5 * np.log(np.maximum(num, eps) / np.maximum(den, eps))

    Bx = J * (mx * I_A + mz * I_B)
    Bz = J * (mx * I_B - mz * I_A)
    return Bx, Bz


def _rect_2d_general(s_obs, w_obs, s_L, s_R, w_T, w_B, J, m_s, m_w):
    """
    Поле 2D-прямоугольника в общем положении: тело занимает
    s ∈ [s_L, s_R], w ∈ [w_T, w_B]; точка наблюдения (s_obs, w_obs)
    может быть произвольной (в т.ч. не на оси w=0). Намагниченность
    в локальной системе (s, w) — J·(m_s, m_w). Возвращает (Bs, Bw).

    Получено через первообразные −arctan(η/ξ) и −(1/2)ln(ξ²+η²),
    где ξ = s_obs − s, η = w_obs − w. Формула корректна для любых
    знаков a, b, c, d при ненулевых a, b (главное значение arctan).
    """
    eps = 1e-12
    a = s_obs - s_L
    b = s_obs - s_R
    c = w_obs - w_T
    d = w_obs - w_B
    # Защита от деления на ноль (наблюдение точно над краем тела
    # в локальной системе — пограничный случай)
    a = np.where(np.abs(a) < eps, eps, a)
    b = np.where(np.abs(b) < eps, eps, b)

    F1 = (-np.arctan(c / a) + np.arctan(d / a)
          + np.arctan(c / b) - np.arctan(d / b))
    num = (a * a + d * d) * (b * b + c * c)
    den = (a * a + c * c) * (b * b + d * d)
    F2 = 0.5 * np.log(np.maximum(num, eps) / np.maximum(den, eps))

    Bs = J * (m_s * F1 + m_w * F2)
    Bw = J * (m_s * F2 - m_w * F1)
    return Bs, Bw


def _rect_2d_general_inf(s_obs, w_obs, s_L, w_T, w_B, J, m_s, m_w):
    """
    То же, что _rect_2d_general, но в пределе s_R → +∞:
    тело s ∈ [s_L, +∞), w ∈ [w_T, w_B].
    """
    eps = 1e-12
    a = s_obs - s_L
    c = w_obs - w_T
    d = w_obs - w_B
    a = np.where(np.abs(a) < eps, eps, a)

    F1 = np.arctan(d / a) - np.arctan(c / a)
    num = a * a + d * d
    den = a * a + c * c
    F2 = 0.5 * np.log(np.maximum(num, eps) / np.maximum(den, eps))

    Bs = J * (m_s * F1 + m_w * F2)
    Bw = J * (m_s * F2 - m_w * F1)
    return Bs, Bw

def compute_anomaly(
    x_obs: np.ndarray,
    x0: float,
    h: float,
    body_type: str,
    size: float,
    J: float,
    Im_deg: float, Dm_deg: float,
    IT_deg: float, DT_deg: float,
    theta_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (Z, H, ΔT) — массивы аномального поля в точках x_obs."""
    Im = np.radians(Im_deg)
    Dm = np.radians(Dm_deg)
    IT = np.radians(IT_deg)
    DT = np.radians(DT_deg)

    # Единичные векторы намагниченности тела и поля Земли
    mx = np.cos(Im) * np.cos(Dm)
    my = np.cos(Im) * np.sin(Dm)
    mz = np.sin(Im)
    tx = np.cos(IT) * np.cos(DT)
    ty = np.cos(IT) * np.sin(DT)
    tz = np.sin(IT)

    # Вектор от центра тела к точке наблюдения. dy = 0 (профиль y = 0)
    dx = x_obs - x0
    dz = -h
    r2 = dx * dx + h * h
    r2 = np.maximum(r2, 1e-9)
    r = np.sqrt(r2)

    Bx = np.zeros_like(dx)
    By = np.zeros_like(dx)
    Bz = np.zeros_like(dx)

    if body_type.startswith("Сфера"):
        # 3D диполь
        V = (4.0 / 3.0) * np.pi * size ** 3
        moment = J * V
        mdotr = mx * dx + mz * dz                  # m·r (dy=0)
        Bx = moment * (3 * mdotr * dx - mx * r2) / r ** 5
        By = moment * (-my * r2) / r ** 5
        Bz = moment * (3 * mdotr * dz - mz * r2) / r ** 5

    elif body_type.startswith("Горизонтальный цилиндр"):
        # 2D диполь (бесконечный по простиранию y)
        A = np.pi * size ** 2
        m_lin = J * A
        mdotr = mx * dx + mz * dz
        Bx = m_lin * (2 * mdotr * dx - mx * r2) / r ** 4
        Bz = m_lin * (2 * mdotr * dz - mz * r2) / r ** 4
        # By = 0 — поле 2D-тела не имеет y-компоненты вне самого тела

    elif body_type.startswith("Точечный"):
        # Монополь: B = m · r̂ / r²
        m_strength = J * size * size
        Bx = m_strength * dx / r ** 3
        Bz = m_strength * dz / r ** 3
        # Для монополя направление вектора намагниченности игнорируется

    elif body_type.startswith("Вертикальный"):
        # Тонкий полубесконечный вертикальный пласт.
        # Эквивалентен 2D-линии полюсов на верхней кромке (z=h),
        # с интенсивностью, пропорциональной вертикальной составляющей J.
        m_strength = J * size * np.sin(Im)
        Bx = 2.0 * m_strength * dx / r2
        Bz = 2.0 * m_strength * dz / r2

    elif body_type.startswith("Куб"):
        # 2D квадратный брус: сечение size×size, центр в (x0, h)
        # (size трактуется как полусторона, чтобы согласоваться
        # с радиусом сферы/цилиндра).
        half = size
        z_top = max(h - half, 1e-3)
        z_bot = h + half
        Bx, Bz = _rect_2d_field(
            x_obs, x0 - half, x0 + half, z_top, z_bot, J, mx, mz)

    elif body_type.startswith("Горизонтальный пласт"):
        # 2D пласт конечной протяжённости: тонкий по z, широкий по x.
        # Полутолщина = size, полуширина = 5·size (соотношение 1:5).
        half_w = 5.0 * size
        half_t = size
        z_top = max(h - half_t, 1e-3)
        z_bot = h + half_t
        Bx, Bz = _rect_2d_field(
            x_obs, x0 - half_w, x0 + half_w, z_top, z_bot, J, mx, mz)

    elif body_type.startswith("Полубесконечная"):
        # Полубесконечный по x пласт: x' > x0, толщина 2·size по z,
        # центр по z на глубине h. x0 — положение вертикального уступа.
        half_t = size
        z_top = max(h - half_t, 1e-3)
        z_bot = h + half_t
        Bx, Bz = _rect_2d_step_field(
            x_obs, x0, z_top, z_bot, J, mx, mz)

    elif body_type.startswith("Наклонный пласт (ограниченный)"):
        # Наклонный брус: повёрнутый прямоугольник.
        # θ ∈ [0°, 180°] — угол падения от горизонтали (от +x), считая по часовой
        # стрелке в плоскости (x, z), z вниз. θ=0° → горизонтальный пласт вправо,
        # θ=90° → вертикальный вниз, θ=180° → горизонтальный пласт влево.
        # Полудлина вдоль падения 5·size, полутолщина = size
        # (то же соотношение, что у горизонтального пласта).
        th = np.radians(theta_deg)
        cs, sn = np.cos(th), np.sin(th)
        half_L = 5.0 * size
        half_t = size
        # Наблюдение в системе тела (повёрнутой)
        dx0 = x_obs - x0
        dz0 = -h
        s_obs = dx0 * cs + dz0 * sn
        w_obs = -dx0 * sn + dz0 * cs
        # Намагниченность в системе тела
        m_s = mx * cs + mz * sn
        m_w = -mx * sn + mz * cs
        Bs, Bw = _rect_2d_general(
            s_obs, w_obs, -half_L, half_L, -half_t, half_t,
            J, m_s, m_w)
        # Обратный поворот в лабораторную систему
        Bx = Bs * cs - Bw * sn
        Bz = Bs * sn + Bw * cs

    elif body_type.startswith("Наклонный пласт (бесконечный)"):
        # Полубесконечный наклонный пласт: от верхней кромки (x0, h)
        # уходит вниз по падению до бесконечности. (x0, h) — центр
        # верхней кромки. Полутолщина = size.
        th = np.radians(theta_deg)
        cs, sn = np.cos(th), np.sin(th)
        half_t = size
        dx0 = x_obs - x0
        dz0 = -h
        s_obs = dx0 * cs + dz0 * sn
        w_obs = -dx0 * sn + dz0 * cs
        m_s = mx * cs + mz * sn
        m_w = -mx * sn + mz * cs
        # В системе тела: s ≥ 0 (вниз по падению), w ∈ [-half_t, half_t]
        Bs, Bw = _rect_2d_general_inf(
            s_obs, w_obs, 0.0, -half_t, half_t,
            J, m_s, m_w)
        Bx = Bs * cs - Bw * sn
        Bz = Bs * sn + Bw * cs

    Z = Bz                              # положительная вниз
    H = Bx                              # вдоль профиля
    T = Bx * tx + By * ty + Bz * tz     # ΔT ≈ B · t̂ (малая аномалия)
    return Z, H, T


# ============================================================
#                            GUI
# ============================================================

class MagneticModelerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Magnetic Modeler — магнитные аномалии типовых тел")
        self.root.geometry("1350x900")

        # ----- защита от рекурсивных обновлений и drag-флаг -----
        self._updating = False
        self._dragging = False

        # ----- параметры профиля -----
        self.x_min = -300.0
        self.x_max = 300.0
        self.z_max = 300.0
        self.n_obs = 400

        # ----- Tk-переменные параметров модели -----
        self.body_var = tk.StringVar(value=BODY_TYPES[0])
        self.x0_var = tk.DoubleVar(value=0.0)
        self.h_var = tk.DoubleVar(value=80.0)
        self.size_var = tk.DoubleVar(value=20.0)
        self.J_var = tk.DoubleVar(value=1.0)
        self.Im_var = tk.DoubleVar(value=65.0)
        self.Dm_var = tk.DoubleVar(value=0.0)
        self.IT_var = tk.DoubleVar(value=65.0)
        self.DT_var = tk.DoubleVar(value=0.0)
        self.theta_var = tk.DoubleVar(value=45.0)

        self.couple_var = tk.BooleanVar(value=True)
        self.show_Z_var = tk.BooleanVar(value=True)
        self.show_H_var = tk.BooleanVar(value=True)
        self.show_T_var = tk.BooleanVar(value=True)
        self.show_arrows_var = tk.BooleanVar(value=True)
        self.equal_aspect_var = tk.BooleanVar(value=False)

        self.xmin_var = tk.StringVar(value=str(self.x_min))
        self.xmax_var = tk.StringVar(value=str(self.x_max))
        self.zmax_var = tk.StringVar(value=str(self.z_max))
        self.nobs_var = tk.StringVar(value=str(self.n_obs))

        # ссылки на слайдеры I_m/D_m (для disable в индуктивном режиме)
        self.Im_scale: Optional[ttk.Scale] = None
        self.theta_scale: Optional[ttk.Scale] = None
        self.Dm_scale: Optional[ttk.Scale] = None

        # последние посчитанные кривые (для экспорта)
        self._last_curves: Optional[tuple] = None

        self._build_ui()
        self._update()

    # ============== Построение интерфейса ==============
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # === Левая колонка: scrollable controls ===
        controls_outer = ttk.Frame(main)
        controls_outer.pack(side="left", fill="y", padx=(0, 6))

        sb = ttk.Scrollbar(controls_outer, orient="vertical")
        sb.pack(side="right", fill="y")

        self.ctrl_canvas = tk.Canvas(
            controls_outer, width=300, highlightthickness=0,
            yscrollcommand=sb.set, takefocus=0)
        self.ctrl_canvas.pack(side="left", fill="y")
        sb.config(command=self.ctrl_canvas.yview)

        controls = ttk.Frame(self.ctrl_canvas)
        self.ctrl_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind(
            "<Configure>",
            lambda e: self.ctrl_canvas.configure(
                scrollregion=self.ctrl_canvas.bbox("all")))

        # колёсико крутит панель только когда курсор над ней
        def _wheel(event):
            self.ctrl_canvas.yview_scroll(int(-event.delta / 120), "units")
        self.ctrl_canvas.bind(
            "<Enter>",
            lambda e: self.ctrl_canvas.bind_all("<MouseWheel>", _wheel))
        self.ctrl_canvas.bind(
            "<Leave>",
            lambda e: self.ctrl_canvas.unbind_all("<MouseWheel>"))

        # === Правая колонка: matplotlib figure ===
        # Создаём фигуру и оси ДО построения контролов, иначе
        # начальный _on_couple_changed() в _build_controls дёрнет
        # _update(), которому нужны self.ax_field/self.ax_section.
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor="white")
        gs = self.figure.add_gridspec(
            2, 1, height_ratios=[1.0, 1.2], hspace=0.08,
            left=0.08, right=0.97, top=0.93, bottom=0.07)
        self.ax_field = self.figure.add_subplot(gs[0])
        self.ax_section = self.figure.add_subplot(gs[1], sharex=self.ax_field)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        # мышь над разрезом
        self.plot_canvas.mpl_connect("button_press_event", self._on_press)
        self.plot_canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.plot_canvas.mpl_connect("button_release_event", self._on_release)

        # Контролы строим в последнюю очередь — их колбэки
        # могут уже спокойно перерисовывать фигуру.
        self._build_controls(controls)

    def _build_controls(self, parent) -> None:
        # === Тип тела ===
        box = ttk.LabelFrame(parent, text="Тип тела")
        box.pack(fill="x", pady=3)
        for bt in BODY_TYPES:
            ttk.Radiobutton(box, text=bt, variable=self.body_var,
                            value=bt, command=self._update)\
                .pack(anchor="w", padx=4)

        # === Геометрия тела ===
        box = ttk.LabelFrame(parent, text="Положение и размер тела")
        box.pack(fill="x", pady=3)
        self._slider(box, "x₀, м",       self.x0_var,   -500.0, 500.0)
        self._slider(box, "h (глубина), м", self.h_var,     5.0, 400.0)
        self._slider(box, "размер a, м", self.size_var,    1.0,  80.0)
        self.theta_scale = self._slider(
            box, "θ (угол падения), °", self.theta_var, 0.0, 180.0)

        # === Вектор намагниченности тела ===
        box = ttk.LabelFrame(parent, text="Вектор намагниченности тела  J")
        box.pack(fill="x", pady=3)
        self._slider(box, "J, у.е.",      self.J_var,    0.05,  10.0)
        self.Im_scale = self._slider(box, "I_m (накл.), °", self.Im_var, -90.0, 90.0)
        self.Dm_scale = self._slider(box, "D_m (скл.), °",  self.Dm_var, -180.0, 180.0)

        # === Поле Земли ===
        box = ttk.LabelFrame(parent, text="Поле Земли  T₀  (для ΔT)")
        box.pack(fill="x", pady=3)
        ttk.Checkbutton(box, text="J ∥ T₀  (индуктивная намагниченность)",
                        variable=self.couple_var,
                        command=self._on_couple_changed)\
            .pack(anchor="w", padx=4, pady=(2, 0))
        self._slider(box, "I_T, °", self.IT_var,  -90.0, 90.0)
        self._slider(box, "D_T, °", self.DT_var, -180.0, 180.0)

        # === Видимость кривых ===
        box = ttk.LabelFrame(parent, text="Кривые на графике")
        box.pack(fill="x", pady=3)
        for txt, var in [
            ("Z — вертикальная",            self.show_Z_var),
            ("H — горизонтальная (вдоль)",  self.show_H_var),
            ("ΔT — полная аномалия",        self.show_T_var),
        ]:
            ttk.Checkbutton(box, text=txt, variable=var,
                            command=self._update).pack(anchor="w", padx=4)
        ttk.Separator(box).pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(box, text="Стрелки J и T₀ на разрезе",
                        variable=self.show_arrows_var,
                        command=self._update).pack(anchor="w", padx=4)
        ttk.Checkbutton(box, text="Равный масштаб осей разреза",
                        variable=self.equal_aspect_var,
                        command=self._update).pack(anchor="w", padx=4)

        # === Профиль ===
        box = ttk.LabelFrame(parent, text="Профиль")
        box.pack(fill="x", pady=3)
        self._entry(box, "x_min, м",        self.xmin_var)
        self._entry(box, "x_max, м",        self.xmax_var)
        self._entry(box, "z_max (глубина), м", self.zmax_var)
        self._entry(box, "точек наблюдения", self.nobs_var)
        ttk.Button(box, text="Применить",
                   command=self._apply_profile).pack(fill="x", padx=4, pady=3)

        # === Действия ===
        ttk.Button(parent, text="Сбросить параметры",
                   command=self._reset).pack(fill="x", pady=(8, 2))
        ttk.Button(parent, text="Сохранить рисунок…",
                   command=self._save_figure).pack(fill="x", pady=2)
        ttk.Button(parent, text="Сохранить кривые…",
                   command=self._save_curves).pack(fill="x", pady=2)

        # === Подсказка ===
        box = ttk.LabelFrame(parent, text="Подсказка")
        box.pack(fill="x", pady=3)
        ttk.Label(box, text=(
            "• Перетащите тело мышью\n"
            "  по нижней панели (разрезу).\n"
            "• Слайдеры — точное задание\n"
            "  параметров.\n"
            "• I_m, D_m — направление\n"
            "  намагниченности тела.\n"
            "• I_T, D_T — направление поля\n"
            "  Земли (нужно для ΔT).\n"
            "• «J ∥ T₀» — индуктивная\n"
            "  намагниченность; I_m, D_m\n"
            "  фиксируются на I_T, D_T."),
            justify="left", foreground="gray30")\
            .pack(padx=4, pady=4)

        # Стартовое состояние индуктивной намагниченности
        self._on_couple_changed()

    # ============== вспомогательные конструкторы виджетов ==============
    def _slider(self, parent, label: str,
                var: tk.DoubleVar, mn: float, mx: float) -> ttk.Scale:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=1)

        top = ttk.Frame(row)
        top.pack(fill="x")
        ttk.Label(top, text=label, width=18, anchor="w").pack(side="left")
        val_label = ttk.Label(top, text="", width=8, anchor="e",
                              font=("Arial", 9, "bold"))
        val_label.pack(side="right")

        def update_label(*_args):
            try:
                val_label.configure(text=f"{var.get():.1f}")
            except Exception:
                pass

        var.trace_add("write", update_label)
        update_label()

        scale = ttk.Scale(row, from_=mn, to=mx, variable=var,
                          orient="horizontal",
                          command=lambda v: self._update())
        scale.pack(fill="x")
        return scale

    def _entry(self, parent, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=var, width=10).pack(side="right")

    # ============== обработчики ==============
    def _on_couple_changed(self) -> None:
        """В индуктивном режиме блокирует слайдеры I_m, D_m."""
        try:
            state = "disabled" if self.couple_var.get() else "normal"
            if self.Im_scale is not None:
                self.Im_scale.configure(state=state)
            if self.Dm_scale is not None:
                self.Dm_scale.configure(state=state)
        except Exception:
            pass
        self._update()

    def _apply_profile(self) -> None:
        try:
            xmin = float(self.xmin_var.get().replace(",", "."))
            xmax = float(self.xmax_var.get().replace(",", "."))
            zmax = float(self.zmax_var.get().replace(",", "."))
            n = max(20, int(self.nobs_var.get()))
            if xmax <= xmin or zmax <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры профиля.")
            return
        self.x_min, self.x_max, self.z_max, self.n_obs = xmin, xmax, zmax, n
        self._update()

    def _reset(self) -> None:
        self.body_var.set(BODY_TYPES[0])
        self.x0_var.set(0.0)
        self.h_var.set(80.0)
        self.size_var.set(20.0)
        self.J_var.set(1.0)
        self.Im_var.set(65.0)
        self.Dm_var.set(0.0)
        self.IT_var.set(65.0)
        self.DT_var.set(0.0)
        self.theta_var.set(45.0)
        self.couple_var.set(True)
        self.show_Z_var.set(True)
        self.show_H_var.set(True)
        self.show_T_var.set(True)
        self.show_arrows_var.set(True)
        self.equal_aspect_var.set(False)
        self._on_couple_changed()

    def _save_figure(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Сохранить рисунок",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")

    def _save_curves(self) -> None:
        if self._last_curves is None:
            messagebox.showwarning("Внимание", "Сначала запустите расчёт.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить кривые",
            defaultextension=".txt",
            filetypes=[("Текст", "*.txt"), ("CSV", "*.csv"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        x_obs, Z, H, T = self._last_curves
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# x_m      Z              H              T\n")
                for x, z, h_v, t_v in zip(x_obs, Z, H, T):
                    f.write(f"{x:10.3f}  {z:+12.5e}  "
                            f"{h_v:+12.5e}  {t_v:+12.5e}\n")
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не сохранилось:\n{exc}")

    # ============== Считывание параметров ==============
    def _get_params(self) -> tuple:
        body = self.body_var.get()
        x0 = float(self.x0_var.get())
        h = max(float(self.h_var.get()), 1.0)
        size = max(float(self.size_var.get()), 0.1)
        J = float(self.J_var.get())
        IT = float(self.IT_var.get())
        DT = float(self.DT_var.get())
        if self.couple_var.get():
            # Принудительно синхронизируем Im,Dm с IT,DT
            Im, Dm = IT, DT
            if abs(self.Im_var.get() - Im) > 1e-6:
                self.Im_var.set(Im)
            if abs(self.Dm_var.get() - Dm) > 1e-6:
                self.Dm_var.set(Dm)
        else:
            Im = float(self.Im_var.get())
            Dm = float(self.Dm_var.get())
        theta = float(self.theta_var.get())
        return body, x0, h, size, J, Im, Dm, IT, DT, theta

    # ============== главный апдейт ==============
    def _update(self, *_) -> None:
        if self._updating:
            return
        # Защита от вызовов до того, как фигура построена
        # (например, из колбэков виджетов на этапе сборки UI).
        if not hasattr(self, "ax_field") or not hasattr(self, "ax_section"):
            return
        self._updating = True
        try:
            body, x0, h, size, J, Im, Dm, IT, DT, theta = self._get_params()

            x_obs = np.linspace(self.x_min, self.x_max, self.n_obs)
            Z, H, T = compute_anomaly(x_obs, x0, h, body, size, J,
                                       Im, Dm, IT, DT, theta)
            self._last_curves = (x_obs, Z, H, T)

            self._draw_field_plot(x_obs, Z, H, T, body,
                                   x0, h, size, J, Im, Dm, IT, DT)
            self._draw_section(body, x0, h, size, Im, Dm, IT, DT, theta)

            # Активируем слайдер угла только для наклонных пластов
            if self.theta_scale is not None:
                state = ("normal" if body.startswith("Наклонный")
                         else "disabled")
                self.theta_scale.configure(state=state)

            self.plot_canvas.draw_idle()
        finally:
            self._updating = False

    def _draw_field_plot(self, x_obs, Z, H, T, body,
                          x0, h, size, J, Im, Dm, IT, DT) -> None:
        ax = self.ax_field
        ax.clear()
        ax.set_ylabel("Магнитная аномалия, у.е.")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", lw=0.6, alpha=0.6)
        ax.axvline(x0, color="gray", lw=0.7, ls=":", alpha=0.7)

        plotted = False
        if self.show_Z_var.get():
            ax.plot(x_obs, Z, color="#1f78b4", lw=2.2,
                    label="Z  (вертикальная)")
            plotted = True
        if self.show_H_var.get():
            ax.plot(x_obs, H, color="#e31a1c", lw=2.2,
                    label="H  (горизонтальная, вдоль профиля)")
            plotted = True
        if self.show_T_var.get():
            ax.plot(x_obs, T, color="#33a02c", lw=2.4,
                    label="ΔT (полная аномалия)")
            plotted = True

        if plotted:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.set_title(
            f"{body}   |   J={J:.2f},  I_m={Im:+.0f}°,  D_m={Dm:+.0f}°"
            f"   |   I_T={IT:+.0f}°,  D_T={DT:+.0f}°"
            f"   |   h={h:.0f} м,  a={size:.1f} м",
            fontsize=10)
        ax.set_xlim(self.x_min, self.x_max)

    def _draw_section(self, body, x0, h, size,
                       Im, Dm, IT, DT, theta) -> None:
        ax = self.ax_section
        ax.clear()
        ax.set_xlabel("x, м (вдоль профиля)")
        ax.set_ylabel("Глубина z, м")
        ax.grid(True, alpha=0.3)

        sky_top = -max(30.0, 0.08 * self.z_max)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.z_max, sky_top)

        # «небо»
        ax.fill_between([self.x_min, self.x_max], sky_top, 0,
                         color="#e8f4fa", alpha=0.6, zorder=0)
        # поверхность
        ax.axhline(0, color="brown", lw=2, zorder=1)
        ax.text(self.x_min + 0.01 * (self.x_max - self.x_min),
                 sky_top * 0.45,
                 "поверхность z = 0", fontsize=8, color="0.3")

        if self.equal_aspect_var.get():
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass
        else:
            ax.set_aspect("auto")

        # тело
        self._draw_body(ax, body, x0, h, size, theta)

        if self.show_arrows_var.get():
            # Стрелка вектора намагниченности тела на самом теле
            arrow_len = 0.08 * (self.x_max - self.x_min)
            Im_r, Dm_r = np.radians(Im), np.radians(Dm)
            # Проекция J на плоскость разреза (x, z=down)
            ax_x = arrow_len * np.cos(Im_r) * np.cos(Dm_r)
            ax_z = arrow_len * np.sin(Im_r)
            ax.annotate(
                "", xy=(x0 + ax_x, h + ax_z), xytext=(x0, h),
                arrowprops=dict(arrowstyle="->", color="red",
                                 lw=2.4, mutation_scale=18),
                zorder=10)
            ax.text(x0 + 1.15 * ax_x, h + 1.15 * ax_z, "J",
                     color="red", fontsize=11, fontweight="bold",
                     ha="left", va="center", zorder=11)

            # Стрелка вектора поля Земли в углу
            T_len = arrow_len * 0.7
            IT_r, DT_r = np.radians(IT), np.radians(DT)
            tx = T_len * np.cos(IT_r) * np.cos(DT_r)
            tz = T_len * np.sin(IT_r)
            corner_x = self.x_min + 0.92 * (self.x_max - self.x_min)
            corner_z = sky_top * 0.55
            ax.annotate(
                "", xy=(corner_x + tx, corner_z + tz),
                xytext=(corner_x, corner_z),
                arrowprops=dict(arrowstyle="->", color="blue",
                                 lw=2, mutation_scale=16),
                zorder=10)
            ax.text(corner_x - 0.04 * (self.x_max - self.x_min),
                     corner_z + 0.45 * sky_top,
                     "T₀", color="blue", fontsize=10,
                     fontweight="bold", zorder=11)

    def _draw_body(self, ax, body: str, x0: float, h: float,
                    size: float, theta: float = 0.0) -> None:
        if body.startswith("Сфера"):
            ax.add_patch(mpatches.Circle(
                (x0, h), size,
                facecolor="#ffb84d", edgecolor="black",
                lw=1.6, alpha=0.85, zorder=5))
            ax.text(x0, h, "S", ha="center", va="center",
                     color="black", fontsize=10, fontweight="bold",
                     zorder=6)
        elif body.startswith("Горизонтальный цилиндр"):
            ax.add_patch(mpatches.Circle(
                (x0, h), size,
                facecolor="#a06cd5", edgecolor="black",
                lw=1.6, alpha=0.85, zorder=5))
            # значок «выходит из плоскости»
            ax.text(x0, h, "⊙", ha="center", va="center",
                     color="white", fontsize=14, fontweight="bold",
                     zorder=6)
        elif body.startswith("Точечный"):
            ax.plot(x0, h, "o", color="black",
                     markersize=14, zorder=5)
            ax.plot(x0, h, "+", color="white",
                     markersize=10, markeredgewidth=2, zorder=6)
        elif body.startswith("Вертикальный"):
            # тонкий вертикальный пласт: визуально — узкий прямоугольник
            # от глубины h до низа разреза. Ширина зависит от size.
            w = max(2.5, 0.4 * size)
            depth = self.z_max - h + 30
            ax.add_patch(mpatches.Rectangle(
                (x0 - w / 2, h), w, depth,
                facecolor="#2e8b57", edgecolor="black",
                lw=1.4, alpha=0.85, zorder=5))
        elif body.startswith("Куб"):
            # 2D квадратный брус (size трактуется как полусторона)
            ax.add_patch(mpatches.Rectangle(
                (x0 - size, h - size), 2 * size, 2 * size,
                facecolor="#5b8def", edgecolor="black",
                lw=1.6, alpha=0.85, zorder=5))
            ax.text(x0, h, "⊠", ha="center", va="center",
                     color="white", fontsize=12, fontweight="bold",
                     zorder=6)
        elif body.startswith("Горизонтальный пласт"):
            # тонкий широкий 2D пласт
            half_w = 5.0 * size
            half_t = size
            ax.add_patch(mpatches.Rectangle(
                (x0 - half_w, h - half_t),
                2 * half_w, 2 * half_t,
                facecolor="#e8a33d", edgecolor="black",
                lw=1.4, alpha=0.80, zorder=5))
        elif body.startswith("Полубесконечная"):
            # полубесконечная ступенька: брус x' > x0, толщина 2·size,
            # уходит вправо за пределы видимой области
            half_t = size
            right_edge = self.x_max + 0.25 * (self.x_max - self.x_min)
            ax.add_patch(mpatches.Rectangle(
                (x0, h - half_t),
                right_edge - x0, 2 * half_t,
                facecolor="#d96666", edgecolor="black",
                lw=1.4, alpha=0.80, zorder=5))
            # стрелка-индикатор «продолжается вправо»
            ax.annotate(
                "→ ∞",
                xy=(self.x_max - 0.06 * (self.x_max - self.x_min), h),
                ha="right", va="center",
                fontsize=11, fontweight="bold",
                color="white", zorder=6)
            # вертикальная линия — фронт уступа
            ax.plot([x0, x0], [h - half_t, h + half_t],
                     color="black", lw=2.5, zorder=6)
        elif body.startswith("Наклонный пласт (ограниченный)"):
            # параллелограмм: повёрнутый прямоугольник, центр в (x0, h)
            th = np.radians(theta)
            cs, sn = np.cos(th), np.sin(th)
            half_L = 5.0 * size
            half_t = size
            # 4 угла в системе тела (s, w), обходим по часовой стрелке
            corners_sw = [(-half_L, -half_t), (half_L, -half_t),
                           (half_L, half_t),  (-half_L, half_t)]
            corners_xz = [
                (x0 + s * cs - w * sn, h + s * sn + w * cs)
                for (s, w) in corners_sw
            ]
            ax.add_patch(mpatches.Polygon(
                corners_xz, closed=True,
                facecolor="#3aa39f", edgecolor="black",
                lw=1.5, alpha=0.85, zorder=5))
            # отметка центра
            ax.plot(x0, h, ".", color="black", markersize=4, zorder=6)
        elif body.startswith("Наклонный пласт (бесконечный)"):
            # полубесконечный параллелограмм: верхняя кромка через (x0, h),
            # уходит вниз по падению на большое расстояние
            th = np.radians(theta)
            cs, sn = np.cos(th), np.sin(th)
            half_t = size
            # длина «вниз по падению» — заведомо за пределы разреза
            s_far = 3.0 * max(self.z_max, self.x_max - self.x_min)
            corners_sw = [(0.0, -half_t), (s_far, -half_t),
                           (s_far, half_t), (0.0, half_t)]
            corners_xz = [
                (x0 + s * cs - w * sn, h + s * sn + w * cs)
                for (s, w) in corners_sw
            ]
            ax.add_patch(mpatches.Polygon(
                corners_xz, closed=True,
                facecolor="#c39bd3", edgecolor="black",
                lw=1.5, alpha=0.85, zorder=5))
            # подчёркиваем верхнюю кромку (фронт уступа)
            top_left = corners_xz[0]
            top_right = corners_xz[3]
            ax.plot([top_left[0], top_right[0]],
                     [top_left[1], top_right[1]],
                     color="black", lw=2.5, zorder=6)

    # ============== перетаскивание тела на разрезе ==============
    def _on_press(self, event) -> None:
        if event.inaxes != self.ax_section or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._dragging = True
        self.x0_var.set(float(event.xdata))
        self.h_var.set(float(max(5.0, event.ydata)))
        # Scale.command не срабатывает на программное var.set, поэтому
        # обновляем графики напрямую:
        self._update()

    def _on_motion(self, event) -> None:
        if not self._dragging:
            return
        if event.inaxes != self.ax_section:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.x0_var.set(float(event.xdata))
        self.h_var.set(float(max(5.0, event.ydata)))
        self._update()

    def _on_release(self, event) -> None:
        self._dragging = False


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    MagneticModelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()