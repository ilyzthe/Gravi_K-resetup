#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI-программа для:
1) загрузки картинки,
2) наложения сетки из квадратных/прямоугольных ячеек,
3) раскраски ячеек по значениям delta rho,
4) загрузки/сохранения сетки в формат .grd (Surfer ASCII Grid / DSAA),
5) перемещения по полотну средней кнопкой мыши,
6) добавления новых значений delta rho в пресеты во время работы.

Зависимости:
    pip install pillow numpy

Запуск:
    python image_to_grd_gui.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk


DEFAULT_ALLOWED = [
    -0.420, -0.080, -0.070, 0.000, 0.010, 0.090,
    0.120, 0.130, 0.140, 0.150, 0.310, 1.730,
]


class GridPainterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Картинка -> сетка delta rho -> .grd")
        self.root.geometry("1350x900")

        self.canvas_width = 1000
        self.canvas_height = 760

        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        self.zoom = 1.0
        self.zoom_min = 0.25
        self.zoom_max = 6.0

        # Координаты области изображения на canvas
        self.img_left = 0
        self.img_top = 0
        self.img_right = 0
        self.img_bottom = 0
        self.img_width = 0
        self.img_height = 0

        self.grid_rect_ids = []
        self.grid_line_ids = []

        self.nx = 100
        self.ny = 30
        self.grid_values = np.zeros((self.ny, self.nx), dtype=float)

        self.drag_painting = False
        self.last_painted_cell = None

        # Пресеты можно пополнять во время работы.
        self.allowed_values = list(DEFAULT_ALLOWED)

        self._build_ui()
        self._reset_grid()
        self._draw_placeholder()

    # --------------------------- UI ---------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        controls = ttk.Frame(main)
        controls.pack(side="left", fill="y", padx=(0, 8))

        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(side="right", fill="both", expand=True)

        # ===== Блок файла =====
        file_box = ttk.LabelFrame(controls, text="Файл")
        file_box.pack(fill="x", pady=4)

        ttk.Button(file_box, text="Загрузить картинку", command=self.load_image).pack(fill="x", padx=6, pady=4)
        ttk.Button(file_box, text="Загрузить .grd", command=self.load_grd).pack(fill="x", padx=6, pady=4)

        zoom_row = ttk.Frame(file_box)
        zoom_row.pack(fill="x", padx=6, pady=4)
        ttk.Button(zoom_row, text="−", command=self.zoom_out, width=4).pack(side="left")
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(zoom_row, textvariable=self.zoom_var, anchor="center").pack(side="left", fill="x", expand=True)
        ttk.Button(zoom_row, text="+", command=self.zoom_in, width=4).pack(side="right")

        ttk.Button(file_box, text="Масштаб 100%", command=self.zoom_reset).pack(fill="x", padx=6, pady=4)
        ttk.Button(file_box, text="Сохранить .grd", command=self.save_grd).pack(fill="x", padx=6, pady=4)
        ttk.Button(file_box, text="Сохранить PNG-превью", command=self.save_preview_png).pack(fill="x", padx=6, pady=4)

        # ===== Блок параметров сетки =====
        grid_box = ttk.LabelFrame(controls, text="Сетка")
        grid_box.pack(fill="x", pady=4)

        self.nx_var = tk.StringVar(value="100")
        self.ny_var = tk.StringVar(value="30")
        self.xmin_var = tk.StringVar(value="0.000")
        self.xmax_var = tk.StringVar(value="1.150")
        self.ymin_var = tk.StringVar(value="0.000")
        self.ymax_var = tk.StringVar(value="1.000")

        self._labeled_entry(grid_box, "Nx (по X)", self.nx_var)
        self._labeled_entry(grid_box, "Ny (по Y/Z)", self.ny_var)
        self._labeled_entry(grid_box, "xmin", self.xmin_var)
        self._labeled_entry(grid_box, "xmax", self.xmax_var)
        self._labeled_entry(grid_box, "ymin / zmin", self.ymin_var)
        self._labeled_entry(grid_box, "ymax / zmax", self.ymax_var)

        ttk.Button(grid_box, text="Создать / обновить сетку", command=self.rebuild_grid).pack(fill="x", padx=6, pady=6)
        ttk.Button(grid_box, text="Очистить сетку (все нули)", command=self.clear_grid).pack(fill="x", padx=6, pady=(0, 6))

        # ===== Блок рисования =====
        paint_box = ttk.LabelFrame(controls, text="Рисование delta rho")
        paint_box.pack(fill="x", pady=4)

        self.value_var = tk.StringVar(value="0.090")
        self.allowed_var = tk.StringVar(value="0.090")
        self.new_preset_var = tk.StringVar(value="")

        row = ttk.Frame(paint_box)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Preset:").pack(side="left")
        self.preset_combo = ttk.Combobox(
            row,
            textvariable=self.allowed_var,
            values=self._preset_strings(),
            state="readonly",
            width=10,
        )
        self.preset_combo.pack(side="right", fill="x", expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self._labeled_entry(paint_box, "Текущее delta rho", self.value_var)

        add_preset_row = ttk.Frame(paint_box)
        add_preset_row.pack(fill="x", padx=6, pady=4)
        ttk.Entry(add_preset_row, textvariable=self.new_preset_var, width=12).pack(side="left", fill="x", expand=True)
        ttk.Button(add_preset_row, text="Добавить preset", command=self.add_preset).pack(side="right", padx=(6, 0))

        ttk.Button(paint_box, text="Добавить текущее значение в пресеты", command=self.add_current_value_to_presets).pack(fill="x", padx=6, pady=4)
        ttk.Button(paint_box, text="Заполнить всё текущим значением", command=self.fill_all).pack(fill="x", padx=6, pady=4)
        ttk.Button(paint_box, text="Инвертировать знак всей сетки", command=self.invert_sign).pack(fill="x", padx=6, pady=4)

        hint = (
            "ЛКМ: закрасить ячейку\n"
            "ПКМ: сделать ячейку 0\n"
            "Shift + ЛКМ: взять значение из ячейки\n"
            "Средняя кнопка: перемещаться по полотну"
        )
        ttk.Label(paint_box, text=hint, justify="left").pack(fill="x", padx=6, pady=6)

        # ===== Информация =====
        info_box = ttk.LabelFrame(controls, text="Информация")
        info_box.pack(fill="x", pady=4)

        self.info_var = tk.StringVar(value="Нет загруженной картинки")
        self.cursor_var = tk.StringVar(value="Ячейка: -, значение: -")
        ttk.Label(info_box, textvariable=self.info_var, wraplength=280, justify="left").pack(fill="x", padx=6, pady=4)
        ttk.Label(info_box, textvariable=self.cursor_var, wraplength=280, justify="left").pack(fill="x", padx=6, pady=4)

        # ===== Canvas =====
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Перемещение по полотну средней кнопкой мыши.
        self.canvas.bind("<Button-2>", self.on_middle_press)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_release)

    def _labeled_entry(self, parent, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=var, width=12).pack(side="right")

    # --------------------------- Пресеты ---------------------------
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
            value = float(raw.replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное значение для пресета.")
            return

        self._add_preset_value(value)
        self.new_preset_var.set("")

    def add_current_value_to_presets(self) -> None:
        try:
            value = float(self.value_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное текущее значение delta rho.")
            return

        self._add_preset_value(value)

    def _add_preset_value(self, value: float) -> None:
        rounded = round(float(value), 6)
        if rounded not in [round(float(v), 6) for v in self.allowed_values]:
            self.allowed_values.append(rounded)
            self._refresh_preset_combo()

        shown = f"{rounded:.3f}"
        self.allowed_var.set(shown)
        self.value_var.set(shown)

    def _on_preset_selected(self, event=None) -> None:
        self.value_var.set(self.allowed_var.get())

    # --------------------------- Обработка изображений ---------------------------
    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_width // 2,
            self.canvas_height // 2,
            text="Загрузите картинку, затем создайте сетку",
            font=("Arial", 18),
            fill="gray40",
        )
        self._update_scroll_region()

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите картинку",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.original_image = Image.open(path).convert("RGB")
            self.zoom = 1.0
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть картинку:\n{exc}")
            return

        self._prepare_display_image()
        self._redraw_scene()
        self.info_var.set(f"Картинка: {Path(path).name}\nРазмер: {self.original_image.width} x {self.original_image.height}")

    def _prepare_display_image(self) -> None:
        if self.original_image is None:
            return

        base = self.original_image.copy()
        base.thumbnail((self.canvas_width - 20, self.canvas_height - 20), Image.Resampling.LANCZOS)

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

        if hasattr(self, "zoom_var"):
            self.zoom_var.set(f"{int(round(self.zoom * 100))}%")

    # --------------------------- Масштаб ---------------------------
    def zoom_in(self) -> None:
        if self.original_image is None:
            return
        self.zoom = min(self.zoom_max, self.zoom * 1.25)
        self._prepare_display_image()
        self._redraw_scene()

    def zoom_out(self) -> None:
        if self.original_image is None:
            return
        self.zoom = max(self.zoom_min, self.zoom / 1.25)
        self._prepare_display_image()
        self._redraw_scene()

    def zoom_reset(self) -> None:
        if self.original_image is None:
            return
        self.zoom = 1.0
        self._prepare_display_image()
        self._redraw_scene()

    def on_mouse_wheel(self, event) -> None:
        if self.original_image is None:
            return
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # --------------------------- Панорамирование полотна ---------------------------
    def on_middle_press(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def on_middle_drag(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_middle_release(self, event) -> None:
        self.canvas.configure(cursor="")

    def _event_canvas_xy(self, event) -> tuple[float, float]:
        """Переводит координаты события окна в координаты canvas с учётом панорамирования."""
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _update_scroll_region(self) -> None:
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return

        x0, y0, x1, y1 = bbox
        margin = 300
        self.canvas.configure(scrollregion=(x0 - margin, y0 - margin, x1 + margin, y1 + margin))

    # --------------------------- Сетка ---------------------------
    def _reset_grid(self) -> None:
        self.grid_values = np.zeros((self.ny, self.nx), dtype=float)

    def rebuild_grid(self) -> None:
        try:
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
            if nx <= 0 or ny <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Nx и Ny должны быть положительными целыми числами.")
            return

        self.nx = nx
        self.ny = ny
        self._reset_grid()
        self._redraw_scene()

    def clear_grid(self) -> None:
        self._reset_grid()
        self._redraw_scene()

    def fill_all(self) -> None:
        try:
            value = float(self.value_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное значение delta rho.")
            return

        self.grid_values[:, :] = value
        self._redraw_scene()

    def invert_sign(self) -> None:
        self.grid_values *= -1.0
        self._redraw_scene()

    def _redraw_scene(self) -> None:
        self.canvas.delete("all")
        self.grid_rect_ids.clear()
        self.grid_line_ids.clear()

        if self.tk_image is not None:
            self.canvas.create_image(self.img_left, self.img_top, anchor="nw", image=self.tk_image)
        else:
            self._draw_placeholder()
            return

        self._draw_colored_cells()
        self._draw_grid_lines()
        self._update_scroll_region()

    def _draw_grid_lines(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            return

        cell_w = self.img_width / self.nx
        cell_h = self.img_height / self.ny

        # Внешняя рамка
        self.grid_line_ids.append(
            self.canvas.create_rectangle(self.img_left, self.img_top, self.img_right, self.img_bottom, outline="black", width=2)
        )

        # Вертикали
        for ix in range(1, self.nx):
            x = self.img_left + ix * cell_w
            self.grid_line_ids.append(
                self.canvas.create_line(x, self.img_top, x, self.img_bottom, fill="black")
            )

        # Горизонтали
        for iy in range(1, self.ny):
            y = self.img_top + iy * cell_h
            self.grid_line_ids.append(
                self.canvas.create_line(self.img_left, y, self.img_right, y, fill="black")
            )

    def _draw_colored_cells(self) -> None:
        cell_w = self.img_width / self.nx
        cell_h = self.img_height / self.ny

        vmin = float(np.min(self.grid_values))
        vmax = float(np.max(self.grid_values))
        absmax = max(abs(vmin), abs(vmax), 1e-12)

        for iy in range(self.ny):
            for ix in range(self.nx):
                val = self.grid_values[iy, ix]
                if abs(val) < 1e-15:
                    continue

                x0 = self.img_left + ix * cell_w
                y0 = self.img_top + iy * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h

                color = self._value_to_color(val, absmax)
                rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color,
                    outline="",
                    stipple="gray50",
                )
                self.grid_rect_ids.append(rect_id)

    # --------------------------- Цвета ---------------------------
    @staticmethod
    def _value_to_color(value: float, absmax: float) -> str:
        """
        Отрицательные значения -> синий диапазон.
        Положительные значения -> красный диапазон.
        Ноль -> белый.
        """
        t = max(-1.0, min(1.0, value / absmax))

        if t >= 0:
            # Белый -> красный
            r = 255
            g = int(255 * (1.0 - t))
            b = int(255 * (1.0 - t))
        else:
            # Синий -> белый
            t = abs(t)
            r = int(255 * (1.0 - t))
            g = int(255 * (1.0 - t))
            b = 255

        return f"#{r:02x}{g:02x}{b:02x}"

    # --------------------------- Работа мышью ---------------------------
    def _point_to_cell(self, x: float, y: float):
        if not (self.img_left <= x <= self.img_right and self.img_top <= y <= self.img_bottom):
            return None

        cell_w = self.img_width / self.nx
        cell_h = self.img_height / self.ny

        ix = int((x - self.img_left) / cell_w)
        iy = int((y - self.img_top) / cell_h)

        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return ix, iy

    def on_left_click(self, event) -> None:
        x, y = self._event_canvas_xy(event)
        cell = self._point_to_cell(x, y)
        if cell is None:
            return

        ix, iy = cell

        if event.state & 0x0001:  # Shift
            self.value_var.set(f"{self.grid_values[iy, ix]:.3f}")
            self.allowed_var.set(f"{self.grid_values[iy, ix]:.3f}")
            return

        self.drag_painting = True
        self.last_painted_cell = None
        self._paint_cell(ix, iy)

    def on_left_drag(self, event) -> None:
        if not self.drag_painting:
            return
        x, y = self._event_canvas_xy(event)
        cell = self._point_to_cell(x, y)
        if cell is None:
            return

        ix, iy = cell
        if self.last_painted_cell == (ix, iy):
            return
        self._paint_cell(ix, iy)

    def on_left_release(self, event) -> None:
        self.drag_painting = False
        self.last_painted_cell = None

    def on_right_click(self, event) -> None:
        x, y = self._event_canvas_xy(event)
        cell = self._point_to_cell(x, y)
        if cell is None:
            return
        ix, iy = cell
        self.grid_values[iy, ix] = 0.0
        self._redraw_scene()
        self.cursor_var.set(f"Ячейка: ({ix}, {iy}), значение: 0.000")

    def on_mouse_move(self, event) -> None:
        x, y = self._event_canvas_xy(event)
        cell = self._point_to_cell(x, y)
        if cell is None:
            self.cursor_var.set("Ячейка: -, значение: -")
            return
        ix, iy = cell
        val = self.grid_values[iy, ix]
        self.cursor_var.set(f"Ячейка: ({ix}, {iy}), значение: {val:.3f}")

    def _paint_cell(self, ix: int, iy: int) -> None:
        try:
            value = float(self.value_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное значение delta rho.")
            self.drag_painting = False
            return

        self.grid_values[iy, ix] = value
        self.last_painted_cell = (ix, iy)
        self._redraw_scene()
        self.cursor_var.set(f"Ячейка: ({ix}, {iy}), значение: {value:.3f}")

    # --------------------------- Загрузка / сохранение ---------------------------
    def load_grd(self) -> None:
        path = filedialog.askopenfilename(
            title="Загрузить .grd",
            filetypes=[("GRD files", "*.grd"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            grid, xmin, xmax, ymin, ymax = read_dsaa(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить .grd:\n{exc}")
            return

        self.grid_values = grid.astype(float)
        self.ny, self.nx = self.grid_values.shape

        self.nx_var.set(str(self.nx))
        self.ny_var.set(str(self.ny))
        self.xmin_var.set(f"{xmin:.3f}")
        self.xmax_var.set(f"{xmax:.3f}")
        self.ymin_var.set(f"{ymin:.3f}")
        self.ymax_var.set(f"{ymax:.3f}")

        # Добавляем ненулевые значения из загруженной сетки в пресеты.
        unique_values = np.unique(np.round(self.grid_values[np.abs(self.grid_values) > 1e-15], 6))
        for value in unique_values:
            if round(float(value), 6) not in [round(float(v), 6) for v in self.allowed_values]:
                self.allowed_values.append(float(value))
        self._refresh_preset_combo()

        self._redraw_scene()

        info = f"Загружена сетка: {Path(path).name}\nРазмер: {self.nx} x {self.ny}"
        if self.original_image is not None:
            info += f"\nКартинка: {self.original_image.width} x {self.original_image.height}"
        self.info_var.set(info)

    def save_grd(self) -> None:
        if self.grid_values is None:
            messagebox.showwarning("Внимание", "Сначала создайте или загрузите сетку.")
            return

        try:
            xmin = float(self.xmin_var.get().replace(",", "."))
            xmax = float(self.xmax_var.get().replace(",", "."))
            ymin = float(self.ymin_var.get().replace(",", "."))
            ymax = float(self.ymax_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные координатные пределы xmin/xmax/ymin/ymax.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить .grd",
            defaultextension=".grd",
            filetypes=[("GRD files", "*.grd"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            write_dsaa(path, self.grid_values, xmin, xmax, ymin, ymax)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить .grd:\n{exc}")
            return

        messagebox.showinfo("Готово", f"Файл сохранён:\n{path}")

    def save_preview_png(self) -> None:
        if self.tk_image is None:
            messagebox.showwarning("Внимание", "Сначала загрузите картинку.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить PNG-превью",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return

        try:
            preview = self._build_preview_image()
            preview.save(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить PNG:\n{exc}")
            return

        messagebox.showinfo("Готово", f"PNG сохранён:\n{path}")

    def _build_preview_image(self) -> Image.Image:
        if self.display_image is None:
            raise RuntimeError("Нет изображения для превью")

        base = self.display_image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))

        cell_w = base.width / self.nx
        cell_h = base.height / self.ny
        absmax = max(abs(float(np.min(self.grid_values))), abs(float(np.max(self.grid_values))), 1e-12)

        from PIL import ImageDraw
        draw = ImageDraw.Draw(overlay)

        for iy in range(self.ny):
            for ix in range(self.nx):
                val = self.grid_values[iy, ix]
                if abs(val) < 1e-15:
                    continue
                x0 = ix * cell_w
                y0 = iy * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                color_hex = self._value_to_color(val, absmax)
                r = int(color_hex[1:3], 16)
                g = int(color_hex[3:5], 16)
                b = int(color_hex[5:7], 16)
                draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 120))

        out = Image.alpha_composite(base, overlay)
        draw2 = ImageDraw.Draw(out)
        # линии сетки
        for ix in range(self.nx + 1):
            x = ix * cell_w
            draw2.line([(x, 0), (x, base.height)], fill=(0, 0, 0, 255), width=1)
        for iy in range(self.ny + 1):
            y = iy * cell_h
            draw2.line([(0, y), (base.width, y)], fill=(0, 0, 0, 255), width=1)

        return out.convert("RGB")


def read_dsaa(path: str) -> tuple[np.ndarray, float, float, float, float]:
    """
    Загружает сетку из DSAA / Surfer ASCII Grid.

    Ожидаемый формат:
        DSAA
        nx ny
        xmin xmax
        ymin ymax
        vmin vmax
        значения сетки
    """
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()

    if len(tokens) < 10:
        raise ValueError("Файл слишком короткий для формата DSAA.")

    if tokens[0].upper() != "DSAA":
        raise ValueError("Это не DSAA-файл: первая строка должна быть DSAA.")

    nx = int(tokens[1])
    ny = int(tokens[2])
    if nx <= 0 or ny <= 0:
        raise ValueError("Некорректные размеры сетки в .grd.")

    xmin = float(tokens[3])
    xmax = float(tokens[4])
    ymin = float(tokens[5])
    ymax = float(tokens[6])

    # tokens[7] и tokens[8] — vmin/vmax, их можно не использовать.
    values_tokens = tokens[9:]
    expected = nx * ny
    if len(values_tokens) < expected:
        raise ValueError(f"Недостаточно значений сетки: найдено {len(values_tokens)}, нужно {expected}.")

    values = [float(v) for v in values_tokens[:expected]]
    grid = np.array(values, dtype=float).reshape((ny, nx))
    return grid, xmin, xmax, ymin, ymax


def write_dsaa(path: str, grid: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """
    Сохраняет сетку в формат DSAA.
    Формат совместим с предыдущими вариантами:
        DSAA
               nx       ny
               xmin     xmax
               ymin     ymax
               vmin     vmax
        далее значения по строкам, по 4 числа в строке.
    """
    ny, nx = grid.shape
    vmin = float(np.min(grid))
    vmax = float(np.max(grid))

    with open(path, "w", encoding="utf-8") as f:
        f.write("DSAA\n")
        f.write(f"{nx:10d}{ny:10d}\n")
        f.write(f"{xmin:12.3f}{xmax:11.3f}\n")
        f.write(f"{ymin:12.3f}{ymax:11.3f}\n")
        f.write(f"{vmin:12.3f}{vmax:10.3f}\n")

        flat = grid.reshape(-1)
        for i in range(0, len(flat), 4):
            vals = flat[i:i + 4]
            f.write(" ".join(f"{v: .3f}" if v >= 0 else f"{v:.3f}" for v in vals) + "\n")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    GridPainterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
