import os
import sys
import csv
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_local
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QFormLayout, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QTextEdit, QGroupBox
)

def segment_image(
    image: np.ndarray,
    kernel_xy: Tuple[int, int],
    area_minmax: Tuple[int, int],
    thresh_offset: int,
    lower_int: int = 0,
    upper_int: int = 65535
) -> np.ndarray:

    image = np.clip(image, lower_int, upper_int)
    imf = image.astype(np.float32)
    imf -= imf.min()
    rng = imf.max() if imf.max() > 0 else 1.0
    im_u8 = (imf * (255.0 / rng)).astype(np.uint8)

    kx, ky = kernel_xy
    sigma = max(1, int(max(kx, ky) / 3))
    bg = cv2.GaussianBlur(im_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = cv2.subtract(im_u8, bg)

    inv = 255 - corrected

    blk = 71 if 71 % 2 == 1 else 72
    bin_img = cv2.adaptiveThreshold(
        inv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blk, thresh_offset
    )

    open_k = 3
    close_k = 4
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)

    h, w = bin_img.shape[:2]
    im_pad = cv2.copyMakeBorder(bin_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(im_pad, mask, (0, 0), 255)
    cv2.floodFill(im_pad, mask, (w + 1, 0), 255)
    cv2.floodFill(im_pad, mask, (0, h + 1), 255)
    cv2.floodFill(im_pad, mask, (w + 1, h + 1), 255)
    filled_inv = cv2.bitwise_not(im_pad[1:-1, 1:-1])
    bin_img = cv2.bitwise_or(bin_img, filled_inv)

    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    bin_bool = bin_img > 0
    bin_filled = ndi.binary_fill_holes(bin_bool)
    bin_img = (bin_filled.astype(np.uint8)) * 255

    _, labels_cc = cv2.connectedComponents(bin_img, connectivity=8)
    labels_cc = labels_cc.astype(np.int32)

    area_min, area_max = area_minmax
    mask_keep = np.zeros_like(labels_cc, dtype=bool)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    for lbl in range(1, num):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area_min <= area <= area_max:
            mask_keep |= (lab == lbl)
    labels_cc = np.where(mask_keep, labels_cc, 0)

    u = np.unique(labels_cc)
    u = u[u > 0]
    out = np.zeros_like(labels_cc, dtype=np.int32)
    for new_id, old_id in enumerate(u, start=1):
        out[labels_cc == old_id] = new_id

    return out



def colored_overlay(gray_image_u8: np.ndarray, labels_im: np.ndarray, alpha: float = 0.35) -> np.ndarray:

    base = cv2.cvtColor(gray_image_u8, cv2.COLOR_GRAY2BGR)
    if labels_im.max() == 0:
        return base
    m = (labels_im.astype(np.float32) / labels_im.max() * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    cells = (labels_im > 0).astype(np.uint8)
    cells3 = np.repeat(cells[..., None], 3, axis=2)
    blend = (base * (1 - alpha) + cmap * alpha).astype(np.uint8)
    return np.where(cells3 == 1, blend, base)


def to_uint8_display(im: np.ndarray) -> np.ndarray:
    """Megjelenítéshez/overlayhez 8-bitre skálázás veszteség nélkül a dinamikán belül."""
    if im.dtype == np.uint8:
        return im
    imf = im.astype(np.float32)
    imf -= imf.min()
    rng = imf.max() if imf.max() > 0 else 1.0
    return (imf * (255.0 / rng)).astype(np.uint8)


@dataclass
class SegPreset:
    name: str
    kernel_x: int
    kernel_y: int
    area_min: int
    area_max: int
    thresh_offset: int
    lower_int: int
    upper_int: int

    def key(self) -> str:
        raw = f"{self.kernel_x}-{self.kernel_y}-{self.area_min}-{self.area_max}-{self.thresh_offset}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


class BatchWorker(QObject):
    progress = pyqtSignal(int, int)
    message = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, in_dir: str, out_dir: str, presets: List[SegPreset]):
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.presets = presets
        self.stop_flag = False

    def log(self, txt: str):
        self.message.emit(txt)

    def run(self):
        try:
            exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
            files = [f for f in os.listdir(self.in_dir) if os.path.splitext(f)[1].lower() in exts]
            files.sort()
            total = len(files) * max(1, len(self.presets))
            done = 0
            if total == 0:
                self.log("Nincs feldolgozható kép a bemeneti mappában.")
                self.finished.emit(False)
                return

            os.makedirs(self.out_dir, exist_ok=True)

            for preset in self.presets:
                if self.stop_flag:
                    break
                preset_dir = os.path.join(self.out_dir, f"preset_{preset.name}_{preset.key()}")
                os.makedirs(preset_dir, exist_ok=True)

                with open(os.path.join(preset_dir, "params.txt"), "w", encoding="utf-8") as fh:
                    for k, v in asdict(preset).items():
                        fh.write(f"{k}: {v}\n")

                for fname in files:
                    if self.stop_flag:
                        break
                    in_path = os.path.join(self.in_dir, fname)
                    stem, _ = os.path.splitext(fname)

                    try:
                        img = cv2.imdecode(np.fromfile(in_path, np.uint8), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            raise RuntimeError("OpenCV nem tudta beolvasni a képet.")
                    except Exception:
                        try:
                            from skimage import io as skio
                            img = skio.imread(in_path)
                            if img.ndim > 2:
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        except Exception as e:
                            self.log(f"[HIBA] {fname} beolvasása sikertelen: {e}")
                            done += 1
                            self.progress.emit(done, total)
                            continue

                    try:
                        labels = segment_image(
                            image=img,
                            kernel_xy=(preset.kernel_x, preset.kernel_y),
                            area_minmax=(preset.area_min, preset.area_max),
                            thresh_offset=preset.thresh_offset,
                            lower_int=preset.lower_int,
                            upper_int=preset.upper_int,
                        )
                    except Exception as e:
                        self.log(f"[HIBA] Szegmentálás hiba {fname}: {e}")
                        done += 1
                        self.progress.emit(done, total)
                        continue

                    ids = np.unique(labels)
                    ids = ids[ids > 0]
                    cell_count = int(ids.size)

                    means = []
                    areas = []
                    if cell_count > 0:
                        means = ndi.mean(img, labels=labels, index=ids)
                        for cid in ids:
                            areas.append(int(np.count_nonzero(labels == cid)))

                    gray8 = to_uint8_display(img)
                    overlay_col = colored_overlay(gray8, labels, alpha=0.35)
                    mask_bin = (labels > 0).astype(np.uint8) * 255

                    out_base = os.path.join(preset_dir, stem)
                    os.makedirs(out_base, exist_ok=True)

                    csv_path = os.path.join(out_base, f"{stem}__cells.csv")
                    try:
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow(["Total Cell Count", cell_count])
                            w.writerow(["Cell ID", "Area (px)", "Mean Intensity"])
                            for i, cid in enumerate(ids):
                                mi = float(means[i]) if len(means) > i else float("nan")
                                w.writerow([int(cid), int(areas[i]), mi])
                    except Exception as e:
                        self.log(f"[HIBA] CSV mentési hiba {fname}: {e}")

                    ov_path = os.path.join(out_base, f"{stem}__overlay_colored.tiff")
                    try:
                        cv2.imwrite(ov_path, overlay_col)
                    except Exception as e:
                        self.log(f"[HIBA] Colored overlay mentési hiba {fname}: {e}")

                    bin_path = os.path.join(out_base, f"{stem}__mask_binary.tiff")
                    try:
                        cv2.imwrite(bin_path, mask_bin)
                    except Exception as e:
                        self.log(f"[HIBA] Binary mask mentési hiba {fname}: {e}")

                    done += 1
                    self.progress.emit(done, total)
                    self.log(f"OK: {fname} – {cell_count} objektum | preset: {preset.name}")

            self.finished.emit(not self.stop_flag)
        except Exception as e:
            self.log(f"[HIBA] Batch futási hiba: {e}")
            self.finished.emit(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brightfield – Batch Segmentation Export")
        self.setMinimumSize(900, 640)

        self.in_dir = ""
        self.out_dir = ""
        self.presets: List[SegPreset] = []

        self.worker_thread: QThread = None
        self.worker: BatchWorker = None

        self._init_ui()

    def _init_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw)

        box_dirs = QGroupBox("Mappák"); root.addWidget(box_dirs)
        ly_dirs = QHBoxLayout(box_dirs)

        self.lbl_in = QLabel("Bemeneti mappa: –")
        self.lbl_out = QLabel("Kimeneti mappa: –")
        btn_in = QPushButton("Bemeneti mappa kiválasztása")
        btn_out = QPushButton("Kimeneti mappa kiválasztása")
        btn_in.clicked.connect(self.select_in_dir)
        btn_out.clicked.connect(self.select_out_dir)

        col1 = QVBoxLayout(); col2 = QVBoxLayout()
        col1.addWidget(self.lbl_in); col1.addWidget(btn_in)
        col2.addWidget(self.lbl_out); col2.addWidget(btn_out)
        ly_dirs.addLayout(col1); ly_dirs.addLayout(col2)

        box_params = QGroupBox("Szegmentálás paraméterek és Presetek")
        root.addWidget(box_params)
        ly_params = QHBoxLayout(box_params)

        form = QFormLayout()
        self.spin_kx = QSpinBox(); self.spin_kx.setRange(1, 1000); self.spin_kx.setValue(21)
        self.spin_ky = QSpinBox(); self.spin_ky.setRange(1, 1000); self.spin_ky.setValue(21)
        self.spin_area_min = QSpinBox(); self.spin_area_min.setRange(1, 1_000_000); self.spin_area_min.setValue(3000)
        self.spin_area_max = QSpinBox(); self.spin_area_max.setRange(1000, 5_000_000); self.spin_area_max.setValue(17000)
        self.spin_off = QSpinBox(); self.spin_off.setRange(-100, 100); self.spin_off.setValue(0)
        self.spin_lower = QSpinBox(); self.spin_lower.setRange(0, 65535); self.spin_lower.setValue(0)
        self.spin_upper = QSpinBox(); self.spin_upper.setRange(1, 65535); self.spin_upper.setValue(65535)
        self.spin_lower = QSpinBox()
        self.spin_lower.setRange(0, 65535)
        self.spin_lower.setValue(0)

        self.spin_upper = QSpinBox()
        self.spin_upper.setRange(1, 65535)
        self.spin_upper.setValue(65535)

        form.addRow("Gaussian kernel X", self.spin_kx)
        form.addRow("Gaussian kernel Y", self.spin_ky)
        form.addRow("Area min (px)", self.spin_area_min)
        form.addRow("Area max (px)", self.spin_area_max)
        form.addRow("Threshold offset", self.spin_off)
        form.addRow("Lower intensity", self.spin_lower)
        form.addRow("Upper intensity", self.spin_upper)

        btn_add = QPushButton("Preset hozzáadása")
        btn_add.clicked.connect(self.add_preset)
        form.addRow(btn_add)

        left = QWidget(); left.setLayout(form)
        ly_params.addWidget(left, 1)

        right = QVBoxLayout()
        self.list_presets = QListWidget()
        self.list_presets.setSelectionMode(self.list_presets.SingleSelection)
        right.addWidget(QLabel("Presets:"))
        right.addWidget(self.list_presets, 1)
        btn_del = QPushButton("Kijelölt preset törlése")
        btn_del.clicked.connect(self.remove_selected_preset)
        right.addWidget(btn_del)

        wright = QWidget(); wright.setLayout(right)
        ly_params.addWidget(wright, 1)

        ly_run = QHBoxLayout()
        self.btn_run = QPushButton("Batch indítása")
        self.btn_stop = QPushButton("Leállítás")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self.start_batch)
        self.btn_stop.clicked.connect(self.stop_batch)
        self.progress = QProgressBar(); self.progress.setMinimum(0); self.progress.setValue(0)
        self.lbl_prog = QLabel("0/0 feldolgozva")
        ly_run.addWidget(self.btn_run); ly_run.addWidget(self.btn_stop)
        ly_run.addWidget(self.progress, 1); ly_run.addWidget(self.lbl_prog)
        root.addLayout(ly_run)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        root.addWidget(QLabel("Napló:"))
        root.addWidget(self.log, 1)

    def select_in_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Bemeneti mappa")
        if d:
            self.in_dir = d
            self.lbl_in.setText(f"Bemeneti mappa: {d}")

    def select_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Kimeneti mappa")
        if d:
            self.out_dir = d
            self.lbl_out.setText(f"Kimeneti mappa: {d}")

    def add_preset(self):
        name = f"kx{self.spin_kx.value()}_ky{self.spin_ky.value()}_A{self.spin_area_min.value()}-{self.spin_area_max.value()}_off{self.spin_off.value()}"
        preset = SegPreset(
            name=name,
            kernel_x=self.spin_kx.value(),
            kernel_y=self.spin_ky.value(),
            area_min=self.spin_area_min.value(),
            area_max=self.spin_area_max.value(),
            thresh_offset=self.spin_off.value(),
            lower_int=self.spin_lower.value(),
            upper_int=self.spin_upper.value(),
        )
        self.presets.append(preset)
        item = QListWidgetItem(f"{preset.name}  [key: {preset.key()}]")
        item.setData(Qt.UserRole, preset)
        self.list_presets.addItem(item)

    def remove_selected_preset(self):
        row = self.list_presets.currentRow()
        if row >= 0:
            item = self.list_presets.takeItem(row)
            preset = item.data(Qt.UserRole)
            self.presets = [p for p in self.presets if p is not preset]

    def start_batch(self):
        if not self.in_dir or not os.path.isdir(self.in_dir):
            QMessageBox.warning(self, "Hiányzó mappa", "Válaszd ki a bemeneti mappát!")
            return
        if not self.out_dir or not os.path.isdir(self.out_dir):
            QMessageBox.warning(self, "Hiányzó mappa", "Válaszd ki a kimeneti mappát!")
            return
        if not self.presets:
            QMessageBox.warning(self, "Nincs preset", "Adj hozzá legalább egy presetet!")
            return
        
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        nfiles = len([f for f in os.listdir(self.in_dir) if os.path.splitext(f)[1].lower() in exts])
        total = nfiles * len(self.presets)
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(0)
        self.lbl_prog.setText(f"0/{total} feldolgozva")

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.clear()

        self.worker_thread = QThread()
        self.worker = BatchWorker(self.in_dir, self.out_dir, self.presets)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.message.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)

        self.worker_thread.start()

    def stop_batch(self):
        if self.worker is not None:
            self.worker.stop_flag = True
            self.append_log("Leállítás kérése ...")

    def on_progress(self, done: int, total: int):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.lbl_prog.setText(f"{done}/{total} feldolgozva")

    def append_log(self, txt: str):
        self.log.append(txt)

    def on_finished(self, ok: bool):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        msg = "Kész!" if ok else "Befejezve (hibával vagy megszakítva)."
        self.append_log(msg)
        QMessageBox.information(self, "Batch", msg)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
