import sys
import cv2
import csv
import numpy as np
from skimage.filters import threshold_triangle
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QPaintEvent, QImage, QPainter, QMouseEvent, QPen, QKeyEvent, QContextMenuEvent, QGuiApplication
from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QAction, QMainWindow, QDockWidget, QFormLayout, QSpinBox, QWidget, QHBoxLayout, QPushButton, QSlider, QFileDialog, QMessageBox, QComboBox, QCheckBox, QDoubleSpinBox
def np_to_qimage_display(img_np):
    import numpy as np
    import cv2
    from PyQt5.QtGui import QImage

    im = img_np
    if im.ndim > 2 and im.shape[2] > 3:
        im = im[..., 0]

    if im.dtype == np.uint16:
        maxv = int(im.max()) if im.max() > 0 else 1
        im8 = (im.astype(np.float32) * (255.0 / maxv)).astype(np.uint8)
    elif im.dtype != np.uint8:
        imf = im.astype(np.float32)
        imf -= imf.min()
        rng = imf.max() if imf.max() > 0 else 1.0
        im8 = (imf * (255.0 / rng)).astype(np.uint8)
    else:
        im8 = im

    if im8.ndim == 3 and im8.shape[2] == 3:
        im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2RGB)
        h, w, ch = im8.shape
        qimg = QImage(im8.data, w, h, im8.strides[0], QImage.Format_RGB888)
    else:
        h, w = im8.shape[:2]
        qimg = QImage(im8.data, w, h, im8.strides[0], QImage.Format_Grayscale8)

    return qimg.copy()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cell Thresholder App")
        self.setCentralWidget(FluoImage(self))

        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        width = screen_geometry.width()
        height = screen_geometry.height()

        self.setFixedSize(width - 50, height - 50)

class LabelCutOperation:
    def __init__(self):
        self.points = []
        self.display_scale = 1.0
        self.draw_x = 0
        self.draw_y = 0
    
    def mousePressEvent(self, event: QMouseEvent):
        if len(self.points) == 2:
            self.points.clear()
        else:
            self.points.append((event.x(), event.y()))
    
    def paintEvent(self, painter: QPainter):
        if len(self.points) == 1:
            (x1, y1) = self.points[0]
            painter.drawPoint(x1, y1)
        elif len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    
    def applyTo(self, image, scale):
        if len(self.points) < 2:
            return image

        height, width = image.shape[:2]
        (x1, y1), (x2, y2) = self.points
        p1_scaled, p2_scaled = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, p1_scaled, p2_scaled, (1), thickness=-1)

        image[mask == 1] = np.average(image)
        return image

class CircleMaskOperation:
    def __init__(self):
        self.points = []
    
    def mousePressEvent(self, event: QMouseEvent):
        if len(self.points) == 3:
            self.points.clear()
        else:
            self.points.append((event.x(), event.y()))
    
    def paintEvent(self, painter: QPainter):
        if len(self.points) < 3:
            for (x, y) in self.points:
                painter.drawPoint(x, y)
        elif len(self.points) == 3:
            cx, cy, radius = self.calculate_circle()
            painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))
    
    def applyTo(self, image, scale):
        if len(self.points) < 3:
            return

        height, width = image.shape[:2]
        cx, cy, radius = self.calculate_circle()
        cx_scaled, cy_scaled, radius_scaled = int(cx * scale), int(cy * scale), int(radius * scale)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (cx_scaled, cy_scaled), radius_scaled, (1), thickness=-1)

        image[mask == 0] = np.average(image)
        return image
    
    def calculate_circle(self):
        (x1, y1), (x2, y2), (x3, y3) = self.points

        D = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        
        cx = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / D
        cy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / D
        
        radius = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        
        return cx, cy, radius

import pandas as pd

class BrightfieldWorker(QObject):
    finished = pyqtSignal(object, object, object)

    def __init__(self, fluo_image,
                 bg_kernel, method, block, offset,
                 open_k, close_k, min_area, max_area, alpha,
                 downsample=1, mode="Fast count (downsample)", skip_overlay=True):
        super().__init__()
        self.fluo_image = fluo_image
        self.bg_kernel = bg_kernel
        self.method = method
        self.block = block
        self.offset = offset
        self.open_k = open_k
        self.close_k = close_k
        self.min_area = min_area
        self.max_area = max_area
        self.alpha = alpha
        self.downsample = max(1, int(downsample))
        self.mode = mode
        self.skip_overlay = bool(skip_overlay)

    def _normalize_uint8(self, im):
        im = im.astype(np.float32)
        im -= im.min()
        rng = im.max() if im.max() > 0 else 1.0
        return (im * (255.0 / rng)).astype(np.uint8)

    def _background_correct(self, im_u8, kernel):
        sigma = max(1, int(kernel/3))
        bg = cv2.GaussianBlur(im_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return cv2.subtract(im_u8, bg)
    
    def _threshold(self, inv_u8):
        if self.method == "adaptive":
            blk = self.block if self.block % 2 == 1 else self.block + 1
            th = cv2.adaptiveThreshold(inv_u8, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                       blk, self.offset)
        else:
            _, th = cv2.threshold(inv_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def _postprocess(self, bin_img):
        if self.open_k > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)
        if self.close_k > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
        return bin_img

    def _filter_area_and_relabel(self, labels_im, min_area, max_area):
        if labels_im.max() == 0:
            return labels_im
        mask = (labels_im > 0).astype(np.uint8) * 255
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        keep = np.zeros_like(labels_im, dtype=bool)
        for lbl in range(1, num):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                keep |= (labels_im > 0) & (lab == lbl)

        filtered = np.where(keep, labels_im, 0).astype(np.int32)
        u = np.unique(filtered); u = u[u > 0]
        out = np.zeros_like(filtered, dtype=np.int32)
        for new, old in enumerate(u, start=1):
            out[filtered == old] = new
        return out
    
    def _colored_overlay(self, gray_u8, labels_im):
        if labels_im.max() == 0:
            return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        m = (labels_im.astype(np.float32) / labels_im.max() * 255).astype(np.uint8)
        color_mask = cv2.applyColorMap(m, cv2.COLORMAP_JET)
        base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        cells = (labels_im > 0).astype(np.uint8)
        cells3 = np.repeat(cells[..., None], 3, axis=2)
        blend = (base * (1 - self.alpha) + color_mask * self.alpha).astype(np.uint8)
        return np.where(cells3 == 1, blend, base)

    def run(self):
        img = self.fluo_image.image
        if img is None:
            self.finished.emit(None, None, None); return

        img_u8 = self._normalize_uint8(img)

        s = self.downsample
        if self.mode.startswith("Fast"):
            if s > 1:
                small = cv2.resize(img_u8, None, fx=1.0/s, fy=1.0/s, interpolation=cv2.INTER_AREA)
            else:
                small = img_u8.copy()
            work = small
            min_area = max(1, int(self.min_area / (s*s)))
            max_area = max(1, int(self.max_area / (s*s)))
        else:
            work = img_u8
            min_area = self.min_area
            max_area = self.max_area

        corrected = self._background_correct(work, self.bg_kernel)
        inv = 255 - self._normalize_uint8(corrected)

        bin_img = self._threshold(inv)
        fg_ratio = (bin_img > 0).mean()
        tw = 512
        th = max(1, int(bin_img.shape[0] * (tw / max(1, bin_img.shape[1]))))
        small = cv2.resize(bin_img, (tw, th), interpolation=cv2.INTER_NEAREST)
        fg_ratio = (small > 0).mean()

        if fg_ratio > 0.70:
            blk = self.block if (self.block % 2 == 1) else (self.block + 1)
            adj_offset = min(0, self.offset + 4)
            if self.method == "adaptive":
                bin_img = cv2.adaptiveThreshold(
                    inv, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                    blk, adj_offset
                )
                small2 = cv2.resize(bin_img, (tw, th), interpolation=cv2.INTER_NEAREST)
                if (small2 > 0).mean() > 0.70:
                    _, bin_img = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bin_img = self._postprocess(bin_img)
        h, w = bin_img.shape[:2]
        im_pad = cv2.copyMakeBorder(bin_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask = np.zeros((h+2+2, w+2+2), np.uint8)

        cv2.floodFill(im_pad, mask, (0, 0), 255)
        cv2.floodFill(im_pad, mask, (w+1, 0), 255)
        cv2.floodFill(im_pad, mask, (0, h+1), 255)
        cv2.floodFill(im_pad, mask, (w+1, h+1), 255)

        filled_inv = cv2.bitwise_not(im_pad[1:-1, 1:-1])
        bin_img = cv2.bitwise_or(bin_img, filled_inv)

        bin_img = self._postprocess(bin_img)


        from scipy import ndimage as ndi
        bin_bool = bin_img > 0
        bin_filled = ndi.binary_fill_holes(bin_bool)
        bin_img = (bin_filled.astype(np.uint8)) * 255

        _, labels_cc = cv2.connectedComponents(bin_img, connectivity=8)
        labels_cc = labels_cc.astype(np.int32)

        labels_clean_small = self._filter_area_and_relabel(labels_cc, min_area, max_area)

        if self.mode.startswith("Fast"):
            if s > 1:
                labels_up = cv2.resize(labels_clean_small, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                labels_up = labels_clean_small

            overlay = None
            if not self.skip_overlay:
                ov_small = self._colored_overlay(work, labels_clean_small)
                overlay = cv2.resize(ov_small, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_NEAREST)

            self.finished.emit(labels_up, overlay, None)
            return

        overlay = self._colored_overlay(work, labels_clean_small)
        self.finished.emit(labels_clean_small, overlay, None)

class ThresholdWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, fluo_image, lower_value, upper_value):
        super().__init__()
        self.fluo_image = fluo_image
        self.lower_value = lower_value
        self.upper_value = upper_value

    def run(self):
        if self.fluo_image.segmentation_mask is None:
            self.finished.emit()
            return

        mask = self.fluo_image.segmentation_mask
        ids = np.unique(mask)[1:]

        if self.fluo_image.intensities is None:
            from scipy import ndimage
            image = self.fluo_image.image
            means = ndimage.mean(image, labels=mask, index=ids)
            self.fluo_image.intensities = dict(zip([int(i) for i in ids], means))

        lo = float(self.lower_value)
        hi = float(self.upper_value)
        if lo > hi:
            lo, hi = hi, lo

        vals = np.array([self.fluo_image.intensities.get(int(i), np.nan) for i in ids])
        keep = (vals >= lo) & (vals <= hi)
        keep_ids = ids[keep]

        processed_mask = np.where(np.isin(mask, keep_ids), mask, 0).astype(mask.dtype)
        self.fluo_image.thresholded_mask = processed_mask
        self.fluo_image.update()

        self.finished.emit()

from skimage.filters import threshold_local
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

class SegmentWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, fluo_image, gaussian_kernel_size, area_threshold, thresh_offset):
        super().__init__()
        self.fluo_image = fluo_image
        self.gaussian_kernel_size = gaussian_kernel_size
        self.area_threshold = area_threshold
        self.thresh_offset = thresh_offset
     
    def run(self):
        image = self.fluo_image.image

        kernel_size = max(self.gaussian_kernel_size) * 4
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(image, background)

        block_size = 321
        local_thresh = threshold_local(corrected, block_size, offset=self.thresh_offset)
        binary_image = (corrected > local_thresh).astype(np.uint8) * 255

        distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

        coords = peak_local_max(
            distance,
            footprint=np.ones((3, 3)),
            labels=binary_image,
            exclude_border=False
        )

        local_maxi = np.zeros_like(distance, dtype=bool)
        local_maxi[tuple(coords.T)] = True

        markers = ndi.label(local_maxi)[0]

        num_labels, labels = cv2.connectedComponents(binary_image)

        label_sizes = np.bincount(labels.ravel())
        invalid_labels = (label_sizes < self.area_threshold[0]) | (label_sizes > self.area_threshold[1])
        labels[np.isin(labels, np.where(invalid_labels)[0])] = 0

        self.fluo_image.segmentation_mask = labels
        self.fluo_image.thresholded_mask = labels
        self.fluo_image.intensities = None
        self.fluo_image.update()

        self.finished.emit()

class FluoImage(QLabel):
    def __init__(self, main_window: QMainWindow):
        super().__init__()

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.main_window = main_window
        self.image = None
        self.segmentation_mask = None
        self.thresholded_mask = None
        self.intensities = None
        self.ongoing_operation = None
        self.scale = None
        self.offset_x = 0
        self.offset_y = 0
        self.display_scale = 1.0
        self.draw_x = 0
        self.draw_y = 0

        self.add_threshold_dock()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drop fluo image")
        self.setStyleSheet("border: 1px solid black")
    
    def paintEvent(self, event: QPaintEvent):
        if self.image is None:
            return super().paintEvent(event)

        painter = QPainter(self)
        pen = QPen(Qt.green, 5)
        painter.setPen(pen)
        
        qimage = np_to_qimage_display(self.image)
        pixmap = QPixmap.fromImage(qimage)

        img_h, img_w = self.image.shape[:2]
        disp_w = int(img_w / self.display_scale)
        disp_h = int(img_h / self.display_scale)
        scaled_pixmap = pixmap.scaled(disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter.drawPixmap(self.draw_x, self.draw_y, scaled_pixmap)

        if self.ongoing_operation is not None:
            self.ongoing_operation.paintEvent(painter)

        if self.segmentation_mask is not None:
            self.overlay_mask(painter, disp_w, disp_h)

        painter.end()
    
    def overlay_mask(self, painter: QPainter, disp_w: int, disp_h: int):
        if self.thresholded_mask is None:
            return
        if np.max(self.thresholded_mask) == 0:
            return
        mask = self.thresholded_mask.astype(np.int32)

        m = mask.copy()
        m[m < 0] = 0
        m8 = (m.astype(np.float32) / (m.max() if m.max() > 0 else 1.0) * 255).astype(np.uint8)
        
        color_mask =cv2.applyColorMap(m8, cv2.COLORMAP_JET)
        qimage_mask = QImage(color_mask.data, color_mask.shape[1], color_mask.shape[0],
                             color_mask.strides[0], QImage.Format_RGB888)
        pixmap_mask = QPixmap.fromImage(qimage_mask)
        scaled_mask = pixmap_mask.scaled(disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter.setOpacity(0.3)
        painter.drawPixmap(self.draw_x + self.offset_x, self.draw_y + self.offset_y, scaled_mask)
        painter.setOpacity(1.0)

    def keyPressEvent(self, event: QKeyEvent):
        if self.ongoing_operation is not None and event.key() == 16777220:
            self.image = self.ongoing_operation.applyTo(self.image, self.scale)
            self.ongoing_operation = None
            self.update()
        elif event.key() == 16777216:
            self.ongoing_operation = None
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.ongoing_operation is not None: 
            self.ongoing_operation.mousePressEvent(event)
            self.update()
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.endswith("npz") and self.image is not None:
                seg = np.load(url)
                self.segmentation_mask = seg["im_markers"]
                self.thresholded_mask = seg["im_markers"]
                self.intensities = None
                self.update()
            else:
                import os
                ext = os.path.splitext(url)[1].lower()
                try:
                    if ext in [".tif", ".tiff"]:
                        from skimage import io as skio
                        im = skio.imread(url)
                        if im.ndim > 2:
                            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                        self.image = im
                    else:
                        self.image = cv2.imdecode(np.fromfile(url, np.uint8), cv2.IMREAD_GRAYSCALE)
                except Exception as e:
                    self.show_error("Load error", f"Could not load image:\n{e}")
                    return

            self.segmentation_mask = None
            self.thresholded_mask = None
            self.intensities = None
            self.ongoing_operation = None

            img_h, img_w = self.image.shape[:2]
            lbl_w, lbl_h = self.width(), self.height()
            scale_w = img_w / lbl_w
            scale_h = img_h / lbl_h
            self.display_scale = max(scale_w, scale_h) if max(scale_w, scale_h) > 0 else 1.0

            disp_w = int(img_w / self.display_scale)
            disp_h = int(img_h / self.display_scale)

            self.draw_x = (lbl_w - disp_w) // 2
            self.draw_y = (lbl_h - disp_h) // 2
            self.scale = self.display_scale

            self.offset_x = 0
            self.offset_y = 0
            self.update()

    def contextMenuEvent(self, event: QContextMenuEvent):
        
        if self.image is not None and self.ongoing_operation is None:
            context_menu = QMenu(self)

            action1 = QAction("Circle mask", self)
            action1.triggered.connect(self.start_circle_mask_operation)
            context_menu.addAction(action1)

            action2 = QAction("Label cut", self)
            action2.triggered.connect(self.start_label_cut_operation)
            context_menu.addAction(action2)

            action3 = QAction("Segment", self)
            action3.triggered.connect(self.add_segmentation_dock)
            context_menu.addAction(action3)

            action_bf = QAction("Brightfield panel", self)
            action_bf.triggered.connect(self.add_brightfield_dock)
            context_menu.addAction(action_bf)

            action4 = QMenu("Export", self)

            csv_action = QAction("Export CSV", self)
            csv_action.triggered.connect(self.start_exporting_csv)
            action4.addAction(csv_action)

            overlay_action = QAction("Export colored overlay", self)
            overlay_action.triggered.connect(self.export_overlay_mask)
            action4.addAction(overlay_action)

            imagej_action = QAction("Export mask for ImageJ overlay", self)
            imagej_action.triggered.connect(self.export_imagej_overlay_mask)
            action4.addAction(imagej_action)

            context_menu.addMenu(action4)
            context_menu.exec_(event.globalPos())
    
    def export_imagej_overlay_mask(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Binary Mask for ImageJ Overlay", "", "TIFF Files (*.tiff);;All Files (*)", options=options
        )
        if not file_path:
            return

        mask = (self.thresholded_mask > 0).astype(np.uint8) * 255

        cv2.imwrite(file_path, mask)

    def start_circle_mask_operation(self):
        self.ongoing_operation = CircleMaskOperation()
    
    def start_label_cut_operation(self):
        self.ongoing_operation = LabelCutOperation()
    
    def start_exporting_csv(self):
        if self.thresholded_mask is None or self.intensities is None:
            self.show_error("Unfinished workflow", "Please segment and threshold the image first.")
            return

        cell_ids = np.unique(self.thresholded_mask)[1:]
        cell_count, areas, integrals = len(cell_ids), {}, {}
        for id in cell_ids:
            areas[id] = np.count_nonzero(self.thresholded_mask == id)
            integrals[id] = self.intensities[id]

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if not file_path:
            return

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["Total Cell Count", cell_count])
            writer.writerow(["Cell ID", "Area", "Integral Intensity"])
            
            for id in cell_ids:
                writer.writerow([id, areas[id], integrals[id]])

    def export_overlay_mask(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Colored Overlay", "", "TIFF Files (*.tiff);;All Files (*)", options=options
        )
        if not file_path:
            return

        img8 = (self.image.astype(np.float32) / self.image.max() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        mask = self.thresholded_mask.astype(np.int32)
        m = mask.copy()
        m[m < 0] = 0
        if m.max() == 0:
            return

        m8 = (m.astype(np.float32) / m.max() * 255).astype(np.uint8)
        color_mask = cv2.applyColorMap(m8, cv2.COLORMAP_JET)

        cell_mask = (mask > 0).astype(np.uint8)
        cell_mask_3ch = np.repeat(cell_mask[:, :, np.newaxis], 3, axis=2)

        overlay = np.where(cell_mask_3ch == 1, color_mask, img_rgb)

        cv2.imwrite(file_path, overlay)

    def add_segmentation_dock(self):
        dock_widget = QDockWidget("Segment", self)
        widget = QWidget()
        dock_widget.setWidget(widget)

        layout = QFormLayout()

        kernel_x = QSpinBox()
        kernel_x.setRange(1, 1000)
        kernel_x.setValue(21)
        kernel_y = QSpinBox()
        kernel_y.setRange(1, 1000)
        kernel_y.setValue(21)
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(kernel_x)
        kernel_layout.addWidget(kernel_y)
        layout.addRow("Gaussian kernel size", kernel_layout)

        area_min_threshold = QSpinBox()
        area_min_threshold.setRange(1, 10000)
        area_min_threshold.setValue(3000)
        layout.addRow("Area min threshold", area_min_threshold)

        area_max_threshold = QSpinBox()
        area_max_threshold.setRange(1000, 1000000)
        area_max_threshold.setValue(17000)
        layout.addRow("Area max threshold", area_max_threshold)

        thresh_offset = QSpinBox()
        thresh_offset.setRange(-100, 100)
        thresh_offset.setValue(0)
        layout.addRow("Threshold offset", thresh_offset)

        apply_segmentation_button = QPushButton("Apply")
        layout.addRow(apply_segmentation_button)

        def start_segment_task():
            apply_segmentation_button.setEnabled(False)
            apply_segmentation_button.setText("Calculating...")

            self.worker_thread = QThread()
            self.worker = SegmentWorker(
                self,
                (kernel_x.value(), kernel_y.value()),
                (area_min_threshold.value(), area_max_threshold.value()),
                thresh_offset.value()
            )
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(lambda: apply_segmentation_button.setText("Apply"))
            self.worker.finished.connect(lambda: apply_segmentation_button.setEnabled(True))
            self.worker_thread.start()

        apply_segmentation_button.clicked.connect(start_segment_task)

        widget.setLayout(layout)
        self.main_window.addDockWidget(2, dock_widget)


    def add_threshold_dock(self):
        dock_widget = QDockWidget("Threshold", self)
        dock_widget.setFeatures(dock_widget.DockWidgetFeature.DockWidgetVerticalTitleBar)

        widget = QWidget()
        dock_widget.setWidget(widget)

        layout = QFormLayout()

        slider_upper = QSlider(Qt.Orientation.Horizontal)
        slider_upper.setRange(0, 255)
        slider_upper.setValue(255)
        layout.addRow("Upper value", slider_upper)

        label_upper = QLabel(f"{slider_upper.value()}")
        label_upper.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(label_upper)
        slider_upper.valueChanged.connect(lambda: label_upper.setText(f"{slider_upper.value()}"))

        slider_lower = QSlider(Qt.Orientation.Horizontal)
        slider_lower.setRange(0, 255)
        layout.addRow("Lower value", slider_lower)

        label_lower = QLabel(f"{slider_lower.value()}")
        label_lower.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(label_lower)
        slider_lower.valueChanged.connect(lambda: label_lower.setText(f"{slider_lower.value()}"))

        apply_button = QPushButton("Apply")
        layout.addRow(apply_button)

        def start_threshold_task():
            if self.image is not None and np.issubdtype(self.image.dtype, np.integer):
                maxval = int(self.image.max())
                if maxval <= 0:
                    maxval = 255
                slider_upper.setRange(0, maxval)
                slider_lower.setRange(0, maxval)
                if slider_upper.value() > maxval:
                    slider_upper.setValue(maxval)
            apply_button.setEnabled(False)
            apply_button.setText("Calculating...")

            self.worker_thread = QThread()
            self.worker = ThresholdWorker(self, slider_lower.value(), slider_upper.value())
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(lambda: apply_button.setText("Apply"))
            self.worker.finished.connect(lambda: apply_button.setEnabled(True))
            self.worker_thread.start()

        apply_button.clicked.connect(start_threshold_task)

        widget.setLayout(layout)
        self.main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_widget)

    def add_brightfield_dock(self):
        dock = QDockWidget("Brightfield", self)
        w = QWidget(); dock.setWidget(w)
        form = QFormLayout()

        mode = QComboBox(); mode.addItems(["Fast count (downsample)", "Segment (full-res)"])
        mode.setCurrentIndex(0)
        form.addRow("Mode", mode)

        downsample = QSpinBox(); downsample.setRange(1, 8); downsample.setValue(2)
        form.addRow("Downsample (×)", downsample)

        skip_overlay = QCheckBox(); skip_overlay.setChecked(True)
        form.addRow("Skip overlay (fast)", skip_overlay)

        bg_kernel = QSpinBox(); bg_kernel.setRange(11, 501); bg_kernel.setSingleStep(10); bg_kernel.setValue(101)
        form.addRow("Background kernel", bg_kernel)

        method = QComboBox(); method.addItems(["adaptive","otsu"]); method.setCurrentText("adaptive")
        form.addRow("Threshold method", method)

        block = QSpinBox(); block.setRange(11, 301); block.setSingleStep(2); block.setValue(71)
        form.addRow("Adaptive block (odd)", block)

        offset = QSpinBox(); offset.setRange(-50, 50); offset.setValue(-6)
        form.addRow("Adaptive offset (C)", offset)

        open_k = QSpinBox(); open_k.setRange(0, 31); open_k.setValue(3)
        form.addRow("Open kernel", open_k)

        close_k = QSpinBox(); close_k.setRange(0, 31); close_k.setValue(4)
        form.addRow("Close kernel", close_k)

        min_area = QSpinBox(); min_area.setRange(1, 1_000_000); min_area.setValue(30)
        form.addRow("Min area (px)", min_area)

        max_area = QSpinBox(); max_area.setRange(10, 5_000_000); max_area.setValue(20_000)
        form.addRow("Max area (px)", max_area)

        alpha = QDoubleSpinBox(); alpha.setRange(0.0, 1.0); alpha.setSingleStep(0.05); alpha.setDecimals(2); alpha.setValue(0.35)
        form.addRow("Overlay alpha", alpha)

        run_btn = QPushButton("Run")
        form.addRow(run_btn)

        def run_brightfield():
            if self.image is None:
                self.show_error("No image", "Load an image first."); return
            run_btn.setEnabled(False); run_btn.setText("Calculating...")

            self.worker_thread = QThread()
            self.worker = BrightfieldWorker(
                self,
                bg_kernel.value(),
                method.currentText(),
                block.value(),
                offset.value(),
                open_k.value(),
                close_k.value(),
                min_area.value(),
                max_area.value(),
                alpha.value(),downsample=downsample.value(),
                mode=mode.currentText(),
                skip_overlay=skip_overlay.isChecked()
            )
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.run)

            def on_done(labels_clean, overlay_bgr, df):
                self.segmentation_mask = labels_clean
                self.thresholded_mask = labels_clean
                self.intensities = None
                run_btn.setText("Run"); run_btn.setEnabled(True)
                self.worker_thread.quit()
                self.update()
                if labels_clean is not None:
                    count = int(labels_clean.max())
                    QMessageBox.information(self, "Brightfield", f"Counted cells: {count}")

            self.worker.finished.connect(on_done)
            self.worker_thread.start()

        run_btn.clicked.connect(run_brightfield)
        w.setLayout(form)
        self.main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)


    def show_error(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    app.exec()

if __name__ == "__main__":
    main()
