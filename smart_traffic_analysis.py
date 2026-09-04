# -*- coding: utf-8 -*-
"""
================================================================================
پروژه: تحلیل هوشمند ترافیک و تقاطع‌ها (Smart Traffic Analysis)
================================================================================
این اسکریپت یک پایپ‌لاین کامل بینایی ماشین برای تحلیل ویدیوهای ترافیکی است.
قابلیت‌های اصلی:
    1) تشخیص خودروها با YOLOv8/YOLOv11 (Ultralytics)
    2) ردیابی چندگانه (Multi-Object Tracking) با الگوریتم ByteTrack
       (از طریق کتابخانه‌ی Supervision - که خودش ByteTrack را پیاده‌سازی کرده)
    3) شمارش خودروها بر اساس عبور از خطوط فرضی (Line Crossing Counting)
       با تفکیک جهت حرکت (بالا->پایین / پایین->بالا)
    4) تخمین سرعت تقریبی و تشخیص خودروهای متوقف/کند بر اساس مسیر حرکت مرکز کادرها
    5) نمایش پنل اطلاعات زنده (Dashboard Overlay) روی فریم خروجی

نویسنده: تیم مهندسی بینایی ماشین
نیازمندی‌ها:
    pip install ultralytics supervision opencv-python numpy pandas
================================================================================
"""

import os
import sys
import time
import argparse
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Deque, Tuple, List, Optional

import numpy as np
import cv2
import pandas as pd

try:
    from ultralytics import YOLO
except ImportError:
    print("خطا: کتابخانه ultralytics نصب نیست. با دستور زیر آن را نصب کنید:")
    print("    pip install ultralytics")
    sys.exit(1)

try:
    import supervision as sv
except ImportError:
    print("خطا: کتابخانه supervision نصب نیست. با دستور زیر آن را نصب کنید:")
    print("    pip install supervision")
    sys.exit(1)


# ==============================================================================
# بخش ۱: تعریف ساختارهای پیکربندی (Configuration)
# ==============================================================================

@dataclass
class TrafficLine:
    """
    نمایانگر یک «خط فرضی» عبور در فریم ویدیو.
    خودروهایی که مرکز کادرشان از این خط عبور کند، شمارش می‌شوند.
    مختصات به صورت (x, y) و بر حسب پیکسل‌های فریم هستند.
    """
    name: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 255)  # رنگ پیش‌فرض: زرد (BGR)

    # شمارنده‌های عبور در دو جهت مختلف (بر اساس علامت حاصل‌ضرب خارجی بردارها)
    count_a_to_b: int = 0
    count_b_to_a: int = 0


@dataclass
class AnalysisConfig:
    """پیکربندی کلی پایپ‌لاین تحلیل ترافیک."""
    model_path: str = "yolov8n.pt"          # مسیر مدل YOLO (از پیش آموزش‌دیده یا سفارشی)
    video_source: str = "input.mp4"          # مسیر ویدیوی ورودی (یا 0 برای وبکم)
    output_path: str = "output_annotated.mp4"
    log_csv_path: str = "traffic_log.csv"

    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5

    # کلاس‌های مورد نظر برای شمارش (بر اساس شناسه کلاس در COCO/مدل سفارشی)
    # مقدار None یعنی همه‌ی کلاس‌ها در نظر گرفته شوند
    target_classes: Optional[List[int]] = None

    # تعداد فریم‌هایی که برای تحلیل مسیر حرکت و سرعت هر شیء نگه‌داری می‌شود
    trajectory_length: int = 30

    # آستانه‌ی سرعت (پیکسل بر فریم) برای تشخیص خودروی «متوقف/کند»
    stationary_speed_threshold: float = 1.5

    # نرخ فریم واقعی ویدیو برای محاسبه‌ی سرعت (در صورت نامشخص بودن، از خود ویدیو خوانده می‌شود)
    assumed_fps: Optional[float] = None

    # ضریب تبدیل پیکسل به متر (اختیاری - برای تخمین سرعت واقعی به‌جای پیکسل بر ثانیه)
    # اگر None باشد، سرعت به واحد «پیکسل بر ثانیه» گزارش می‌شود.
    meters_per_pixel: Optional[float] = None

    # نمایش زنده‌ی پنجره‌ی پیش‌نمایش در حین پردازش (در محیط‌های بدون نمایشگر غیرفعال کنید)
    show_live_preview: bool = False


# ==============================================================================
# بخش ۲: کلاس تحلیل مسیر حرکت و سرعت (Trajectory & Speed Analyzer)
# ==============================================================================

class TrajectoryAnalyzer:
    """
    این کلاس مسئول ذخیره‌ی تاریخچه‌ی مکان هر خودرو (بر اساس ID یکتای ردیاب) است
    و از روی این تاریخچه، سرعت لحظه‌ای و وضعیت توقف را محاسبه می‌کند.
    """

    def __init__(self, max_len: int, fps: float, meters_per_pixel: Optional[float]):
        self.max_len = max_len
        self.fps = fps if fps and fps > 0 else 30.0
        self.meters_per_pixel = meters_per_pixel

        # برای هر شناسه‌ی ردیابی، یک صف با آخرین موقعیت‌های مرکز کادر نگه می‌داریم
        self.history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.max_len)
        )

    def update(self, tracker_id: int, center: Tuple[int, int]) -> None:
        """ثبت موقعیت جدید یک خودرو در تاریخچه‌ی حرکتش."""
        self.history[tracker_id].append(center)

    def get_speed(self, tracker_id: int) -> float:
        """
        محاسبه‌ی سرعت تقریبی بر اساس جابه‌جایی بین اولین و آخرین نقطه‌ی ثبت‌شده
        در پنجره‌ی زمانی موجود. خروجی بر حسب پیکسل بر ثانیه (یا متر بر ثانیه در صورت کالیبراسیون).
        """
        points = self.history.get(tracker_id)
        if not points or len(points) < 2:
            return 0.0

        (x1, y1), (x2, y2) = points[0], points[-1]
        distance_px = float(np.hypot(x2 - x1, y2 - y1))

        # مدت زمان طی‌شده بین اولین و آخرین نمونه بر حسب ثانیه
        elapsed_frames = len(points) - 1
        elapsed_seconds = elapsed_frames / self.fps if self.fps > 0 else 1.0
        if elapsed_seconds <= 0:
            return 0.0

        speed_px_per_sec = distance_px / elapsed_seconds

        if self.meters_per_pixel:
            speed_mps = speed_px_per_sec * self.meters_per_pixel
            return speed_mps * 3.6  # تبدیل متر بر ثانیه به کیلومتر بر ساعت
        return speed_px_per_sec

    def is_stationary(self, tracker_id: int, threshold_px_per_frame: float) -> bool:
        """تشخیص اینکه آیا خودرو در چند فریم اخیر عملاً بی‌حرکت بوده است یا خیر."""
        points = self.history.get(tracker_id)
        if not points or len(points) < 2:
            return False

        # میانگین جابه‌جایی بین فریم‌های متوالی
        displacements = [
            np.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            for i in range(1, len(points))
        ]
        avg_displacement = float(np.mean(displacements))
        return avg_displacement < threshold_px_per_frame

    def cleanup(self, active_ids: set) -> None:
        """پاک‌سازی تاریخچه‌ی خودروهایی که دیگر در فریم فعلی دیده نمی‌شوند (کاهش مصرف حافظه)."""
        stale_ids = [tid for tid in self.history.keys() if tid not in active_ids]
        for tid in stale_ids:
            # تاریخچه را کاملاً حذف نمی‌کنیم، بلکه اجازه می‌دهیم خودش رشد نکند؛
            # در پروژه‌های واقعی و طولانی‌مدت می‌توان با شمارنده‌ی غیبت، حذف واقعی انجام داد.
            pass


# ==============================================================================
# بخش ۳: کلاس شمارش عبور از خط (Line Crossing Counter)
# ==============================================================================

class LineCrossingCounter:
    """
    این کلاس با استفاده از حاصل‌ضرب خارجی بردارها (Cross Product) تشخیص می‌دهد
    که آیا یک شیء از یک خط فرضی عبور کرده است یا نه، و در چه جهتی.
    """

    def __init__(self, lines: List[TrafficLine]):
        self.lines = lines
        # برای هر شیء و هر خط، آخرین «سمت» خط که شیء در آن قرار داشته را نگه می‌داریم
        # sign مثبت یعنی یک سمت خط، منفی یعنی سمت دیگر
        self._last_side: Dict[Tuple[int, str], float] = {}
        # مجموعه‌ای برای جلوگیری از شمارش تکراری یک شیء از یک خط در یک بازه‌ی زمانی
        self._counted_recently: Dict[Tuple[int, str], int] = {}

    @staticmethod
    def _side_of_line(point: Tuple[int, int], line: TrafficLine) -> float:
        """
        محاسبه‌ی علامت قرارگیری یک نقطه نسبت به خط (سمت راست یا چپ بردار خط).
        از فرمول حاصل‌ضرب خارجی دو بردار دو‌بعدی استفاده می‌شود.
        """
        (x1, y1), (x2, y2) = line.start, line.end
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return cross

    def update(self, tracker_id: int, center: Tuple[int, int], frame_index: int) -> None:
        """
        بررسی عبور یک خودروی مشخص از تمامی خطوط تعریف‌شده و به‌روزرسانی شمارنده‌ها.
        """
        for line in self.lines:
            key = (tracker_id, line.name)
            current_side = self._side_of_line(center, line)

            if key not in self._last_side:
                # اولین باری که این شیء دیده می‌شود؛ فقط سمت فعلی را ثبت می‌کنیم
                self._last_side[key] = current_side
                continue

            previous_side = self._last_side[key]

            # اگر علامت تغییر کرده باشد یعنی خط را قطع کرده است
            crossed = (previous_side <= 0 and current_side > 0) or \
                      (previous_side >= 0 and current_side < 0)

            # جلوگیری از شمارش چندباره‌ی یک عبور در فریم‌های پیاپی (نویز نزدیک خط)
            last_counted_frame = self._counted_recently.get(key, -999)
            cooldown_ok = (frame_index - last_counted_frame) > 5

            if crossed and previous_side != 0 and cooldown_ok:
                if previous_side < 0 and current_side > 0:
                    line.count_a_to_b += 1
                else:
                    line.count_b_to_a += 1
                self._counted_recently[key] = frame_index

            self._last_side[key] = current_side


# ==============================================================================
# بخش ۴: کلاس رسم پنل اطلاعات روی فریم (Dashboard Renderer)
# ==============================================================================

class DashboardRenderer:
    """مسئول رسم اطلاعات آماری و گرافیکی روی فریم خروجی با OpenCV."""

    def __init__(self, lines: List[TrafficLine]):
        self.lines = lines
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=30)

    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """رسم خطوط فرضی شمارش روی فریم."""
        for line in self.lines:
            cv2.line(frame, line.start, line.end, line.color, thickness=3)
            mid_point = (
                (line.start[0] + line.end[0]) // 2,
                (line.start[1] + line.end[1]) // 2,
            )
            cv2.putText(
                frame, line.name, mid_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, line.color, 2, cv2.LINE_AA
            )
        return frame

    def draw_stat_panel(
        self, frame: np.ndarray, fps: float, active_tracks: int,
        total_counted: int, stationary_count: int
    ) -> np.ndarray:
        """رسم پنل شفاف اطلاعات آماری در گوشه‌ی بالا-چپ فریم."""
        panel_h = 40 + 30 * (2 + len(self.lines))
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, panel_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        y = 35
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28
        cv2.putText(frame, f"Active IDs: {active_tracks} | Stationary: {stationary_count}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
        cv2.putText(frame, f"Total Crossed: {total_counted}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

        for line in self.lines:
            y += 26
            text = f"{line.name}  A->B: {line.count_a_to_b}  B->A: {line.count_b_to_a}"
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, line.color, 1, cv2.LINE_AA)

        return frame

    def draw_detections(
        self, frame: np.ndarray, detections: "sv.Detections", labels: List[str]
    ) -> np.ndarray:
        """رسم کادرهای تشخیص، شناسه‌ها و مسیر حرکت خودروها با استفاده از Supervision."""
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return frame


# ==============================================================================
# بخش ۵: کلاس اصلی پایپ‌لاین تحلیل ترافیک (Main Pipeline)
# ==============================================================================

class SmartTrafficAnalyzer:
    """
    کلاس هماهنگ‌کننده‌ی اصلی که تمام مراحل (تشخیص، ردیابی، شمارش، تحلیل سرعت،
    رسم داشبورد و ذخیره‌ی خروجی) را به هم متصل می‌کند.
    """

    def __init__(self, config: AnalysisConfig, lines: List[TrafficLine]):
        self.config = config
        self.lines = lines

        # بارگذاری مدل YOLO (وزن‌های از پیش آموزش‌دیده یا مدل سفارشی آموزش‌دیده روی داده‌ی خودمان)
        self.model = YOLO(config.model_path)

        # ByteTrack از طریق کتابخانه‌ی Supervision؛ الگوریتمی سبک و دقیق برای ردیابی چندگانه
        self.tracker = sv.ByteTrack()

        self.line_counter = LineCrossingCounter(lines)

        # این مقدار بعد از باز کردن ویدیو مقداردهی نهایی می‌شود (چون به FPS واقعی ویدیو نیاز دارد)
        self.trajectory_analyzer: Optional[TrajectoryAnalyzer] = None
        self.dashboard: Optional[DashboardRenderer] = None

        # لاگ آماری برای ذخیره‌ی خروجی نهایی در فایل CSV
        self.log_records: List[dict] = []

    def _resolve_video_source(self):
        """
        در صورتی که مسیر ورودی رشته‌ی عددی باشد (مثلاً '0')، آن را به عدد صحیح
        تبدیل می‌کند تا OpenCV بتواند وبکم را باز کند.
        """
        src = self.config.video_source
        if isinstance(src, str) and src.isdigit():
            return int(src)
        return src

    def run(self) -> None:
        """اجرای کامل پایپ‌لاین از ابتدا (خواندن ویدیو) تا انتها (ذخیره‌ی خروجی و لاگ)."""

        video_source = self._resolve_video_source()
        cap = cv2.VideoCapture(video_source)

        # مدیریت خطا: بررسی موفقیت‌آمیز بودن باز شدن ویدیو
        if not cap.isOpened():
            raise IOError(
                f"امکان باز کردن منبع ویدیویی وجود ندارد: {self.config.video_source}\n"
                "لطفاً مسیر فایل یا شماره‌ی دوربین را بررسی کنید."
            )

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_for_speed = self.config.assumed_fps or (source_fps if source_fps > 0 else 30.0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.trajectory_analyzer = TrajectoryAnalyzer(
            max_len=self.config.trajectory_length,
            fps=fps_for_speed,
            meters_per_pixel=self.config.meters_per_pixel,
        )
        self.dashboard = DashboardRenderer(self.lines)

        # آماده‌سازی نویسنده‌ی ویدیوی خروجی
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.config.output_path, fourcc, source_fps if source_fps > 0 else 30.0,
            (frame_width, frame_height)
        )
        if not writer.isOpened():
            cap.release()
            raise IOError(f"امکان ایجاد فایل خروجی وجود ندارد: {self.config.output_path}")

        frame_index = 0
        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # پایان طبیعی ویدیو یا خطای خواندن فریم
                    break

                frame_index += 1

                # ---------- مرحله‌ی ۱: تشخیص خودروها با YOLO ----------
                results = self.model.predict(
                    source=frame,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    classes=self.config.target_classes,
                    verbose=False,
                )[0]

                detections = sv.Detections.from_ultralytics(results)

                # ---------- مرحله‌ی ۲: ردیابی چندگانه با ByteTrack ----------
                detections = self.tracker.update_with_detections(detections)

                active_ids = set()
                stationary_count = 0
                labels: List[str] = []

                # ---------- مرحله‌ی ۳ و ۴: شمارش عبور از خط + تحلیل مسیر/سرعت ----------
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                    class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                    confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0

                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    if tracker_id == -1:
                        labels.append(f"#{class_id} {confidence:.2f}")
                        continue

                    active_ids.add(tracker_id)

                    # به‌روزرسانی مسیر حرکت این خودرو
                    self.trajectory_analyzer.update(tracker_id, center)

                    # بررسی عبور از خطوط فرضی
                    self.line_counter.update(tracker_id, center, frame_index)

                    # محاسبه‌ی سرعت و وضعیت توقف
                    speed = self.trajectory_analyzer.get_speed(tracker_id)
                    is_stationary = self.trajectory_analyzer.is_stationary(
                        tracker_id, self.config.stationary_speed_threshold
                    )
                    if is_stationary:
                        stationary_count += 1

                    speed_unit = "km/h" if self.config.meters_per_pixel else "px/s"
                    status_tag = "STOPPED" if is_stationary else ""
                    labels.append(f"ID:{tracker_id} {speed:.1f}{speed_unit} {status_tag}".strip())

                    # ثبت رکورد آماری برای خروجی CSV نهایی
                    self.log_records.append({
                        "frame": frame_index,
                        "tracker_id": tracker_id,
                        "class_id": class_id,
                        "confidence": round(confidence, 3),
                        "center_x": center[0],
                        "center_y": center[1],
                        "speed": round(speed, 2),
                        "is_stationary": is_stationary,
                    })

                # پاک‌سازی حافظه‌ی خودروهایی که دیگر دیده نمی‌شوند
                self.trajectory_analyzer.cleanup(active_ids)

                # ---------- مرحله‌ی ۵: رسم داشبورد و خروجی گرافیکی ----------
                annotated_frame = self.dashboard.draw_detections(frame.copy(), detections, labels)
                annotated_frame = self.dashboard.draw_lines(annotated_frame)

                now = time.time()
                live_fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                total_counted = sum(l.count_a_to_b + l.count_b_to_a for l in self.lines)
                annotated_frame = self.dashboard.draw_stat_panel(
                    annotated_frame, live_fps, len(active_ids), total_counted, stationary_count
                )

                writer.write(annotated_frame)

                # نمایش زنده اختیاری (در محیط‌های بدون نمایشگر مانند سرور، این بخش را غیرفعال کنید)
                if self.config.show_live_preview:
                    cv2.imshow("Smart Traffic Analysis", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except Exception as error:
            # مدیریت خطا: هرگونه خطای غیرمنتظره در حین پردازش گزارش و ویدیو تا همان لحظه ذخیره می‌شود
            print(f"خطا در حین پردازش فریم شماره {frame_index}: {error}")
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            self._export_log()
            self._print_summary()

    def _export_log(self) -> None:
        """ذخیره‌ی تمام رکوردهای آماری جمع‌آوری‌شده در یک فایل CSV."""
        if not self.log_records:
            print("هشدار: هیچ رکورد آماری‌ای برای ذخیره‌سازی وجود ندارد.")
            return
        df = pd.DataFrame(self.log_records)
        df.to_csv(self.config.log_csv_path, index=False, encoding="utf-8-sig")
        print(f"لاگ آماری با موفقیت ذخیره شد: {self.config.log_csv_path}")

    def _print_summary(self) -> None:
        """چاپ خلاصه‌ی نهایی نتایج شمارش در ترمینال."""
        print("\n" + "=" * 60)
        print("خلاصه‌ی نتایج تحلیل ترافیک")
        print("=" * 60)
        for line in self.lines:
            print(f"خط «{line.name}» :  جهت A->B = {line.count_a_to_b}  |  جهت B->A = {line.count_b_to_a}")
        print(f"ویدیوی پردازش‌شده در مسیر ذخیره شد: {self.config.output_path}")
        print("=" * 60)


# ==============================================================================
# بخش ۶: نقطه‌ی ورود برنامه (CLI Entry Point)
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """پردازش آرگومان‌های خط فرمان برای اجرای انعطاف‌پذیر اسکریپت."""
    parser = argparse.ArgumentParser(
        description="Smart Traffic Analysis System with YOLO and ByteTrack / "
                    "سیستم تحلیل هوشمند ترافیک با YOLO و ByteTrack"
    )
    parser.add_argument("--source", type=str, default="input.mp4", help="مسیر ویدیوی ورودی یا شماره‌ی وبکم")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="مسیر وزن‌های مدل YOLO")
    parser.add_argument("--output", type=str, default="output_annotated.mp4", help="مسیر ویدیوی خروجی")
    parser.add_argument("--log", type=str, default="traffic_log.csv", help="مسیر فایل CSV لاگ آماری")
    parser.add_argument("--conf", type=float, default=0.35, help="آستانه‌ی اطمینان تشخیص")
    parser.add_argument("--show", action="store_true", help="نمایش زنده‌ی پردازش در یک پنجره")
    return parser.parse_args()


def build_default_lines(frame_width: int = 1280, frame_height: int = 720) -> List[TrafficLine]:
    """
    تعریف پیش‌فرض یک خط افقی وسط فریم برای شمارش عبور خودروها.
    در پروژه‌ی واقعی، این مختصات باید متناسب با زاویه‌ی دوربین و تقاطع مورد نظر تنظیم شود.
    """
    return [
        TrafficLine(
            name="Line-1",
            start=(0, int(frame_height * 0.6)),
            end=(frame_width, int(frame_height * 0.6)),
            color=(0, 255, 255),
        )
    ]


def main() -> None:
    args = parse_args()

    config = AnalysisConfig(
        model_path=args.model,
        video_source=args.source,
        output_path=args.output,
        log_csv_path=args.log,
        confidence_threshold=args.conf,
        show_live_preview=args.show,
    )

    # مدیریت خطا: بررسی وجود فایل مدل و ویدیو پیش از شروع پردازش سنگین
    if not os.path.isfile(config.model_path) and not config.model_path.endswith(".pt"):
        print(f"هشدار: مسیر مدل معتبر به نظر نمی‌رسد: {config.model_path}")

    if isinstance(config.video_source, str) and not config.video_source.isdigit():
        if not os.path.isfile(config.video_source):
            print(f"خطا: فایل ویدیوی ورودی یافت نشد: {config.video_source}")
            sys.exit(1)

    # می‌توانید در اینجا چند خط دلخواه با مختصات دقیق تعریف کنید؛
    # مقادیر پیش‌فرض صرفاً یک نمونه‌ی نمایشی هستند.
    lines = build_default_lines()

    analyzer = SmartTrafficAnalyzer(config, lines)
    analyzer.run()


if __name__ == "__main__":
    main()
