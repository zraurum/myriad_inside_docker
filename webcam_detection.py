#!/usr/bin/env python3
"""
Скрипт для детекции объектов в реальном времени с веб-камеры
На основе object-detection.ipynb
"""

import sys
import os
from time import sleep

# САМОЕ ПЕРВОЕ - настройка вывода
sys.stderr.write("=== СКРИПТ ЗАПУЩЕН ===\n")
sys.stderr.flush()

# Убедимся, что вывод не буферизуется
os.environ['PYTHONUNBUFFERED'] = '1'
# Отключаем GUI для работы в Docker
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''

# Функция для вывода отладочной информации в stderr
def debug_print(msg):
    try:
        sys.stderr.write(str(msg) + "\n")
        sys.stderr.flush()
    except:
        pass

debug_print("=== Начало выполнения скрипта ===")
try:
    debug_print(f"Python версия: {sys.version}")
    debug_print(f"Рабочая директория: {os.getcwd()}")
except Exception as e:
    debug_print(f"Ошибка при выводе информации: {e}")

try:
    debug_print("Импорт cv2...")
    import cv2
    cv2.setNumThreads(0)
    debug_print(f"cv2 версия: {cv2.__version__}")
except Exception as e:
    debug_print(f"ОШИБКА импорта cv2: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

try:
    debug_print("Импорт numpy...")
    import numpy as np
    debug_print(f"numpy версия: {np.__version__}")
except Exception as e:
    debug_print(f"ОШИБКА импорта numpy: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

try:
    debug_print("Импорт pathlib...")
    from pathlib import Path
except Exception as e:
    debug_print(f"ОШИБКА импорта pathlib: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

try:
    debug_print("Импорт ultralytics...")
    from ultralytics import YOLO
    debug_print("ultralytics импортирован успешно")
except Exception as e:
    debug_print(f"ОШИБКА импорта ultralytics: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

# Попытка импорта OpenVINO (новый API)
OPENVINO_NEW_API = False
OPENVINO_OLD_API = False
OPENVINO_EXPORT_SUPPORTED = False

try:
    debug_print("Попытка импорта нового API OpenVINO...")
    import openvino as ov
    OPENVINO_NEW_API = True
    debug_print("Новый API OpenVINO доступен")
    try:
        ov_version = getattr(ov, "__version__", None)
        if ov_version is None:
            debug_print("Не удалось определить версию OpenVINO, экспорт будет пропущен")
        else:
            def _parse_version_tuple(version_str):
                parts = []
                for item in version_str.replace("-", ".").split("."):
                    if item.isdigit():
                        parts.append(int(item))
                    else:
                        break
                while len(parts) < 3:
                    parts.append(0)
                return tuple(parts[:3])
            version_tuple = _parse_version_tuple(str(ov_version))
            if version_tuple >= (2024, 5, 0):
                OPENVINO_EXPORT_SUPPORTED = True
                debug_print(f"Экспорт в OpenVINO доступен (версия {ov_version})")
            else:
                debug_print(f"Версия OpenVINO ({ov_version}) ниже 2024.5.0, экспорт будет пропущен")
    except Exception as version_err:
        debug_print(f"Не удалось определить возможность экспорта в OpenVINO: {version_err}")
except ImportError as e:
    OPENVINO_NEW_API = False
    debug_print(f"Новый API OpenVINO недоступен: {e}")
    debug_print("Попробуем старый API...")
    try:
        from openvino.inference_engine import IECore
        OPENVINO_OLD_API = True
        debug_print("Старый API OpenVINO доступен")
    except ImportError as e2:
        OPENVINO_OLD_API = False
        debug_print(f"Старый API OpenVINO недоступен: {e2}")
        debug_print("OpenVINO API недоступен, будет использоваться только CPU через PyTorch")
except Exception as e:
    debug_print(f"Неожиданная ошибка при импорте OpenVINO: {e}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()

import time
import gc
import shutil


def compile_model_openvino(det_model_path, device="CPU"):
    """Компиляция модели OpenVINO для указанного устройства"""
    if OPENVINO_NEW_API:
        debug_print(f"Использование нового API OpenVINO для устройства {device}")
        try:
            core = ov.Core()
            debug_print(f"Доступные устройства: {core.available_devices}")
            det_ov_model = core.read_model(str(det_model_path))
            
            ov_config = {}
            if device != "CPU":
                det_ov_model.reshape({0: [1, 3, 640, 640]})
            if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
                ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
            
            det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
            debug_print(f"Модель скомпилирована для устройства {device}")
            return det_compiled_model
        except Exception as e:
            debug_print(f"Ошибка при компиляции модели: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    elif OPENVINO_OLD_API:
        debug_print(f"Использование старого API OpenVINO для устройства {device}")
        try:
            ie = IECore()
            debug_print(f"Доступные устройства: {ie.available_devices}")
            network = ie.read_network(model=str(det_model_path), weights=str(det_model_path).replace('.xml', '.bin'))
            exec_network = ie.load_network(network=network, device_name=device)
            debug_print(f"Модель загружена для устройства {device}")
            return exec_network
        except Exception as e:
            debug_print(f"Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    else:
        raise RuntimeError("OpenVINO API недоступен")


def load_model_with_openvino(det_model_path, device="CPU"):
    """Загрузка модели YOLO с использованием OpenVINO"""
    if not OPENVINO_NEW_API and not OPENVINO_OLD_API:
        debug_print("OpenVINO недоступен, используем стандартную загрузку YOLO")
        return YOLO(str(det_model_path).replace('_openvino_model', '.pt').replace('.xml', '.pt'))
    
    debug_print(f"Компиляция модели OpenVINO для устройства {device}...")
    compiled_model = compile_model_openvino(det_model_path, device)
    
    # Загружаем YOLO модель из директории OpenVINO
    model_dir = det_model_path.parent
    debug_print(f"Загрузка YOLO модели из {model_dir}...")
    det_model = YOLO(str(model_dir), task="detect")
    
    # Настраиваем predictor для использования OpenVINO
    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)
    
    # Присваиваем скомпилированную модель OpenVINO
    if OPENVINO_NEW_API:
        det_model.predictor.model.ov_compiled_model = compiled_model
    else:
        # Для старого API нужно использовать другой подход
        debug_print("Внимание: старый API OpenVINO может не поддерживаться ultralytics напрямую")
        # Попробуем установить через атрибут модели
        if hasattr(det_model.predictor.model, 'ov_compiled_model'):
            det_model.predictor.model.ov_compiled_model = compiled_model
    
    return det_model


def run_webcam_detection(model_path, device="CPU", cam_id=0, flip=False, 
                         save_output=False, output_dir="output", show_gui=False,
                         host_output_dir=None, save_after_frames=100, stop_after_save=False):
    """
    Запуск детекции объектов с веб-камеры в реальном времени
    
    Args:
        model_path: Путь к XML файлу модели OpenVINO или .pt файлу
        device: Устройство для inference (CPU, MYRIAD, GPU, или cpu для PyTorch)
        cam_id: ID веб-камеры (по умолчанию 0)
        flip: Зеркально отображать кадр (для фронтальных камер)
        save_output: Сохранять обработанные кадры в файл
        output_dir: Директория для сохранения результатов
        show_gui: Показывать GUI окно (не работает в Docker без X11)
        host_output_dir: Директория хоста для копирования результата
        save_after_frames: Количество кадров до начала сохранения (0 — сразу)
        stop_after_save: Останавливать запись сразу после достижения лимита кадров
    """
    model_path = Path(model_path)
    
    # Определяем, это OpenVINO модель или PyTorch
    use_openvino = model_path.suffix == '.xml' or '_openvino_model' in str(model_path)
    
    # Загрузка модели
    if use_openvino and (OPENVINO_NEW_API or OPENVINO_OLD_API):
        debug_print(f"Загрузка модели OpenVINO из {model_path}...")
        try:
            model = load_model_with_openvino(model_path, device)
            debug_print("Модель OpenVINO загружена!")
        except Exception as e:
            debug_print(f"Ошибка при загрузке модели OpenVINO: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            debug_print("Попытка загрузки через стандартный YOLO...")
            model = YOLO(str(model_path).replace('_openvino_model', '.pt').replace('.xml', '.pt'))
    else:
        debug_print(f"Загрузка модели YOLO из {model_path}...")
        try:
            # Для PyTorch используем device="cpu", так как MYRIAD не поддерживается напрямую
            pytorch_device = "cpu" if device.upper() in ["MYRIAD", "GPU"] else device.lower()
            model = YOLO(str(model_path))
            debug_print(f"Модель YOLO загружена!")
        except Exception as e:
            debug_print(f"Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return
    
    # Инициализация веб-камеры
    debug_print(f"Инициализация веб-камеры {cam_id}...")
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        debug_print(f"Ошибка: Не удалось открыть веб-камеру {cam_id}")
        debug_print("Проверьте, что веб-камера подключена и доступна в контейнере")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    debug_print("Веб-камера открыта. Обработка кадров...")
    debug_print("Нажмите Ctrl+C для выхода...")
    
    # Создание директории для сохранения результатов
    video_writer = None
    video_output_path = None
    if save_output:
        if output_dir:
            output_path = Path(output_dir).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path.cwd()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_output_path = (output_path / f"detections_{timestamp}.avi").resolve()
        debug_print(f"Результаты будут сохранены в: {video_output_path}")
    
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                debug_print("Ошибка: Не удалось получить кадр")
                break
            
            if flip:
                frame = cv2.flip(frame, 1)

            stop_requested = False

            try:
                # Для OpenVINO моделей не передаем device, так как он уже скомпилирован
                if use_openvino and (OPENVINO_NEW_API or OPENVINO_OLD_API):
                    results = model(frame, verbose=False)
                else:
                    # Для PyTorch моделей используем device
                    pytorch_device = "cpu" if device.upper() in ["MYRIAD", "GPU"] else device.lower()
                    results = model(frame, device=pytorch_device, verbose=False)
                
                result = results[0]
                detections = result.boxes
                num_detections = len(detections)
                
                if num_detections > 0:
                    classes = detections.cls.cpu().numpy()
                    confidences = detections.conf.cpu().numpy()
                    names = result.names
                    
                    detected_objects = []
                    for i in range(num_detections):
                        class_id = int(classes[i])
                        confidence = float(confidences[i])
                        class_name = names[class_id]
                        detected_objects.append(f"{class_name}({confidence:.2f})")
                    
                    debug_print(f"Кадр {frame_count}: Обнаружено {num_detections} объектов - {', '.join(detected_objects)}")
                


                annotated_frame = result.plot()
                
                processed_frames = frame_count + 1
                ready_to_save = save_output and (save_after_frames <= 0 or processed_frames >= save_after_frames)

                if ready_to_save:
                    try:
                        if video_writer is None:
                            frame_height, frame_width = annotated_frame.shape[:2]
                            video_fps = cap.get(cv2.CAP_PROP_FPS)
                            if not video_fps or video_fps <= 1:
                                video_fps = 30.0
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            video_writer = cv2.VideoWriter(
                                str(video_output_path),
                                fourcc,
                                1,
                                (frame_width, frame_height)
                            )
                            if not video_writer.isOpened():
                                debug_print("Не удалось инициализировать VideoWriter, сохранение отключено")
                                video_writer.release()
                                video_writer = None
                                save_output = False
                        if video_writer is not None:
                            video_writer.write(annotated_frame)
                    except Exception as video_err:
                        debug_print(f"Ошибка при записи видео: {video_err}")
                        save_output = False
                
                if stop_after_save and save_after_frames > 0 and processed_frames >= save_after_frames:
                    stop_requested = True
                
                if show_gui:
                    try:
                        cv2.imshow("Object Detection - Press ESC to Exit", annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            debug_print("Выход по нажатию ESC")
                            break
                    except cv2.error:
                        show_gui = False
                        debug_print("GUI недоступен, продолжаем без отображения")
            except Exception as e:
                debug_print(f"Ошибка при детекции: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            if fps_frame_count >= 30:
                elapsed = time.time() - fps_start_time
                fps = fps_frame_count / elapsed
                debug_print(f"FPS: {fps:.2f}")
                fps_start_time = time.time()
                fps_frame_count = 0
            
            if stop_requested:
                debug_print(f"Достигнут лимит кадров ({save_after_frames}). Останавливаем запись.")
                break
                
    except KeyboardInterrupt:
        debug_print("\nПрервано пользователем")
    except Exception as e:
        debug_print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        if video_writer is not None:
            try:
                video_writer.release()
                debug_print(f"Видео сохранено в: {video_output_path}")
            except Exception as release_err:
                debug_print(f"Ошибка при закрытии видеофайла: {release_err}")
        if save_output and host_output_dir:
            try:
                host_path = Path(host_output_dir).expanduser().resolve()
                host_path.mkdir(parents=True, exist_ok=True)
                if video_output_path and video_output_path.exists():
                    target_file = host_path / video_output_path.name
                    shutil.copy2(video_output_path, target_file)
                    debug_print(f"Видео скопировано в директорию хоста: {target_file}")
                else:
                    debug_print("Видео не найдено для копирования на хост")
            except Exception as host_err:
                debug_print(f"Ошибка при сохранении видео на хост: {host_err}")
        cap.release()
        if show_gui:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        debug_print(f"Ресурсы освобождены. Обработано кадров: {frame_count}")


def main():
    """Главная функция"""
    debug_print("=== Запуск функции main() ===")
    
    model_name = "yolov8n"
    device = os.getenv("DEVICE", "MYRIAD").upper()  # MYRIAD, CPU, GPU
    
    # Путь к модели OpenVINO
    openvino_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
    pt_model_path = Path(f"{model_name}.pt")
    
    debug_print(f"Проверка модели OpenVINO: {openvino_model_path}")
    debug_print(f"Существует: {openvino_model_path.exists()}")
    
    # Если модель OpenVINO не существует, создаем её
    model_path = None

    if not openvino_model_path.exists():
        debug_print(f"Модель OpenVINO не найдена: {openvino_model_path}")
        if OPENVINO_EXPORT_SUPPORTED:
            debug_print("Скачивание и конвертация модели в формат OpenVINO...")
            try:
                if not pt_model_path.exists():
                    debug_print("Скачивание PyTorch модели...")
                    pt_model = YOLO(f"{model_name}.pt")
                else:
                    pt_model = YOLO(str(pt_model_path))
                
                debug_print("Экспорт модели в формат OpenVINO...")
                pt_model.export(format="openvino", dynamic=True, half=True)
                del pt_model
                gc.collect()
                debug_print(f"Модель OpenVINO создана: {openvino_model_path}")
                if openvino_model_path.exists():
                    model_path = openvino_model_path
            except Exception as e:
                debug_print(f"Ошибка при создании модели OpenVINO: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            debug_print("Экспорт в OpenVINO пропущен (нет подходящей версии OpenVINO). Будем использовать PyTorch модель.")
    
    if model_path is None:
        if openvino_model_path.exists():
            model_path = openvino_model_path
        else:
            if not pt_model_path.exists():
                debug_print("PyTorch модель отсутствует, выполняем загрузку...")
                YOLO(f"{model_name}.pt")
            model_path = pt_model_path
            if device != "CPU":
                pass
                # debug_print("MYRIAD/GPU не поддерживается напрямую PyTorch моделью, переключаемся на CPU")
            device = "MYRIAD"
    else:
        model_path = openvino_model_path

    device = "MYRIAD"
    
    debug_print(f"Используемое устройство: {device}")
    debug_print(f"Используемая модель: {model_path}")
    
    
    
    # Запуск детекции
    save_output_env = os.getenv("SAVE_OUTPUT")
    save_output = True if save_output_env is None else save_output_env.lower() == "true"
    output_dir_env = os.getenv("OUTPUT_DIR", "").strip()
    save_after_frames_env = os.getenv("SAVE_AFTER_FRAMES", "").strip()
    stop_after_save_env = os.getenv("STOP_AFTER_SAVE", "true").strip().lower()

    try:
        save_after_frames = int(save_after_frames_env) if save_after_frames_env else 100
        if save_after_frames < 0:
            save_after_frames = 0
    except ValueError:
        save_after_frames = 100

    stop_after_save = stop_after_save_env not in ["0", "false", "no"]

    run_webcam_detection(
        model_path=model_path,
        device=device,
        cam_id=0,
        flip=True,
        save_output=True,
        output_dir=output_dir_env,
        show_gui=False,
        host_output_dir=os.getenv("HOST_OUTPUT_DIR", "").strip(),
        save_after_frames=save_after_frames,
        stop_after_save=stop_after_save
    )


if __name__ == "__main__":
    debug_print("=== Точка входа ===")
    try:
        main()
    except Exception as e:
        debug_print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    sleep(20000000)
    debug_print("=== Конец выполнения ===")
