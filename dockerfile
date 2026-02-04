# Dockerfile
FROM openvino/ubuntu20_runtime:2021.4.1_20210416

# Убедимся, что мы работаем от root
USER root

# Удаление проблемного репозитория Intel Graphics перед обновлением
RUN rm -f /etc/apt/sources.list.d/*intel*graphics*.list 2>/dev/null || true

# Обновление и установка дополнительных пакетов (игнорируя ошибки репозиториев)
RUN chmod 755 /var/lib/apt 2>/dev/null || true && \
    mkdir -p /var/lib/apt/lists/partial 2>/dev/null || true && \
    apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false 2>&1 | grep -v "Failed to fetch" || true && \
    apt-get install -y --allow-unauthenticated \
    python3-pip \
    git \
    wget \
    curl \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения OpenVINO
# Обычно OpenVINO устанавливается в /opt/intel/openvino
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino

# Настройка LD_LIBRARY_PATH для OpenVINO и OpenCV библиотек
# Используем синтаксис без $ для начального значения, чтобы избежать предупреждений
ENV LD_LIBRARY_PATH=/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/ngraph/lib:/opt/intel/openvino/opencv/lib

# Настройка PYTHONPATH - сначала системные пакеты, потом OpenVINO
# Это позволит использовать opencv-python из pip, если он установлен
ENV PYTHONPATH=/opt/intel/openvino/python/python3

# Установка дополнительных Python-библиотек
# НЕ устанавливаем opencv-python, используем версию из OpenVINO
RUN pip3 install --no-cache-dir \
    numpy \
    pillow

# Создание рабочей директории
WORKDIR /workspace

# Установка ultralytics и других зависимостей
RUN pip3 install --no-cache-dir \
    "ultralytics==8.3.59" \
    requests \
    tqdm

# Копирование скрипта
COPY webcam_detection.py /workspace/

# Создание entrypoint скрипта для активации OpenVINO окружения
RUN printf '#!/bin/bash\n\
set -e\n\
# Активация OpenVINO окружения\n\
if [ -f /opt/intel/openvino/bin/setupvars.sh ]; then\n\
    source /opt/intel/openvino/bin/setupvars.sh\n\
fi\n\
# Запуск переданной команды\n\
exec "$@"\n' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Установка entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]