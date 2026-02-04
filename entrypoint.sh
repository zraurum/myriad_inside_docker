#!/bin/bash
set -e

# Активация OpenVINO окружения
if [ -f /opt/intel/openvino/bin/setupvars.sh ]; then
    source /opt/intel/openvino/bin/setupvars.sh
fi

# Запуск переданной команды
exec "$@"
