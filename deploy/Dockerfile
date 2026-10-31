# The pick planner service: perception, candidates, risk, and monitor sessions on the C++ monitor.
FROM python:3.12-slim AS native
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make && rm -rf /var/lib/apt/lists/*
WORKDIR /src/cpp
COPY cpp/ .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2 && ./build/test_monitor

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PODPICK_ROOT=/app
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY src/ src/
RUN pip install ".[service]"
COPY config/ config/
COPY models/ models/
COPY cpp/monitor.cpp cpp/monitor.h cpp/
COPY --from=native /src/cpp/build/libpodmonitor.so cpp/build/libpodmonitor.so
# keep the prebuilt library newer than its source so nothing tries to compile at runtime
RUN touch cpp/build/libpodmonitor.so && useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8080
CMD ["uvicorn", "podpick.service.app:app", "--host", "0.0.0.0", "--port", "8080"]
