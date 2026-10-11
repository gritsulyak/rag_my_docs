# llama.cpp server image с выбором бэкенда ускорения.
#
# Соответствует подходу из ~/github/llm_eng (Vulkan-бинарники llama.cpp,
# prism-vulkan/prism-rocm) и официальной сборке ggml-org/llama.cpp.
#
# Бэкенды (--build-arg BACKEND=...):
#   vulkan — AMD/Intel/NVIDIA через Vulkan (целевой профиль AI MAX+ 395 / Radeon 8060S);
#   cuda   — NVIDIA (RTX 3060, sm_86) через CUDA;
#   rocm   — AMD через HIP/ROCm (альтернатива Vulkan на Strix Halo, gfx1151);
#   cpu    — без ускорения (универсальный фолбэк).
#
# Примеры сборки:
#   docker build -f docker/llama/Dockerfile --build-arg BACKEND=vulkan -t normdoc-llama:vulkan .
#   docker build -f docker/llama/Dockerfile --build-arg BACKEND=cuda   -t normdoc-llama:cuda   .
#   docker build -f docker/llama/Dockerfile --build-arg BACKEND=rocm   -t normdoc-llama:rocm   .
#
# Обычно образы собираются автоматически из docker-compose.yml (профили ai395 / rtx3060 / cpu).
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS build

ARG BACKEND=cpu
ARG LLAMA_REF=master
ARG CUDA_ARCHITECTURES=86
ARG AMDGPU_TARGETS=gfx1151

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates curl \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Бэкенд-специфичные зависимости.
RUN set -eux; \
    if [ "$BACKEND" = "vulkan" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libvulkan-dev vulkan-tools glslc spirv-tools libshaderc-dev \
        && rm -rf /var/lib/apt/lists/*; \
    elif [ "$BACKEND" = "rocm" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            hip-dev rocm-hip-sdk \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN git clone --depth 1 --branch "${LLAMA_REF}" https://github.com/ggml-org/llama.cpp /llama.cpp

# Флаги cmake по бэкенду.
RUN set -eux; \
    case "$BACKEND" in \
        vulkan) FLAGS="-DGGML_VULKAN=ON" ;; \
        cuda)   FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" ;; \
        rocm)   FLAGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=${AMDGPU_TARGETS}" ;; \
        *)      FLAGS="" ;; \
    esac; \
    cmake -S /llama.cpp -B /llama.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
        -DBUILD_SHARED_LIBS=ON \
        $FLAGS; \
    cmake --build /llama.cpp/build --config Release -j"$(nproc)" --target llama-server llama-bench

# ---------------------------------------------------------------------------
# Рантайм-база: для CUDA/ROCm задаётся отдельно (например, nvidia/cuda:*-runtime,
# rocm/dev-ubuntu-22.04). По умолчанию — тот же Ubuntu, что и сборка.
ARG RUNTIME_BASE_IMAGE=ubuntu:24.04
FROM ${RUNTIME_BASE_IMAGE} AS runtime
ARG BACKEND=cpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# Для Vulkan/ROCm в рантайме нужны клиентские библиотеки (монтируются /dev/dri,/dev/kfd).
RUN set -eux; \
    if [ "$BACKEND" = "vulkan" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=build /llama.cpp/build/bin /opt/llama/bin
ENV LD_LIBRARY_PATH=/opt/llama/bin
ENV PATH=/opt/llama/bin:$PATH

COPY docker/llama/entrypoint.sh /opt/llama/entrypoint.sh
RUN chmod +x /opt/llama/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/opt/llama/entrypoint.sh"]
