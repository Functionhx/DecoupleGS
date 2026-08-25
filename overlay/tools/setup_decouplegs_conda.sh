#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 SOURCE_ENV [TARGET_ENV]" >&2
    exit 2
fi

source_env="$1"
target_env="${2:-decouplegs}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-$(command -v conda)}"

if ! "$conda_exe" run -n "$source_env" python -c 'import sys' >/dev/null 2>&1; then
    echo "Source conda environment does not exist: $source_env" >&2
    exit 2
fi

if ! "$conda_exe" run -n "$target_env" python -c 'import sys' >/dev/null 2>&1; then
    "$conda_exe" create --name "$target_env" --clone "$source_env" -y
fi

env_prefix="$("$conda_exe" run -n "$target_env" python -c 'import sys; print(sys.prefix)' | tail -n 1)"
env_pip="$env_prefix/bin/pip"
env_python="$env_prefix/bin/python"

cuda_root="${DECUPLEGS_CUDA_ROOT:-/usr/local/cuda-11.8}"
cc_path="${DECUPLEGS_CC:-/usr/bin/gcc-11}"
cxx_path="${DECUPLEGS_CXX:-/usr/bin/g++-11}"
if [[ ! -x "$cuda_root/bin/nvcc" || ! -x "$cc_path" || ! -x "$cxx_path" ]]; then
    echo "CUDA 11.8 and GCC/G++ 11 are required to build HUGSIM extensions" >&2
    exit 3
fi

"$env_pip" install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
"$env_pip" install --upgrade \
    -c "$repo_root/constraints-decouplegs.txt" \
    -r "$repo_root/requirements-decouplegs.txt"

"$env_pip" install -c "$repo_root/constraints-decouplegs.txt" \
    'git+https://github.com/hyzhou404/trajdata.git@dea018df54dd4917165429932ec4f8b645e04f07' \
    'git+https://github.com/hyzhou404/nuscenes-devkit.git@adf6972147a474185915fd68cae4ccb2f0355957' \
    'git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf' \
    'git+https://github.com/ChristophReich1996/Optical-Flow-Visualization-PyTorch.git@9177370c7c00b4b7dbe4deda6fed734fdff48b2c'

# UniDepth's wheel omits the sources for its optional evaluation KNN op. Keep
# one pinned checkout long enough to install both the Python package and op.
unidepth_build_dir="$(mktemp -d -t decouplegs-unidepth-XXXXXX)"
trap 'rm -rf -- "$unidepth_build_dir"' EXIT
git clone --filter=blob:none --no-checkout \
    https://github.com/hyzhou404/UniDepth.git "$unidepth_build_dir/UniDepth"
git -C "$unidepth_build_dir/UniDepth" checkout 1d3a0bb6cfa715c76335606b6585a2a9551b55f2
"$env_pip" install -c "$repo_root/constraints-decouplegs.txt" \
    "$unidepth_build_dir/UniDepth"

cuda_arch="${DECUPLEGS_CUDA_ARCH_LIST:-}"
if [[ -z "$cuda_arch" ]]; then
    cuda_arch="$("$env_python" -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}")')"
fi
tcnn_cuda_arch="${cuda_arch//./}"
max_jobs="${DECUPLEGS_MAX_JOBS:-8}"

build_env=(
    CC="$cc_path"
    CXX="$cxx_path"
    CUDAHOSTCXX="$cxx_path"
    CUDA_HOME="$cuda_root"
    TORCH_CUDA_ARCH_LIST="$cuda_arch"
    MAX_JOBS="$max_jobs"
)
env "${build_env[@]}" "$env_pip" install --no-build-isolation --no-deps --force-reinstall \
    "$unidepth_build_dir/UniDepth/unidepth/ops/knn"
env "${build_env[@]}" "$env_pip" install --no-build-isolation --no-deps --force-reinstall \
    "$repo_root/submodules/simple-knn"
env "${build_env[@]}" "$env_pip" install --no-build-isolation --no-deps --force-reinstall \
    'git+https://github.com/hyzhou404/HUGSIM_splat.git@88f2a40c4e2f6bafde2beeaba6c43bbb0ccb1f5f'
env "${build_env[@]}" TCNN_CUDA_ARCHITECTURES="$tcnn_cuda_arch" "$env_pip" install \
    --no-build-isolation --no-deps --force-reinstall \
    'git+https://github.com/NVlabs/tiny-cuda-nn.git@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch'
"$env_pip" install --no-deps -e "$repo_root/sim"
# An inherited ONNX toolchain may conflict with HUGSIM's explicit protobuf
# 3.20.2 pin. These packages are not used by this project.
"$env_pip" uninstall -y onnx onnx-tool || true

cd "$repo_root"
"$env_python" -m unittest discover -s tests -v
"$env_python" tools/smoke_test_decouplegs_cuda.py
"$env_pip" check

echo "Environment ready. Activate it with: conda activate $target_env"
