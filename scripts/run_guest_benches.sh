#!/bin/bash

# Script to collect some numbers from our OpenVM crush vs RISC-V guest examples.
# Mostly for CI usage, but can be easily modified for manual tests.

# NOTE: The script expects the python environment to be set up with the required
# dependencies. Should be run from the project root, will create a `results`
# directory.

set -e

# Parse flags
CUDA_FLAGS=""
BENCHMARKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA_FLAGS="--features cuda"; shift ;;
    --*) echo "Unknown flag: $1"; exit 1 ;;
    *) BENCHMARKS+=("$1"); shift ;;
  esac
done
# Default: run all benchmarks
if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
  BENCHMARKS=(keccak keccak_precompile u256)
fi

should_run() {
  local name="$1"
  for b in "${BENCHMARKS[@]}"; do
    [[ "$b" == "$name" ]] && return 0
  done
  return 1
}

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPTS_DIR=$(dirname "$SCRIPT_PATH")

# Download shared analysis scripts from powdr upstream.
POWDR_SCRIPTS_URL="https://raw.githubusercontent.com/powdr-labs/powdr/main/openvm-riscv/scripts"
curl -sL "$POWDR_SCRIPTS_URL/basic_metrics.py" -o "$SCRIPTS_DIR/basic_metrics.py"
curl -sL "$POWDR_SCRIPTS_URL/metrics_utils.py" -o "$SCRIPTS_DIR/metrics_utils.py"
curl -sL "$POWDR_SCRIPTS_URL/plot_trace_cells.py" -o "$SCRIPTS_DIR/plot_trace_cells.py"

# Convert space-separated values into --input flags for the CLI.
# E.g. "0 0 10 155" -> "--input 0 --input 0 --input 10 --input 155"
make_input_flags() {
  local flags=()
  for val in $1; do
    flags+=(--input "$val")
  done
  echo "${flags[@]}"
}

# Run a command, capture its wall time, and append {label: seconds} to a JSON file.
# Usage: timed <json_file> <label> <command...>
timed() {
    local json_file="$1"; shift
    local label="$1"; shift
    local start end elapsed
    start=$(date +%s.%N)
    "$@"
    local exit_code=$?
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    # Append to JSON file (builds up object entries, finalized later)
    echo "\"${label}\": ${elapsed}" >> "${json_file}.tmp"
    return $exit_code
}

# Finalize wall_times JSON from collected entries.
finalize_wall_times() {
    local json_file="$1"
    echo "{" > "$json_file"
    # Join entries with commas
    sed '$!s/$/,/' "${json_file}.tmp" >> "$json_file"
    echo "}" >> "$json_file"
    rm -f "${json_file}.tmp"
}

# Compile+prove pipeline: compile is excluded from metrics, prove is measured.
run_bench_wasm() {
  local guest="$1"
  local input="$2"
  local run_name="$3"
  local apc_count="$4"
  local extra_flags="${5:-}"  # e.g. "--keccak"

  echo ""
  echo "==== ${run_name} ===="
  echo ""

  mkdir -p "${run_name}"

  local input_flags
  input_flags=($(make_input_flags "$input"))

  local compiled_dir="${run_name}/compiled"
  local wall_times="${run_name}/wall_times.json"

  # Compile step (not included in metrics)
  timed "$wall_times" "compile" \
    cargo run -r $CUDA_FLAGS -- compile $extra_flags --apc-count "$apc_count" --apc-candidates-dir "${run_name}" --output-dir "$compiled_dir" "$guest" "main" "${input_flags[@]}" &>"${run_name}/compile_log.txt"

  # Prove step (metrics captured here)
  timed "$wall_times" "prove" \
    cargo run -r $CUDA_FLAGS -- prove --compiled-dir "$compiled_dir" --recursion "${input_flags[@]}" --metrics "${run_name}/metrics.json" &>"${run_name}/log.txt"

  finalize_wall_times "$wall_times"

  python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json >"${run_name}"/trace_cells.txt
}

run_bench_riscv() {
  local guest="$1"
  local input="$2"
  local run_name="$3"
  local apc_count="$4"

  echo ""
  echo "==== ${run_name} ===="
  echo ""

  mkdir -p "${run_name}"

  local input_flags
  input_flags=($(make_input_flags "$input"))

  local compiled_dir="${run_name}/compiled"
  local wall_times="${run_name}/wall_times.json"

  # Compile step (not included in metrics)
  timed "$wall_times" "compile" \
    cargo run -r $CUDA_FLAGS -- compile-riscv --apc-count "$apc_count" --apc-candidates-dir "${run_name}" --output-dir "$compiled_dir" "$guest" "${input_flags[@]}" &>"${run_name}/compile_log.txt"

  # Prove step (metrics captured here)
  timed "$wall_times" "prove" \
    cargo run -r $CUDA_FLAGS -- prove-riscv --compiled-dir "$compiled_dir" "${input_flags[@]}" --metrics "${run_name}/metrics.json" &>"${run_name}/log.txt"

  finalize_wall_times "$wall_times"

  python3 "$SCRIPTS_DIR"/plot_trace_cells.py -o "${run_name}"/trace_cells.png "${run_name}"/metrics.json >"${run_name}"/trace_cells.txt

  # Clean up some files that we don't want to push.
  rm -f debug.pil
}

ROOT_DIR=$(pwd)

### Keccak 1000 iterations (software)
if should_run keccak; then
dir="results/keccak_1000"
# The RISC-V guest takes as input the number of iterations.
# It returns the first byte of the result as a public.
input_riscv="1000"
# The WASM guest takes as input the main arguments [argc, argv], the input number of iterations,
# and the first byte of the result to be asserted inside the guest program.
input_wasm="0 0 1000 39"

mkdir -p "$dir"
pushd "$dir"

run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv" "0"
run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv_apc_1" "1"
run_bench_riscv "$ROOT_DIR/sample-programs/keccak_with_inputs" "$input_riscv" "riscv_apc_10" "10"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "crush" "0"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "crush_apc_1" "1"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_with_inputs/target/wasm32-unknown-unknown/release/keccak_with_inputs.wasm" "$input_wasm" "crush_apc_10" "10"

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json >basic_metrics.csv
python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush/metrics.json" "riscv/metrics.json" >crush_apc_0_vs_riscv.txt
python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush_apc_1/metrics.json" "riscv_apc_1/metrics.json" >crush_apc_1_vs_riscv_apc_1.txt
python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush_apc_10/metrics.json" "riscv_apc_10/metrics.json" >crush_apc_10_vs_riscv_apc_10.txt
popd
fi # keccak

### Keccak 1000 iterations with manual precompile
if should_run keccak_precompile; then
dir="results/keccak_precompile_1000"
input_riscv="1000"
input_wasm="0 0 1000 39"

# Build the WASM guest binary first.
cargo build --release --target wasm32-unknown-unknown --manifest-path "${ROOT_DIR}/sample-programs/keccak_precompile/Cargo.toml"

mkdir -p "$dir"
pushd "$dir"

run_bench_riscv "$ROOT_DIR/sample-programs/keccak_precompile" "$input_riscv" "riscv" "0"
run_bench_wasm "$ROOT_DIR/sample-programs/keccak_precompile/target/wasm32-unknown-unknown/release/keccak_precompile.wasm" "$input_wasm" "crush" "0" "--keccak"

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json >basic_metrics.csv
popd
fi # keccak_precompile

### U256 matrix multiply (10x10 identity * constant, 1 repetition)
if should_run u256; then
dir="results/u256"
# The RISC-V guest takes as input the number of repetitions.
input_riscv="1"
# The WASM guest takes as input the main arguments [argc, argv] and the number of repetitions.
input_wasm="0 0 1"

# Build the WASM guest binary first.
cargo build --release --target wasm32-unknown-unknown --manifest-path "${ROOT_DIR}/sample-programs/u256_matmul/Cargo.toml"

mkdir -p "$dir"
pushd "$dir"

run_bench_riscv "$ROOT_DIR/sample-programs/u256_matmul" "$input_riscv" "riscv" "0"
run_bench_riscv "$ROOT_DIR/sample-programs/u256_matmul" "$input_riscv" "riscv_apc_2" "2"
run_bench_riscv "$ROOT_DIR/sample-programs/u256_matmul" "$input_riscv" "riscv_apc_10" "10"
run_bench_wasm "$ROOT_DIR/sample-programs/u256_matmul/target/wasm32-unknown-unknown/release/u256_matmul.wasm" "$input_wasm" "crush" "0"
run_bench_wasm "$ROOT_DIR/sample-programs/u256_matmul/target/wasm32-unknown-unknown/release/u256_matmul.wasm" "$input_wasm" "crush_apc_2" "2"
run_bench_wasm "$ROOT_DIR/sample-programs/u256_matmul/target/wasm32-unknown-unknown/release/u256_matmul.wasm" "$input_wasm" "crush_apc_10" "10"

python3 "$SCRIPTS_DIR"/basic_metrics.py summary-table --csv */metrics.json >basic_metrics.csv
python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush/metrics.json" "riscv/metrics.json" >crush_apc_0_vs_riscv.txt
python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush_apc_2/metrics.json" "riscv_apc_2/metrics.json" >crush_apc_2_vs_riscv_apc_2.txt
python3 "$SCRIPTS_DIR"/crush_vs_riscv.py "crush_apc_10/metrics.json" "riscv_apc_10/metrics.json" >crush_apc_10_vs_riscv_apc_10.txt
popd
fi # u256
