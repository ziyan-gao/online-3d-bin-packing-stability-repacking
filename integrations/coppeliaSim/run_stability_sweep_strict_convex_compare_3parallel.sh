#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/gao/anaconda3/envs/packing-toolkit/bin/python}"
COPPELIASIM_DIR="${COPPELIASIM_DIR:-/home/gao/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04}"
COPPELIASIM_SETTINGS_SOURCE="${COPPELIASIM_SETTINGS_SOURCE:-${HOME}/.CoppeliaSim}"
COPPELIASIM_ISOLATED_SETTINGS="${COPPELIASIM_ISOLATED_SETTINGS:-0}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${SCRIPT_DIR}/experiment_results_strict_convex_compare_${RUN_ID}"
RESULT_DIR="${ARTIFACT_DIR}/results"
SUMMARY_DIR="${ARTIFACT_DIR}/summaries"
LOG_DIR="${ARTIFACT_DIR}/logs"

PORTS=(23000 23001 23002)
ENGINES=(bullet_2_83 ode newton)
PIDS=()

EPOCHS="${EPOCHS:-20}"
MAX_ITEMS="${MAX_ITEMS:-500}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SIMULATION_TIME_STEP="${SIMULATION_TIME_STEP:-0.01}"
BIN_Z_VALUES=(0.6 1.2 1.8)
Z_MODES=(same variable)
CONVEX_THRESHOLDS=(0.1 0.2 0.3 0.4 0.5)
HEURISTICS=(convex_hull convex_hull_old)

mkdir -p "$RESULT_DIR" "$SUMMARY_DIR" "$LOG_DIR"

cleanup() {
  if ((${#PIDS[@]} > 0)); then
    echo "Stopping active experiment jobs: ${PIDS[*]}"
    for PID in "${PIDS[@]}"; do
      kill "$PID" 2>/dev/null || true
    done
  fi
}

port_is_open() {
  local port="$1"
  bash -c ":</dev/tcp/127.0.0.1/${port}" 2>/dev/null
}

wait_for_port() {
  local port="$1"
  local timeout_s="${2:-60}"
  local start_s
  start_s="$(date +%s)"
  while true; do
    if port_is_open "$port"; then
      return 0
    fi
    if (( $(date +%s) - start_s >= timeout_s )); then
      return 1
    fi
    sleep 0.5
  done
}

sync_coppeliasim_registration() {
  local settings_suffix="$1"
  local target_dir="${HOME}/.CoppeliaSim${settings_suffix}"
  local source_usrset="${COPPELIASIM_SETTINGS_SOURCE}/usrset.txt"
  local target_usrset="${target_dir}/usrset.txt"
  if [[ ! -f "$source_usrset" ]]; then
    echo "WARNING: no source CoppeliaSim settings found at ${source_usrset}; registration will not be copied." >&2
    return
  fi
  mkdir -p "$target_dir"
  if [[ ! -f "$target_usrset" ]]; then
    cp -p "$source_usrset" "$target_usrset"
  fi
  local key value
  for key in license_lite license_edu license_pro; do
    value="$(sed -n "s/^${key} = //p" "$source_usrset" | tail -n 1)"
    if [[ -n "$value" ]]; then
      if grep -q "^${key} =" "$target_usrset"; then
        sed -i "s|^${key} =.*|${key} = ${value}|" "$target_usrset"
      else
        printf '%s = %s\n' "$key" "$value" >> "$target_usrset"
      fi
    fi
  done
  if [[ -f "${COPPELIASIM_SETTINGS_SOURCE}/persistentData.dat" && ! -f "${target_dir}/persistentData.dat" ]]; then
    cp -p "${COPPELIASIM_SETTINGS_SOURCE}/persistentData.dat" "${target_dir}/persistentData.dat"
  fi
}

start_coppeliasim() {
  local port="$1"
  local engine="$2"
  local log_file="${LOG_DIR}/coppeliasim_${engine}_port${port}.log"
  local __pid_var="$3"
  local settings_suffix=""

  if port_is_open "$port"; then
    echo "ERROR: port ${port} is already in use. Close the existing CoppeliaSim on that port first." >&2
    exit 1
  fi
  if [[ "$COPPELIASIM_ISOLATED_SETTINGS" == "1" ]]; then
    settings_suffix="_strict_convex_${engine}_${port}"
    sync_coppeliasim_registration "$settings_suffix"
  fi

  (
    cd "$COPPELIASIM_DIR"
    if [[ -n "$settings_suffix" ]]; then
      export COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX="$settings_suffix"
    fi
    export QT_PLUGIN_PATH="$COPPELIASIM_DIR"
    export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_DIR}/platforms"
    unset QT_QPA_FONTDIR || true
    ./coppeliaSim.sh -GzmqRemoteApi.rpcPort="${port}"
  ) > "$log_file" 2>&1 &

  local pid=$!
  printf -v "$__pid_var" "%s" "$pid"
  echo "Started CoppeliaSim PID=${pid} engine=${engine} port=${port} log=${log_file}"

  if ! wait_for_port "$port" 90; then
    echo "ERROR: CoppeliaSim did not open port ${port}. Check ${log_file}" >&2
    exit 1
  fi
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: CoppeliaSim on port ${port} exited after startup. Check ${log_file}" >&2
    exit 1
  fi
}

trap cleanup EXIT INT TERM

run_experiment_csv() {
  local port="$1"
  local engine="$2"
  local bin_z="$3"
  local z_mode="$4"
  local z_flag
  if [[ "$z_mode" == "same" ]]; then
    z_flag="--same-object-z"
  else
    z_flag="--variable-object-z"
  fi
  local label="${engine}_bin${bin_z}_${z_mode}"
  local result_csv="${RESULT_DIR}/results_${RUN_ID}_${label}.csv"
  local summary_csv="${SUMMARY_DIR}/summary_${RUN_ID}_${label}.csv"

  echo "Started strict convex comparison engine=${engine}, bin_z=${bin_z}, z_mode=${z_mode}, port=${port}, run_id=${RUN_ID}"
  "$PYTHON_BIN" "${SCRIPT_DIR}/stability_experiment.py" \
    --epochs "$EPOCHS" \
    --max-items "$MAX_ITEMS" \
    --bin-z "$bin_z" \
    $z_flag \
    --remote-api-port "$port" \
    --engines "$engine" \
    --heuristics "${HEURISTICS[@]}" \
    --convex-thresholds "${CONVEX_THRESHOLDS[@]}" \
    --drop-heights 0.005 \
    --load-delay 0.0 \
    --sim-object-xy-scale 0.999 \
    --simulation-time-step "$SIMULATION_TIME_STEP" \
    --stepping \
    --settle-by-velocity \
    --linear-velocity-threshold 0.01 \
    --angular-velocity-threshold 0.05 \
    --settle-stable-duration 0.5 \
    --max-settle-wait-time 5.0 \
    --settle-poll-interval 0.05 \
    --settle-time 0.5 \
    --sample-time 0.5 \
    --drift-tolerance 0.001 \
    --save-every "$SAVE_EVERY" \
    --output-csv "$result_csv" \
    --summary-csv "$summary_csv"
  echo "Finished strict convex comparison engine=${engine}, bin_z=${bin_z}, z_mode=${z_mode}, port=${port}"
}

run_engine_job() {
  local port="$1"
  local engine="$2"
  local sim_pid=""
  cleanup_sim() {
    if [[ -n "$sim_pid" ]]; then
      echo "Stopping CoppeliaSim PID=${sim_pid} for engine=${engine}"
      kill "$sim_pid" 2>/dev/null || true
      wait "$sim_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_sim EXIT INT TERM
  start_coppeliasim "$port" "$engine" sim_pid
  for BIN_Z in "${BIN_Z_VALUES[@]}"; do
    for Z_MODE in "${Z_MODES[@]}"; do
      run_experiment_csv "$port" "$engine" "$BIN_Z" "$Z_MODE"
    done
  done
}

for IDX in "${!ENGINES[@]}"; do
  ENGINE="${ENGINES[$IDX]}"
  PORT="${PORTS[$IDX]}"
  LOG_FILE="${LOG_DIR}/experiment_${ENGINE}_port${PORT}.log"
  run_engine_job "$PORT" "$ENGINE" > "$LOG_FILE" 2>&1 &
  PID=$!
  PIDS+=("$PID")
  echo "Launched PID=${PID} engine=${ENGINE} port=${PORT} log=${LOG_FILE}"
done

echo "Running ${#PIDS[@]} strict convex comparison jobs in parallel."
echo "Settings: epochs=${EPOCHS}, max_items=${MAX_ITEMS}, dt=${SIMULATION_TIME_STEP}, stepping=true, velocity_settle=true"
echo "Strict stability: settle_stable_duration=0.5, max_settle_wait_time=5.0, settle_time=0.5, sample_time=0.5, drift_tolerance=0.001"
echo "Sim object scale: xy=0.999, z=1.0"
for PID in "${PIDS[@]}"; do
  wait "$PID"
done
PIDS=()
trap - EXIT INT TERM

echo "Finished strict convex comparison sweep: ${RUN_ID}"
echo "Results: ${RESULT_DIR}"
echo "Summaries: ${SUMMARY_DIR}"
echo "Logs: ${LOG_DIR}"
