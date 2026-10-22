#!/usr/bin/env bash
# train_bg.sh — 后台挂起训练（无 tmux 环境用 setsid + nohup）
#
#   ./train_bg.sh start sg3 2                    # 在 2 号卡后台训练 sg3
#   ./train_bg.sh start sh1 3 trainer.max_steps=1001
#   ./train_bg.sh start sg3 2 checkpointing.load=outputs/x/checkpoint.ckpt checkpointing.resume=true
#   ./train_bg.sh status                         # 三个变体状态
#   ./train_bg.sh logs sg3                       # 看日志末尾
#   ./train_bg.sh follow sg3                     # 实时 tail -f
#   ./train_bg.sh stop sg3
#
# 环境变量：CONDA_ENV(默认 ecosplat) / CONDA_SH / LOG_DIR / RUN_DIR
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-ecosplat}"
CONDA_SH="${CONDA_SH:-/data1/xz/miniconda3/etc/profile.d/conda.sh}"
RUN_DIR="${RUN_DIR:-$REPO/.run}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
VARIANTS="sg3 sh1 sh2 sh4"

mkdir -p "$RUN_DIR" "$LOG_DIR"

variant_experiment() {
    case "$1" in
        sg3) echo "spfsplat/re10k_10view_sg" ;;
        sh1) echo "spfsplat/re10k_10view_sh1" ;;
        sh2) echo "spfsplat/re10k_10view_sh2" ;;
        sh4) echo "spfsplat/re10k_10view_sh4" ;;
        *) return 1 ;;
    esac
}

pidfile() { echo "$RUN_DIR/$1.pid"; }
gpu_file() { echo "$RUN_DIR/$1.gpu"; }
is_running() {
    local f; f=$(pidfile "$1")
    [ -f "$f" ] && kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null
}

start() {
    local variant="${1:?usage: $0 start <sg3|sh1|sh2|sh4> [gpu_id] [hydra overrides...]}"; shift
    local gpu="${1:-0}"; shift || true
    local exp; exp=$(variant_experiment "$variant") || { echo "unknown variant: $variant" >&2; exit 2; }

    if is_running "$variant"; then
        echo "[$variant] already running (pid $(cat "$(pidfile "$variant")"), gpu $(cat "$(gpu_file "$variant")" 2>/dev/null))"
        return 0
    fi

    local ts log
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOG_DIR/${variant}_${ts}.log"

    setsid nohup bash -c "
        source '$CONDA_SH'
        conda activate '$CONDA_ENV'
        cd '$REPO'
        echo \"[train_bg] \$(date '+%F %T') variant=$variant gpu=$gpu\"
        echo \"[train_bg] overrides: $*\"
        export PYTHONUNBUFFERED=1
        export CUDA_VISIBLE_DEVICES=$gpu
        exec python -u -m src.main +experiment='$exp' \
            wandb.mode=offline wandb.name='$variant' $*
    " >>"$log" 2>&1 &

    local pid=$!
    echo "$pid" > "$(pidfile "$variant")"
    echo "$gpu" > "$(gpu_file "$variant")"
    ln -sfn "$log" "$LOG_DIR/${variant}.latest.log"

    sleep 5
    if is_running "$variant"; then
        echo "[$variant] started  pid=$pid  gpu=$gpu"
        echo "  log : $log"
        echo "  查看: $0 follow $variant"
    else
        echo "[$variant] FAILED to start; last log lines:"
        tail -n 25 "$log"
        rm -f "$(pidfile "$variant")"
        exit 1
    fi
}

stop() {
    local variant="${1:?usage: $0 stop <variant>}"
    if ! is_running "$variant"; then
        echo "[$variant] not running"
        rm -f "$(pidfile "$variant")"
        return 0
    fi
    local pid; pid=$(cat "$(pidfile "$variant")")
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null
        echo "[$variant] force-killed (pid $pid)"
    else
        echo "[$variant] stopped (pid $pid)"
    fi
    rm -f "$(pidfile "$variant")"
}

status_one() {
    local v=$1
    if is_running "$v"; then
        local pid; pid=$(cat "$(pidfile "$v")")
        local et; et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
        printf "%-4s RUNNING  pid=%-8s gpu=%-2s elapsed=%-10s\n" \
            "$v" "$pid" "$(cat "$(gpu_file "$v")" 2>/dev/null || echo '?')" "$et"
        # 最后一条含 step/loss 的日志行
        tail -n 40 "$LOG_DIR/$v.latest.log" 2>/dev/null \
            | grep -E "train step|it/s|Epoch" | tail -n 1 | sed 's/^/       /'
    else
        printf "%-4s stopped\n" "$v"
    fi
}

status() {
    if [ $# -gt 0 ] && [ -n "${1:-}" ]; then
        status_one "$1"
    else
        for v in $VARIANTS; do status_one "$v"; done
    fi
}

logs() {
    local variant="${1:?usage: $0 logs <variant> [n]}"
    local n="${2:-30}"
    ls -1t "$LOG_DIR/${variant}"_*.log 2>/dev/null | head -1 | xargs -r tail -n "$n"
}

follow() {
    local variant="${1:?usage: $0 follow <variant>}"
    tail -f "$LOG_DIR/${variant}.latest.log"
}

case "${1:-}" in
    start)  shift; start "$@" ;;
    stop)   shift; stop "$@" ;;
    status) shift; status "$@" ;;
    logs)   shift; logs "$@" ;;
    follow) shift; follow "$@" ;;
    *)
        cat >&2 <<EOF
usage: $(basename "$0") <command>

  start <sg3|sh1|sh2|sh4> [gpu_id] [hydra overrides...]   后台启动
  stop  <variant>                                     停止
  status [variant]                                    状态 + 最近进度
  logs  <variant> [n=30]                              日志末尾
  follow <variant>                                    tail -f 日志

示例:
  $(basename "$0") start sg3 2
  $(basename "$0") start sh1 3 trainer.max_steps=1001
  $(basename "$0") start sg3 2 checkpointing.load=outputs/<run>/checkpoint.ckpt checkpointing.resume=true
EOF
        exit 2
        ;;
esac
