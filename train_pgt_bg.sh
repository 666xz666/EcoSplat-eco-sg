#!/usr/bin/env bash
# train_pgt_bg.sh — 第一阶段 PGT（10 视角、全参数、400K 步）后台启动器
#
#   ./train_pgt_bg.sh probe 3               # 51 步探针：看显存/步速，约 10 分钟
#   ./train_pgt_bg.sh start 3               # 单卡，1 x accum 4（与 Eco 完全一致）
#   ./train_pgt_bg.sh start 3,6             # 双卡，每卡 1 x accum 2
#   ./train_pgt_bg.sh start 3,6,4,2         # 四卡，每卡 1，不累积
#   ./train_pgt_bg.sh sync                  # 每小时把新检查点推到对象存储（本地只留 3 个）
#   ./train_pgt_bg.sh status                # 训练与同步是否在跑 + 最近两行进度
#   ./train_pgt_bg.sh follow                # tail -f 训练日志
#   ./train_pgt_bg.sh stop                  # 停训练与同步（整组信号）
#
# 有效 batch 固定为 4（每卡 micro-batch 1，靠 accumulate 补到 4），与
# EcoSplat 的 10 视角配方一致；所以卡数只能取 4 的约数（1/2/4）。
# 环境变量：CONDA_ENV / CONDA_SH / LOG_DIR / RUN_DIR / OUT_ROOT
#           SYNC_REMOTE（rclone 目标，如 oss:ckpt/pgt_sg3_10v）/ SYNC_INTERVAL（秒，默认 3600）
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-ecosplat}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RUN_DIR="${RUN_DIR:-$REPO/.run}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
OUT_ROOT="${OUT_ROOT:-/data14T/xz/outputs}"
DATA_ROOT="${DATA_ROOT:-/data14T/xz/data/re10k}"
EXPERIMENT="spfsplat/sg/re10k_pgt_sg"
TAG="pgt_sg3_10v"
EFFECTIVE_BATCH=4

mkdir -p "$RUN_DIR" "$LOG_DIR"
pidfile() { echo "$RUN_DIR/$TAG.pid"; }
syncpidfile() { echo "$RUN_DIR/$TAG.sync.pid"; }
is_running() { local f; f=$(pidfile); [ -f "$f" ] && kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null; }
sync_running() { local f; f=$(syncpidfile); [ -f "$f" ] && kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null; }

launch() {
    local gpus="${1:?用法: $0 start|probe <gpu_ids> [hydra overrides...]}"; shift
    local mode="$1"; shift
    local n_gpu accum suffix extra run_dir
    n_gpu=$(awk -F',' '{print NF}' <<<"$gpus")
    if (( EFFECTIVE_BATCH % n_gpu != 0 )); then
        echo "卡数必须是 ${EFFECTIVE_BATCH} 的约数（1/2/4），收到 $n_gpu 张：$gpus" >&2
        exit 2
    fi
    accum=$(( EFFECTIVE_BATCH / n_gpu ))
    suffix=""; extra=""; run_dir="$OUT_ROOT/$TAG"
    if [ "$mode" = "probe" ]; then
        # 51 步足够读出显存峰值与步速；不落 checkpoint，避免白写一个 7 GB 文件。
        suffix="_probe"; run_dir="$OUT_ROOT/${TAG}_probe"
        extra="trainer.max_steps=51 train.print_log_every_n_steps=1 checkpointing.every_n_train_steps=100000"
    fi

    local ts log payload
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOG_DIR/${TAG}${suffix}_${ts}.log"
    payload="$RUN_DIR/${TAG}${suffix}.cmd.sh"
    cat > "$payload" <<EOF
#!/usr/bin/env bash
source '$CONDA_SH'
conda activate '$CONDA_ENV'
cd '$REPO'
echo "[pgt] \$(date '+%F %T') mode=$mode gpus=$gpus micro=1 accum=$accum effective=$((1 * $n_gpu * $accum))"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m src.main +experiment='$EXPERIMENT' \\
    wandb.mode=offline wandb.name='${TAG}${suffix}' \\
    dataset.re10k.roots=[$DATA_ROOT] \\
    data_loader.train.num_workers=4 \\
    data_loader.train.batch_size=1 \\
    trainer.accumulate_grad_batches=$accum \\
    hydra.run.dir='$run_dir' \\
    $extra $*
EOF
    chmod +x "$payload"

    # 每行前面加时间戳，日志里就能直接算出 s/step（步速估计要用）。
    setsid nohup bash -c "bash '$payload' 2>&1 | awk '{ print strftime(\"%F %T\"), \$0; fflush() }'" \
        >>"$log" 2>&1 &
    local pid=$!
    echo "$pid" > "$(pidfile)"
    echo "$gpus" > "$RUN_DIR/$TAG.gpu"
    ln -sfn "$log" "$LOG_DIR/$TAG.latest.log"

    sleep 10
    if is_running; then
        echo "[$TAG] started pid=$pid gpus=$gpus micro=1 accum=$accum effective=$((1 * n_gpu * accum))"
        echo "  log : $log"
        echo "  查看: $0 follow"
    else
        echo "[$TAG] 启动失败，日志末尾："
        tail -n 30 "$log"
        rm -f "$(pidfile)"
        exit 1
    fi
}

start_sync() {
    if [ -z "${SYNC_REMOTE:-}" ]; then
        echo "先用 SYNC_REMOTE=<rclone 目标> 指定落点，例如：" >&2
        echo "  SYNC_REMOTE=oss:ckpt/$TAG $0 sync" >&2
        exit 2
    fi
    if ! command -v rclone >/dev/null; then
        echo "找不到 rclone，先装或用对象存储客户端替代" >&2
        exit 2
    fi
    if sync_running; then
        echo "[$TAG] sync 已在运行 pid=$(cat "$(syncpidfile)")"; return 0
    fi

    local ckpt_dir="$OUT_ROOT/$TAG/checkpoints"
    local hydra_dir="$OUT_ROOT/$TAG/.hydra"
    local payload="$RUN_DIR/$TAG.sync.sh"
    local slog="$LOG_DIR/$TAG.sync.log"
    # 只往上加不删本地：--ignore-existing 让每小时的重跑变成增量，检查点写完就不再碰。
    cat > "$payload" <<EOF
#!/usr/bin/env bash
while :; do
    echo "[\$(date '+%F %T')] sync $ckpt_dir -> $SYNC_REMOTE"
    if [ -d '$ckpt_dir' ]; then
        rclone copy --ignore-existing --transfers 4 --checkers 8 --stats 2m \\
            '$ckpt_dir' '$SYNC_REMOTE' || echo "[warn] checkpoint copy rc=\$?"
    fi
    if [ -d '$hydra_dir' ]; then
        rclone copy --ignore-existing '$hydra_dir' '$SYNC_REMOTE/.hydra' || true
    fi
    sleep ${SYNC_INTERVAL:-3600}
done
EOF
    chmod +x "$payload"
    setsid nohup bash "$payload" >>"$slog" 2>&1 &
    echo $! > "$(syncpidfile)"
    ln -sfn "$slog" "$LOG_DIR/$TAG.sync.latest.log"
    sleep 2
    if sync_running; then
        echo "[$TAG] sync started pid=$(cat "$(syncpidfile)") -> $SYNC_REMOTE, every ${SYNC_INTERVAL:-3600}s"
        echo "  log : $slog"
    else
        echo "[$TAG] sync 启动失败，日志末尾："; tail -n 20 "$slog"; rm -f "$(syncpidfile)"; exit 1
    fi
}

status() {
    if is_running; then
        local pid; pid=$(cat "$(pidfile)")
        printf "%-14s RUNNING pid=%-8s gpus=%-12s elapsed=%s\n" "$TAG" "$pid" \
            "$(cat "$RUN_DIR/$TAG.gpu" 2>/dev/null || echo '?')" \
            "$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
        tail -n 80 "$LOG_DIR/$TAG.latest.log" 2>/dev/null | grep "train step" | tail -n 2 | sed 's/^/    /'
    else
        printf "%-14s stopped\n" "$TAG"
        [ -f "$LOG_DIR/$TAG.latest.log" ] && tail -n 3 "$LOG_DIR/$TAG.latest.log" | sed 's/^/    /'
    fi
    if sync_running; then
        printf "%-14s RUNNING pid=%s\n" "sync" "$(cat "$(syncpidfile)")"
        tail -n 2 "$LOG_DIR/$TAG.sync.latest.log" 2>/dev/null | sed 's/^/    /'
    else
        printf "%-14s stopped\n" "sync"
    fi
    local ckpt_dir="$OUT_ROOT/$TAG/checkpoints"
    if [ -d "$ckpt_dir" ]; then
        printf "checkpoints %s 个, %s\n" "$(find "$ckpt_dir" -name '*.ckpt' | wc -l)" \
            "$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)"
    fi
}

stop() {
    local stopped=0
    if is_running; then
        local pid; pid=$(cat "$(pidfile)")
        # 整组发信号，否则 DDP 的其它 rank 会留在卡上。
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 -- -"$pid" 2>/dev/null
        rm -f "$(pidfile)"
        echo "[$TAG] stopped"; stopped=1
    fi
    if sync_running; then
        kill "$(cat "$(syncpidfile)")" 2>/dev/null
        rm -f "$(syncpidfile)"
        echo "[sync] stopped"; stopped=1
    fi
    [ "$stopped" = 0 ] && echo "[$TAG] nothing running"
}

case "${1:-}" in
    start) shift; launch "${1:?用法: $0 start <gpu_ids> [overrides...]}" run ;;
    probe) shift; launch "${1:?用法: $0 probe <gpu_ids> [overrides...]}" probe ;;
    sync) start_sync ;;
    status) status ;;
    follow) tail -f "$LOG_DIR/$TAG.latest.log" ;;
    stop) stop ;;
    *) sed -n '2,16p' "$0" >&2; exit 2 ;;
esac
