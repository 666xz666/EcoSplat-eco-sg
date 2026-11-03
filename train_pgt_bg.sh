#!/usr/bin/env bash
# train_pgt_bg.sh — 第一阶段 PGT（全参数、2 视角、400K 步）后台启动器
#
#   ./train_pgt_bg.sh probe 4              # 50 步探针：看显存/步速，约 5 分钟
#   ./train_pgt_bg.sh start 4              # 单卡，batch 12
#   ./train_pgt_bg.sh start 4,3            # 双卡，每卡 batch 6（有效 batch 12）
#   ./train_pgt_bg.sh start 4,3,2,6        # 四卡，每卡 batch 3
#   ./train_pgt_bg.sh status               # 是否在跑 + 最近两行进度
#   ./train_pgt_bg.sh follow               # tail -f
#   ./train_pgt_bg.sh stop                 # 停整个进程组（含 DDP worker）
#
# 有效 batch 固定为 12（EcoSplat 第一阶段配方的 batch_size），所以卡数只能取
# 12 的约数（1/2/3/4/6），每卡 batch = 12 / 卡数。
# 环境变量：CONDA_ENV(默认 ecosplat) / CONDA_SH / LOG_DIR / RUN_DIR
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-ecosplat}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RUN_DIR="${RUN_DIR:-$REPO/.run}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
OUT_ROOT="${OUT_ROOT:-/data14T/xz/outputs}"
DATA_ROOT="${DATA_ROOT:-/data14T/xz/data/re10k}"
EXPERIMENT="spfsplat/sg/re10k_pgt_sg"
TAG="pgt_sg3"

mkdir -p "$RUN_DIR" "$LOG_DIR"
pidfile() { echo "$RUN_DIR/$TAG.pid"; }
is_running() { local f; f=$(pidfile); [ -f "$f" ] && kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null; }

launch() {
    local gpus="${1:?用法: $0 start|probe <gpu_ids> [hydra overrides...]}"; shift
    local mode="$1"; shift
    local n_gpu batch suffix extra run_dir
    n_gpu=$(awk -F',' '{print NF}' <<<"$gpus")
    if (( 12 % n_gpu != 0 )); then
        echo "卡数必须是 12 的约数（1/2/3/4/6），收到 $n_gpu 张：$gpus" >&2
        exit 2
    fi
    batch=$(( 12 / n_gpu ))
    suffix=""; extra=""; run_dir="$OUT_ROOT/$TAG"
    if [ "$mode" = "probe" ]; then
        # 50 步足够读出显存峰值与步速；不落 checkpoint，避免白写一个 7 GB 文件。
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
echo "[pgt] \$(date '+%F %T') mode=$mode gpus=$gpus batch/rank=$batch effective=$(($batch * $n_gpu))"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m src.main +experiment='$EXPERIMENT' \\
    wandb.mode=offline wandb.name='${TAG}${suffix}' \\
    dataset.re10k.roots=[$DATA_ROOT] \\
    data_loader.train.num_workers=4 \\
    data_loader.train.batch_size=$batch \\
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
        echo "[$TAG] started pid=$pid gpus=$gpus batch/rank=$batch effective=$(($batch * $n_gpu))"
        echo "  log : $log"
        echo "  查看: $0 follow"
    else
        echo "[$TAG] 启动失败，日志末尾："
        tail -n 30 "$log"
        rm -f "$(pidfile)"
        exit 1
    fi
}

status() {
    if is_running; then
        local pid; pid=$(cat "$(pidfile)")
        printf "%-8s RUNNING pid=%-8s gpus=%-12s elapsed=%s\n" "$TAG" "$pid" \
            "$(cat "$RUN_DIR/$TAG.gpu" 2>/dev/null || echo '?')" \
            "$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
        tail -n 80 "$LOG_DIR/$TAG.latest.log" 2>/dev/null | grep "train step" | tail -n 2 | sed 's/^/    /'
    else
        printf "%-8s stopped\n" "$TAG"
        [ -f "$LOG_DIR/$TAG.latest.log" ] && tail -n 3 "$LOG_DIR/$TAG.latest.log" | sed 's/^/    /'
    fi
}

stop() {
    if ! is_running; then echo "[$TAG] not running"; rm -f "$(pidfile)"; return 0; fi
    local pid; pid=$(cat "$(pidfile)")
    # 整组发信号，否则 DDP 的其它 rank 会留在卡上。
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$pid" 2>/dev/null
    rm -f "$(pidfile)"
    echo "[$TAG] stopped"
}

case "${1:-}" in
    start) shift; launch "${1:?用法: $0 start <gpu_ids> [overrides...]}" run ;;
    probe) shift; launch "${1:?用法: $0 probe <gpu_ids> [overrides...]}" probe ;;
    status) status ;;
    follow) tail -f "$LOG_DIR/$TAG.latest.log" ;;
    stop) stop ;;
    *) sed -n '2,14p' "$0" >&2; exit 2 ;;
esac
