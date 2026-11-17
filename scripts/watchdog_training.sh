#!/usr/bin/env bash
# 训练看门狗：在云端每隔几十秒检查一次训练进程与指标。
# 不是常驻监控 —— 它会在「异常」或「到达里程碑」时退出，
# 由调用方（Claude Code 的后台任务）收到通知后再汇报、再重启。
#
# 用法：
#   bash scripts/watchdog_training.sh [输出目录] [训练日志] [里程碑步数] [轮询秒数]
#   bash scripts/watchdog_training.sh outputs/E1_grpo_lora outputs/train_lora_500.log 25 30
#
# 退出码：
#   0 = 到达下一个里程碑（默认每 100 步），输出该点摘要
#   2 = 训练进程消失（可能是正常结束，也可能是崩溃）
#   3 = 指标出现 nan / inf
#   4 = 梯度连续 10 步为 0（梯度断了）
#   5 = 日志里出现 OOM
set -u

OUT_DIR="${1:-outputs/E1_grpo_lora}"
LOG_FILE="${2:-outputs/train_lora_500.log}"
MILESTONE_EVERY="${3:-25}"
INTERVAL="${4:-30}"

METRICS="$OUT_DIR/logs/metrics.jsonl"
LAST_MILESTONE=0
ZERO_GRAD_STREAK=0

read_metrics() {
    python - "$METRICS" <<'PY'
import json, sys
try:
    lines = open(sys.argv[1], encoding="utf-8").read().strip().splitlines()
    rec = json.loads(lines[-1])
except Exception:
    print("0|0|0|0|0")
    raise SystemExit
print("|".join(str(rec.get(key, 0)) for key in (
    "step", "kl", "grad_norm", "reward_mean", "policy_entropy")))
PY
}

summarize() {
    local step="$1" kl="$2" grad="$3" reward="$4" entropy="$5"
    echo "STATUS=milestone step=$step kl=$kl grad_norm=$grad reward_mean=$reward policy_entropy=$entropy"
}

echo "看门狗启动：metrics=$METRICS 里程碑每 $MILESTONE_EVERY 步，轮询 ${INTERVAL}s"

# 从当前步数开始对齐里程碑，重启看门狗时不会立刻触发
if [ -f "$METRICS" ]; then
    IFS='|' read -r START_STEP _ _ _ _ <<< "$(read_metrics)"
    LAST_MILESTONE=$(( START_STEP / MILESTONE_EVERY * MILESTONE_EVERY ))
fi

while true; do
    if ! pgrep -f "[t]rain_grpo_standalone.py" > /dev/null; then
        echo "STATUS=process_gone"
        echo "--- 训练日志末尾 ---"
        tail -20 "$LOG_FILE" 2>/dev/null
        exit 2
    fi

    if [ -f "$LOG_FILE" ] && grep -q "OutOfMemoryError" "$LOG_FILE"; then
        echo "STATUS=oom"
        grep -A5 "OutOfMemoryError" "$LOG_FILE" | tail -12
        exit 5
    fi

    if [ -f "$METRICS" ]; then
        IFS='|' read -r STEP KL GRAD REWARD ENTROPY <<< "$(read_metrics)"

        # nan / inf：用 awk 判断，避免依赖 bc
        if awk -v v="$KL" 'BEGIN{exit !(v ~ /nan|inf|Inf|NaN/)}'; then
            echo "STATUS=nan kl=$KL"
            exit 3
        fi

        # 梯度连续为 0
        if awk -v v="$GRAD" 'BEGIN{exit !(v+0 == 0)}'; then
            ZERO_GRAD_STREAK=$((ZERO_GRAD_STREAK + 1))
            if [ "$ZERO_GRAD_STREAK" -ge 10 ]; then
                echo "STATUS=zero_grad streak=$ZERO_GRAD_STREAK"
                exit 4
            fi
        else
            ZERO_GRAD_STREAK=0
        fi

        # 里程碑（向上取整到 MILESTONE_EVERY 的整数倍）
        if [ "$STEP" -ge $((LAST_MILESTONE + MILESTONE_EVERY)) ]; then
            LAST_MILESTONE=$(( STEP / MILESTONE_EVERY * MILESTONE_EVERY ))
            summarize "$STEP" "$KL" "$GRAD" "$REWARD" "$ENTROPY"
            exit 0
        fi
    fi

    sleep "$INTERVAL"
done
