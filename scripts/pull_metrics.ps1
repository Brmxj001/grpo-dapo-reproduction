# 把云端训练指标同步到本地，并打印一行摘要
#
# 用法：
#   .\scripts\pull_metrics.ps1                    # 每 5 分钟同步一次，Ctrl+C 停止
#   .\scripts\pull_metrics.ps1 -IntervalSeconds 60
#   .\scripts\pull_metrics.ps1 -Once              # 只同步一次
#
# 产出（全部在本地 D:\grpo-dapo-reproduction\outputs\<RemoteOut>\）：
#   logs\metrics.jsonl   每步一行的原始指标
#   logs\metrics.csv     同内容转成 CSV，Excel 可直接打开
#   train_lora_500.log   训练脚本的每步日志

param(
    [int]$IntervalSeconds = 300,
    [string]$RemoteOut = "outputs/E1_grpo_lora",
    [string]$LocalRoot = "D:\grpo-dapo-reproduction",
    [switch]$Once
)

$ErrorActionPreference = "Continue"
$RemoteProject = "/root/autodl-tmp/grpo-dapo-reproduction"
$LocalOut = Join-Path $LocalRoot $RemoteOut
$LogsDir = Join-Path $LocalOut "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Sync-Once {
    $stamp = Get-Date -Format "HH:mm:ss"
    $localJsonl = Join-Path $LogsDir "metrics.jsonl"

    scp -q "autodl:$RemoteProject/$RemoteOut/logs/metrics.jsonl" $localJsonl 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Output "[$stamp] 还没有 metrics.jsonl（训练可能尚未写出第一步）"
        return
    }

    scp -q "autodl:$RemoteProject/outputs/train_lora_500.log" (Join-Path $LocalOut "train_lora_500.log") 2>$null

    # 转 CSV，方便 Excel / 画图
    try {
        $rows = Get-Content $localJsonl | ForEach-Object { $_ | ConvertFrom-Json }
        if ($rows) {
            $rows | Export-Csv -Path (Join-Path $LogsDir "metrics.csv") -NoTypeInformation -Encoding UTF8
        }
    } catch {}

    $latest = Get-Content $localJsonl -Tail 1 | ConvertFrom-Json
    $count = (Get-Content $localJsonl | Measure-Object -Line).Lines
    $fmt = { param($v, $d) if ($null -eq $v) { "-" } else { [math]::Round([double]$v, $d) } }

    Write-Output ("[$stamp] {0} 步 | kl={1} ent={2} rew={3} zvar={4} ratio={5} clip={6} grad={7} len={8} trunc={9} mem={10}G lr={11}" -f `
        $count,
        (& $fmt $latest.kl 4), (& $fmt $latest.policy_entropy 3),
        (& $fmt $latest.reward_mean 2), (& $fmt $latest.zero_variance_ratio 2),
        (& $fmt $latest.ratio_mean 4), (& $fmt $latest.clip_fraction 4),
        (& $fmt $latest.grad_norm 3), (& $fmt $latest.avg_response_length 0),
        (& $fmt $latest.truncation_rate 2), (& $fmt $latest.gpu_mem_gb 1),
        (& $fmt $latest.learning_rate 8))
}

if ($Once) {
    Sync-Once
    exit 0
}

Write-Output "每 $IntervalSeconds 秒同步到 $LogsDir（Ctrl+C 停止，不影响云端训练）"
while ($true) {
    Sync-Once
    Start-Sleep -Seconds $IntervalSeconds
}
