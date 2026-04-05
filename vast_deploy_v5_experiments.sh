#!/bin/bash
set -euo pipefail

# Deploy two experiments in parallel on separate vast.ai instances:
#   1. V5 long training (5M steps, engineered-config branch)
#   2. V5 frame stacking (4 frames, 5M steps, frame-stack branch)

launch_instance() {
    local TRAIN_SCRIPT="$1"
    local LABEL="$2"
    local REMOTE_NAME="$3"

    OFFER_ID=$(vastai search offers \
        'gpu_ram>=10 num_gpus=1 cpu_cores>=16 inet_down>=200 rentable=true reliability>0.95 disk_space>=30 compute_cap>=60 cuda_vers>=12.4' \
        -o 'dph' --raw 2>/dev/null \
        | python3 -c "
import json, sys
offers = json.load(sys.stdin)
for o in offers:
    name = o.get('gpu_name','')
    if any(x in name for x in ['RTX 3','RTX 4','RTX 2080','A4000','A5000']):
        print(o['id'])
        exit()
print(offers[0]['id'])
" 2>/dev/null)
    echo "[$LABEL] Offer: $OFFER_ID"

    RESULT=$(vastai create instance "$OFFER_ID" \
        --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime \
        --disk 30 --label "$LABEL" --raw 2>/dev/null)
    INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract',''))" 2>/dev/null)
    echo "[$LABEL] Instance: $INSTANCE_ID"

    MAX_WAIT=300; ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status',''))" 2>/dev/null)
        [ "$STATUS" = "running" ] && break
        echo "  [$LABEL] $STATUS (${ELAPSED}s)"; sleep 15; ELAPSED=$((ELAPSED + 15))
    done

    sleep 5
    SSH_INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print(f'{d.get(\"ssh_host\",\"\")} {d.get(\"ssh_port\",\"\")}')" 2>/dev/null)
    SSH_HOST=$(echo "$SSH_INFO" | awk '{print $1}')
    SSH_PORT=$(echo "$SSH_INFO" | awk '{print $2}')
    SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT root@$SSH_HOST"

    for i in $(seq 1 12); do
        $SSH_CMD "echo ssh_ok" 2>/dev/null | grep -q "ssh_ok" && break; sleep 10
    done

    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$TRAIN_SCRIPT" "root@$SSH_HOST:/workspace/${REMOTE_NAME}"
    $SSH_CMD "chmod +x /workspace/${REMOTE_NAME} && nohup bash /workspace/${REMOTE_NAME} > /workspace/training_output.log 2>&1 &"

    echo ""
    echo "=== [$LABEL] Deployed ==="
    echo "Instance: $INSTANCE_ID | SSH: ssh -p $SSH_PORT root@$SSH_HOST"
    echo "Destroy: vastai destroy instance $INSTANCE_ID"
    echo ""
}

echo "=== Deploying V5 long training ==="
launch_instance "vast_train_v5_long.sh" "rl-v5-long" "vast_train_v5_long.sh"

echo "=== Deploying V5 frame-stack ==="
launch_instance "vast_train_v5_fs.sh" "rl-v5-fs" "vast_train_v5_fs.sh"

echo "=== Both instances deployed ==="
echo "Monitor progress:"
echo "  V5 long:  ssh to instance and: tail -3 /workspace/rl-thesis/runs/engineered_v5_long/seed_*/logs/eval.csv"
echo "  V5 fs:    ssh to instance and: tail -3 /workspace/rl-thesis/runs/engineered_v5_fs/seed_*/logs/eval.csv"
