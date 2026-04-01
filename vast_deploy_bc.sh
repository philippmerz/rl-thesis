#!/bin/bash
set -euo pipefail

TRAIN_SCRIPT="vast_train_bc.sh"
LABEL="rl-thesis-bc-rl"

echo "=== Searching for GPU instances ==="
OFFER_ID=$(vastai search offers \
    'gpu_ram>=8 num_gpus=1 cpu_cores>=12 inet_down>=200 rentable=true reliability>0.95 disk_space>=30 compute_cap>=60 cuda_vers>=12.4' \
    -o 'dph' --raw 2>/dev/null \
    | python3 -c "
import json, sys
offers = json.load(sys.stdin)
for o in offers:
    name = o.get('gpu_name','')
    if any(x in name for x in ['RTX','A4000','A5000','Tesla_V100','Tesla_T4']):
        print(o['id'])
        exit()
print(offers[0]['id'])
" 2>/dev/null)

if [ -z "$OFFER_ID" ]; then
    echo "ERROR: No suitable offers found"
    exit 1
fi
echo "Selected offer: $OFFER_ID"

echo "=== Creating instance ==="
RESULT=$(vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime \
    --disk 30 \
    --label "$LABEL" \
    --raw 2>/dev/null)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract',''))" 2>/dev/null)
if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Failed to create instance. Response: $RESULT"
    exit 1
fi
echo "Instance created: $INSTANCE_ID"

echo "=== Waiting for instance to start ==="
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status',''))" 2>/dev/null)
    if [ "$STATUS" = "running" ]; then
        echo "Instance is running!"
        break
    fi
    echo "  Status: $STATUS (${ELAPSED}s elapsed)"
    sleep 15
    ELAPSED=$((ELAPSED + 15))
done

if [ "$STATUS" != "running" ]; then
    echo "ERROR: Instance did not start within ${MAX_WAIT}s"
    exit 1
fi

sleep 5
SSH_INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'{d.get(\"ssh_host\",\"\")} {d.get(\"ssh_port\",\"\")}')
" 2>/dev/null)
SSH_HOST=$(echo "$SSH_INFO" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_INFO" | awk '{print $2}')
echo "SSH: $SSH_HOST:$SSH_PORT"

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT root@$SSH_HOST"
SCP_CMD="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "=== Waiting for SSH ==="
for i in $(seq 1 12); do
    if $SSH_CMD "echo ssh_ok" 2>/dev/null | grep -q "ssh_ok"; then
        echo "SSH ready!"
        break
    fi
    echo "  SSH not ready yet (attempt $i/12)"
    sleep 10
done

echo "=== Uploading training script ==="
$SCP_CMD "$TRAIN_SCRIPT" "root@$SSH_HOST:/workspace/vast_train_bc.sh"

echo "=== Launching training ==="
$SSH_CMD "chmod +x /workspace/vast_train_bc.sh && nohup bash /workspace/vast_train_bc.sh > /workspace/training_output.log 2>&1 &"

echo ""
echo "=== Deployment complete ==="
echo "Instance ID: $INSTANCE_ID"
echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo "Monitor: ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/training_output.log'"
echo "Progress: ssh -p $SSH_PORT root@$SSH_HOST 'tail -3 /workspace/rl-thesis/runs/engineered/seed_*/logs/eval.csv'"
echo ""
echo "Download results:"
echo "  scp -r -P $SSH_PORT root@$SSH_HOST:/workspace/rl-thesis/runs/ ./runs_vast_bc/"
echo "  vastai destroy instance $INSTANCE_ID"
