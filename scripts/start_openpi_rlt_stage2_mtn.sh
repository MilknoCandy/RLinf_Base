#!/usr/bin/env bash
set -euo pipefail

# Resolve repo paths and tunables.
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
RAY_PORT=${RAY_PORT:-29500}  # Default port for Ray, can be modified if needed
CLUSTER_NODES=${CLUSTER_NODES:-2}
ELECTION_TIMEOUT=${ELECTION_TIMEOUT:-300}
# CONFIG_NAME=${CONFIG_NAME:-maniskill_rlt_stage2_ac_mlp}
# ENTRY_SCRIPT=${ENTRY_SCRIPT:-examples/embodiment/run_embodiment.sh}

# Shared state used for leader election and run completion.
STATE_DIR="${SHARED_STATE_DIR:-${REPO_PATH}/ray_state}"
CANDIDATE_DIR="${STATE_DIR}/candidates"
HEAD_IP_FILE="${STATE_DIR}/ray_head_ip.txt"
DONE_FILE="${STATE_DIR}/train.done"

# Prepare shared state directories and clear leftovers from older runs.
mkdir -p "${STATE_DIR}" "${CANDIDATE_DIR}"
rm -f "${HEAD_IP_FILE}" "${DONE_FILE}"

# Record this node as a candidate and derive its local address.
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
LOCAL_IP="${NODE_IP:-$(hostname -I | awk '{print $1}')}"
CANDIDATE_FILE="${CANDIDATE_DIR}/${HOSTNAME_SHORT}.env"

# Always stop Ray and remove the candidate file on exit.
cleanup() {
  ray stop >/dev/null 2>&1 || true
  rm -f "${CANDIDATE_FILE}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

# Publish the candidate metadata used by the leader election step.
printf 'hostname=%s\nip=%s\npid=%s\nstarted_at=%s\n' \
  "${HOSTNAME_SHORT}" "${LOCAL_IP}" "$$" "$(date +%s)" > "${CANDIDATE_FILE}"

# Collect live candidates that are still within the election window.
list_recent_candidates() {
  local now
  now="$(date +%s)"

  for candidate in "${CANDIDATE_DIR}"/*.env; do
    [ -f "${candidate}" ] || continue

    local started_at hostname ip age
    started_at="$(sed -n 's/^started_at=//p' "${candidate}" | head -n1)"
    hostname="$(sed -n 's/^hostname=//p' "${candidate}" | head -n1)"
    ip="$(sed -n 's/^ip=//p' "${candidate}" | head -n1)"

    [ -n "${started_at}" ] || continue
    [ -n "${hostname}" ] || continue
    [ -n "${ip}" ] || continue

    age=$((now - started_at))
    if [ "${age}" -le "${ELECTION_TIMEOUT}" ]; then
      printf '%s\t%s\t%s\t%s\n' "${ip}" "${hostname}" "${started_at}" "${candidate}"
    fi
  done
}

# Wait until enough nodes have registered, then choose the smallest IP/hostname
# pair as the head so every node reaches the same decision.
deadline="$(( $(date +%s) + ELECTION_TIMEOUT ))"
HEAD_RECORD=""
while :; do
  RECENT_CANDIDATES="$(list_recent_candidates)"
  CANDIDATE_COUNT="$(printf '%s\n' "${RECENT_CANDIDATES}" | sed '/^$/d' | wc -l | awk '{print $1}')"
  if [ "${CANDIDATE_COUNT}" -ge "${CLUSTER_NODES}" ]; then
    HEAD_RECORD="$(printf '%s\n' "${RECENT_CANDIDATES}" | sort -t $'\t' -k1,1V -k2,2 | head -n1)"
    break
  fi

  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "Timed out waiting for ${CLUSTER_NODES} recent candidate nodes in ${CANDIDATE_DIR}."
    exit 1
  fi

  sleep 1
done

# Split the chosen head record into IP and hostname.
HEAD_IP="$(printf '%s' "${HEAD_RECORD}" | awk -F '\t' '{print $1}')"
HEAD_HOSTNAME="$(printf '%s' "${HEAD_RECORD}" | awk -F '\t' '{print $2}')"

# Decide whether this process is the head or a worker.
if [ "${HOSTNAME_SHORT}" = "${HEAD_HOSTNAME}" ] && [ "${LOCAL_IP}" = "${HEAD_IP}" ]; then
  ROLE="head"
else
  ROLE="worker"
fi

if [ "${ROLE}" = "head" ]; then
  # Head node starts Ray in head mode and then launches the training script.
  export RLINF_NODE_RANK=0
  ray start --head --port="${RAY_PORT}" --node-ip-address="${HEAD_IP}"
  echo "${HEAD_IP}" > "${HEAD_IP_FILE}"

  # Check RLINF_CODE_WORKING_DIR; without sync, rlinf/ must match on all nodes. With sync, only the rlinf/ package is shipped (avoid large files under rlinf/); configs under examples/, models, and simulator assets must still be local or on shared storage.
  # export RLINF_CODE_WORKING_DIR=auto
  bash scripts/run_openpi_rlt_stage2_mtn.sh

  # Signal workers that training is done.
  touch "${DONE_FILE}"
else
  # Worker node waits for the head IP, joins the Ray cluster, and then idles.
  export RLINF_NODE_RANK=1
  echo "Waiting for head IP..."
  while [ ! -s "${HEAD_IP_FILE}" ]; do
    sleep 1
  done

  HEAD_IP="$(cat "${HEAD_IP_FILE}")"
  ray start --address="${HEAD_IP}:${RAY_PORT}"

  echo "Waiting for training to finish..."
  while [ ! -f "${DONE_FILE}" ]; do
    sleep 10
  done
fi