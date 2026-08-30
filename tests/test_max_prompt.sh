#!/usr/bin/env bash
# ----------------------------------------------------------------------
# test_max_prompt.sh
# Find the maximum request size the Llama server will accept via the proxy.
# ----------------------------------------------------------------------
set -euo pipefail

PROXY_URL="http://localhost:8000/v1/chat/completions"
MODEL="qwen3-coder"

# Function to generate a random payload of a given size (in characters)
generate_payload() {
    local size=$1
    # Use Python to reliably generate a random ASCII string of the requested length
    # This avoids locale issues with tr and ensures proper quoting
    python3 -c "import sys, random, string; print(''.join(random.choices(string.ascii_letters + string.digits + ' ', k=int(sys.argv[1]))), end='')" "$size"
}

# Test sizes (characters). Adjust the step size if you want finer granularity.
# The loop will stop once a request fails.
sizes=(
  19500   # ~4.875k tokens
)

for size in "${sizes[@]}"; do
    echo "------------------------------------------------------------"
    echo "Testing payload size: $size characters"
    payload=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {
      "role": "user",
      "content": "$(generate_payload "$size")"
    }
  ],
  "stream": false
}
EOF
)

    # Measure elapsed time
    start=$(date +%s.%N)
    # Use a generous timeout for the curl request (300 s matches the proxy's Llama timeout)
    response=$(curl -s -w "\n%{http_code}" --max-time 300 -X POST "$PROXY_URL" \
        -H "Content-Type: application/json" \
        -d "$payload")
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    # Split response and HTTP code
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    echo "HTTP status: $http_code"
    echo "Elapsed time: $elapsed seconds"

    # Show a short excerpt of the body (first 200 characters)
    echo "Response (truncated): ${body:0:200}"

    # If we got a 4xx/5xx error, stop the test
    if [[ $http_code -ge 400 ]]; then
        echo "❌ Request failed at $size characters. Stopping test."
        break
    else
        echo "✅ Request succeeded at $size characters."
    fi
done

echo "----------------------------------------------------------------"
 echo "Test finished."
