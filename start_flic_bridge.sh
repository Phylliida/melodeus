#!/bin/bash
# Start Flic bridge, killing any existing process on the port first

PORT=11235

echo "🔍 Checking for existing processes on port $PORT..."

# Find PIDs using the port
PIDS=$(lsof -ti:$PORT)

if [ ! -z "$PIDS" ]; then
    echo "⚠️  Found process(es) on port $PORT: $PIDS"
    for PID in $PIDS; do
        echo "🔪 Killing PID $PID..."
        kill -9 $PID
    done
    echo "✅ Killed existing processes"
    # Wait a moment for port to be released
    sleep 1
else
    echo "✅ Port $PORT is free"
fi

echo "🚀 Starting Flic bridge..."
python flic_bridge.py