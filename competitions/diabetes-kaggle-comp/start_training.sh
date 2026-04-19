#!/bin/bash
# Script to start optimized training in background

cd /Users/saba/Desktop/diabetes-kaggle-comp

echo "🚀 Starting Optimized Training for 0.78 AUC..."
echo "⏱️  Expected time: ~75 minutes"
echo ""

# Start training in background
nohup python3 run_optimized_0.78.py > training.log 2>&1 &

PID=$!
echo "✅ Training started!"
echo "   Process ID: $PID"
echo "   Log file: training.log"
echo ""
echo "📊 To check progress:"
echo "   tail -f training.log"
echo ""
echo "🔍 To check if running:"
echo "   ps aux | grep $PID"
echo ""
echo "⏹️  To stop:"
echo "   kill $PID"





