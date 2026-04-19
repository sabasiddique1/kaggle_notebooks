#!/bin/bash
# Real-time training monitor

cd /Users/saba/Desktop/diabetes-kaggle-comp

echo "🔍 Training Monitor"
echo "=================="

# Check if process is running
PID=$(ps aux | grep "run_optimized_0.78.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ Training is NOT running"
    echo ""
    echo "🚀 To start: ./start_training.sh"
    exit 1
fi

echo "✅ Training IS running!"
echo "   Process ID: $PID"
echo ""

# Get process info
ps aux | grep $PID | grep -v grep | awk '{
    print "   CPU Usage: " $3 "%"
    print "   Memory: " $4 "%"
    print "   Runtime: " $10
}'

# Check for output files
echo ""
echo "📄 Output Files:"
if [ -f "submission_optimized_0.78.csv" ]; then
    echo "   ✅ submission_optimized_0.78.csv exists"
    ls -lh submission_optimized_0.78.csv | awk '{print "      Size: " $5 ", Modified: " $6 " " $7 " " $8}'
else
    echo "   ⏳ submission_optimized_0.78.csv not created yet (training in progress)"
fi

# Estimate progress
echo ""
echo "📈 Progress Estimate:"
echo "   Expected total: ~75 minutes"
echo "   Check runtime above for current progress"

echo ""
echo "💡 To see detailed output (if redirected):"
echo "   tail -f training.log"
echo "   (or check terminal where script was started)"





