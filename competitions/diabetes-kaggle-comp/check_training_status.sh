#!/bin/bash
# Script to check if training is running

cd /Users/saba/Desktop/diabetes-kaggle-comp

echo "🔍 Checking Training Status..."
echo "=" * 60

# Check for running processes
if ps aux | grep -E "python.*run_optimized" | grep -v grep > /dev/null; then
    echo "✅ Training IS running!"
    echo ""
    ps aux | grep -E "python.*run_optimized" | grep -v grep
    echo ""
    echo "📊 Check log file:"
    if [ -f training.log ]; then
        echo "   Last 10 lines:"
        tail -10 training.log
    fi
else
    echo "❌ Training is NOT running"
    echo ""
    echo "📄 Check log files:"
    ls -lht *.log 2>/dev/null | head -5
    echo ""
    echo "🚀 To start training:"
    echo "   ./start_training.sh"
fi

echo ""
echo "📈 To monitor progress:"
echo "   tail -f training.log"





