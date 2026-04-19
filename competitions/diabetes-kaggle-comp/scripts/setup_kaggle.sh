#!/bin/bash

# Setup script for Kaggle API credentials

echo "🔧 Setting up Kaggle API credentials..."
echo ""

# Add kaggle to PATH
export PATH="$HOME/Library/Python/3.9/bin:$PATH"

# Create .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Check if kaggle.json already exists
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✅ Kaggle credentials already exist!"
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "📝 Please follow these steps to get your Kaggle API token:"
    echo ""
    echo "1. Go to: https://www.kaggle.com/account"
    echo "2. Scroll down to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. This will download 'kaggle.json' to your Downloads folder"
    echo ""
    echo "Once downloaded, run this command:"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Press Enter after you've moved kaggle.json to ~/.kaggle/"
    
    if [ -f ~/.kaggle/kaggle.json ]; then
        chmod 600 ~/.kaggle/kaggle.json
        echo "✅ Credentials set up successfully!"
    else
        echo "❌ kaggle.json not found in ~/.kaggle/"
        exit 1
    fi
fi

# Test the setup
echo ""
echo "🧪 Testing Kaggle API..."
kaggle --version

echo ""
echo "✅ Setup complete! You can now download data with:"
echo "   kaggle competitions download -c playground-series-s5e12"

