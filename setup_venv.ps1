# Remove existing virtual environment if it exists
if (Test-Path -Path "venv") {
    Remove-Item -Recurse -Force "venv"
}

# Create and activate new virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

Write-Host "`nVirtual environment setup complete.`nTo activate the virtual environment, run:
.\venv\Scripts\Activate`n"
