#!/bin/sh
# Parse flags for extras
while test $# != 0
do
    case "$1" in
        --microrts) microrts=t ;;
        --lux) lux=t ;;
    esac
    shift
done

# System-level dependencies:
# On many HPC systems you won't have sudo privileges.
# Uncomment these lines if you're on a system where you can run sudo.
#: <<'EOF'
#sudo apt update
#sudo apt install -y python-opengl
#sudo apt install -y ffmpeg
#sudo apt install -y xvfb
#sudo apt install -y swig
#sudo apt install -y default-jdk
#
#curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
#sudo apt install -y git-lfs
#git lfs pull
#EOF

# Ensure pip is upgraded (using the system pip within your venv)
python3 -m pip install --upgrade pip

# Upgrade torch-related packages (you can adjust versions as needed)
pip install --upgrade torch torchvision torchaudio

# Install pipx if not already installed
python3 -m pip install --upgrade pipx
python3 -m pipx ensurepath --force

# Install Poetry using pipx so that it’s isolated
pipx install poetry

# Upgrade pip within Poetry’s environment
poetry run pip install --upgrade pip

# Build extra flags for Poetry based on command-line options
poetry_extras=""
if [ "$microrts" = "t" ]; then
    poetry_extras="$poetry_extras -E microrts"
fi
if [ "$lux" = "t" ]; then
    # For lux, install the extra dependency 'vec-noise' first (not installable via Poetry)
    poetry run pip install vec-noise
    poetry_extras="$poetry_extras -E lux"
fi

# Install project dependencies with extras specified
poetry install $poetry_extras

if [ "$lux" = "t" ]; then
    # Check CUDA version and install appropriate jax version
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    # Check if CUDA version is 11 or 12 and install the appropriate jax version
    if [[ $cuda_version == 11.* ]]; then
        echo "CUDA version 11 detected. Installing jax for CUDA 11."
        poetry run pip install --upgrade "jax[cuda11_pip]==0.4.7" \
          -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    elif [[ $cuda_version == 12.* ]]; then
        echo "CUDA version 12 detected. Installing jax for CUDA 12."
        poetry run pip install --upgrade "jax[cuda12_pip]==0.4.7" \
          -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "Unsupported CUDA version: $cuda_version"
    fi
fi

# Log in to Weights & Biases using Poetry's environment
poetry run wandb login
