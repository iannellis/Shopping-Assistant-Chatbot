#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Has variables we need
source $SCRIPT_DIR/docker/.env

declare -A MODEL_FILES MODEL_GOOGLE_IDS CHROMA_FILES CHROMA_GOOGLE_IDS

# Array of BLIP-2 models mapping to their file names and Google Drive IDs
MODEL_FILES["pretrain"]="blip-2-pretrain.pt"
MODEL_FILES["gs"]="blip-2-gs.pt"
MODEL_FILES["abo"]="blip-2-abo.pt"
MODEL_FILES["coco"]="blip-2-coco.pt"
MODEL_GOOGLE_IDS["pretrain"]="1xLfjTUf4MuBl1FeDpmClt04etF-PaZCZ"
MODEL_GOOGLE_IDS["gs"]="1jsuiImeloqeQN99gULJbCZCHL8Q8mR92"
MODEL_GOOGLE_IDS["abo"]="1kNkkk2Q6922a9oXQUol19hg16z_4JuE5"
MODEL_GOOGLE_IDS["coco"]="15CuzLdzyhORtVFMLhuKZ16E50ENJavY4"

CHROMA_FILES["pretrain"]="chroma_pretrain.tar"
CHROMA_FILES["gs"]="chroma_gs.tar"
CHROMA_FILES["abo"]="chroma_abo.tar"
CHROMA_FILES["coco"]="chroma_coco.tar"
CHROMA_GOOGLE_IDS["pretrain"]="13io3F5krf2HF8Ym__40unrNI8liZtG5b"
CHROMA_GOOGLE_IDS["gs"]="1GFjwaOqIecG3F6LZC9fIO_zb7qNJISWi"
CHROMA_GOOGLE_IDS["abo"]="13iPFG_QuOydoyGKZpFvZCD4Tl7RybEBC"
CHROMA_GOOGLE_IDS["coco"]="1AcjQmAimO04Dj8pSCkwoVapBDB5zl2LX"

# ABO metadata Pandas dataframe file name and Google Drive ID
ABO_METADATA_DF=("abo-listings-final-draft.pkl", "1hChAT7PL_3c9YQugQJFFOAElbRPV7yqg")

# ABO images
ABO_IMAGES=("abo-images-small.tar", "1wvEJbRL4hZ5de8Mm2sIvoZZ_frrl5MFl")

# Ensure the selected model is valid
if [ -z "${MODEL_FILES[$BLIP_2_MODEL]}" ]; then
    echo "Error: BLIP-2 Model '$BLIP_2_MODEL' is not defined. Please check your BLIP_2_MODEL in the .env file."
    exit 1
fi

# Check if the 'models' directory exists; create it if not
if [ ! -d "$BLIP_2_DIR_LOCAL" ]; then
    echo "Directory '$BLIP_2_DIR_LOCAL' does not exist. Creating it..."
    mkdir "$BLIP_2_DIR_LOCAL"
else
    echo "Directory '$BLIP_2_DIR_LOCAL' already exists."
fi

# Check if the 'OLlama' directory exists; create it if not
if [ ! -d "$OLLAMA_DIR_LOCAL" ]; then
    echo "Directory '$OLLAMA_DIR_LOCAL' does not exist. Creating it..."
    mkdir "$OLLAMA_DIR_LOCAL"
else
    echo "Directory '$OLLAMA_DIR_LOCAL' already exists."
fi

# Check if the 'database' directory exists; create it if not
if [ ! -d "$CHROMA_DIR_LOCAL" ]; then
    echo "Directory '$CHROMA_DIR_LOCAL' does not exist. Creating it..."
    mkdir "$CHROMA_DIR_LOCAL"
else
    echo "Directory '$CHROMA_DIR_LOCAL' already exists."
fi

# Check if the 'ABO dataset' directory exists; create it if not
if [ ! -d "$ABO_DIR_LOCAL" ]; then
    echo "Directory '$ABO_DIR_LOCAL' does not exist. Creating it..."
    mkdir "$ABO_DIR_LOCAL"
else
    echo "Directory '$ABO_DIR_LOCAL' already exists."
fi

sudo apt update

# For downloading models from Google Drive
sudo apt install pipx -y
pipx ensurepath
source ~/.bashrc
pipx install gdown

# For progress bars
sudo apt install pv -y

# Check and download a file if it doesn't exist
gdownload_file() {
    local file_name=$1
    local file_id=$2

    if [ ! -f "$file_name" ]; then
        echo "File '$file_name' not found. Downloading..."
        ~/.local/bin/gdown "$file_id" -O "$file_name"
        if [ $? -eq 0 ]; then
            echo "File '$file_name' downloaded successfully."
        else
            echo "Error downloading '$file_name'."
        fi
    else
        echo "File '$file_name' already exists. Skipping download."
    fi
}

# Download the model file
cd "$BLIP_2_DIR_LOCAL"
gdownload_file "${MODEL_FILES[$BLIP_2_MODEL]}" "${MODEL_GOOGLE_IDS[$BLIP_2_MODEL]}"

# Download the Chroma database and extract it
cd "$CHROMA_DIR_LOCAL"
gdownload_file "${CHROMA_FILES[$BLIP_2_MODEL]}" "${CHROMA_GOOGLE_IDS[$BLIP_2_MODEL]}"

if [ ! -f "chroma.sqlite3" ]; then
    echo "Extracting Chroma database..."
    tar -xf "${CHROMA_FILES[$BLIP_2_MODEL]}"
fi

# Get the ABO dataset processed metadata
cd "$ABO_DIR_LOCAL"
gdownload_file "${ABO_METADATA_DF[0]}" "${ABO_METADATA_DF[1]}"

# Get the ABO dataset images and extract them
gdownload_file "${ABO_IMAGES[0]}" "${ABO_IMAGES[1]}"

if [ ! -d "images" ]; then
    echo "Extracting abo-images-small.tar..."
    pv "${ABO_IMAGES[0]}" | tar -xf-
fi

if [ -f "images/metadata/images.csv.gz" ] && [ ! -f "images/metadata/images.csv" ]; then
    echo "Extracting image metadata..."
    gunzip "images/metadata/images.csv.gz"
fi

# install Docker via apt
if snap list | grep -q docker; then
    echo "Docker is installed via Snap. Will not work with CUDA. Uninstalling..."
    sudo snap remove --purge docker
fi

echo "Installing Docker via apt so it works with CUDA..."
sudo apt remove containerd.io docker-compose-plugin -y
sudo apt install docker.io docker-compose-v2 -y

# build and run the docker system
cd "$SCRIPT_DIR"
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up