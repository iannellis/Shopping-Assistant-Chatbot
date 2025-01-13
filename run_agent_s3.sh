#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Has variables we need
source $SCRIPT_DIR/docker/.env

declare -A MODEL_FILES MODEL_S3_OBJS CHROMA_FILES CHROMA_S3_OBJS

S3_BUCKET="bri-shoptalk"

# Array of BLIP-2 models mapping to their file names and S3 objects
MODEL_FILES["pretrain"]="blip-2-pretrain.pt"
MODEL_FILES["gs"]="blip-2-gs.pt"
MODEL_FILES["abo"]="blip-2-abo.pt"
MODEL_FILES["coco"]="blip-2-coco.pt"
MODEL_S3_OBJS["pretrain"]="model_files/blip-2-pretrain.pt"
MODEL_S3_OBJS["gs"]="model_files/blip-2-gs.pt"
MODEL_S3_OBJS["abo"]="model_files/blip-2-abo.pt"
MODEL_S3_OBJS["coco"]="model_files/blip-2-coco.pt"

CHROMA_FILES["pretrain"]="chroma_pretrain.tar"
CHROMA_FILES["gs"]="chroma_gs.tar"
CHROMA_FILES["abo"]="chroma_abo.tar"
CHROMA_FILES["coco"]="chroma_coco.tar"
CHROMA_S3_OBJS["pretrain"]="chroma/chroma_pretrain.tar"
CHROMA_S3_OBJS["gs"]="chroma/chroma_gs.tar"
CHROMA_S3_OBJS["abo"]="chroma/chroma_abo.tar"
CHROMA_S3_OBJS["coco"]="chroma/chroma_coco.tar"

# ABO metadata Pandas dataframe file name and Google Drive ID
ABO_METADATA_DF=("abo-listings-final-draft.pkl" "abo-listings-final-draft.pkl")

# ABO images
ABO_IMAGES=("abo-images-small.tar" "abo-images-small.tar")

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

# For progress bars
sudo apt install pv -y

# Check and download a file if it doesn't exist
s3download_file() {
    local file_name=$1
    local s3_obj=$2

    if [ ! -f "$file_name" ]; then
        echo "File '$file_name' not found. Downloading..."
        aws s3 cp s3://"$S3_BUCKET"/"$s3_obj" "$file_name" --no-sign-request
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
s3download_file "${MODEL_FILES[$BLIP_2_MODEL]}" "${MODEL_S3_OBJS[$BLIP_2_MODEL]}"

# Download the Chroma database and extract it
cd "$CHROMA_DIR_LOCAL"
s3download_file "${CHROMA_FILES[$BLIP_2_MODEL]}" "${MODEL_S3_OBJS[$BLIP_2_MODEL]}"

if [ ! -f "chroma.sqlite3" ]; then
    echo "Extracting Chroma database..."
    tar -xf "${CHROMA_FILES[$BLIP_2_MODEL]}"
fi

# Get the ABO dataset processed metadata
cd "$ABO_DIR_LOCAL"
s3download_file "${ABO_METADATA_DF[0]}" "${ABO_METADATA_DF[1]}"

# Get the ABO dataset images and extract them
s3download_file "${ABO_IMAGES[0]}" "${ABO_IMAGES[1]}"

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
echo "Downloading and building Docker images..."
docker compose -f docker/docker-compose.yml build
echo "Starting Docker..."
docker compose -f docker/docker-compose.yml up