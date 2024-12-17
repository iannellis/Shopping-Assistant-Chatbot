#!/bin/bash

# Has variables we need
source docker/.env

declare -A MODELS

# Array of BLIP-2 models mapping to their file names and Google Drive IDs
MODELS["pretrain"]=("blip-2-pretrain.pt" "1xLfjTUf4MuBl1FeDpmClt04etF-PaZCZ")
MODELS["gs"]=("blip-2-gs-trained-1epoch.pt", "1jsuiImeloqeQN99gULJbCZCHL8Q8mR92")
MODELS["abo"]=("blip-2-abo-trained-2epochs.pt", "1kNkkk2Q6922a9oXQUol19hg16z_4JuE5")

# ABO metadata Pandas dataframe file name and Google Drive ID
ABO_METADATA_DF=("abo-listings-final-draft.pkl", "1hChAT7PL_3c9YQugQJFFOAElbRPV7yqg")

# Ensure the selected model is valid
if [ -z "${MODELS[$BLIP_2_MODEL]}" ]; then
    echo "Error: BLIP-2 Model '$BLIP_2_MODEL' is not defined. Please check your BLIP_2_MODEL in the .evn file."
    exit 1
fi

# Check if the 'models' directory exists; create it if not
if [ ! -d "$MODEL_DIR_LOCAL" ]; then
    echo "Directory '$MODEL_DIR_LOCAL' does not exist. Creating it..."
    mkdir "$MODEL_DIR_LOCAL"
else
    echo "Directory '$MODEL_DIR_LOCAL' already exists."
fi

# Check if the 'Hugging Face' directory exists; create it if not
if [ ! -d "$HF_LOCAL" ]; then
    echo "Directory '$HF_LOCAL' does not exist. Creating it..."
    mkdir "$HF_LOCAL"
else
    echo "Directory '$HF_LOCAL' already exists."
fi

# Check if the 'ABO dataset' directory exists; create it if not
if [ ! -d "$ABO_DIR_LOCAL" ]; then
    echo "Directory '$ABO_DIR_LOCAL' does not exist. Creating it..."
    mkdir "$ABO_DIR_LOCAL"
else
    echo "Directory '$ABO_DIR_LOCAL' already exists."
fi

# Change to the 'models' directory
cd "$MODEL_DIR_LOCAL"

# For downloading models from Google Drive
pip3 install gdown

# For progress bars
apt install pv

# Get the filename and ID for the chosen model
FILE_NAME="${MODELS[$BLIP_2_MODEL][0]}"
FILE_ID="${MODELS[$BLIP_2_MODEL][1]}"

# Check and download the file if it doesn't exist
if [ ! -f "$FILE_NAME" ]; then
    echo "File '$FILE_NAME' for model '$BLIP_2_MODEL' not found. Downloading..."
    gdown "$FILE_ID" -O "$FILE_NAME"
    if [ $? -eq 0 ]; then
        echo "File '$FILE_NAME' for model '$BLIP_2_MODEL' downloaded successfully."
    else
        echo "Error downloading file '$FILE_NAME' for model '$BLIP_2_MODEL'."
    fi
else
    echo "File '$FILE_NAME' for model '$BLIP_2_MODEL' already exists. Skipping download."
fi

# Get the ABO dataset
cd ${ABO_DIR_LOCAL}
FILE_NAME="abo-images-small.tar"
if [ ! -f "./$FILE_NAME" ]; then
    echo "File '$FILE_NAME' not found. Downloading..."
    aws s3 cp s3://amazon-berkeley-objects/archives/$FILE_NAME ./$FILE_NAME
    if [ $? -eq 0 ]; then
        echo "File '$FILE_NAME' downloaded successfully."
    else
        echo "Error downloading file '$FILE_NAME' for model '$FILE_NAME'."
    fi
else
    echo "File '$FILE_NAME' already exists. Skipping download."
fi

if [ ! -d "images" ]; then
    echo "Extracting abo-images-small.tar..."
    pv "$FILE_NAME" | tar -xf-
fi

if [ -f "images/metadata/listings/images.csv.gz" ] && [ ! -f "images/metadata/listings/images.csv.gz" ]; then
    echo "Extracting image metadata..."
    gunzip "images/metadata/listings/images.csv.gz"
fi

# The processed metadata
FILE_NAME="${ABO_METADATA_DF[0]}"
FILE_ID="${ABO_METADATA_DF[1]}"

if [ ! -f "$FILE_NAME"]; then
    echo "File '$FILE_NAME' not found. Downloading..."
    gdown "$FILE_ID" -O "$FILE_NAME"
    if [ $? -eq 0 ]; then
        echo "File '$FILE_NAME' downloaded successfully."
    else
        echo "Error downloading file '$FILE_NAME'."
    fi
else
    echo "File '$FILE_NAME' already exists. Skipping download."
fi

# build and run the docker system
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up