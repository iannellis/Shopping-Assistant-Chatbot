## Overview
This implementation of the ShopTalk shopping assistant chat-bot uses a Streamlit front-end and LangGraph to tie together a Llama-3.1-8B-Instruct LLM, a Chroma vector database, and a BLIP-2 embedding model.


## Running ShopTalk
This ShopTalk implementation is designed to be easy to run on an AWS EC2 instance with a video card (tested on a g6.2xlarge instance with 150GB EBS storage) and running an Ubuntu 22.04 image with CUDA (Deep Learning Base OSS Nvidia Driver GPU AMI). The process is designed to be simple:
1. Extract ShopTalk-main.zip into the home directory.
2. Change directories (cd) into ShopTalk-main.
3. Run the `run_agent_s3.sh` script found in the ShopTalk root directory. If the data files are no longer available on s3, then `run_agent_google_drive.sh` may work.
4. Give Docker ~30s to get everything running after `docker compose up` is called.
5. Connect to the UI on port 8501 (default).

The `run_agent_s3.sh` script will set up the directory structure required, download all the additional files (BLOBs) required, build the Docker images, and run them. 

Provided all the elements are in place, ShopTalk can also be run from the root of the ShopTalk directory structure by running `docker compose -f docker/docker-compose.yml build` followed by `docker compose -f docker/docker-compose.yml up`.

### Configuration
Changes to environmental variables that control various aspects of the agent can be found in `docker/.env`. The configuration options are (variables in **bold**):

- **BLIP_2_MODEL**, which has existing options of `gs`, `abo`, `pretrain`, or `coco`. It determines the name of the BLIP-2 model file that is downloaded, which takes the form `blip-2-`**BLIP_2_MODEL**`.pt`, the name of the tarfile containing the Chroma database that is downloaded, which takes the form `chroma_`**BLIP_2_MODEL**`.tar`, and the name of the multimodal Chroma collection that is loaded, which takes the form `blip_2_`**BLIP_2_MODEL**`_multimodal`.
- **BLIP_2_DIR_LOCAL**, which specifies the directory where the models are stored on the local machine, which gets mapped to **BLIP_2_DIR_CONTAINER** inside the Docker container.
- **OLLAMA_DIR_LOCAL**, which specifies where the Ollama should download its models to. It is mapped to `/root/.ollama` inside the Docker container.
- **OLLAMA_MODEL**, which specifies which LLM from Ollama to use. The format is the same as requried by the `ollama run` command. The model must support tools.
- **CHROMA_DIR_LOCAL**, which is where the Chroma database is stored. It is mapped to `/chroma/chroma` inside the Docker container.
- **CHROMA_MAX_IMAGES_PER_ITEM**, the maximum number of images any item in the dataset has.
- **CHROMA_MAX_ITEMS**, the number of items to return to the user.
- **ABO_LISTINGS_FILE** containing the processed metadata for all the items in the dataset. It must be directly inside **ABO_DIR_LOCAL**, described below.
- **ABO_DIR_LOCAL**, where the ABO dataset is located, which is mapped to **ABO_DIR_CONTAINER** in the Docker container. It must have two directories inside.
    - `images/small` containing all the images from the dataset.
    - `images/metadata` containing `images.csv` mapping the image_ids in the metadata in **ABO_LISTINGS_FILE** to the images in the `images/small`.
- ***_PORT**, of which there are five: one for each container. Only **BLIP_2_PORT** and **AGENT_PORT** may be changed without further code changes. The rest are set to the defaults of their contained software.

Note that, upon changing **BLIP_2_MODEL**, a new database needs to be extracted, and `run_agent_s3.sh` has no way to detect that it must be done. Therefore, the simplest way to fix this problem is to delete the directory at **CHROMA_DIR_LOCAL** before running the script again.

## User Interface and Agent
The user interface is a simple Streamlit interface. The first chat thread created by default is *First Chat*. Once the first prompt is sent, the chat will show in the list on the left sidebar. The same process occurs whenever a new chat thread is started by entering a name into the box on the left. The chat thread does not show in the list until the first prompt is sent. As long as a chat thread has at least one prompt, it can be accessed via the list on the left.

The user can upload one image per chat thread. It can be uploaded at any time during the chat thread, but it is only submitted to the agent with the next prompt and never processed again.

The agent accepts prompts for products and general chat prompts, and will respond to both. If the LLM thinks the user is querying about a product, it will then search for it. If an image is provided, the agent is instructed to assume that the user is querying about a product.

The agent returns three results per product query. It returns images with most results (there are a few products in the database without images). The user can reference one of the results that is returned in a future query, but the agent will only process the text description and will not process the image that was returned with it.

## Further Details
Further details, including how to run the data pipeline, fine-tuning, and explanations of notebooks are found in `documentation.docx`