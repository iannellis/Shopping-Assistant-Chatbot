# ShopTalk — Multimodal Shopping Assistant

> Streamlit UI + LangGraph agent orchestrating an Ollama LLM, a BLIP‑2 multimodal embedding service, and a Chroma vector DB — all tied together with Docker.

**Note:** The asset files necessary to run the agent are no longer available online. However, all the code required to re-create them is available in this repository.

## What is this?
ShopTalk is a multimodal shopping assistant chatbot that performs retrieval augmented generation (RAG) on products in the Amazon Berkeley Objects (ABO) dataset. It lets a user search for products using **text**, an **image**, or both. Under the hood it:
- embeds user queries with a **BLIP‑2** model (fine‑tuned options included),
- retrieves candidate products from **ChromaDB** using multimodal embeddings,
- has a conversational **LLM (Ollama: Llama‑3.1‑8B‑Instruct)** summarize/clarify what the user wants and decide when to call tools, and
- returns the top matching products (images + titles + descriptions), while maintaining **chat memory** across threads.

This implementation focuses on small-ish models that fit within a **single 12 GB GPU** (e.g., RTX 4070 Ti), with clean Dockerized components so you can spin it up quickly on a GPU machine (local or cloud).

---

## Architecture at a glance
Five containers work together:

1. **UI (Streamlit)** — `code/UI/ui2.py`  
   Presents chat, image upload, and feedback controls on **port `${UI_PORT}`** (default 8501).

2. **Agent (FastAPI + LangGraph)** — `code/Agent/app.py`, `code/Agent/LangGraph_agent.py`  
   Orchestrates the conversation, decides when to use tools, calls retrieval, and streams responses on **port `${AGENT_PORT}`**.

3. **BLIP‑2 Embedding Service (FastAPI)** — `code/Blip-2_API_endpoint/app.py`  
   Loads processors and a BLIP‑2 model (pretrain/COCO or fine‑tuned **GS/ABO** variants) and serves **multimodal embeddings** on **port `${BLIP_2_PORT}`**.

4. **ChromaDB** — persistent vector store containing **image–item pairs** and metadata.  
   Backed by files in `${CHROMA_DIR_LOCAL}` mapped into the container.

5. **Ollama** — serves **Llama‑3.1‑8B‑Instruct** (configurable) on **port 11434** with GPU acceleration.

All components are described in `docker/docker-compose.yml` and built with `docker/*.dockerfile`. Environment is governed by `docker/.env`.

---

## How it works (high‑level)
1. **User prompt (text and/or image) → UI**  
   The UI posts to the Agent. One image is allowed per thread and is consumed on the next prompt.

2. **Agent (LangGraph) decides**  
   The agent uses a **system prompt** to force a call to the `retrieve_products` tool when it detects a product query. It also supports general chat (no tool call) when appropriate.

3. **Retrieve products → Chroma**  
   - If the user supplied an image, the BLIP‑2 endpoint builds a **multimodal embedding** (text+image).  
   - Otherwise it builds **text‑only** embeddings.  
   - The Agent queries Chroma and filters down to the top **k items** (configurable).

4. **LLM composes the reply**  
   Using the retrieved results (titles, images, descriptions) the LLM writes a concise answer and returns **three** candidate products by default.

5. **Memory**  
   Threads are tracked with **LangGraph’s `MemorySaver`**. The system stores chat text plus uploaded/returned images. *Current limitation:* session isolation is not implemented; multiple UI users would share the same memory store.

---

## Features
- **Multimodal search**: text‑only, image‑only, or unified text+image embedding with BLIP‑2.
- **Fine‑tuned BLIP‑2**: recipes and code for **GS (Marqo Google Shopping)** and **ABO (Amazon Berkeley Objects)** variants (see `code/Blip-2_fine_tune`). Only the **Q‑Former** is fine‑tuned; the LLM portion is not used for embeddings.
- **Data pipeline**: scripts to preprocess large product catalogs and build a Chroma database (`code/Data_pipeline` + `pipeline_config.toml`).
- **Dockerized**: spin up UI, Agent, BLIP‑2, Chroma, and Ollama in one go.
- **Streaming responses** from the Agent API.
- **Feedback API** to store user feedback per thread.

---

## Quick start (Docker, GPU recommended)

> Prereqs: Docker + NVIDIA Container Toolkit; a machine with at least **1× 12 GB NVIDIA GPU**.

1. **Clone/extract the project** onto a GPU host.  
2. **From the repo root**, run one of the bootstrap scripts (these pull assets and then compose up):  
   - `./run_agent_s3.sh` *(preferred; downloads BLOBs from S3)*  
   - `./run_agent_google_drive.sh` *(fallback if S3 assets are unavailable)*
3. Wait ~30 seconds after `docker compose up` for services to be ready.
4. Open the UI: **http://localhost:8501** (or your `${UI_PORT}`).

> If you already have the models and DB locally, you can also run directly:  
> ```bash
> docker compose -f docker/docker-compose.yml up --build
> ```

---

## Configuration (`docker/.env`)
Key variables (non‑exhaustive; see the file for full list):

### BLIP‑2 service
- `BLIP_2_MODEL`, which has existing options of `gs`, `abo`, `pretrain`, or `coco`. It determines the name of the BLIP-2 model file that is downloaded, which takes the form `blip-2-<BLIP_2_MODEL>.pt`, the name of the tarfile containing the Chroma database that is downloaded, which takes the form `chroma_<BLIP_2_MODEL>.tar`, and the name of the multimodal Chroma collection that is loaded, which takes the form `blip_2_<BLIP_2_MODEL>_multimodal`.
- `BLIP_2_DIR_LOCAL`/`BLIP_2_DIR_CONTAINER`, the models directory on the host and in the Docker container, respectively.

### Ollama / LLM
- `OLLAMA_DIR_LOCAL`, which specifies where the Ollama should download its models to. It is mapped to `/root/.ollama` inside the Docker container.
- `OLLAMA_MODEL`, which specifies which LLM from Ollama to use. The format is the same as requried by the `ollama run` command. The model must support tools.

### ChromaDB
- `CHROMA_DIR_LOCAL`, which is where the Chroma database is stored. It is mapped to `/chroma/chroma` inside the Docker container.
- `CHROMA_MAX_IMAGES_PER_ITEM`, the maximum number of images any item in the dataset has.
- `CHROMA_MAX_ITEMS`, the number of items to return to the user.

### Dataset (ABO)

- `ABO_DIR_LOCAL`/`ABO_DIR_CONTAINER`, where the ABO dataset is located on the host and in the Docker container, respectively. It must contain the following:
    - `ABO_LISTINGS_FILE` containing the processed metadata for all the items in the dataset.
    - `images/small` containing all the images from the dataset.
    - `images/metadata` containing `images.csv` mapping the image_ids in the metadata in `ABO_LISTINGS_FILE` to the images in `images/small`.

### Ports
- `*_PORT`, of which there are five: one for each container. Only `BLIP_2_PORT` and `AGENT_PORT` may be changed without further code changes. The rest are set to the defaults of their contained software.

**Note:** Upon changing `BLIP_2_MODEL`, a new database needs to be extracted, and `run_agent_s3.sh` has no way to detect that it must be done. Therefore, the simplest way to fix this problem is to delete the directory at `CHROMA_DIR_LOCAL` before running the script again.

---

## Using the UI
- Start a new chat (left sidebar) and **optionally upload an image**.  
- Type what you’re shopping for; the system will **assume a product search** unless your message clearly isn’t one.  
- The agent returns **three** results with images, titles, and descriptions.  
- Uploaded images are **used once** (on the next prompt). Continued chat uses **text‑only** retrieval.

---

## Data pipeline (building your own Chroma DB)
Run the pipeline from the repo root:
```bash
./run_pipeline.sh
```

Configure via `pipeline_config.toml`:

- `working_dir` — where intermediate pickle files are written  
- `abo_dataset_dir` — dataset root with expected structure:  
  - `listings/metadata/*.json[.gz]` — product metadata  
  - `images/small/` — images referenced by metadata  
  - `images/metadata/images.csv[.gz]` — image ID ↔ file mapping
- `models_dir` — directory containing BLIP‑2 checkpoints named `blip-2-<model_selection>.pt`  
- `model_selection` — `gs | abo | pretrain | coco`  
- `embeddings_dir` — temporary embedding batches  
- `chroma_dir` — final Chroma database destination

Pipeline stages (see `code/Data_pipeline`):
1. **english_tags_1.py** — load metadata, keep English‑tagged items.  
2. **product_type_spaces_2.py** — clean/normalize product types.  
3. **english_check_3.py** — local and optional Google Cloud language detection.  
4. **product_type_verification_4.py** — category consistency checks with checkpoints.  
5. **embed_to_chroma_5.py** — batch BLIP‑2 embeddings → Chroma DB.

> BLIP‑2 embedding is run on **Python 3.11** (see the script and Dockerfile).

---

## Fine‑tuning BLIP‑2 (optional)
Code lives in `code/Blip-2_fine_tune` (Linux + PyTorch with **nccl** backend; Python 3.11). Two entrypoints:
- **`run_gs.py`** — Marqo Google Shopping dataset (expects `images/` and `marqo-gs-dataset/marqo_gs_full_10m/{query_0_product_id_0.csv, query_1_product_id_1.csv}` under `marqo_gs_data_dir`).  
- **`run_abo.py`** — Amazon Berkeley Objects dataset.

Only the **Q‑Former** is fine‑tuned; we use the **image encoder + Q‑Former** to emit embeddings. Checkpoints and training loss are saved under `save_dir` every N batches; validation logs loss only.

---

## API surfaces
### Agent (FastAPI)
- `GET /api/v1/` — health check
- `POST /api/v1/prompt` — stream responses for a prompt (text/image base64)
- `GET /api/v1/threads` — list thread IDs
- `GET /api/v1/thread/{thread_id}` — get conversation + uploaded image
- `PUT /api/v1/feedback/{thread_id}` — store feedback for an agent response
- `GET /api/v1/feedback/{thread_id}` — retrieve stored feedback

### BLIP‑2 (FastAPI)
- `POST /api/v1/embed` — JSON body with optional `image_b64` and `text`; returns a **flattened multimodal embedding**

---

## Repo layout
```
Shopping-Assistant-Chatbot-main/
├─ assets/                     # processors + wheels + example assets
├─ code/
│  ├─ Agent/                   # FastAPI + LangGraph agent
│  ├─ Blip-2_API_endpoint/     # FastAPI service for BLIP‑2 embeddings
│  ├─ Blip-2_embeddings/       # embedding utilities
│  ├─ Blip-2_fine_tune/        # fine‑tuning entrypoints + trainers
│  ├─ Data_pipeline/           # ETL to Chroma
│  ├─ Llama_data_utils/        # helpers for LLM data/formatting
│  ├─ UI/                      # Streamlit app
│  └─ notebooks/               # helper notebooks (Chroma, LangGraph experiments)
├─ docker/                     # Dockerfiles + docker-compose + .env
├─ python_envs/                # pinned requirements by service (3.11/3.12)
├─ testing/                    # sample images
├─ run_agent_s3.sh             # bootstrap + compose up (S3)
├─ run_agent_google_drive.sh   # bootstrap + compose up (GDrive)
├─ run_pipeline.sh             # build a Chroma DB from datasets
├─ run-perf-test.py            # simple throughput/latency test
└─ Documentation.pdf           # full technical documentation
```

---

## Requirements
- **GPU**: 1× NVIDIA GPU (12 GB VRAM recommended).  
- **OS**: Linux/macOS/Windows with Docker; fine‑tuning requires **Linux**.  
- **Python**: Services use 3.12 (Agent/UI) and 3.11 (BLIP‑2) as pinned in `docker/*.dockerfile` and `python_envs/*`.  
- **Disk**: Enough space for datasets, embeddings, and the Chroma DB.

---

## Troubleshooting
- **Model switch doesn’t “take”**: When changing `BLIP_2_MODEL`, **delete** `${CHROMA_DIR_LOCAL}` and download the DB for the new model before restarting.
- **Agent/UI 404s on startup**: The UI waits for the Agent to be reachable. Give it a few seconds after `docker compose up`.
- **No results**: Verify your Chroma DB exists at `${CHROMA_DIR_LOCAL}` and that `CHROMA_MAX_ITEMS` > 0.
- **GPU not visible**: Confirm the host has NVIDIA Container Toolkit and your compose stack requests a GPU device.
- **Single image per thread**: This is by design in this release; the uploaded image is used once for retrieval.

---

## License
See `LICENSE`.

---

## Credits
Built by **BRI ShopTalk** (Brahmeswara Yerrabolu, Rajat Sharma, Ian Ellis).  
BLIP‑2 via **Salesforce LAVIS**; LLM served by **Ollama**; vector store via **ChromaDB**; orchestration via **LangGraph**; UI by **Streamlit**.
