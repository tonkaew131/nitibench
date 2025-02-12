# NitiBench

This repository hosts the evaluation script for the proposed benchmark in the paper:  
**NitiBench: A Comprehensive Study of LLM Frameworks’ Capabilities for Thai Legal Question Answering**.  

It contains two main scripts:  
1. **Generating responses** using the setup proposed in the paper.  
2. **Evaluating responses** in both retrieval and end-to-end aspects.  

---

## 📌 Getting Started

### 1️⃣ Clone the Repository  
Clone this repository to your local machine:

```bash
git clone [REPO_URL]
cd NitiBench
```

### 2️⃣ Configure API Keys  
Edit the environment settings file (`setting.env`) to store all your API keys.  
An example configuration is provided in `setting.env.example`.

### 3️⃣ Build and Run the Docker Container  
Use the following command to build the Docker image and create a container:

```bash
docker build -t nitibench . & 
docker run -dit --rm --network=host --gpus all --shm-size=10gb --name nitibench-container nitibench bash
```

### 4️⃣ Expected File Structure  
Once inside the container, the file structure should look like this:

```plaintext
app/
|---LRG/
|   |---[packages]
|---test_data/
|   |---hf_tax.csv
|   |---hf_wcx.csv
|   |---lclm_sample.csv
|   |---hf_tax_reduced_section.csv
|   |---hf_wcx_reduced_section.csv
|---llama_index/
```

- `hf_tax.csv` & `hf_wcx.csv` → Tax Case and WCX-CCL datasets.  
- `hf_tax_reduced_section.csv` & `hf_wcx_reduced_section.csv` → Reduced versions containing only queries that use sections within naive chunking strategy.  
- `lclm_sample.csv` → A 20% stratified sample of the WCX-CCL dataset.  

---

## 🚀 Using the Benchmark

### 1️⃣ Generating Responses  
To generate responses, use the configuration files inside:  
📂 `/app/LRG/config/all_e2e_config/`  

Run the following command:  

```bash
python script/response_e2e.py --config_path=[PATH_TO_YOUR_CONFIG]
```

- You can adjust the config file to match your preferences.  
- The generated responses will be saved as:  
  - `tax_response.json`  
  - `wcx_response.json`  

---

### 2️⃣ Evaluating Responses  
To evaluate the responses, create a config file inside:  
📂 `/app/LRG/config/all_e2e_metric_config/`  

Run the evaluation script:

```bash
python script/metric_e2e.py --config_path=[PATH_TO_YOUR_CONFIG]
```

The evaluation results will be saved in:  
- **Per-query metrics:**  
  - `tax_e2e_metrics.json`  
  - `wcx_e2e_metrics.json`  
- **Global metrics:**  
  - `tax_global_metrics.json`  
  - `wcx_global_metrics.json` 