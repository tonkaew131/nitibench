# NitiBench

[[Technical Report](https://arxiv.org/pdf/2502.10868)] | [[🤗 Hugging Face Dataset](https://huggingface.co/datasets/VISAI-AI/nitibench)]

This repository hosts the evaluation script for the proposed benchmark in the paper:  
[**NitiBench: A Comprehensive Study of LLM Frameworks’ Capabilities for Thai Legal Question Answering**](https://arxiv.org/pdf/2502.10868)

It contains two main scripts:  

1. **Generating responses** using the setup proposed in the paper.  
2. **Evaluating responses** in both retrieval and end-to-end aspects.  

---

## 📌 Getting Started

### 1️⃣ Clone the Repository

Clone this repository to your local machine:

```bash
git clone [REPO_URL]
cd nitiBench
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

When the image is created, the script `setup_data.py` will be executed to pull the data from [HuggingFace](https://huggingface.co/datasets/VISAI-AI/nitibench), preprocess and store in `/app/test_data`

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

## Models

|Model Name|URL|
|Human-Finetuned BGE-M3|[🤗 HuggingFace Model](https://huggingface.co/VISAI-AI/nitibench-ccl-human-finetuned-bge-m3)|
|Auto-Finetuned BGE-M3|[🤗 HuggingFace Model](https://huggingface.co/VISAI-AI/nitibench-ccl-auto-finetuned-bge-m3)|

---

## Acknowledgement  

We would like to express our sincere gratitude to **Supavich Punchun** for facilitating WCX-CCL data preparation, and to **Apiwat Sukthawornpradit, Watcharit Boonying, and Tawan Tantakull** for scraping, preprocessing, and preparing the **Tax Case Dataset**. We also thank all **VISAI.AI** company members for assisting in quality control for **LLM-as-a-judge** metric validation.  

We are deeply thankful to the **legal expert annotators** for their meticulous work in annotating samples, which was essential for validating the **LLM-as-a-judge** metrics.  

Special thanks to **Prof. Keerakiat Pratai** (Faculty of Law, Thammasat University) for insightful consultations on Thai legal information and background knowledge, which significantly enriched our research.  

We sincerely thank **PTT, SCB, and SCBX**, the main sponsors of the **WangchanX project**, for their generous support. Their contributions have been instrumental in advancing research on Thai legal AI.  

Next, we extend our appreciation to the **research assistants at VISTEC** for their valuable guidance in constructing benchmarks for LLM systems, particularly in retrieval and end-to-end (E2E) metrics.  

Lastly, if you use our code in your research, please cite our work:  

```bibtex
@misc{akarajaradwong2025nitibenchcomprehensivestudiesllm,
      title={NitiBench: A Comprehensive Studies of LLM Frameworks Capabilities for Thai Legal Question Answering}, 
      author={Pawitsapak Akarajaradwong and Pirat Pothavorn and Chompakorn Chaksangchaichot and Panuthep Tasawong and Thitiwat Nopparatbundit and Sarana Nutanong},
      year={2025},
      eprint={2502.10868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.10868}, 
}
```

## Contribution  

We welcome contributions from the community! Whether it's **bug fixes, feature additions, or documentation improvements**, your input is valuable.  

### How to Contribute  

1. **Fork** the repository  
2. **Create your feature branch**  

   ```bash
   git checkout -b feature/NewFeature
   ```

3. **Commit your changes**  

   ```bash
   git commit -m 'Add some NewFeature'
   ```

4. **Push to the branch**  

   ```bash
   git push origin feature/NewFeature
   ```

5. **Open a Pull Request**  

We look forward to your contributions! 🚀
