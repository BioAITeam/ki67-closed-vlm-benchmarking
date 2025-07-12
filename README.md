# Benchmarking of Closed Vision-Language Models for Ki-67 Index Prediction in Breast Cancer Histopathology Images

This repository provides an end-to-end pipeline for a comparative study of leading closed-source Vision-Language Models (VLMs)—including OpenAI (GPT-4.5, GPT-4.1 mini, GPT-4.1, GPT-4o), Google (Gemini 1.5 Pro, Gemini 1.5 Flash), xAI (Grok-2 Vision), and Anthropic (Claude 3.5 Sonnet)—for automated Ki-67 index estimation from breast cancer histopathology images using the BCData and SHIDC-B-Ki-67 datasets.

---

## Project Structure

The project follows a step-by-step workflow, organized into logical directories:

- `1.data_access/`: Scripts and documentation for accessing and understanding the BCData and SHIDC-B-Ki-67 datasets.
- `2.preprocess/`: Tools for image conversion and annotation extraction.
- `3.vlm_processing/`: Core scripts for VLM inference and results extraction (for each model and provider).
- `4.utils/`: Utility scripts for metrics, validation, and results analysis.
- `5.results/`: Directory for outputs, including CSVs, logs, and visualizations.

---

## 0. Getting Started: Environment Setup

Before running the project, it's recommended to set up a dedicated Python virtual environment to manage dependencies and avoid conflicts.

### 0.1 Create and Activate Virtual Environment

1. **Navigate to the project root directory** in your terminal or command prompt.

2. **Create the virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **On Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

   You should see `(venv)` prepended to your command prompt, indicating that the virtual environment is active.

### 0.2 Configure your environment variables

1. **Create your personal `.env` file**

   Windows (PowerShell / Git Bash):
   ```bash
   cp .env.example .env
   ```

   Windows CMD:
   ```bash
   copy .env.example .env
   ```

   macOS / Linux:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys for OpenAI, Google, xAI and Anthropic**
   ```dotenv
   # .env
   OPENAI_API_KEY="sk-..."
   GOOGLE_API_KEY="AIza..."
   XAI_API_KEY="xai-..."
   ANTHROPIC_API_KEY="sk-ant-api03-..."
   ```

### 0.3 Install Dependencies

With your virtual environment activated, install all necessary project dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 1. Data Access

To begin, download the **BCData** and **SHIDC-B-Ki-67** datasets. The `1.data_access/` directory includes all the necessary information and scripts to help you understand the data structure, download the datasets, and view sample data.

Please refer to the instructions provided in this folder to obtain and organize the raw data.

---

## 2. Data Preprocessing

This repository contains two slightly different pipelines, one for **BCData** (images + HDF5 annotations) and another for **SHIDC-B-Ki-67** (images + JSON annotations).  
All scripts live inside **`2.preprocess/`**.

---

### 2.1 BCData

The raw BCData archive is organised as follows:

```text
1.images/            # histopathology images
2.annotations/
├─ positive/         # *.h5 files with positive cells
└─ negative/         # *.h5 files with negative cells
```

#### 2.1.1 Convert images to JPG

`1.convert_images.py` converts every image to JPG and writes the result to the chosen output folder.

Structure:
```bash
python 2.preprocess/1.convert_images.py <images_src> <processed_dataset>
```

Example:
```bash
python 2.preprocess/1.convert_images.py 1.data_access/data_sample_BCData/1.images/test 1.data_access/data_sample_BCData/3.data_processed
```

#### 2.1.2 Generate JSON annotations

`2.generate_json.py` reads the positive/negative *.h5* files and merges them into a single JSON file per image.
A typical output looks like:

```json
[
    {
        "x": 487,
        "y": 27,
        "label_id": 1
    },
    {
        "x": 426,
        "y": 42,
        "label_id": 1
    },
]
```

Structure:
```bash
python 2.preprocess/2.generate_json.py <positive_dir> <negative_dir> <processed_dataset>
```

Example:
```bash
python 2.preprocess/2.generate_json.py 1.data_access/data_sample_BCData/2.annotations/test/positive 1.data_access/data_sample_BCData/2.annotations/test/negative 1.data_access/data_sample_BCData/3.data_processed
```

After both scripts have run, the **`<processed_dataset>`** folder contains every image in JPG format and its matching JSON file, ready for evaluation.

---

### 2.2 SHIDC-B-Ki-67

The SHIDC-B-Ki-67 dataset have images already in JPG and one JSON per image.
The only required step is to **drop ambiguous cells whose `label_id` == 3**.
While doing so, copy the original image so that the cleaned JSON and its picture stay together.

#### 2.2.1 Remove `label_id == 3` and copy images

Use `3.delete_label_3.py`:

Structure:
```bash
python 2.preprocess/3.delete_label_3.py <annotations_src> <processed_dataset>
```

Example:
```bash
python 2.preprocess/3.delete_label_3.py 1.data_access/data_sample_SHIDC-B-Ki-67/1.bare_images/Test 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed
```

* **`<annotations_src>`** — folder that already contains paired `*.json` + `*.jpg`.
* **`<processed_dataset>`** — destination where the script will write the **filtered JSON** and **copy the matching image**.

The resulting directory preserves the original filenames but **excludes** every object whose `label_id` is 3, keeping only positive (`1`) and negative (`2`) label cells.

> **No other preprocessing is needed** for SHIDC-B-Ki-67; image conversion and HDF5 handling are exclusive to BCData.

---

## 3. VLM Processing and Evaluation

The `3.vlm_processing/` directory contains the core logic for evaluating the VLMs. In this stage, the models are used to calculate the Ki-67 proliferation index from the processed images. These predictions are then compared against the actual Ki-67 values extracted from the JSON annotation files generated in the data preprocessing step.

**This process generates several important outputs:**
- A **CSV file** that provides a detailed breakdown of the evaluation, including the predicted Ki-67 value and the actual Ki-67 value for each image (`image,predicted,true`).
- A **log file** that mirrors the structure of the CSV.
- An **`llm_responses` file** which stores the complete, raw responses received directly from the VLM.
- A **results graph** that visually compares the model's predictions against the actual values.

For prompting the VLM, `.txt` prompt files are included in this directory.

---

### Main VLM scripts

**OpenAI**
```bash
python 3.vlm_processing/1_1.main_openai.py <processed_dataset> [<output_parent_dir>]
```
Example BCData:
```bash
python 3.vlm_processing/1_1.main_openai.py 1.data_access/data_sample_BCData/3.data_processed 5.results
```
Example SHIDC-B-Ki-67:
```bash
python 3.vlm_processing/1_1.main_openai.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results
```

**Google**
```bash
python 3.vlm_processing/1_2.main_google.py <processed_dataset> [<output_parent_dir>]
```
Example BCData:
```bash
python 3.vlm_processing/1_2.main_google.py 1.data_access/data_sample_BCData/3.data_processed 5.results
```
Example SHIDC-B-Ki-67:
```bash
python 3.vlm_processing/1_2.main_google.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results
```

**xAI**
```bash
python 3.vlm_processing/1_3.main_xai.py <processed_dataset> [<output_parent_dir>]
```
Example BCData:
```bash
python 3.vlm_processing/1_3.main_xai.py 1.data_access/data_sample_BCData/3.data_processed 5.results
```
Example SHIDC-B-Ki-67:
```bash
python 3.vlm_processing/1_3.main_xai.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results
```

**Anthropic**
```bash
python 3.vlm_processing/1_4.main_anthropic.py <processed_dataset> [<output_parent_dir>]
```
Example BCData:
```bash
python 3.vlm_processing/1_4.main_anthropic.py 1.data_access/data_sample_BCData/3.data_processed 5.results
```
Example SHIDC-B-Ki-67:
```bash
python 3.vlm_processing/1_4.main_anthropic.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results
```

---

Resume unfinished run (process only the remaining images):

Example
```bash
python 3.vlm_processing/1_1.main_openai.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed --resume 5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67 
```

---

### Run the VLM processing for a **single image**

```bash
python 3.vlm_processing/2.ki67_single_image.py <image_path> [--model <model_id>]
```

Replace `<image_path>` with the path to your `.jpg`/`.jpeg`/`.png` file.
If `--model` is omitted, the script defaults to **`gpt-4.1-mini-2025-04-14`** (OpenAI).

---

#### BCData

OpenAI

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4o-2024-11-20
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.1-2025-04-14
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.1-mini-2025-04-14
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gpt-4.5-preview
```

Gemini

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gemini-1.5-pro
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model gemini-1.5-flash
```

xAI
```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model grok-2-vision-latest
``` 

Anthropic
```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_BCData/3.data_processed/8.jpg --model claude-3-5-sonnet-20240620
``` 

#### SHIDC-B-Ki-67

OpenAI

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model gpt-4o-2024-11-20
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model gpt-4.1-2025-04-14
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model gpt-4.1-mini-2025-04-14
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model gpt-4.5-preview
```

Gemini

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model gemini-1.5-pro
```

```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model gemini-1.5-flash
```

xAI
```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model grok-2-vision-latest
``` 

Anthropic
```bash
python 3.vlm_processing/2.ki67_single_image.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed/p1_0317_1.jpg --model claude-3-5-sonnet-20240620
``` 

## 4. Utilities

The `utils/` directory houses a collection of auxiliary scripts designed to support various tasks related to data processing, results analysis, and validation. These scripts provide functionalities that complement the main project workflow.

Here's a breakdown of the available utility scripts:

- ### `calculate_ki_from_json.py`

  This script processes a single JSON annotation file (corresponding to a specific case) and calculates the Ki-67 index. It also returns the counts of immunopositive and immunonegative cells.

  **Usage:**

  structure  
  ```bash
  python 4.utils/calculate_ki_from_json.py <json_path>
  ```

  example  
  ```bash
  python 4.utils/calculate_ki_from_json.py 1.data_access/data_sample_BCData/3.data_processed/8.json
  ```

- ### `calculate_metrics.py`

  This utility calculates key evaluation metrics (R-squared, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE)) based on the model's results recorded in a CSV file.

  **Output (example):**

  ```
  Metrics
  R²   : val1
  MSE  : val2
  RMSE : val3
  MAE  : val4
  ```

  **Usage:**

  structure  
  ```bash
  python 4.utils/calculate_metrics.py <results.csv>
  ```

  example  
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.5/BCData/ki67_results.csv
  ```

  Example BCData:
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.5/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.1-2025-04-14/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4o/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/gemini1.5pro/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/gemini1.5flash/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/grok2vision/BCData/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/claude-3-5-sonnet/BCData/ki67_results.csv
  ```

  Example SHIDC-B-Ki-67:

  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/4o/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv
  ```
  ```bash
  python 4.utils/calculate_metrics.py 5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv
  ```

- ### `calculate_time_average.py`

  Benchmarks **execution time** and **token usage** for any supported Vision-Language Model (VLM) on a subset – or on all – of your processed images.

  **Usage**

  ```bash
  python 4.utils/calculate_time_average.py <processed_dataset> <output_parent_dir> [--model <model_id>] [--n <int|all>]
  ```

  *Arguments*

  * `<processed_dataset>` – folder containing the processed `.jpg` images **and** their matching `.json` annotations (from **Step 2**).
  * `<output_parent_dir>` – folder where you want the benchmark results to be written.
  * `--model` – VLM identifier; the prefix decides the provider. Defaults to **`gpt-4.1-mini-2025-04-14`**.
  * `--n` – number of images to evaluate.

    * **positive integer** → sample that many.
    * **`all`** or **`-1`** → use *every* image.
    * **negative value** (e.g. `-2`) → “all minus |n|”.
    * Default is **10**.

  **Example invocations BCData**

  *OpenAI*
  ```bash  
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_BCData/3.data_processed 5.results --model gpt-4.5-preview --n 10
  ```  
  ```bash  
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_BCData/3.data_processed 5.results --model gpt-4.1-mini-2025-04-14 --n all   # default model
  ```  
  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_BCData/3.data_processed 5.results --model gpt-4.1-2025-04-14 --n all
  ```    
  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_BCData/3.data_processed 5.results --model gpt-4o-2024-11-20 --n all
  ```
  
  *Google Gemini*
  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_BCData/3.data_processed 5.results --model gemini-1.5-pro --n all
  ```
  ```bash  
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_BCData/3.data_processed 5.results --model gemini-1.5-flash --n all
  ```

  *xAI Grok*

  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model grok-2-vision-latest --n all
  ```

  *Anthropic Claude*

  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model claude-3-5-sonnet-20240620 --n -1
  ```

  **Example invocations SHIDC-B-Ki-67**

  *OpenAI*
  ```bash  
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.5-preview --n 10
  ```  
  ```bash  
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.1-mini-2025-04-14 --n all   # default model
  ```  
  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4.1-2025-04-14 --n all
  ```    
  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gpt-4o-2024-11-20 --n all
  ```
  
  *Google Gemini*
  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gemini-1.5-pro --n all
  ```
  ```bash  
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model gemini-1.5-flash --n all
  ```

  *xAI Grok*

  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model grok-2-vision-latest --n all
  ```

  *Anthropic Claude*

  ```bash
  python 4.utils/calculate_time_average.py 1.data_access/data_sample_SHIDC-B-Ki-67/2.data_processed 5.results --model claude-3-5-sonnet-20240620 --n -1
  ```

  **Outputs**

  A new, timestamped folder is created inside `<output_parent_dir>` containing:

  * `time_stats.csv` – per-image timings and token counts
  * `raw_responses.txt` – full, raw model outputs

  When the run finishes the script prints a console summary with the average time and average prompt/completion/total token counts across the sample.

- ### `check_duplicates_in_csv.py`

  This script verifies the integrity of a CSV file by checking for any duplicate entries within it.

  **Usage:**

  structure  
  ```bash
  python 4.utils/check_duplicates_in_csv.py <results.csv>
  ```

  example  
  ```bash
  python 4.utils/check_duplicates_in_csv.py 5.results/4.5/bcdata/ki67_results.csv
  ```

- ### `check_duplicates_in_txt.py`

  This script verifies the integrity of a TXT file by checking for any duplicate entries within it.

  **Usage:**

  structure  
  ```bash
  python 4.utils/check_duplicates_in_txt.py <ki67_log.txt>
  ```

  example  
  ```bash
  python 4.utils/check_duplicates_in_txt.py 5.results/4.5/BCData/ki67_log.txt
  ```

- ### `check_range_in_csv.py`

  This utility ensures data completeness within the results CSV. It checks a specified range of image IDs (e.g., from 0 to 401) to confirm that no images are missing their predicted and real values in the CSV.

  **Usage:**

  structure  
  ```bash
  python 4.utils/check_range_in_csv.py <results.csv> <start_id> <end_id>
  ```

  example  
  ```bash
  python 4.utils/check_range_in_csv.py 5.results/4.5/bcdata/ki67_results.csv 0 25
  ```

- ### `compare_txt_vs_csv.py`

  This script compares the raw `llm_responses.txt` (which holds all model outputs) against the `ki67_results.csv` to identify any instances where the model provided a response that was not successfully logged into the CSV file.

  **Usage:**

  structure  
  ```bash
  python 4.utils/compare_txt_vs_csv.py <results.csv> <llm_responses.txt>
  ```

  example  
  ```bash
  python 4.utils/compare_txt_vs_csv.py 5.results/4.5/bcdata/ki67_results.csv 5.results/4.5/bcdata/llm_responses.txt
  ```

- ### `count_jsons.py`

  A simple utility to count the total number of JSON files present within a given directory.

  **Usage:**
  structure  
  ```bash
  python 4.utils/count_jsons.py <folder_with_jsons>
  ```

  example  
  ```bash
  python 4.utils/count_jsons.py 1.data_access/data_sample_BCData/3.data_processed
  ```

- ### `evaluate_ki67_classification.py`

  Classification Metrics Based on Cut-Off Ranges

  BCData
  ```bash
  python 4.utils/evaluate_ki67_classification.py  --gt 1.data_access/data_full_BCData_summary.xlsx  --pred   5.results/4.5/BCData/ki67_results.csv   5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv   5.results/4.1-2025-04-14/BCData/ki67_results.csv   5.results/4o/BCData/ki67_results.csv   5.results/gemini1.5pro/BCData/ki67_results.csv   5.results/gemini1.5flash/BCData/ki67_results.csv   5.results/grok2vision/BCData/ki67_results.csv   5.results/claude-3-5-sonnet/BCData/ki67_results.csv  --labels "GPT-4.5,GPT-4.1 mini,GPT-4.1,GPT-4o,Gemini 1.5 Pro,Gemini 1.5 Flash,Grok-2 Vision,Claude 3.5 Sonnet"  --low 16 --mid 30  --cm-out 5.results/conf_matrix_grid_bcdata.pdf  --cm-rows 1 --cm-cols 8
  ```

  SHIDC-B-Ki-67
  ```bash
  python 4.utils/evaluate_ki67_classification.py  --gt 1.data_access/data_full_SHIDC-B-Ki-67_summary.xlsx  --pred   5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv   5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv   5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv   5.results/4o/SHIDC-B-Ki-67/ki67_results.csv   5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv   5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv   5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv   5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv  --labels "GPT-4.5,GPT-4.1 mini,GPT-4.1,GPT-4o,Gemini 1.5 Pro,Gemini 1.5 Flash,Grok-2 Vision,Claude 3.5 Sonnet"  --low 16 --mid 30  --cm-out 5.results/conf_matrix_grid_shidc_b_ki67.pdf  --cm-rows 1 --cm-cols 8
  ```

- ### `fill_csv_from_txt.py`

  This script helps to rectify the results CSV. It identifies cases from the `llm_responses.txt` that were correctly responded to by the model but, due to extraction errors, were not fully recorded in the initial CSV. It then populates these missing entries into the output CSV.

  **Usage:**

  structure  
  ```bash
  python 4.utils/fill_csv_from_txt.py <results.csv> <llm_responses.txt> <json_folder>
  ```

  example  
  ```bash
  python 4.utils/fill_csv_from_txt.py 5.results/4.5/bcdata/ki67_results.csv 5.results/4.5/bcdata/llm_responses.txt 1.data_access/data_sample/3.data_processed
  ```

- ### `plot_multiple_models.py`

  This script generates a consolidated graph visualizing the results from multiple models. It takes the CSV result files from various models as input and plots their performance for comparative analysis.

  **Usage:**

  structure  
  ```bash
  python 4.utils/plot_multiple_models.py <csv1> [<csv2> ...] [--rows N] [--cols M] [--out <output.pdf>]
  ```

  Example BCData
  ```bash
  python 4.utils/plot_multiple_models.py 5.results/4.5/BCData/ki67_results.csv 5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv 5.results/4.1-2025-04-14/BCData/ki67_results.csv 5.results/4o/BCData/ki67_results.csv 5.results/gemini1.5pro/BCData/ki67_results.csv 5.results/gemini1.5flash/BCData/ki67_results.csv 5.results/grok2vision/BCData/ki67_results.csv 5.results/claude-3-5-sonnet/BCData/ki67_results.csv --rows 1 --cols 8 --out 5.results/ki67_comparison_plot_bcdata.pdf
  ```

  Example SHIDC-B-Ki-67
  ```bash
  python 4.utils/plot_multiple_models.py 5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv 5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv 5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv 5.results/4o/SHIDC-B-Ki-67/ki67_results.csv 5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv 5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv 5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv 5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv --rows 1 --cols 8 --out 5.results/ki67_comparison_plot_shidc-b-ki-67.pdf
  ```

  *(If you call the script **without arguments** it will fall back to that same default list.)*

- ### `verify_images_in_csv.py`

  Checks that every image file in a given directory is listed in the image column of a results CSV.  
  Useful for spotting images that were processed but never logged.

  **Usage:**
  structure  
  ```bash
  python 4.utils/verify_images_in_csv.py <image_folder> <results.csv>
  ```

  example  
  ```bash
  python 4.utils/verify_images_in_csv.py 1.data_access/data_sample_BCData/3.data_processed 5.results/4.5/bcdata/ki67_results.csv
  ```

## 5. Results

The `5.results` directory is dedicated to storing the outputs. After running the VLM processing and evaluation (Step 3), this folder will contain the generated CSV files, logs, raw VLM responses, and the graphical plots for each model's performance on the Ki-67 index calculation.
