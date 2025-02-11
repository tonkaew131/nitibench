import argparse
import sys
import asyncio
import os
import yaml
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append("/app/LRG")
import pandas as pd

from lrg.data import EvalDataset
from lrg.retrieval.retrieval_init import init_retriever
from lrg.prompting import PromptManager
from lrg.llm import init_llm
from lrg.augmenter import AugmenterConfig, Augmenter
from lrg.e2e import Ragger

from tqdm import tqdm
import torch

import time

from typing import List

import asyncio
import gc

from IPython.display import display

tqdm.pandas()

from llama_index.core.evaluation.retrieval.metrics import resolve_metrics

retrieval_metric = ["multi_hit_rate", "hit_rate", "recall", "mrr", "multimrr"]
retrieval_metric = {m: r() for r, m in zip(resolve_metrics(retrieval_metric), retrieval_metric)}

def display_case_final(sample_tax_case, key_to_show: list = ["ข้อหารือ", "แนววินิจฉัย"]):
    display_str = []
    for key, value in sample_tax_case.items():
        if key not in key_to_show:
            continue
        display_str.append(f"<{key}> {value} </{key}>")

    return "\n".join(display_str)

def citation_score(reference_citations, generated_citations):
    #What we need to do is two things, micro and macro
    #For micro, count TP, FP, FN and sum before calculating precision, recall and f1
    #As for macro, count TP, FP, FN and calculate precision, recall and f1 before averaging
    tp = 0
    fp = 0
    fn = 0

    precisions = []
    recalls = []
    f1s = []

    assert len(reference_citations) == len(generated_citations), "Reference citations and generated citations must have the same length"
    assert len(reference_citations) > 0, "There must be at least one citation"

    #List comprehension, change to set
    reference_citations = [set([(x["law"], x["sections"]) for x in r]) for r in reference_citations]
    generated_citations = [set([(x["law"], x["section"]) for x in g]) for g in generated_citations]

    for r, g in zip(reference_citations, generated_citations):
        local_tp = len(r & g)
        local_fp = len(r - g)
        local_fn = len(g - r)

        tp += local_tp
        fp += local_fp
        fn += local_fn

        local_precision = local_tp / (local_tp + local_fp) if local_tp + local_fp > 0 else 0
        local_recall = local_tp / (local_tp + local_fn) if local_tp + local_fn > 0 else 0
        local_f1 = 2*local_precision*local_recall/(local_precision+local_recall) if local_precision + local_recall > 0 else 0

        precisions.append(local_precision)
        recalls.append(local_recall)
        f1s.append(local_f1)

    micro_precision = tp / (tp + fp) if tp + fp > 0 else 0
    micro_recall = tp / (tp + fn) if tp + fn > 0 else 0
    micro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall) if micro_precision + micro_recall > 0 else 0

    macro_precision = sum(precisions)/len(precisions)
    macro_recall = sum(recalls)/len(recalls)
    macro_f1 = sum(f1s)/len(f1s)

    return {"micro_precision": micro_precision, 
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "local_precision": precisions,
            "local_recall": recalls,
            "local_f1s": f1s}

async def calc_metric(row: pd.Series, 
                      col_names: dict,
                      llm,
                      pm: PromptManager,
                      metric_name: List[str] = list(retrieval_metric.keys()), 
                      model_name: str = "gpt-4o-2024-08-06", 
                      dataset_name: str = "tax",
                      eval_retrieval: bool = False,
                      mapping = None):
    
    task = "coverage-contradiction"
    
    # all_laws = [(r["law"], r["sections"]) if "sections" in r else (r["law"], r["section"]) for r in row[col_names["reference_laws"]] + row[col_names["student_laws"]] ]

    query = display_case_final(row, key_to_show=[col_names["question"], col_names["reference_answer"], col_names["student_answer"]])
    
    augmented_query = query

    formatted_prompts = pm.get_formatted_prompt(query=augmented_query, task=task, dataset=dataset_name, model=model_name)
    # print(formatted_prompts["messages"])
    
    name = model_name.split("-")[0]
    if name == "gemini":
        structure = pm.response_structure[task][2]
    elif name == "gpt":
        structure = pm.response_structure[task][1]
    elif name == "claude":
        structure = pm.response_structure[task][0]
    else:
        raise ValueError("Unrecognized model name: {}".format(model_name))
    
    #Then, generate the response
    response = await llm.complete(**formatted_prompts, structure=structure)
    
    if eval_retrieval:
        expected_ids = [f"{r['law']}-{r['sections']}" for r in row[col_names["reference_laws"]]]
        retrieved_ids = row[col_names["retrieved_ids"]]
        
        retrieval_result = {m: retrieval_metric[m].compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids, mapping = mapping).score for m in metric_name}
        
        response["retrieval_result"] = retrieval_result
        
    return response

async def main(args):
    
    #First, load config
    #Read config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        
    #Next, load actual dataset
    dataset = EvalDataset(node_path = config["chunk_node_path"])
    chunk_mapping = dataset.mapper
    
    del dataset
        
    #Create node dict
    dataset = EvalDataset(node_path = config["golden_node_path"], 
                          wangchan_data_path = "/app/test_data/final_full_data_reduced_sections.csv",
                          tax_data_path = "/app/test_data/usable_tax_case_reduced_sections.csv")
    
    
    tax_result_path = os.path.join(config["result_dir"], "tax_response.json")
    wangchan_result_path = os.path.join(config["result_dir"], "wangchan_response.json")
    
    tax_col_names = {"question": "ข้อหารือ", "reference_answer": "reference_answer", "student_answer": "student_answer", 
            "reference_laws": "actual_relevant_laws", "student_laws": "student_citations",
            "retrieved_ids": "retrieved_ids"}
    
    wangchan_col_names = {"question": "question", "reference_answer": "reference_answer", "student_answer": "student_answer", 
            "reference_laws": "relevant_laws", "student_laws": "student_citations",
            "retrieved_ids": "retrieved_ids"}

    pm = PromptManager()

    llm = init_llm(config = config["llm_config"])
    
    
    #Read Answer
    wangchan_result = pd.read_json(wangchan_result_path)

    result = []
    for i, row in wangchan_result.iterrows():

        if not isinstance(row["content"], dict):
            print(i)
            result.append({**row["usage"], "retrieved_ids": row["retrieved_ids"], "idx": row["idx"], "tries": row["tries"]})
        else:
            result.append({**row["content"], **row["usage"], "retrieved_ids": row["retrieved_ids"], "idx": row["idx"], "tries": row["tries"]})

    wangchan_result = pd.DataFrame(result).rename(columns = {"answer": "student_answer", "citations": "student_citations"})
    
    tax_result = pd.read_json(tax_result_path)

    result = []
    for i, row in tax_result.iterrows():

        if not isinstance(row["content"], dict):
            print(i)
            result.append({**row["usage"], "retrieved_ids": row["retrieved_ids"], "idx": row["idx"], "tries": row["tries"]})
        else:
            result.append({**row["content"], **row["usage"], "retrieved_ids": row["retrieved_ids"], "idx": row["idx"], "tries": row["tries"]})

    tax_result = pd.DataFrame(result).rename(columns = {"answer": "student_answer", "citations": "student_citations"})
    # display(tax_result)
    
    #Then, merge together with actual answer
    wangchan_df = pd.merge(dataset.wangchan_df, wangchan_result[["idx", "student_answer", "student_citations", "retrieved_ids"]], left_on="idx", right_on="idx", how="inner")
    print(wangchan_result.shape)
    print(dataset.wangchan_df.shape)
    assert wangchan_df.shape[0] == dataset.wangchan_df.shape[0], "Merge wangchan got incorrect shape"
    
    #Then, merge together with actual answer
    tax_df = pd.merge(dataset.tax_df, tax_result[["idx", "student_answer", "student_citations", "retrieved_ids"]], left_on="file_name", right_on="idx", how="inner")
    assert tax_df.shape[0] == dataset.tax_df.shape[0], "Merge tax got incorrect shape"
    
    # display(tax_df)
    
    #Then, just call for each
    eval_retrieval = config.get("eval_retrieval", False)
    is_golden_chunk = "golden" in os.path.basename(config["result_dir"])
    model_name = config["llm_config"]["model"]
    wangchan_e2e_path = os.path.join(config["result_dir"], "wangchan_e2e_metrics.json")
    tax_e2e_path = os.path.join(config["result_dir"], "tax_e2e_metrics.json")
    batch_size = config.get("batch_size", 50)
    
    tax_e2e_metrics = []
    
    #Fill answer na with empty string
    #Fill empty citations with empty list
    tax_df = tax_df.fillna({tax_col_names["student_answer"]: "", tax_col_names["student_laws"]: ""})
    tax_df[tax_col_names["student_laws"]] = tax_df[tax_col_names["student_laws"]].apply(lambda x: [] if x == "" else x)
    
    for i in tqdm(range(0, len(tax_df), batch_size)):
        #Create jobs
        jobs = [calc_metric(row, 
                            col_names = tax_col_names, 
                            llm = llm,
                            pm = pm,
                            dataset_name = "tax",
                            model_name = model_name,
                            eval_retrieval = eval_retrieval,
                            mapping = chunk_mapping if not is_golden_chunk else None) for _, row in tax_df.iloc[i: i+batch_size].iterrows()]
        
        tax_e2e_metrics.extend(await asyncio.gather(*jobs))
        
    #Dump
    print("TAX DONE")
    print("DUMPING TO ", tax_result_path)
    with open(tax_e2e_path, "w") as f:
        json.dump(tax_e2e_metrics, f)
        
        
    wangchan_e2e_metrics = []
    
    wangchan_df = wangchan_df.fillna({wangchan_col_names["student_answer"]: "", wangchan_col_names["student_laws"]: ""})
    # print(wangchan_df[wangchan_df[wangchan_col_names["student_laws"]] == ""])
    wangchan_df[wangchan_col_names["student_laws"]] = wangchan_df[wangchan_col_names["student_laws"]].apply(lambda x: [] if x == "" else x)
    
    for i in tqdm(range(0, len(wangchan_df), batch_size)):
        #Create jobs
        jobs = [calc_metric(row, 
                            col_names = wangchan_col_names, 
                            llm = llm,
                            pm = pm,
                            dataset_name = "wangchan",
                            model_name = model_name,
                            eval_retrieval = eval_retrieval,
                            mapping = chunk_mapping if not is_golden_chunk else None) for _, row in wangchan_df.iloc[i: i+batch_size].iterrows()]
        
        wangchan_e2e_metrics.extend(await asyncio.gather(*jobs))
        
        #Dump
        with open(wangchan_e2e_path, "w") as f:
            json.dump(wangchan_e2e_metrics, f)
    
    
    
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/app/LRG/results/e2e/v0.1/chunk_vary")
    args = parser.parse_args()
    asyncio.run(main(args))