import sys
sys.path.append("../")
import argparse
import numpy as np
from truthfulqa import metrics, models, utilities
from truthfulqa.evaluate import data_to_dict, format_frame
from truthfulqa.models import MC_calcs, find_subsequence, set_columns
from truthfulqa.presets import COMPARE_PRIMER, preset_map
from truthfulqa.utilities import (find_start, format_best, format_prompt,
                                  format_prompt_with_answer_strings,
                                  split_multi_answer)
import pandas as pd
from utils import run_ce_loss, run_kl_wrt_orig
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama_7B", type=str)
    parser.add_argument("--file", default="", type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--judge_name", type=str, default="")
    parser.add_argument("--info_name", type=str, default="")
    args = parser.parse_args()
    model = args.model
    file = args.file
    judge_name = args.judge_name
    info_name = args.info_name

    questions = utilities.load_questions(filename=file)
    questions = metrics.run_end2end_GPT3(model, 'GPT-judge', judge_name, questions, info=False)
    utilities.save_questions(questions, file)

    questions = metrics.run_end2end_GPT3(model, 'GPT-info', info_name, questions, info=True)
    utilities.save_questions(questions, file)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'GPT-judge acc',
                                              'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')


    # save results
    results.to_csv(args.output_file, index=False)
