"""Evaluation script for Semeval 2020 Task 3: DeftEval Relation Extraction
How to use:
    Used by semeval2020_06_evaluation_main.py, which is the main entry point into the evaluation_subtask_3 scripts

Input files:
    gold_fname: tab-separated .deft files for Subtask 3 with the following columns (no column headers):
        token filename token_start token_end label tag_id relation_root relation_name
    pred_fname: same format as gold_fname above

Notes:
    1. pred_fname must have the same columns as gold_fname in the same order
    2. all columns in pred_fname, except for relation_root and relation_name, must contain the same data as in gold_fname
    3. pred_fname may not contain any relations that did not appear in the training data
"""

import csv
from pathlib import Path
import sys
import os
from sklearn.metrics import classification_report
import warnings


def get_token(row):
    """Get the token from the row
    Inputs:
        row: list of strings
    Returns:
        token: string
    """
    return row[0]


def get_relation(row):
    """Get the relation from the row
    Inputs:
        row: list of strings
    Returns:
        relation: string
    """

    # TODO: shouldn't need to strip spaces, check where they're coming from
    return row[-1].strip()


def get_relation_from(row):
    """Get the root of the relation
    Inputs:
        row: list of strings
    Returns:
        relation_from: string
    """
    return row[-2].strip()


def get_relation_to(row):
    """Get the destination of the relation
    Inputs:
        row: list of strings
    Returns:
        relation_to: string
    """
    return row[-3].strip()

def remove_relation(row):
    row[-2] = "0"
    row[-3] = "-1"
    return row

def is_root(row):
    """Is this token the root of the relation?"""
    return row[-2] == "0"


def has_relation(row):
    """Does this token participate in a relation?"""

    return get_relation(row) != "0"


def validate_length(gold_rows, pred_rows):
    """Check that gold and pred files have same number of rows
    Inputs:
        gold_rows: list of lists of strings
        pred_rows: list of lists of strings
    """
    if len(pred_rows) != len(gold_rows):
        raise ValueError("Row length mismatch: Pred {} Gold {}".format(len(pred_rows), len(gold_rows)))


def validate_columns(gold_rows, pred_rows):
    """Check correct number of (and correct data in) columns
     Inputs:
        gold_rows: list of lists of strings
        pred_rows: list of lists of strings
    """
    gold_column_counts = dict()
    pred_column_counts = dict()
    for row in gold_rows:
        n_columns = len(row)
        gold_column_counts[n_columns] = gold_column_counts.get(n_columns, 0) + 1
    for row in pred_rows:
        n_columns = len(row)
        pred_column_counts[n_columns] = pred_column_counts.get(n_columns, 0) + 1
    if len(gold_column_counts.keys()) > 1:
        raise TypeError("Uneven number of Gold columns: {}".format(gold_column_counts))
    if len(pred_column_counts.keys()) > 1:
        raise TypeError("Uneven number of Pred columns: {}".format(pred_column_counts))


def validate_tokens(gold_rows, pred_rows):
    """Check that gold and pred files have same tokens in each row
     Inputs:
        gold_rows: list of lists of strings
        pred_rows: list of lists of strings
    """
    for row_index in range(len(pred_rows)):
        gold_token = get_token(gold_rows[row_index]).split()
        pred_token = get_token(pred_rows[row_index]).split()

        if len(gold_token) != len(pred_token):
            raise ValueError("Token mismatch row {}: Pred {} Gold {}".format(row_index, pred_token, gold_token))


def validate_pred_relations(pred_rows, known_relations):
    """Check that pred file doesn't have any unknown relations
      Inputs:
        gold_rows: list of lists of strings
        pred_rows: list of lists of strings
    """
    unknown_relations = list()  # [(row, relation)]
    for row_index in range(len(pred_rows)):
        pred_relation = get_relation(pred_rows[row_index])
        if pred_relation.strip() != '0' and pred_relation not in known_relations:
            unknown_relations.append((row_index, pred_relation))

    if unknown_relations:
        raise ValueError("Encountered unknown relation in the following rows {}".format(unknown_relations))

def validate_ref_relations(eval_labels, ref_rows):
    validated_rows = []
    for row_index in range(len(ref_rows)):
        ref_relation = get_relation(ref_rows[row_index])
        if ref_relation is not '0' and ref_relation not in eval_labels:
            warnings.warn("Labels exist in the reference files that will not be scored given the current evaluation_subtask_3 labels.")
            row = remove_relation(ref_rows[row_index])
        else:
            row = ref_rows[row_index]
        validated_rows.append(row)
    return validated_rows

def validate_data(gold_rows, pred_rows, eval_relations):
    """Make sure the data is OK
      Inputs:
        gold_rows: list of lists of strings
        pred_rows: list of lists of strings
    """
    gold_rows = validate_ref_relations(eval_relations, gold_rows)
    validate_pred_relations(pred_rows, eval_relations)
    validate_length(gold_rows, pred_rows)
    validate_columns(gold_rows, pred_rows)
    validate_tokens(gold_rows, pred_rows)
    return gold_rows
    # validate_relations(gold_rows, pred_rows)


def get_gold_and_pred_relations(gold_fname, pred_fname, eval_relations, eval_labels):
    """Get the relation pairs for evaluation_subtask_3
    Inputs:
        gold_fname: path to .deft file
        pred_fname: path to .deft file
    Returns:
        y_gold_rel_pairs: list of (tag1, tag2, relation) tuples
        y_pred_rel_pairs: list of (tag1, tag2, relation) tuples
    """
    y_gold_rel_pairs = set()  # [(elem1, elem2, rel)]
    y_pred_rel_pairs = set()  # [(elem1, elem2, rel)]


    with gold_fname.open(encoding='utf-8') as gold_source:
        gold_reader = csv.reader(gold_source, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
        gold_rows = [row for row in gold_reader if row]

    with pred_fname.open(encoding='utf-8') as pred_source:
        pred_reader = csv.reader(pred_source, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
        pred_rows = [row for row in pred_reader if row]

    gold_rows = validate_data(gold_rows, pred_rows, eval_relations)

    gold_relation_rows = [row for row in gold_rows if has_relation(row)]
    pred_relation_rows = [row for row in pred_rows if has_relation(row)]



    for row in gold_relation_rows:
        if row[4].strip() not in eval_labels:
            continue
        relation = get_relation(row)
        relation_from = get_relation_from(row)
        relation_to = get_relation_to(row)
        y_gold_rel_pairs.add((relation_from, relation_to, relation))

    for row in pred_relation_rows:
        if row[4].strip() not in eval_labels:
            continue
        relation = get_relation(row)
        relation_from = get_relation_from(row)
        relation_to = get_relation_to(row)
        y_pred_rel_pairs.add((relation_from, relation_to, relation))

    return y_gold_rel_pairs, y_pred_rel_pairs


def evaluate(y_gold_rel_pairs, y_pred_rel_pairs, eval_relations):
    """Get the scores
    Inputs:
        y_gold_rel_pairs: list of (tag1, tag2, relation) tuples
        y_pred_rel_pairs: list of (tag1, tag2, relation) tuples
        eval_relations: set of relations to evaluate on (e.g. excluding '0', etc.)
    Returns:
        report: dict
    """

    report = dict()
    running_p = 0
    running_r = 0
    running_f = 0

    for eval_relation in eval_relations:
        report[eval_relation] = dict()
        gold_matches = set([pair for pair in y_gold_rel_pairs if pair[-1] == eval_relation])
        pred_matches = set([pair for pair in y_pred_rel_pairs if pair[-1] == eval_relation])
        tp = len([match for match in pred_matches if match in gold_matches])
        fp = len([match for match in pred_matches if match not in gold_matches])
        fn = len([match for match in gold_matches if match not in pred_matches])
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0
        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = 0
        try:
            f = (2 * p * r) / (p + r)
        except ZeroDivisionError:
            f = 0
        report[eval_relation]['p'] = p
        report[eval_relation]['r'] = r
        report[eval_relation]['f'] = f
        running_p += p
        running_r += r
        running_f += f

    macro_p = running_p / len(eval_relations)
    macro_r = running_r / len(eval_relations)
    macro_f = running_f / len(eval_relations)
    report['macro'] = dict()
    report['macro']['p'] = macro_p
    report['macro']['r'] = macro_r
    report['macro']['f'] = macro_f

    return report

def write_to_scores(report, output_fname):
    """ Prints scores to scores.txt
      Inputs:
          report: classification_report generated from evaluate()
      Returns:
          None
      """
    with open(output_fname, 'a+') as scores_file:
        # scores_file.write('precision_macro: ' + report['macro']['p'])
        # scores_file.write('recall_macro: ' + report['macro']['r'])

        if report is not None:
            scores_file.write('subtask_3_f1-score_macro: ' + str(report['macro']['f']) + '\n')
        else:
            scores_file.write('subtask_3_f1-score_macro: -1\n')


def task_3_eval_main(ref_path, res_path,  output_dir, eval_relations, eval_labels):
    """Evaluate the data
    Inputs:
        gold_fname: path of .deft file
        pred_fname: path of .deft file
        eval_labels: set of labels to evaluate on
    Returns:
        report: dict of macro average scores
    """

    y_gold = []
    y_pred = []

    reference_files = []
    results_files = []

    for child in res_path.iterdir():
        if child.name.startswith('task_3'):
            results_files.append(child.name)
            if not os.path.exists(ref_path.joinpath(child.name)):
                message = "Expected submission file '{0}', found files {1}"
                sys.exit(message.format(child, os.listdir(child.parents[0])))

            temp_y_gold, temp_y_pred = get_gold_and_pred_relations(ref_path.joinpath(child.name), child, eval_relations, eval_labels)
            y_gold.extend(temp_y_gold)
            y_pred.extend(temp_y_pred)

    if len(results_files) == 0:
        message = "No subtask 3 files to evaluate."
        warnings.warn(message)
        write_to_scores(None, Path(output_dir).joinpath('scores.txt'))
        return None

    for child in ref_path.iterdir():
        if child.name.startswith('task_3'):
            reference_files.append(child.name)

    missing = [file for file in reference_files if file not in results_files]
    if len(missing) > 0:
        message = "Missing evaluation_subtask_3 files {0}"
        sys.exit(message.format(str(missing)))

    report = evaluate(y_gold, y_pred, eval_relations)
    write_to_scores(report, Path(output_dir).joinpath('scores.txt'))

    return report


