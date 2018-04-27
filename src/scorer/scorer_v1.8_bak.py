#!/usr/bin/python

"""
    A simple scorer that reads the CMU Event Mention Format (tbf)
    data and produce a mention based F-Scores.

    It could also call the CoNLL coreference implementation and
    produce coreference results.

    This scorer also require token files to conduct evaluation.

    Author: Zhengzhong Liu ( liu@cs.cmu.edu )
"""
# Change log v1.8
# 1. Adding supports for Event Sequencing evaluation by calling TIMEML evaluators.

# Change log v1.7.3
# 1. Allow user to configure specific types for evaluation.

# Change log v1.7.2
# 1. Remove invisible word list, it is too arbitrary.

# Change log v1.7.1
# 1. Add character based evaluation back, which can support languages such as Chinese.
# 2. Within cluster duplicate check is currently disabled because of valid cases might exist:
#   a. If the argument is an appositive (multiple nouns), sometimes multiple event mentions are annotated.

# Change log v1.7.0
# 1. Changing the way of mention mapping to coreference, so that it will not favor too much on recall.
# 2. Speed up on the coreference scoring since we don't need to select the best, we can convert in one single step.
# 3. Removing "what", "it" from invisible list.
# 4. Small changes on the way of looking for the conll scorer.
# 5. Small changes on the layout of the scores.
# 6. New per mention type scoring is also provided in the score report.

# Change log v1.6.2:
# 1. Add clusters in the comparison output. No substantial changes in scoring.

# Change log v1.6.1:
# 1. Minor change that remove punctuation and whitespace in attribute types and lowercase all types to make system
# output more flexible.

# Change log v1.6:
# 1. Because there are too many double annotation, now such ambiguity are resolved arbitrarily:
#    a. For mention scoring, the system mention is mapped to a gold mention greedily.
#    b. The coreference evaluation relies on the mapping produced by mention mapping at mention type level. This means
#        that a system mention can only be mapped to a gold mention when their mention type matches.

# Change log v1.5:
# 1. Given that the CoNLL scorer only score exact matched mentions, we convert input format.
#    to a simplified form. We produce a mention mappings and feed to the scorer.
#    In case of double tagging, there are multiple way of mention mappings, we will produce all
#    possible ways, and use the highest final score mapping.
# 2. Fix a bug that crashes when generating text output from empty responses.
# 3. Write out the coreference scores into the score output.
# 4. Move global variables into class wrappers.
# 5. Current issue: gold standard coreference cannot be empty! Maybe file a bug to them.

# Change log v1.4:
# 1. Global mention span check: do not allow duplicate mention span with same type.
# 2. Within cluster mention span check : do not allow duplicate span in one cluster.

# Change log v1.3:
# 1. Add ability to convert input format to conll format, and feed it to the coreference resolver.
# 2. Clean up and remove global variables.

# Change log v1.2:
# 1. Change attribute scoring, combine it with mention span scoring.
# 2. Precision for span is divided by #SYS instead of TP + FP.
# 3. Plain text summary is made better.
# 4. Separate the visualization code out into anther file.

# Change log v1.1:
# 1. If system produce no mentions, the scorer should penalize it instead of ignore it.
# 2. Enhance the output of the comparison file, add the system actual output side by side for easy debug.
# 3. Add the ability to compare system and gold mentions using Brat embedded visualization.
# 4. For realis type not annotated, give full credit as long as system give a result.
# 5. Add more informative error message.

import argparse
import heapq
import itertools
import logging
import math
import os
import re
import sys

import utils
from config import Config, MutableConfig, EvalMethod, EvalState
from conll_coref import ConllEvaluator
from temporal import TemporalEval

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
logger.addHandler(stream_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Event mention scorer, provides support to Event Nugget scoring, Event Coreference and Event "
                    "Sequencing scoring.")
    parser.add_argument("-g", "--gold", help="Golden Standard", required=True)
    parser.add_argument("-s", "--system", help="System output", required=True)
    parser.add_argument("-d", "--comparison_output",
                        help="Compare and help show the difference between "
                             "system and gold")
    parser.add_argument(
        "-o", "--output", help="Optional evaluation result redirects, put eval result to file")
    parser.add_argument(
        "-c", "--coref", help="Eval Coreference result output, need to put the reference"
                              "conll coref scorer in the same folder with this scorer")
    parser.add_argument(
        "-a", "--sequencing", help="Eval Event sequencing result output (After and Subevent)"
    )
    parser.add_argument(
        "-nv", "--no_script_validation", help="Whether to turn off script validation", action="store_true"
    )
    parser.add_argument(
        "-t", "--token_path", help="Path to the directory containing the token mappings file, only used in token mode.")
    parser.add_argument(
        "-m", "--coref_mapping", help="Which mapping will be used to perform coreference mapping.", type=int
    )
    parser.add_argument(
        "-of", "--offset_field", help="A pair of integer indicates which column we should "
                                      "read the offset in the token mapping file, index starts"
                                      "at 0, default value will be %s" % Config.default_token_offset_fields
    )
    parser.add_argument(
        "-te", "--token_table_extension",
        help="any extension appended after docid of token table files. Default is [%s], only used in token mode."
             % Config.default_token_file_ext)
    parser.add_argument("-ct", "--coreference_threshold", type=float, help="Threshold for coreference mention mapping")
    parser.add_argument("-b", "--debug", help="turn debug mode on", action="store_true")

    parser.add_argument("--eval_mode", choices=["char", "token"], default="char",
                        help="Use Span or Token mode. The Span mode will take a span as range [start:end], while the "
                             "Token mode consider each token is provided as a single id.")

    parser.add_argument("-wl", "--type_white_list", type=argparse.FileType('r'),
                        help="Provide a file, where each line list a mention type subtype pair to be evaluated. Types "
                             "that are out of this white list will be ignored.")

    parser.add_argument(
        "-dn", "--doc_id_to_eval", help="Provide one single doc id to evaluate."
    )

    parser.set_defaults(debug=False)
    args = parser.parse_args()

    if args.debug:
        stream_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Entered debug mode.")
    else:
        stream_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    if args.type_white_list is not None:
        logger.info("Only the following types in the white list will be evaluated.")
        EvalState.white_listed_types = set()
        for line in args.type_white_list:
            logger.info(line.strip())
            EvalState.white_listed_types.add(canonicalize_string(line))

    if args.eval_mode == "char":
        MutableConfig.eval_mode = EvalMethod.Char
    else:
        MutableConfig.eval_mode = EvalMethod.Token

    if args.output is not None:
        out_path = args.output
        utils.create_parent_dir(out_path)
        mention_eval_out = open(out_path, 'w')
        logger.info("Evaluation output will be saved at %s" % out_path)
    else:
        mention_eval_out = sys.stdout
        logger.info("Evaluation output at standard out.")

    if os.path.isfile(args.gold):
        gf = open(args.gold)
    else:
        logger.error("Cannot find gold standard file at " + args.gold)
        sys.exit(1)

    if args.coref is not None:
        Config.conll_out = args.coref
        Config.conll_gold_file = args.coref + "_gold.conll"
        Config.conll_sys_file = args.coref + "_sys.conll"

        logger.info("CoNLL script output will be output at " + Config.conll_out)

        logger.info(
            "Gold and system conll files will generated at " + Config.conll_gold_file + " and " + Config.conll_sys_file)

    if args.sequencing is not None:
        Config.script_result_dir = args.sequencing

        logger.info("Temporal files will be output at " + Config.script_result_dir)
        utils.supermakedirs(Config.script_result_dir)

        logger.info("Will evaluate link type: %s." % ",".join(Config.script_types))
        for t in Config.script_types:
            utils.supermakedirs(os.path.join(Config.script_result_dir, t))

        utils.remove_file_by_extension(Config.script_result_dir, ".tml")
        utils.remove_file_by_extension(Config.script_result_dir, ".tml")

        if args.no_script_validation:
            Config.no_script_validation = True

    if os.path.isfile(args.system):
        sf = open(args.system)
    else:
        logger.error("Cannot find system file at " + args.system)
        sys.exit(1)

    if args.coref_mapping is not None:
        if args.coref_mapping < 4:
            Config.coref_criteria = Config.possible_coref_mapping[args.coref_mapping]
        else:
            logger.error("Possible mapping : 0: Span only 1: Mention Type 2: Realis 3 Type and Realis")
            utils.terminate_with_error("Must provide a mapping between 0 to 3")
    else:
        Config.coref_criteria = Config.possible_coref_mapping[1]

    diff_out = None
    if args.comparison_output is not None:
        diff_out_path = args.comparison_output
        utils.create_parent_dir(diff_out_path)
        diff_out = open(diff_out_path, 'w')

    token_dir = "."
    if args.token_path is not None:
        if args.eval_mode == EvalMethod.Token:
            utils.terminate_with_error("Token table (-t) must be provided in token mode")
        if os.path.isdir(args.token_path):
            logger.debug("Will search token files in " + args.token_path)
            token_dir = args.token_path
        else:
            logger.debug("Cannot find given token directory at [%s], "
                         "will try search for current directory" % args.token_path)

    token_offset_fields = Config.default_token_offset_fields
    if args.offset_field is not None:
        try:
            token_offset_fields = [int(x) for x in args.offset_field.split(",")]
        except ValueError as _:
            logger.error("Token offset argument should be two integer with comma in between, i.e. 2,3")

    if args.coreference_threshold is not None:
        MutableConfig.coref_mention_threshold = args.coreference_threshold

    # Read all documents.
    read_all_doc(gf, sf, args.doc_id_to_eval)

    # Take all attribute combinations, which will be used to produce scores.
    attribute_comb = get_attr_combinations(Config.attribute_names)

    logger.info("Coreference mentions need to match %s before consideration" % Config.coref_criteria[0][1])

    while True:
        if not evaluate(token_dir, args.coref, attribute_comb,
                        token_offset_fields, args.token_table_extension,
                        diff_out):
            break

    # Run the CoNLL script on the combined files, which is concatenated from the best alignment of all documents.
    if args.coref is not None:
        logger.debug("Running coreference script for the final scores.")
        ConllEvaluator.run_conll_script(Config.conll_gold_file, Config.conll_sys_file, Config.conll_out)
        # Get the CoNLL scores from output
        EvalState.overall_coref_scores = ConllEvaluator.get_conll_scores(Config.conll_out)

    # Run the TimeML evaluation script.
    if Config.script_result_dir:
        TemporalEval.eval_time_ml()

    print_eval_results(mention_eval_out, attribute_comb)

    # Clean up, close files.
    close_if_not_none(diff_out)

    logger.info("Evaluation Done.")


def close_if_not_none(f):
    if f is not None:
        f.close()


def get_combined_attribute_header(all_comb, size):
    header_list = [pad_char_before_until("plain", size)]
    for comb in all_comb:
        attr_header = []
        for attr_pair in comb:
            attr_header.append(attr_pair[1])
        header_list.append(pad_char_before_until("+".join(attr_header), size))
    return header_list


def get_cell_width(scored_infos):
    max_doc_name = 0
    for info in scored_infos:
        doc_id = info[5]
        if len(doc_id) > max_doc_name:
            max_doc_name = len(doc_id)
    return max_doc_name


def pad_char_before_until(s, n, c=" "):
    return c * (n - len(s)) + s


def print_eval_results(mention_eval_out, all_attribute_combinations):
    total_gold_mentions = 0
    total_system_mentions = 0
    valid_docs = 0

    plain_global_scores = [0.0] * 4
    attribute_based_global_scores = [[0.0] * 4 for _ in xrange(len(all_attribute_combinations))]

    doc_id_width = get_cell_width(EvalState.doc_mention_scores)

    mention_eval_out.write("========Document Mention Detection Results==========\n")
    small_header_item = "Prec  \tRec  \tF1   "
    attribute_header_list = get_combined_attribute_header(all_attribute_combinations, len(small_header_item))
    small_headers = [small_header_item] * (len(all_attribute_combinations) + 1)
    mention_eval_out.write(pad_char_before_until("", doc_id_width) + "\t" + "\t|\t".join(attribute_header_list) + "\n")
    mention_eval_out.write(pad_char_before_until("Doc ID", doc_id_width) + "\t" + "\t|\t".join(small_headers) + "\n")

    for (tp, fp, attribute_based_counts, num_gold_mentions, num_sys_mentions, docId) in EvalState.doc_mention_scores:
        tp *= 100
        fp *= 100
        prec = safe_div(tp, num_sys_mentions)
        recall = safe_div(tp, num_gold_mentions)
        doc_f1 = compute_f1(prec, recall)

        attribute_based_doc_scores = []

        for comb_index, comb in enumerate(all_attribute_combinations):
            counts = attribute_based_counts[comb_index]
            attr_tp = counts[0] * 100
            attr_fp = counts[1] * 100
            attr_prec = safe_div(attr_tp, num_sys_mentions)
            attr_recall = safe_div(attr_tp, num_gold_mentions)
            attr_f1 = compute_f1(attr_prec, attr_recall)

            attribute_based_doc_scores.append("%.2f\t%.2f\t%.2f" % (attr_prec, attr_recall, attr_f1))

            for score_index, score in enumerate([attr_tp, attr_fp, attr_prec, attr_recall]):
                if not math.isnan(score):
                    attribute_based_global_scores[comb_index][score_index] += score

        mention_eval_out.write(
            "%s\t%.2f\t%.2f\t%.2f\t|\t%s\n" % (
                pad_char_before_until(docId, doc_id_width), prec, recall, doc_f1,
                "\t|\t".join(attribute_based_doc_scores)))

        # Compute the denominators:
        # 1. Number of valid doc does not include gold standard files that contains no mentions.
        # 2. Gold mention count and system mention count are accumulated, used to compute prec, recall.
        if math.isnan(recall):
            # gold produce no mentions, do nothing
            pass
        elif math.isnan(prec):
            # system produce no mentions, accumulate denominator
            logger.warning('System produce nothing for document [%s], assigning 0 scores' % docId)
            valid_docs += 1
            total_gold_mentions += num_gold_mentions
        else:
            valid_docs += 1
            total_gold_mentions += num_gold_mentions
            total_system_mentions += num_sys_mentions

            for score_index, score in enumerate([tp, fp, prec, recall]):
                plain_global_scores[score_index] += score

    if len(EvalState.doc_coref_scores) > 0:
        mention_eval_out.write("\n\n========Document Mention Corefrence Results (CoNLL Average)==========\n")
        for coref_score, doc_id in EvalState.doc_coref_scores:
            mention_eval_out.write("%s\t%.2f\n" % (doc_id, coref_score))

    per_type_precision, per_type_recall, per_type_f1 = summarize_type_scores()

    mention_eval_out.write("\n\n========Mention Type Results==========\n")
    if len(per_type_f1) > 0:
        max_type_name_width = len(max(per_type_f1.keys(), key=len))
        mention_eval_out.write("%s\tPrec\tRec\tF1\t#Gold\t#Sys\n" % pad_char_before_until("Type", max_type_name_width))
        for mention_type, f1 in sorted(per_type_f1.items()):
            mention_eval_out.write("%s\t%.2f\t%.2f\t%.2f\t%d\t%d\n" % (
                pad_char_before_until(mention_type, max_type_name_width),
                utils.nan_as_zero(utils.get_or_else(per_type_precision, mention_type, 0)),
                utils.nan_as_zero(utils.get_or_else(per_type_recall, mention_type, 0)),
                utils.nan_as_zero(utils.get_or_else(per_type_f1, mention_type, 0)),
                utils.nan_as_zero(utils.get_or_else(EvalState.per_type_num_gold, mention_type, 0)),
                utils.nan_as_zero(utils.get_or_else(EvalState.per_type_num_response, mention_type, 0))
            ))

    # Use the denominators above to calculate the averages.
    plain_average_scores = get_averages(plain_global_scores, total_gold_mentions, total_system_mentions, valid_docs)

    mention_eval_out.write("\n=======Final Mention Detection Results=========\n")
    max_attribute_name_width = len(max(attribute_header_list, key=len))
    attributes_name_header = pad_char_before_until("Attributes", max_attribute_name_width)

    final_result_big_header = ["Micro Average", "Macro Average"]

    mention_eval_out.write(
        pad_char_before_until("", max_attribute_name_width, " ") + "\t" + "\t".join(
            [pad_char_before_until(h, len(small_header_item)) for h in final_result_big_header]) + "\n")
    mention_eval_out.write(attributes_name_header + "\t" + "\t".join([small_header_item] * 2) + "\n")
    mention_eval_out.write(pad_char_before_until(attribute_header_list[0], max_attribute_name_width) + "\t" + "\t".join(
        "%.2f" % f for f in plain_average_scores) + "\n")
    for attr_index, attr_based_score in enumerate(attribute_based_global_scores):
        attr_average_scores = get_averages(attr_based_score, total_gold_mentions, total_system_mentions, valid_docs)
        mention_eval_out.write(
            pad_char_before_until(attribute_header_list[attr_index + 1],
                                  max_attribute_name_width) + "\t" + "\t".join(
                "%.2f" % f for f in attr_average_scores) + "\n")

    if len(EvalState.overall_coref_scores) > 0:
        mention_eval_out.write("\n=======Final Mention Coreference Results=========\n")
        conll_sum = 0.0
        num_metric = 0
        for metric, score in EvalState.overall_coref_scores.iteritems():
            formatter = "Metric : %s\tScore\t%.2f\n"
            if metric in Config.skipped_metrics:
                formatter = "Metric : %s\tScore\t%.2f *\n"
            else:
                conll_sum += score
                num_metric += 1
            mention_eval_out.write(formatter % (metric, score))
        mention_eval_out.write(
            "Overall Average CoNLL score\t%.2f\n" % (conll_sum / num_metric))
        mention_eval_out.write("\n* Score not included for final CoNLL score.\n")

    if Config.script_result_dir is not None:
        mention_eval_out.write("\n")

        for eval_type in Config.script_types + ["All"]:
            for filename in os.listdir(os.path.join(Config.script_result_dir, eval_type)):
                script_eval_path = os.path.join(Config.script_result_dir, eval_type, filename)
                if os.path.isfile(script_eval_path):
                    if filename == Config.script_out:
                        with open(script_eval_path, 'r') as out:
                            mention_eval_out.write("=======Event Sequencing Results for %s =======\n" % eval_type)
                            for l in out:
                                mention_eval_out.write(l)

                    if Config.eval_cluster_level_links:
                        if filename == Config.script_out_cluster:
                            with open(script_eval_path, 'r') as out:
                                mention_eval_out.write(
                                    "=======Event Sequencing Results for %s (Cluster) =======\n" % eval_type)
                                for l in out:
                                    mention_eval_out.write(l)

    if mention_eval_out is not None:
        mention_eval_out.flush()
    if not mention_eval_out == sys.stdout:
        mention_eval_out.close()


def get_averages(scores, num_gold, num_sys, num_docs):
    micro_prec = safe_div(scores[0], num_sys)
    micro_recall = safe_div(scores[0], num_gold)
    micro_f1 = compute_f1(micro_prec, micro_recall)
    macro_prec = safe_div(scores[2], num_docs)
    macro_recall = safe_div(scores[3], num_docs)
    macro_f1 = compute_f1(macro_prec, macro_recall)
    return micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1


def read_token_ids(token_dir, g_file_name, provided_token_ext, token_offset_fields):
    tf_ext = Config.default_token_file_ext if provided_token_ext is None else provided_token_ext

    invisible_ids = set()
    id2token = {}
    id2span = {}

    token_file_path = os.path.join(token_dir, g_file_name + tf_ext)

    logger.debug("Reading token for " + g_file_name)

    try:
        token_file = open(token_file_path)

        # Discard the header.
        # _ = token_file.readline()

        for tline in token_file:
            fields = tline.rstrip().split("\t")
            if len(fields) < 4:
                logger.error("Weird token line " + tline)
                continue

            token = fields[1].lower().strip().rstrip()
            token_id = fields[0]

            id2token[token_id] = token

            try:
                token_span = (int(fields[token_offset_fields[0]]), int(fields[token_offset_fields[1]]))
                id2span[token_id] = token_span
            except ValueError as _:
                logger.warn("Token file is wrong at for file [%s], cannot parse token span here." % g_file_name)
                logger.warn("  ---> %s" % tline.strip())
                logger.warn(
                    "Field %d and Field %d are not integer spans" % (
                        token_offset_fields[0], token_offset_fields[1]))

            if token in Config.invisible_words:
                invisible_ids.add(token_id)

    except IOError:
        logger.error(
            "Cannot find token file for doc [%s] at [%s], "
            "will use empty invisible words list" % (g_file_name, token_file_path))
        pass

    return invisible_ids, id2token, id2span


def safe_div(n, dn):
    return 1.0 * n / dn if dn > 0 else float('nan')


def compute_f1(p, r):
    return safe_div(2 * p * r, (p + r))


def read_all_doc(gf, sf, single_doc_id_to_eval):
    """
    Read all the documents, collect the document ids that are shared by both gold and system. It will populate the
    gold_docs and system_docs, stored as map from doc id to raw annotation strings.

    The document ids considered to be scored are those presented in the gold documents.

    :param gf: Gold standard file
    :param sf:  System response file
    :param single_doc_id_to_eval: If not None, we will evaluate only this doc id.
    :return:
    """
    EvalState.gold_docs, _ = read_docs_with_doc_id_and_name(gf)
    EvalState.system_docs, EvalState.system_id = read_docs_with_doc_id_and_name(sf)

    g_doc_ids = EvalState.gold_docs.keys()
    s_doc_ids = EvalState.system_docs.keys()

    g_id_set = set(g_doc_ids)
    s_id_set = set(s_doc_ids)

    common_id_set = g_id_set.intersection(s_id_set)

    if single_doc_id_to_eval is not None:
        logger.info("Evaluate only file [%s]" % single_doc_id_to_eval)
        if single_doc_id_to_eval not in g_id_set:
            logger.error("This document is not found in gold standard.")
        if single_doc_id_to_eval not in s_id_set:
            logger.error("This document is not found in system standard")

        EvalState.doc_ids_to_score = [single_doc_id_to_eval]
    else:
        g_minus_s = g_id_set - common_id_set
        s_minus_g = s_id_set - common_id_set

        if len(g_minus_s) > 0:
            logger.warning("The following document are not found in system but in gold standard")
            for d in g_minus_s:
                logger.warning("  - " + d)

        if len(s_minus_g) > 0:
            logger.warning("\tThe following document are not found in gold standard but in system")
            for d in s_minus_g:
                logger.warning("  - " + d)

        if len(common_id_set) == 0:
            logger.warning("No document to score, file names are all different!")

        EvalState.doc_ids_to_score = sorted(g_id_set)


def read_docs_with_doc_id_and_name(f):
    """
    Parse file into a map from doc id to mention and relation raw strings
    :param f: The annotation file
    :return: A map from doc id to corresponding mention and relation annotations, which are stored as raw string
    """
    all_docs = {}
    mention_lines = []
    relation_lines = []
    doc_id = ""
    run_id = os.path.basename(f.name)
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip().rstrip()

        if line.startswith(Config.comment_marker):
            if line.startswith(Config.bod_marker):
                doc_id = line[len(Config.bod_marker):].strip()
            elif line.startswith(Config.eod_marker):
                all_docs[doc_id] = mention_lines, relation_lines
                mention_lines = []
                relation_lines = []
        elif line.startswith(Config.relation_marker):
            relation_lines.append(line[len(Config.relation_marker):].strip())
        elif line == "":
            pass
        else:
            mention_lines.append(line)

    return all_docs, run_id


def get_next_doc():
    """
    Get next document pair of gold standard and system response.
    :return: A tuple of 4 element
        (has_next, gold_annotation, system_annotation, doc_id)
    """
    if EvalState.has_next_doc():  # A somewhat redundant check
        doc_id = EvalState.doc_ids_to_score[EvalState.evaluating_index]
        EvalState.advance_index()
        if doc_id in EvalState.system_docs:
            return True, EvalState.gold_docs[doc_id], EvalState.system_docs[doc_id], doc_id, EvalState.system_id
        else:
            return True, EvalState.gold_docs[doc_id], ([], []), doc_id, EvalState.system_id
    else:
        logger.error("Reaching end of all documents")
        return False, ([], []), ([], []), "End_Of_Documents"


def parse_characters(s):
    """
    Method to parse the character based span
    :param s:
    """
    span_strs = s.split(Config.span_seperator)
    characters = []
    for span_strs in span_strs:
        span = list(map(int, span_strs.split(Config.span_joiner)))
        for c in range(span[0], span[1]):
            characters.append(c)

    return characters


def parse_token_ids(s, invisible_ids):
    """
     Method to parse the token ids (instead of a span).
    :param s: The input token id field string.
    :param invisible_ids: Ids that should be regarded as invisible.
    :return: The token ids and filtered token ids.
    """
    filtered_token_ids = set()
    original_token_ids = s.split(Config.token_joiner)
    for token_id in original_token_ids:
        if token_id not in invisible_ids:
            filtered_token_ids.add(token_id)
        else:
            logger.debug("Token Id %s is filtered" % token_id)
            pass
    return filtered_token_ids, original_token_ids


def parse_line(l, invisible_ids):
    """
    Parse the line, get the token ids, remove invisible ones.
    :param l: A line in the tbf file.
    :param invisible_ids: Set of invisible ids to remove.
    """
    fields = l.split("\t")
    num_attributes = len(Config.attribute_names)
    if len(fields) < 5 + num_attributes:
        utils.terminate_with_error("System line has too few fields:\n ---> %s" % l)

    if MutableConfig.eval_mode == EvalMethod.Token:
        spans, original_spans = parse_token_ids(fields[3], invisible_ids)
        if len(spans) == 0:
            logger.warn("Find mention with only invisible words, will not be mapped to anything")
    else:
        # There is no filtering thing in the character mode.
        spans = parse_characters(fields[3])
        original_spans = spans

    attributes = [canonicalize_string(a) for a in fields[5:5 + num_attributes]]

    if EvalState.white_listed_types:
        if attributes[0] not in EvalState.white_listed_types:
            return None

    event_id = fields[2]
    text = fields[4]
    # span_id = fields[script_column] if len(fields) > script_column else None

    return spans, attributes, event_id, original_spans, text


def canonicalize_string(str):
    if Config.canonicalize_types:
        return "".join(c.lower() for c in str if c.isalnum())
        # return "".join(str.lower().split()).translate(string.maketrans("", ""), string.punctuation)
    else:
        return str


def span_overlap(span1, span2):
    """
    Compute the number of characters that overlaps
    :param span1:
    :param span2:
    :return: number of overlapping spans
    """
    characters1 = set()
    characters2 = set()
    for s in span1:
        for i in range(s[0], s[1]):
            characters1.add(i)
    for s in span2:
        for i in range(s[0], s[1]):
            characters2.add(i)

    return compute_dice(characters1, characters2)


def compute_token_overlap_score(g_tokens, s_tokens):
    """
    token based overlap score
    It is a set F1 score, which is the same as Dice coefficient
    :param g_tokens: Gold tokens
    :param s_tokens: System tokens
    :return: The Dice Coefficient between two sets of tokens
    """
    return compute_dice(g_tokens, s_tokens)


def compute_dice(items1, items2):
    if len(items1) + len(items2) == 0:
        return 0
    intersect = set(items1).intersection(set(items2))
    return 2.0 * len(intersect) / (len(items1) + len(items2))


def compute_overlap_score(system_outputs, gold_annos):
    return compute_dice(system_outputs, gold_annos)


def get_attr_combinations(attr_names):
    """
    Generate all possible combination attributes.
    :param attr_names: List of attribute names
    :return:
    """
    attribute_names_with_id = list(enumerate(attr_names))
    comb = []
    for L in range(1, len(attribute_names_with_id) + 1):
        comb.extend(itertools.combinations(attribute_names_with_id, L))
    logger.debug("Will score on the following attribute combinations : ")
    logger.debug(", ".join([str(x) for x in comb]))
    return comb


def attribute_based_match(target_attributes, gold_attrs, sys_attrs, doc_id):
    """
    Return whether the two sets of attributes match on all the given attributes
    :param target_attributes: The target attributes to check
    :param gold_attrs: Gold standard attributes
    :param sys_attrs: System response attributes
    :param doc_id: Document ID, used mainly for logging
    :return: True if two sets of attributes matches on given attributes
    """
    for (attribute_index, attribute_name) in target_attributes:
        gold_attr = canonicalize_string(gold_attrs[attribute_index])
        if gold_attr == canonicalize_string(Config.missing_attribute_place_holder):
            logger.warning(
                "Found one attribute [%s] in file [%s] not annotated, give full credit to all system." % (
                    Config.attribute_names[attribute_index], doc_id))
            continue
        sys_attr = canonicalize_string(sys_attrs[attribute_index])
        if gold_attr != sys_attr:
            return False
    return True


def write_if_provided(out_file, text):
    if out_file is not None:
        out_file.write(text)


def write_gold_and_system_mappings(system_id, gold_2_system_mapping, gold_table, system_table, diff_out):
    mapped_system_mentions = set()

    for gold_index, (system_index, score) in enumerate(gold_2_system_mapping):
        score_str = "%.2f" % score if gold_index >= 0 and system_index >= 0 else "-"

        gold_info = "-"
        if gold_index != -1:
            gold_spans, gold_attributes, gold_mention_id, gold_origin_spans, text = gold_table[gold_index]
            gold_info = "%s\t%s\t%s\t%s" % (
                gold_mention_id, ",".join(str(x) for x in gold_origin_spans), "\t".join(gold_attributes), text)

        sys_info = "-"
        if system_index != -1:
            system_spans, system_attributes, sys_mention_id, sys_origin_spans, text = system_table[system_index]
            sys_info = "%s\t%s\t%s\t%s" % (
                sys_mention_id, ",".join(str(x) for x in sys_origin_spans), "\t".join(system_attributes), text)
            mapped_system_mentions.add(system_index)

        write_if_provided(diff_out, "%s\t%s\t|\t%s\t%s\n" % (system_id, gold_info, sys_info, score_str))

    # Write out system mentions that does not map to anything.
    for system_index, (system_spans, system_attributes, sys_mention_id, sys_origin_spans, text) in enumerate(
            system_table):
        if system_index not in mapped_system_mentions:
            sys_info = "%s\t%s\t%s\t%s" % (
                sys_mention_id, ",".join(str(x) for x in sys_origin_spans), "\t".join(system_attributes), text)
            write_if_provided(diff_out, "%s\t%s\t|\t%s\t%s\n" % (system_id, "-", sys_info, "-"))


def write_gold_and_system_corefs(diff_out, gold_coref, sys_coref, gold_id_2_text, sys_id_2_text):
    for c in gold_coref:
        write_if_provided(diff_out, "@coref\tgold\t%s\n" %
                          ",".join([c + ":" + gold_id_2_text[c].replace(",", "") for c in c[2]]))
    for c in sys_coref:
        write_if_provided(diff_out, "@coref\tsystem\t%s\n" %
                          ",".join([c + ":" + sys_id_2_text[c].replace(",", "") for c in c[2]]))


def get_tp_greedy(all_gold_system_mapping_scores, all_attribute_combinations, gold_mention_table,
                  system_mention_table, doc_id):
    tp = 0.0  # span only true positive
    attribute_based_tps = [0.0] * len(all_attribute_combinations)  # attribute based true positive

    # For mention only and attribute augmented true positives.
    greedy_all_attributed_mapping = [[(-1, 0)] * len(gold_mention_table) for _ in
                                     xrange(len(all_attribute_combinations))]
    greedy_mention_only_mapping = [(-1, 0)] * len(gold_mention_table)

    # Record already mapped system index for each case.
    mapped_system = set()
    mapped_gold = set()
    mapped_system_with_attributes = [set() for _ in xrange(len(all_attribute_combinations))]
    mapped_gold_with_attributes = [set() for _ in xrange(len(all_attribute_combinations))]

    while len(all_gold_system_mapping_scores) != 0:
        neg_mapping_score, system_index, gold_index = heapq.heappop(all_gold_system_mapping_scores)
        score = -neg_mapping_score
        if system_index not in mapped_system and gold_index not in mapped_gold:
            tp += score
            greedy_mention_only_mapping[gold_index] = (system_index, score)
            mapped_system.add(system_index)
            mapped_gold.add(gold_index)

        # For each attribute combination.
        gold_attrs = gold_mention_table[gold_index][1]
        system_attrs = system_mention_table[system_index][1]
        for attr_comb_index, attr_comb in enumerate(all_attribute_combinations):
            if system_index not in mapped_system_with_attributes[attr_comb_index] and gold_index not in \
                    mapped_gold_with_attributes[attr_comb_index]:
                if attribute_based_match(attr_comb, gold_attrs, system_attrs, doc_id):
                    attribute_based_tps[attr_comb_index] += score
                    greedy_all_attributed_mapping[attr_comb_index][gold_index] = (system_index, score)
                    mapped_system_with_attributes[attr_comb_index].add(system_index)
                    mapped_gold_with_attributes[attr_comb_index].add(gold_index)
    return tp, attribute_based_tps, greedy_mention_only_mapping, greedy_all_attributed_mapping


def per_type_eval(system_mention_table, gold_mention_table, type_mapping):
    """
    Accumulate per type statistics.
    :param system_mention_table:
    :param gold_mention_table:
    :param type_mapping:
    :return:
    """
    # print type_mapping

    for gold_index, (sys_index, score) in enumerate(type_mapping):
        attributes = gold_mention_table[gold_index][1]
        mention_type = attributes[0]

        # print sys_index, gold_index, score
        # print "Gold", gold_mention_table[gold_index]
        # print "System", system_mention_table[sys_index]

        if sys_index >= 0:
            utils.put_or_increment(EvalState.per_type_tp, mention_type, score)

    for gold_row in gold_mention_table:
        attributes = gold_row[1]
        mention_type = attributes[0]
        utils.put_or_increment(EvalState.per_type_num_gold, mention_type, 1)

    for sys_row in system_mention_table:
        attributes = sys_row[1]
        mention_type = attributes[0]
        utils.put_or_increment(EvalState.per_type_num_response, mention_type, 1)

        # print EvalState.per_type_tp
        # print EvalState.per_type_num_gold
        # print EvalState.per_type_num_response
        #
        # sys.stdin.readline()


def summarize_type_scores():
    """
    Calculate the overall type scores from the accumulated statistics.
    :return:
    """
    per_type_precision = {}
    per_type_recall = {}
    per_type_f1 = {}

    for mention_type, num_gold in EvalState.per_type_num_gold.iteritems():
        tp = utils.get_or_else(EvalState.per_type_tp, mention_type, 0)
        num_sys = utils.get_or_else(EvalState.per_type_num_response, mention_type, 0)
        prec = safe_div(tp, num_sys)
        recall = safe_div(tp, num_gold)
        f_score = safe_div(2 * prec * recall, prec + recall)
        per_type_precision[mention_type] = prec
        per_type_recall[mention_type] = recall
        per_type_f1[mention_type] = f_score

    return per_type_precision, per_type_recall, per_type_f1


def evaluate(token_dir, coref_out, all_attribute_combinations, token_offset_fields, token_file_ext, diff_out):
    """
    Conduct the main evaluation steps.
    :param token_dir:
    :param coref_out:
    :param all_attribute_combinations:
    :param token_offset_fields:
    :param token_file_ext:
    :param diff_out:
    :return:
    """
    if EvalState.has_next_doc():
        res, (g_mention_lines, g_relation_lines), (
            s_mention_lines, s_relation_lines), doc_id, system_id = get_next_doc()
    else:
        return False

    logger.info("Evaluating Document %s" % doc_id)

    if len(g_mention_lines) == 0:
        logger.warn(
            "[%s] does not contain gold standard mentions. Document level F score will not be valid, but the micro "
            "score will be fine." % doc_id)

    invisible_ids = []
    if MutableConfig.eval_mode == EvalMethod.Token:
        invisible_ids, id2token, id2span = read_token_ids(token_dir, doc_id, token_file_ext, token_offset_fields)

    # Parse the lines and save them as a table from id to content.
    system_mention_table = []
    gold_mention_table = []

    # Save the raw text for visualization.
    sys_id_2_text = {}
    gold_id_2_text = {}

    logger.debug("Reading gold and response mentions.")

    remaining_sys_ids = set()
    num_system_mentions = 0
    for sl in s_mention_lines:
        parse_result = parse_line(sl, invisible_ids)

        # If parse result is rejected, we ignore this line.
        if not parse_result:
            continue

        num_system_mentions += 1

        sys_attributes = parse_result[1]
        sys_mention_id = parse_result[2]
        text = parse_result[4]

        system_mention_table.append(parse_result)
        EvalState.all_possible_types.add(sys_attributes[0])
        remaining_sys_ids.add(sys_mention_id)
        sys_id_2_text[sys_mention_id] = text

    if not num_system_mentions == len(remaining_sys_ids):
        logger.warn("Duplicated mention id for doc %s, one of them is randomly removed." % doc_id)

    remaining_gold_ids = set()
    for gl in g_mention_lines:
        parse_result = parse_line(gl, invisible_ids)

        # If parse result is rejected, we ignore this line.
        if not parse_result:
            continue

        gold_attributes = parse_result[1]
        gold_mention_id = parse_result[2]
        text = parse_result[4]

        gold_mention_table.append(parse_result)
        EvalState.all_possible_types.add(gold_attributes[0])
        gold_id_2_text[gold_mention_id] = text
        remaining_gold_ids.add(gold_mention_id)

    num_system_predictions = len(system_mention_table)
    num_gold_predictions = len(gold_mention_table)

    # Store list of mappings with the score as a priority queue. Score is stored using negative for easy sorting.
    all_gold_system_mapping_scores = []

    # Debug purpose printing.
    print_score_matrix = False

    logger.debug("Computing overlap scores.")
    for system_index, (sys_spans, sys_attributes, sys_mention_id, _, _) in enumerate(system_mention_table):
        if print_score_matrix:
            print system_index, sys_mention_id,
        for index, (gold_spans, gold_attributes, gold_mention_id, _, _) in enumerate(gold_mention_table):
            if len(gold_spans) == 0:
                logger.warning("Found empty span gold standard at doc : %s, mention : %s" % (doc_id, gold_mention_id))
            if len(sys_spans) == 0:
                logger.warning("Found empty span system standard at doc : %s, mention : %s" % (doc_id, sys_mention_id))

            overlap = compute_overlap_score(gold_spans, sys_spans)

            if print_score_matrix:
                print "%.1f" % overlap,

            if overlap > 0:
                # maintaining a max heap based on overlap score
                heapq.heappush(all_gold_system_mapping_scores, (-overlap, system_index, index))
        if print_score_matrix:
            print

    greedy_tp, greedy_attribute_tps, greedy_mention_only_mapping, greedy_all_attribute_mapping = get_tp_greedy(
        all_gold_system_mapping_scores, all_attribute_combinations, gold_mention_table,
        system_mention_table, doc_id)

    write_if_provided(diff_out, Config.bod_marker + " " + doc_id + "\n")
    if diff_out is not None:
        # Here if you change the mapping used, you will see what's wrong on different level!

        # write_gold_and_system_mappings(doc_id, system_id, greedy_all_attribute_mapping[0], gold_mention_table,
        #                                system_mention_table, diff_out)

        write_gold_and_system_mappings(system_id, greedy_mention_only_mapping, gold_mention_table, system_mention_table,
                                       diff_out)

    attribute_based_fps = [0.0] * len(all_attribute_combinations)
    for attribute_comb_index, abtp in enumerate(greedy_attribute_tps):
        attribute_based_fps[attribute_comb_index] = num_system_predictions - abtp

    # Unmapped system mentions and the partial scores are considered as false positive.
    fp = len(remaining_sys_ids) - greedy_tp

    EvalState.doc_mention_scores.append((greedy_tp, fp, zip(greedy_attribute_tps, attribute_based_fps),
                                         num_gold_predictions, num_system_predictions, doc_id))

    # Select a computed mapping, we currently select the mapping based on mention type. This means that in order to get
    # coreference right, your mention type should also be right. This can be changed by change Config.coref_criteria
    # settings.
    mention_mapping = None
    type_mapping = None
    for attribute_comb_index, attribute_comb in enumerate(all_attribute_combinations):
        if attribute_comb == Config.coref_criteria:
            mention_mapping = greedy_all_attribute_mapping[attribute_comb_index]
            logger.debug("Select mapping that matches criteria [%s]" % (Config.coref_criteria[0][1]))
        if attribute_comb[0][1] == "mention_type":
            type_mapping = greedy_all_attribute_mapping[attribute_comb_index]

    if Config.coref_criteria == "span_only":
        mention_mapping = greedy_mention_only_mapping

    if mention_mapping is None:
        # In case when we don't do attribute scoring.
        mention_mapping = greedy_mention_only_mapping

    # Evaluate how the performance of each type.
    per_type_eval(system_mention_table, gold_mention_table, type_mapping)

    gold_directed_relations, gold_corefs = utils.parse_relation_lines(g_relation_lines, remaining_gold_ids)
    sys_directed_relations, sys_corefs = utils.parse_relation_lines(s_relation_lines, remaining_sys_ids)

    # # Parse relations.
    # g_relations = [utils.parse_relation_line(l) for l in g_relation_lines]
    # s_relations = [utils.parse_relation_line(l) for l in s_relation_lines]
    #
    # if EvalState.white_listed_types:
    #     g_relations = filter_relations(g_relations, remaining_gold_ids)
    #     s_relations = filter_relations(s_relations, remaining_sys_ids)
    #
    #
    # gold_relations_by_type = separate_relations(g_relations)
    # sys_relations_by_type = separate_relations(s_relations)
    #
    # # Evaluate other directed links.
    # gold_directed_relations = {}
    # sys_directed_relations = {}
    #
    # for name in Config.directed_relations:
    #     if name in gold_relations_by_type:
    #         gold_directed_relations[name] = gold_relations_by_type[name]
    #
    #     if name in sys_relations_by_type:
    #         sys_directed_relations[name] = sys_relations_by_type[name]
    #
    # gold_corefs = []
    # if Config.coreference_relation_name in gold_relations_by_type:
    #     gold_corefs = gold_relations_by_type[Config.coreference_relation_name]
    #
    # sys_corefs = []
    # if Config.coreference_relation_name in sys_relations_by_type:
    #     sys_corefs = sys_relations_by_type[Config.coreference_relation_name]

    if Config.script_result_dir:
        seq_eval = TemporalEval(mention_mapping, gold_mention_table, gold_directed_relations, system_mention_table,
                                sys_directed_relations, gold_corefs, sys_corefs)

        if not Config.no_script_validation:
            if not seq_eval.validate_gold():
                logger.error("The gold edges cannot form a valid script graph.")
                utils.exit_on_fail()

            if not seq_eval.validate_sys():
                logger.error("The system edges cannot form a valid script graph.")
                utils.exit_on_fail()

        seq_eval.write_time_ml(doc_id)

    # Evaluate coreference links.
    if coref_out is not None:
        logger.debug("Start preparing coreference files.")

        # Prepare CoNLL style coreference input for this document.
        conll_converter = ConllEvaluator(doc_id, system_id, sys_id_2_text, gold_id_2_text)
        gold_conll_lines, sys_conll_lines = conll_converter.prepare_conll_lines(gold_corefs, sys_corefs,
                                                                                gold_mention_table,
                                                                                system_mention_table,
                                                                                mention_mapping,
                                                                                MutableConfig.coref_mention_threshold)

        # If we are selecting among multiple mappings, it is easy to write in our file.
        write_mode = 'w' if EvalState.claim_write_flag() else 'a'
        g_conll_out = open(Config.conll_gold_file, write_mode)
        s_conll_out = open(Config.conll_sys_file, write_mode)
        g_conll_out.writelines(gold_conll_lines)
        s_conll_out.writelines(sys_conll_lines)

        if diff_out is not None:
            write_gold_and_system_corefs(diff_out, gold_corefs, sys_corefs, gold_id_2_text, sys_id_2_text)

    write_if_provided(diff_out, Config.eod_marker + " " + "\n")

    return True


def natural_order(key):
    """
    Compare order based on the numeric values in key, for example, 't1 < t2'
    :param key:
    :return:
    """
    if type(key) is int:
        return key
    convert = lambda text: int(text) if text.isdigit() else text
    return [convert(c) for c in re.split('([0-9]+)', key)]


if __name__ == "__main__":
    main()
