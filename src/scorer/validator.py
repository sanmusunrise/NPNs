#!/usr/bin/python

"""
    A simple validator that check whether the format can at be parsed by the scorer
    It might not be able to detect all the problems so please double-check the results
    carefully.

    Author: Zhengzhong Liu ( liu@cs.cmu.edu )
"""

import argparse
import logging
import os
import re
import sys
from temporal import TemporalEval
from config import Config, EvalMethod, MutableConfig
import utils

logger = logging.getLogger()

comment_marker = "#"
bod_marker = "#BeginOfDocument"  # mark begin of a document
eod_marker = "#EndOfDocument"  # mark end of a document
relation_marker = "@"  # mark start of a relation

conll_bod_marker = "#begin document"
conll_eod_marker = "#end document"

default_token_file_ext = ".tab"
default_token_offset_fields = [2, 3]

# run this on an annotation to confirm
invisible_words = set()

# attribute names
attribute_names = ["mention_type", "realis_status"]

gold_docs = {}
doc_ids_to_score = []
all_possible_types = set()
evaluating_index = 0

token_joiner = ","
span_separator = ";"
span_joiner = ","

missingAttributePlaceholder = "NOT_ANNOTATED"

total_mentions = 0

unrecognized_relation_count = 0
total_tokens_not_found = 0


def exit_on_fail():
    logger.error("Validation failed.")
    logger.error("Please fix the warnings/errors.")
    sys.exit(255)


def main():
    parser = argparse.ArgumentParser(
        description="The validator check whether the supplied 'tbf' file follows assumed structure . The validator"
                    " will exit at status 255 if any errors are found, validation logs will be written at the same "
                    "directory of the validator with 'errlog' as extension.")
    parser.add_argument("-s", "--system", help="System output", required=True)
    parser.add_argument("-tm", "--token_mode", help="Token mode, default is false.", action="store_true")
    parser.add_argument(
        "-t", "--token_path", help="Path to the directory containing the token mappings file, only in token mode.")
    parser.add_argument(
        "-of", "--offset_field", help="A pair of integer indicates which column we should read the offset in the token "
                                      "mapping file, index starts at 0, default value will be %s. Only used in token "
                                      "mode." % default_token_offset_fields
    )
    parser.add_argument(
        "-te", "--token_table_extension",
        help="any extension appended after docid of token table files. Default is [%s]" % default_token_file_ext)
    parser.add_argument(
        "-wc", "--word_count_file",
        help="A word count file that can be used to help validation, such as the character_counts.tsv in LDC2016E64."
    )
    parser.add_argument("-ty", "--type_file",
                        help="If provided, the validator will check whether the type subtype pair is valid.")

    parser.add_argument(
        "-b", "--debug", help="turn debug mode on", action="store_true")

    parser.set_defaults(debug=False)
    args = parser.parse_args()
    validator_log = os.path.basename(args.system) + ".errlog"
    handler = logging.FileHandler(validator_log)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if os.path.isfile(args.system):
        sf = open(args.system)
    else:
        logger.error("Cannot find system file at " + args.system)
        exit_on_fail()

    doc_lengths = None
    if args.word_count_file is not None and os.path.isfile(args.word_count_file):
        doc_lengths = get_document_length(open(args.word_count_file))
    else:
        logger.warn("Word count file not provided, will not validate document id.")

    possible_types = None
    if args.type_file is not None and os.path.isfile(args.type_file):
        possible_types = read_type_file(open(args.type_file))
    else:
        logger.warn("Will not validate mention type, all type will be considered valid.")

    if args.debug:
        handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    # token based eval mode
    MutableConfig.eval_mode = EvalMethod.Token if args.token_mode else  EvalMethod.Char

    token_dir = "."
    if MutableConfig.eval_mode == EvalMethod.Token:
        if args.token_path is not None:
            if os.path.isdir(args.token_path):
                logger.debug("Will search token files in " + args.token_path)
                token_dir = args.token_path
            else:
                logger.debug("Cannot find given token directory at [%s], "
                             "will try search for current directory" % args.token_path)

    token_offset_fields = default_token_offset_fields
    if args.offset_field is not None:
        try:
            token_offset_fields = [int(x) for x in args.offset_field.split(",")]
        except ValueError as _:
            logger.error("Should provide two integer with comma in between")

    if not read_all_doc(sf):
        exit_on_fail()

    validation_success = True
    while has_next_doc():
        if not validate_next(doc_lengths, possible_types, token_dir, token_offset_fields,
                             args.token_table_extension):
            validation_success = False

    if not validation_success:
        exit_on_fail()
    else:
        logger.info("Validation did not find obvious errors.")

    logger.info("Submission contains %d files, %d mentions" % (len(gold_docs), total_mentions))
    logger.info("Validation Finished.")


def read_type_file(type_file):
    all_types = set()
    for line in type_file:
        all_types.add("".join(line.split()).lower())
    return all_types


def get_document_length(wc):
    doc_lengths = {}
    for line in wc:
        fields = line.split()
        doc_lengths[fields[0]] = int(fields[1])
    return doc_lengths


def read_token_ids(token_dir, g_file_name, provided_token_ext, token_offset_fields):
    tf_ext = default_token_file_ext if provided_token_ext is None else provided_token_ext
    invisible_ids = set()
    id2token_map = {}
    id2span_map = {}
    token_file_path = os.path.join(token_dir, g_file_name + tf_ext)

    logger.debug("Reading token for " + g_file_name)
    try:
        token_file = open(token_file_path)
        # discard the header
        header = token_file.readline()

        for tline in token_file:
            fields = tline.rstrip().split("\t")
            if len(fields) < 4:
                logger.error("Token line should have 4 fields, found the following : ")
                logger.error(tline)
                continue

            token = fields[1].lower().strip().rstrip()
            token_id = fields[0]
            id2token_map[token_id] = token

            try:
                token_span = (int(fields[token_offset_fields[0]]), int(fields[token_offset_fields[1]]))
                id2span_map[token_id] = token_span
            except ValueError as e:
                logger.error("Cannot find field %s and %s in token file %s in the following line: " % (
                    token_offset_fields[0], token_offset_fields[1], token_file))
                logger.error(tline)
            if token in invisible_words:
                invisible_ids.add(token_id)
    except IOError:
        logger.error("Cannot find token file for doc [%s] at [%s], did you use correct file paths?" % (
            g_file_name, token_file_path))
        pass
    return invisible_ids, id2token_map, id2span_map


def read_all_doc(gf):
    global gold_docs
    global doc_ids_to_score
    gold_docs = read_docs_with_doc_id(gf)
    if "" in gold_docs:
        gold_docs.pop("")
    g_doc_ids = gold_docs.keys()
    g_id_set = set(g_doc_ids)

    doc_ids_to_score = sorted(g_id_set)

    if len(g_doc_ids) == 0:
        logger.error("No document id found for [%s], please check begin and end marker.")
        return False
    return True


def check_unique(keys):
    return len(keys) == len(set(keys))


def read_docs_with_doc_id(f):
    all_docs = {}
    mention_lines = []
    relation_lines = []
    doc_id = ""
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip().rstrip()

        if line.startswith(comment_marker):
            if line.startswith(bod_marker):
                doc_id = line[len(bod_marker):].strip()
            elif line.startswith(eod_marker):
                all_docs[doc_id] = mention_lines, relation_lines
                mention_lines = []
                relation_lines = []
        elif line.startswith(relation_marker):
            relation_lines.append(line[len(relation_marker):].strip())
        elif line == "":
            pass
        else:
            mention_lines.append(line)

    return all_docs


def has_next_doc():
    return evaluating_index < len(doc_ids_to_score)


def get_next_doc():
    global gold_docs
    global evaluating_index
    if evaluating_index < len(doc_ids_to_score):
        doc_id = doc_ids_to_score[evaluating_index]
        evaluating_index += 1
        return True, gold_docs[doc_id], ([], []), doc_id
    else:
        logger.error("Reaching end of all documents")
        return False, ([], []), ([], []), "End_Of_Documents"


def parse_token_ids(s, invisible_ids):
    """
    Method to parse the token ids (instead of a span)
    """
    filtered_token_ids = set()
    for token_id in s.split(token_joiner):
        if token_id not in invisible_ids:
            filtered_token_ids.add(token_id)
        else:
            logger.debug("Token Id %s is filtered" % token_id)
            pass
    return filtered_token_ids


def parse_characters(s):
    """
    Method to parse the character based span
    :param s:
    """
    span_strs = s.split(span_separator)
    characters = []
    for span_strs in span_strs:
        span = list(map(int, span_strs.split(span_joiner)))
        for c in range(span[0], span[1]):
            characters.append(c)

    return characters


def parse_line(l, invisible_ids):
    fields = l.split("\t")
    num_attributes = len(attribute_names)
    min_len = num_attributes + 5
    if len(fields) < min_len:
        logger.error("System line has too few fields:\n ---> %s" % l)
        exit_on_fail()

    if MutableConfig.eval_mode == EvalMethod.Token:
        spans, original_spans = parse_token_ids(fields[3], invisible_ids)
        if len(spans) == 0:
            logger.warn("Find mention with only invisible words, will not be mapped to anything")
    else:
        spans = parse_characters(fields[3])

    return fields[2], spans, fields[5:]


# def parse_relation(relation_line):
#     parts = relation_line.split("\t")
#     if len(parts) < 3:
#         logger.error("Relation should have at least 3 fields, found the following:")
#         logger.error(parts)
#         exit_on_fail()
#     relation_arguments = parts[2].split(",")
#     return parts[0], parts[1], relation_arguments


def natural_order(key):
    convert = lambda text: int(text) if text.isdigit() else text
    return [convert(c) for c in re.split('([0-9]+)', key)]


def get_eid_2_sorted_token_map(mention_table):
    event_mention_id_2_sorted_tokens = {}
    for mention in mention_table:
        tokens = sorted(mention[0], key=natural_order)
        event_id = mention[2]
        event_mention_id_2_sorted_tokens[event_id] = tokens
    return event_mention_id_2_sorted_tokens


def get_eid_2_character_span(mention_table):
    event_mention_id_2_span = {}
    for mention in mention_table:
        spans = sorted(mention[0])
        event_id = mention[2]
        event_mention_id_2_span[event_id] = spans
    return event_mention_id_2_span


def validate_next(doc_lengths, possible_types, token_dir, token_offset_fields, token_file_ext):
    global total_mentions
    global unrecognized_relation_count

    success = True

    res, (mention_lines, relation_lines), (_, _), doc_id = get_next_doc()

    max_length = None
    if doc_lengths is not None:
        if doc_id not in doc_lengths:
            logger.error("Document id not listed in evaluation set : %s", doc_id)
            success = False
        else:
            max_length = doc_lengths[doc_id]

    if MutableConfig.eval_mode == EvalMethod.Token:
        invisible_ids, id2token_map, id2span_map = read_token_ids(token_dir, doc_id, token_file_ext,
                                                                  token_offset_fields)
    else:
        invisible_ids = set()
        id2token_map = {}

    # Parse the lines in file.
    mention_table = []

    mention_ids = []
    remaining_gold_ids = set()

    for l in mention_lines:
        mention_id, spans, attributes = parse_line(l, invisible_ids)

        if max_length is not None and not check_range(spans, max_length):
            logger.error(
                "The following mention line exceed the character range %d of document [%s]" % (max_length, doc_id))
            logger.error(l)
            success = False

        if possible_types is not None:
            mtype = canonicalize_string(attributes[0])
            if not check_type(possible_types, mtype):
                logger.error("Submission contains type [%s] that is not in evaluation." % mtype)
                success = False

        mention_table.append((spans, attributes, mention_id))
        mention_ids.append(mention_id)
        all_possible_types.add(attributes[0])
        remaining_gold_ids.add(mention_id)

    total_mentions += len(mention_table)

    if not check_unique(mention_ids):
        logger.error("Duplicated mention id for doc %s" % doc_id)
        success = False

    if MutableConfig.eval_mode == EvalMethod.Token and has_invented_token(id2token_map, mention_table):
        logger.error("Invented token id was found for doc %s" % doc_id)
        logger.error("Tokens not in tbf not found in token map : %d" % total_tokens_not_found)
        success = False

    clusters = {}
    cluster_id = 0
    for l in relation_lines:
        relation = utils.parse_relation_line(l)
        if relation[0] == Config.coreference_relation_name:
            clusters[cluster_id] = set(relation[2])
            cluster_id += 1
        elif relation[0] not in Config.all_relations:
            unrecognized_relation_count += 1
            logger.warning("Relation [%s] is not recognized, this task only takes: [%s]", relation[0],
                           ";".join(Config.all_relations))

    if unrecognized_relation_count > 10:
        logger.error("Too many unrecognized relations : %d" % unrecognized_relation_count)
        success = False

    if transitive_not_resolved(clusters):
        logger.error("Coreference transitive closure is not resolved! Please resolve before submitting.")
        logger.error("Problem was found in file %s" % doc_id)
        success = False

    if EvalMethod.Char:
        event_mention_id_2_span = get_eid_2_character_span(mention_table)
    else:
        event_mention_id_2_span = get_eid_2_sorted_token_map(mention_table)

    for cluster_id, cluster in clusters.iteritems():
        if invented_mention_check(cluster, event_mention_id_2_span):
            logger.error("Found invented id in clusters at doc [%s]" % doc_id)
            success = False

    directed_relations, corefs = utils.parse_relation_lines(relation_lines, remaining_gold_ids)

    seq_eval = TemporalEval([], mention_table, directed_relations, [], {}, corefs, [])
    if not seq_eval.validate_gold():
        logger.error("The edges cannot form a valid script graph.")
        utils.exit_on_fail()

    return success


def canonicalize_string(str):
    return "".join(c.lower() for c in str if c.isalnum())


def check_type(possible_types, mtype):
    if mtype not in possible_types:
        return False
    return True


def check_range(spans, max_length):
    for span in spans:
        if span < 0 or span >= max_length:
            return False
    return True


def has_invented_token(id2token_map, gold_mention_table):
    for gold_mention in gold_mention_table:
        spans = gold_mention[0]
        for tid in spans:
            if tid not in id2token_map:
                logger.error("Token Id [%s] is not in the given token map" % tid)
                return True
    return False


def invented_mention_check(cluster, event_mention_id_2_sorted_tokens):
    for eid in cluster:
        if not eid in event_mention_id_2_sorted_tokens:
            logger.error("Cluster contains ID not in event mention list [%s]" % eid)
            return True
        else:
            return False


def within_cluster_span_duplicate(cluster, event_mention_id_2_sorted_tokens):
    # print cluster
    # print event_mention_id_2_sorted_tokens
    span_map = {}
    for eid in cluster:
        span = tuple(event_mention_id_2_sorted_tokens[eid])
        if span in span_map:
            logger.error("Span within the same cluster cannot be the same.")
            logger.error("%s->[%s]" % (eid, ",".join(span)))
            logger.error("%s->[%s]" % (span_map[span], ",".join(span)))
            return True
        else:
            span_map[span] = eid


def transitive_not_resolved(clusters):
    ids = clusters.keys()
    for i in range(0, len(ids) - 1):
        for j in range(i + 1, len(ids)):
            if len(clusters[i].intersection(clusters[j])) != 0:
                logger.error("Non empty intersection between clusters found.")
                logger.error(clusters[i])
                logger.error(clusters[j])
                return True
    return False


if __name__ == "__main__":
    main()
