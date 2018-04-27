#!/usr/bin/python
from scorer_v1_8 import *


def transform_to_score_list(golden_dict):
    lines = []
    s = "SunriseSystem\t"
    e_cnt = 0
    doc2data = {}
    for key in golden_dict:
        for label in golden_dict[key]:
            doc_id,offset,length = key
            if not doc_id in doc2data:
                doc2data[doc_id] = []
            doc2data[doc_id].append((doc_id,offset,length,label))
    
    for d in doc2data:
        lines.append("#BeginOfDocument " + d + "\n")
        for doc_id,offset,length,label in doc2data[d]:
            ss = s
            ss += str(doc_id) + "\t"
            ss += "E" + str(e_cnt) + "\t"
            ss += str(offset) + "," + str(offset + length) + "\t"
            ss += "token" + "\t"
            ss += label + "\t" + "actual" + "\n"
            lines.append(ss)
            e_cnt +=1
            #print ss
        lines.append("#EndOfDocument\n")
    return lines
            
def score(glod_list, system_list):
    """
    :param
    same format in glod_list, system_list; each line is an element in list
    split symbol in one line: \t
    #BeginOfDocument docID
    SystemName  docID   eventId start,end   token   Type    RealisType
    SystemName  docID   eventId start,end   token   Type    RealisType
    @Coreference    relationId      eventId,eventId,...
    #EndOfDocument

    For Example:
    ['#BeginOfDocument sample',
     'sys     sample  E1      2,8     murder  Conflict_Attack Actual',
     'sys     sample  E2      12,16   kill    Conflict_Attack Actual',
     '@Coreference    R1      E1,E2',
     '#EndOfDocument',]

    :return:
    each_doc_result { 'docId1':
                        {'plain': (Pre, Rec, F1),
                         'mention_type': (Pre, Rec, F1),
                         'realis_status': (Pre, Rec, F1),
                         'mention_type+realis_status': (Pre, Rec, F1),
                         },
                      'docId2': ...
                    }

    type_result     {'typename1': (Pre, Rec, F1), 'typename2': (Pre, Rec, F1), ...}
    final_result    {'plain': {'micro': (Pre, Rec, F1), 'macro': (Pre, Rec, F1)},
                     'mention_type': {'micro': (Pre, Rec, F1), 'macro': (Pre, Rec, F1)},
                     'realis_status': {'micro': (Pre, Rec, F1), 'macro': (Pre, Rec, F1)},
                     'mention_type+realis_status': {'micro': (Pre, Rec, F1), 'macro': (Pre, Rec, F1)},
                     }
    """
    MutableConfig.eval_mode = EvalMethod.Char
    # Read all documents.
    read_all_doc_from_list(glod_list, system_list, None)

    # Take all attribute combinations, which will be used to produce scores.
    attribute_comb = get_attr_combinations(Config.attribute_names)

    while True:
        if not evaluate(None, None, attribute_comb,
                        None, None,
                        None):
            break

    each_doc_result, type_result, final_result = get_eval_results(attribute_comb)

    # Renew EvalState
    EvalState.gold_docs = {}
    EvalState.system_docs = {}
    EvalState.doc_ids_to_score = []
    EvalState.all_possible_types = set()
    EvalState.evaluating_index = 0

    EvalState.doc_mention_scores = []
    EvalState.doc_coref_scores = []
    EvalState.overall_coref_scores = {}

    EvalState.per_type_tp = {}
    EvalState.per_type_num_response = {}
    EvalState.per_type_num_gold = {}

    return each_doc_result, type_result, final_result


def get_eval_results(all_attribute_combinations, mention_eval_out=None):
    # mention_eval_out = sys.stdout
    docs_result = dict()
    type_result = dict()
    final_result = dict()

    total_gold_mentions = 0
    total_system_mentions = 0
    valid_docs = 0

    plain_global_scores = [0.0] * 4
    attribute_based_global_scores = [[0.0] * 4 for _ in xrange(len(all_attribute_combinations))]

    doc_id_width = get_cell_width(EvalState.doc_mention_scores)
    if mention_eval_out is not None:
        mention_eval_out.write("========Document Mention Detection Results==========\n")
    small_header_item = "Prec  \tRec  \tF1   "
    attribute_header_list = get_combined_attribute_header(all_attribute_combinations, len(small_header_item))
    small_headers = [small_header_item] * (len(all_attribute_combinations) + 1)
    if mention_eval_out is not None:
        mention_eval_out.write(pad_char_before_until("", doc_id_width) + "\t" + "\t|\t".join(attribute_header_list) + "\n")
        mention_eval_out.write(pad_char_before_until("Doc ID", doc_id_width) + "\t" + "\t|\t".join(small_headers) + "\n")

    for (tp, fp, attribute_based_counts, num_gold_mentions, num_sys_mentions, docId) in EvalState.doc_mention_scores:
        tp *= 100
        fp *= 100
        prec = safe_div(tp, num_sys_mentions)
        recall = safe_div(tp, num_gold_mentions)
        doc_f1 = compute_f1(prec, recall)

        docs_result[docId] = dict()
        docs_result[docId]['plain'] = prec, recall, doc_f1

        attribute_based_doc_scores = []

        for comb_index, comb in enumerate(all_attribute_combinations):
            counts = attribute_based_counts[comb_index]
            attr_tp = counts[0] * 100
            attr_fp = counts[1] * 100
            attr_prec = safe_div(attr_tp, num_sys_mentions)
            attr_recall = safe_div(attr_tp, num_gold_mentions)
            attr_f1 = compute_f1(attr_prec, attr_recall)

            attribute_based_doc_scores.append("%.2f\t%.2f\t%.2f" % (attr_prec, attr_recall, attr_f1))
            attr_name = attribute_header_list[comb_index + 1].strip()
            docs_result[docId][attr_name] = attr_prec, attr_recall, attr_f1

            for score_index, score in enumerate([attr_tp, attr_fp, attr_prec, attr_recall]):
                if not math.isnan(score):
                    attribute_based_global_scores[comb_index][score_index] += score
        if mention_eval_out is not None:
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
        if mention_eval_out is not None:
            for coref_score, doc_id in EvalState.doc_coref_scores:
                mention_eval_out.write("%s\t%.2f\n" % (doc_id, coref_score))

    per_type_precision, per_type_recall, per_type_f1 = summarize_type_scores()

    if mention_eval_out is not None:
        mention_eval_out.write("\n\n========Mention Type Results==========\n")
    if len(per_type_f1) > 0:
        max_type_name_width = len(max(per_type_f1.keys(), key=len))
        if mention_eval_out is not None:
            mention_eval_out.write("%s\tPrec\tRec\tF1\t#Gold\t#Sys\n" % pad_char_before_until("Type", max_type_name_width))
        for mention_type, f1 in sorted(per_type_f1.items()):
            prec = utils.nan_as_zero(utils.get_or_else(per_type_precision, mention_type, 0))
            rec = utils.nan_as_zero(utils.get_or_else(per_type_recall, mention_type, 0))
            f1 = utils.nan_as_zero(utils.get_or_else(per_type_f1, mention_type, 0))
            type_result[mention_type] = prec * 100, rec * 100, f1 * 100
            if mention_eval_out is not None:
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
    final_result['plain'] = dict()
    final_result['plain']['micro'] = plain_average_scores[:3]
    final_result['plain']['macro'] = plain_average_scores[3:]

    if mention_eval_out is not None:
        mention_eval_out.write("\n=======Final Mention Detection Results=========\n")
    max_attribute_name_width = len(max(attribute_header_list, key=len))
    attributes_name_header = pad_char_before_until("Attributes", max_attribute_name_width)

    final_result_big_header = ["Micro Average", "Macro Average"]

    if mention_eval_out is not None:
        mention_eval_out.write(
            pad_char_before_until("", max_attribute_name_width, " ") + "\t" + "\t".join(
                [pad_char_before_until(h, len(small_header_item)) for h in final_result_big_header]) + "\n")
        mention_eval_out.write(attributes_name_header + "\t" + "\t".join([small_header_item] * 2) + "\n")
        mention_eval_out.write(pad_char_before_until(attribute_header_list[0], max_attribute_name_width) + "\t" + "\t".join(
            "%.2f" % f for f in plain_average_scores) + "\n")
    for attr_index, attr_based_score in enumerate(attribute_based_global_scores):
        attr_average_scores = get_averages(attr_based_score, total_gold_mentions, total_system_mentions, valid_docs)
        att_name = attribute_header_list[attr_index + 1].strip()
        final_result[att_name] = dict()
        final_result[att_name]['micro'] = attr_average_scores[:3]
        final_result[att_name]['macro'] = attr_average_scores[3:]
        if mention_eval_out is not None:
            mention_eval_out.write(
                pad_char_before_until(attribute_header_list[attr_index + 1],
                                      max_attribute_name_width) + "\t" + "\t".join(
                    "%.2f" % f for f in attr_average_scores) + "\n")

    if len(EvalState.overall_coref_scores) > 0:
        if mention_eval_out is not None:
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
            if mention_eval_out is not None:
                mention_eval_out.write(formatter % (metric, score))
        if mention_eval_out is not None:
            mention_eval_out.write(
                "Overall Average CoNLL score\t%.2f\n" % (conll_sum / num_metric))
            mention_eval_out.write("\n* Score not included for final CoNLL score.\n")

    if Config.script_result_dir is not None:
        if mention_eval_out is not None:
            mention_eval_out.write("\n")

        for eval_type in Config.script_types + ["All"]:
            for filename in os.listdir(os.path.join(Config.script_result_dir, eval_type)):
                script_eval_path = os.path.join(Config.script_result_dir, eval_type, filename)

                if mention_eval_out is not None:
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
        if mention_eval_out is not None:
            mention_eval_out.flush()
        if not mention_eval_out == sys.stdout:
            mention_eval_out.close()
    return docs_result, type_result, final_result


def read_all_doc_from_list(gf_list, sf_list, single_doc_id_to_eval):
    """
    Read all the documents, collect the document ids that are shared by both gold and system. It will populate the
    gold_docs and system_docs, stored as map from doc id to raw annotation strings.

    The document ids considered to be scored are those presented in the gold documents.

    :param gf: Gold standard file
    :param sf:  System response file
    :param single_doc_id_to_eval: If not None, we will evaluate only this doc id.
    :return:
    """
    EvalState.gold_docs = read_docs_with_doc_id_and_name_from_list(gf_list)
    EvalState.system_id = "from_list"
    EvalState.system_docs = read_docs_with_doc_id_and_name_from_list(sf_list)

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


def read_docs_with_doc_id_and_name_from_list(f_list):
    """
    Parse file into a map from doc id to mention and relation raw strings
    :param f: The annotation file
    :return: A map from doc id to corresponding mention and relation annotations, which are stored as raw string
    """
    all_docs = {}
    mention_lines = []
    relation_lines = []
    doc_id = ""
    for line in f_list:
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

    return all_docs


def get_plain_score(gold_list, system_list):
    _, _, result = score(gold_list, system_list)
    return result['plain']

def get_typed_score(gold_list, system_list):
    _, _, result = score(gold_list, system_list)
    return result['mention_type']

def get_realis_score(gold_list, system_list):
    _, _, result = score(gold_list, system_list)
    return result['realis_status']

def get_typed_realis_score(gold_list, system_list):
    _, _, result = score(gold_list, system_list)
    return result['mention_type+realis_status']


if __name__ == "__main__":
    l1 = open(sys.argv[1]).readlines()
    l2 = open(sys.argv[2]).readlines()
    print score(l1, l2)
