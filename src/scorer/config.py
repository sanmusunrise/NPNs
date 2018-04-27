import os


class Config:
    """
    Hold configuration variable for evaluation. These variables
    should not be changed during evaluation.
    """

    def __init__(self):
        pass

    # TBF file formats.
    comment_marker = "#"
    bod_marker = "#BeginOfDocument"  # mark begin of a document
    eod_marker = "#EndOfDocument"  # mark end of a document
    relation_marker = "@"  # mark start of a relation
    coreference_relation_name = "Coreference"  # mark coreference
    after_relation_name = "After"  # mark after

    directed_relations = {"After", "Subevent"}

    all_relations = {"After", "Subevent", "Coreference"}

    token_joiner = ","
    span_seperator = ";"
    span_joiner = ","

    missing_attribute_place_holder = "NOT_ANNOTATED"

    default_token_file_ext = ".tab"
    default_token_offset_fields = [2, 3]

    # We should probably remove this as a whole.
    invisible_words = {}

    # Attribute names, these are the same order as they appear in submissions.
    attribute_names = ["mention_type", "realis_status"]

    # Conll related settings.
    conll_bod_marker = "#begin document"
    conll_eod_marker = "#end document"

    conll_gold_file = None
    conll_sys_file = None

    conll_out = None

    # By default, this reference scorer is shipped with the script. We do it this way so that we can call the script
    # successfully from outside scripts.
    relative_perl_script_path = "/reference-coreference-scorers-8.01/scorer.pl"
    conll_scorer_executable = os.path.dirname(os.path.realpath(__file__)) + relative_perl_script_path

    skipped_metrics = {"ceafm"}

    zero_for_empty_metrics = {"muc"}

    token_miss_msg = "Token ID [%s] not found in token list, the score file provided is incorrect."

    coref_criteria = ((0, "mention_type"),)

    possible_coref_mapping = [((-1, "span_only"),),
                              ((0, "mention_type"),), ((1, "realis_status"),),
                              ((0, "mention_type"), (1, "realis_status"))]

    canonicalize_types = True

    # script link settings.

    script_result_dir = None

    script_gold_dir = "gold"

    script_sys_dir = "sys"

    script_out = "seq.out"

    script_out_cluster = "seq_cluster.out"

    temp_eval_executable = os.path.dirname(os.path.realpath(__file__)) + "/evaluation-relations/temporal_evaluation.py"

    no_script_validation = False

    script_types = ["Subevent", "After"]

    eval_cluster_level_links = False


class EvalMethod:
    """
    Two different evaluation methods
    Char based evaluation is not supported and is only here for legacy reasons.
    """

    def __init__(self):
        pass

    Token, Char = range(2)


class MutableConfig:
    """
    Some configuration that might be changed at setup. Default
    values are set here. Do not modify these variables outside
    the Main() function (i.e. outside the setup stage)
    """

    def __init__(self):
        pass

    remove_conll_tmp = False
    eval_mode = EvalMethod.Char
    coref_mention_threshold = 1.0


class EvalState:
    """
    Hold evaluation state variables.
    """

    def __init__(self):
        pass

    gold_docs = {}
    system_docs = {}
    doc_ids_to_score = []
    all_possible_types = set()
    evaluating_index = 0

    doc_mention_scores = []
    doc_coref_scores = []
    overall_coref_scores = {}

    per_type_tp = {}
    per_type_num_response = {}
    per_type_num_gold = {}

    use_new_conll_file = True

    system_id = "_id_"

    white_listed_types = None

    @staticmethod
    def advance_index():
        EvalState.evaluating_index += 1

    @staticmethod
    def has_next_doc():
        return EvalState.evaluating_index < len(EvalState.doc_ids_to_score)

    @staticmethod
    def claim_write_flag():
        r = EvalState.use_new_conll_file
        EvalState.use_new_conll_file = False
        return r
