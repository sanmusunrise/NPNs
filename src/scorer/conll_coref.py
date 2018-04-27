import logging
import os
import subprocess

import utils
from config import Config

logger = logging.getLogger(__name__)


class ConllEvaluator:
    def __init__(self, doc_id, system_id, sys_id_2_text, gold_id_2_text):
        """
        :param doc_id: the document id
        :param system_id: The id of the participant system
        :param sys_id_2_text
        :param gold_id_2_text
        :return:
        """
        self.doc_id = doc_id
        self.system_id = system_id
        self.sys_id_2_text = sys_id_2_text
        self.gold_id_2_text = gold_id_2_text

    @staticmethod
    def run_conll_script(gold_path, system_path, script_out):
        """
        Run the Conll script and output result to the path given
        :param gold_path:
        :param system_path:
        :param script_out: Path to output the scores
        :return:
        """
        logger.info("Running reference CoNLL scorer.")
        with open(script_out, 'wb', 0) as out_file:
            subprocess.call(
                ["perl", Config.conll_scorer_executable, "all", gold_path, system_path],
                stdout=out_file)
        logger.info("Done running CoNLL scorer.")

    @staticmethod
    def get_conll_scores(score_path):
        metric = "UNKNOWN"

        scores_by_metric = {}

        with open(score_path, 'r') as f:
            for l in f:
                if l.startswith("METRIC"):
                    metric = l.split()[-1].strip().strip(":")
                if l.startswith("Coreference: ") or l.startswith("BLANC: "):
                    f1 = float(l.split("F1:")[-1].strip().strip("%"))
                    scores_by_metric[metric] = f1

        return scores_by_metric

    @staticmethod
    def create_aligned_tables(gold_2_system_one_2_one_mapping, gold_mention_table, system_mention_table,
                              threshold=1.0):
        """
        Create coreference alignment for gold and system mentions by taking an alignment threshold.
        :param gold_2_system_one_2_one_mapping: Gold index to (system index, score) mapping, indexed by gold index.
        :param gold_mention_table:
        :param system_mention_table:
        :param threshold:
        :return:
        """
        aligned_gold_table = []
        aligned_system_table = []

        aligned_system_mentions = set()

        for gold_index, system_aligned in enumerate(gold_2_system_one_2_one_mapping):
            aligned_gold_table.append((gold_mention_table[gold_index][0], gold_mention_table[gold_index][2]))
            if system_aligned is None:
                # Indicate nothing aligned with this gold mention.
                aligned_system_table.append(None)
                continue
            system_index, alignment_score = system_aligned
            if alignment_score >= threshold:
                aligned_system_table.append(
                    (system_mention_table[system_index][0], system_mention_table[system_index][2]))
                aligned_system_mentions.add(system_index)
            else:
                aligned_system_table.append(None)

        for system_index, system_mention in enumerate(system_mention_table):
            # Add unaligned system mentions.
            if system_index not in aligned_system_mentions:
                aligned_gold_table.append(None)
                aligned_system_table.append(
                    (system_mention_table[system_index][0], system_mention_table[system_index][2]))

        return aligned_gold_table, aligned_system_table

    @staticmethod
    def generate_temp_conll_file_base(temp_header, system_id, doc_id):
        return "%s_%s_%s" % (temp_header, system_id, doc_id)

    @staticmethod
    def extract_token_map(mention_table):
        event_mention_id2sorted_tokens = {}
        for mention in mention_table:
            tokens = sorted(mention[0], key=utils.natural_order)
            event_mention_id2sorted_tokens[mention[2]] = tokens
        return event_mention_id2sorted_tokens

    def prepare_conll_lines(self, gold_corefs, sys_corefs, gold_mention_table, system_mention_table,
                            gold_2_system_one_2_one_mapping, threshold=1.0):
        """
        Convert to ConLL style lines
        :param gold_corefs: gold coreference chain
        :param sys_corefs: system coreferenc chain
        :param gold_mention_table:  gold mention table
        :param system_mention_table: system mention table
        :param gold_2_system_one_2_one_mapping: a mapping between gold and system
        :param threshold: To what extent we treat two mention can be aligned, default 1 for exact match
        :return:
        """
        aligned_gold_table, aligned_system_table = self.create_aligned_tables(gold_2_system_one_2_one_mapping,
                                                                              gold_mention_table,
                                                                              system_mention_table,
                                                                              threshold)
        logger.debug("Preparing CoNLL files using mapping threhold %.2f" % threshold)

        gold_conll_lines = self.prepare_lines(gold_corefs, aligned_gold_table, self.gold_id_2_text)

        sys_conll_lines = self.prepare_lines(sys_corefs, aligned_system_table, self.sys_id_2_text)

        if not gold_conll_lines:
            utils.terminate_with_error("Gold standard has data problem for doc [%s], please refer to log. Quitting..."
                                       % self.doc_id)

        if not sys_conll_lines:
            utils.terminate_with_error("System has data problem for doc [%s], please refer to log. Quitting..."
                                       % self.doc_id)

        return gold_conll_lines, sys_conll_lines

    def prepare_lines(self, corefs, mention_table, id_2_text):
        clusters = {}
        for cluster_id, one_coref_cluster in enumerate(corefs):
            clusters[cluster_id] = set(one_coref_cluster[2])

        if utils.transitive_not_resolved(clusters):
            return False

        singleton_cluster_id = len(corefs)

        coref_fields = []
        for mention in mention_table:
            if mention is None:
                coref_fields.append(("None", "-"))
                continue

            event_mention_id = mention[1]

            non_singleton_cluster_id = None

            for cluster_id, cluster_mentions in clusters.iteritems():
                if event_mention_id in cluster_mentions:
                    non_singleton_cluster_id = cluster_id
                    break

            if non_singleton_cluster_id is not None:
                output_cluster_id = non_singleton_cluster_id
            else:
                output_cluster_id = singleton_cluster_id
                singleton_cluster_id += 1

            merged_mention_str = "_".join(id_2_text[event_mention_id].split())

            # merged_mention_str = "_".join([get_or_terminate(self.id2token, tid, Config.token_miss_msg % tid)
            #                                for tid in event_mention_id2sorted_tokens[event_mention_id]])

            coref_fields.append((merged_mention_str, output_cluster_id))

        # for cluster_id, cluster in clusters.iteritems():
        #     if within_cluster_span_duplicate(cluster, event_mention_id2sorted_tokens):
        #         return False

        lines = []

        lines.append("%s (%s); part 000%s" % (Config.conll_bod_marker, self.doc_id, os.linesep))
        for index, (merged_mention_str, cluster_id) in enumerate(coref_fields):
            lines.append("%s\t%s\t%s\t(%s)\n" % (self.doc_id, index, merged_mention_str, cluster_id))
        lines.append(Config.conll_eod_marker + os.linesep)

        return lines
