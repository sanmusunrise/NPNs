#!/usr/bin/python

"""
Utilities to convert AFTER relations into TimeML format and compute the scoring using TimeML tools.
"""

import logging
import os
import subprocess
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

from config import Config
from utils import TransitiveGraph
import utils

logger = logging.getLogger()


def validate(nuggets, edges_by_type, gold_cluster_lookup, gold_clusters):
    """
        Validate whether the edges are valid. Currently, the validation only check whether the reverse is also included,
    which is not possible for after links.

    :param nuggets: 
    :param edges_by_type: 
    :param gold_cluster_lookup: 
    :param gold_clusters: 
    :param logger: 
    :return: 
    """
    for name, edges in edges_by_type.iteritems():
        reverse_edges = set()

        for edge in edges:
            left, right, t = edge

            if left not in nuggets:
                logger.error("Relation contains unknown event %s." % left)
            if right not in nuggets:
                logger.error("Relation contains unknown event %s." % right)

            if edge in reverse_edges:
                left_cluster = gold_cluster_lookup[left]
                right_cluster = gold_cluster_lookup[right]

                logger.error("There is link from clusters A to cluster B, and from cluster B to cluster A. "
                                      "This is create a cyclic graph, which is not allowed.")
                logger.error("Cluster A contains: %s." % ",".join(gold_clusters[left_cluster]))
                logger.error("Cluster B contains: %s." % ",".join(gold_clusters[right_cluster]))

                return False
            reverse_edges.add((right, left, t))

    return True


def make_event(parent, eid):
    event = SubElement(parent, "EVENT")
    event.set("eid", eid)


def create_root():
    timeml = Element('TimML')
    timeml.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    timeml.set("xsi:noNamespaceSchemaLocation", "http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd")

    # Add a dummy DCT (document creation time).
    dct = SubElement(timeml, "DCT")
    timex3 = SubElement(dct, "TIMEX3")
    timex3.set("tid", "t0")
    timex3.set("type", "TIME")
    timex3.set("value", "")
    timex3.set("scriptFunction", "false")
    timex3.set("functionInDocument", "CREATION_TIME")

    return timeml


def convert_links(links_by_name):
    all_converted = {}
    for name, links in links_by_name.iteritems():
        converted = []
        for l in links:
            relation_name = convert_name(l[2])
            converted.append((l[0], l[1], relation_name))
        all_converted[name] = converted

    return all_converted


def convert_name(name):
    """
    Convert the Event Sequencing names to an corresponding TimeML name for evaluation.
    Note that, the meaning of After in event sequencing is different from TimeML specification.

    In Event Sequencing task, E1 --after--> E2 represent a directed link, where the latter one happens later. In TIMEML,
    E1 --after--> E2 actually says E1 is after E2. So we use the BEFORE tag instead.

    This conversion is just for the sake of logically corresponding, in fact, converting to "BEFORE" and "AFTER" will
    produce the same scores.

    In addiction, in TIMEML, there is no definition for Subevent, but "INCLUDES" have a similar semantic with that.

    :param name:
    :return:
    """
    if name == "After":
        return "BEFORE"
    elif name == "Subevent":
        return "INCLUDES"
    else:
        logger.warn("Unsupported relations name %s found." % name)


def pretty_xml(element):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def find_equivalent_sets(clusters, nodes):
    node_2_set = {}
    set_2_nodes = {}
    non_singletons = set()

    set_index = 0

    for cluster in clusters:
        for element in cluster[2]:
            node_2_set[element] = set_index
            non_singletons.add(element)

            try:
                set_2_nodes[set_index].append(element)
            except KeyError:
                set_2_nodes[set_index] = [element]

        set_index += 1

    for node in nodes:
        if node not in non_singletons:
            node_2_set[node] = set_index
            try:
                set_2_nodes[set_index].append(node)
            except KeyError:
                set_2_nodes[set_index] = [node]

            set_index += 1

    return set_2_nodes, node_2_set


def convert_to_cluster_links(node_links_by_name, cluster_lookup):
    cluster_links_by_name = {}

    unknown_nodes = set()

    for name, node_links in node_links_by_name.iteritems():
        cluster_links = set()

        for node1, node2, link_type in node_links:
            if node1 not in cluster_lookup:
                unknown_nodes.add(node1)
                continue
            if node2 not in cluster_lookup:
                unknown_nodes.add(node2)
                continue

            cluster1 = cluster_lookup[node1]
            cluster2 = cluster_lookup[node2]
            cluster_links.add((cluster1, cluster2, link_type))

        cluster_links_by_name[name] = cluster_links

    for node in unknown_nodes:
        logger.warn("Unknown node %s found in links, probably not declared as mention." % node)

    return cluster_links_by_name


def propagate_through_equivalence(links_by_name, set_2_nodes, node_2_set):
    # set_2_nodes, node_2_set = find_equivalent_sets(equivalent_links, nuggets)

    all_expanded_links = {}

    for name, links in links_by_name.iteritems():
        set_links = []

        for link in links:
            relation = link[0]
            arg1, arg2 = link[2]
            set1 = node_2_set[arg1]
            set2 = node_2_set[arg2]

            if set1 == set2:
                logger.warn("Link between %s and %s will create a self link by propagation, ignored." % (arg1, arg2))
            else:
                set_links.append((set1, set2, relation))

        reduced_set_links = compute_reduced_graph(set_links)

        expanded_links = set()

        for link in reduced_set_links:
            arg1, arg2, relation = link

            for node1 in set_2_nodes[arg1]:
                for node2 in set_2_nodes[arg2]:
                    expanded_links.add((node1, node2, relation))

        all_expanded_links[name] = list(expanded_links)

    return all_expanded_links


def compute_reduced_graph(set_links):
    node_indices = utils.get_nodes(set_links)

    graph = TransitiveGraph(len(node_indices))

    for arg1, arg2, relation in set_links:
        node_index1 = node_indices[arg1]
        node_index2 = node_indices[arg2]
        graph.add_edge(node_index1, node_index2)

    closure_matrix = graph.transitive_closure()

    indirect_links = set()

    for from_node, to_nodes in enumerate(closure_matrix):
        for to_node, reachable in enumerate(to_nodes):
            if from_node != to_node and reachable == 1:
                for indirect_node, indirect_reachable in enumerate(closure_matrix[to_node]):
                    if indirect_node != to_node:
                        if indirect_reachable == 1:
                            indirect_links.add((from_node, indirect_node))

    reduced_links = []

    for arg1, arg2, relation in set_links:
        node_index1 = node_indices[arg1]
        node_index2 = node_indices[arg2]

        if (node_index1, node_index2) not in indirect_links:
            reduced_links.append((arg1, arg2, relation))

    return reduced_links


def run_eval(link_type, script_output, gold_dir, sys_dir):
    gold_sub_dir = os.path.join(Config.script_result_dir, link_type, gold_dir)
    sys_sub_dir = os.path.join(Config.script_result_dir, link_type, sys_dir)

    with open(script_output, 'wb', 0) as out_file:
        logger.info("Evaluating directory: %s" % sys_sub_dir)
        subprocess.call(["python", Config.temp_eval_executable, gold_sub_dir, sys_sub_dir,
                         '0', "implicit_in_recall"], stdout=out_file)


def store_cluster_nodes(gold_clusters, gold_cluster_lookup, gold_nuggets, sys_nuggets, g2s_mapping):
    """
    Store cluster as nodes in TimeML. In addition, links between system nuggets are considered as a link in the
    corresponding gold cluster. The correspondence is obtained by mapping the system nugget to gold nugget, then map
    to a gold cluster.

    For system mentions that does not correspond to any gold cluster, new nodes are made for them.
    :param gold_clusters:
    :param gold_cluster_lookup:
    :param gold_nuggets:
    :param sys_nuggets:
    :param g2s_mapping:
    :return:
    """
    # Store another set of time ML nodes that represents clusters.
    cluster_nodes_in_gold = []
    cluster_id_to_gold_node = {}

    cluster_nodes_in_sys = []
    cluster_id_to_sys_node = {}

    rewritten_lookup = {}

    gold_id_2_system_id = {}
    mapped_system_nuggets = set()
    for gold_index, (sys_index, _) in enumerate(g2s_mapping):
        gold_nugget_id = gold_nuggets[gold_index]
        sys_nugget_id = sys_nuggets[sys_index]
        gold_id_2_system_id[gold_nugget_id] = sys_nugget_id

    for gold_nugget_id, cluster_id in gold_cluster_lookup.iteritems():
        sys_nugget_id = gold_id_2_system_id[gold_nugget_id]
        rewritten_lookup[sys_nugget_id] = cluster_id
        mapped_system_nuggets.add(sys_nugget_id)

    tid = 0
    max_cluster_id = 0
    for cluster_id, cluster in gold_clusters.iteritems():
        node_id = "te%d" % tid
        cluster_nodes_in_gold.append(node_id)
        cluster_nodes_in_sys.append(node_id)

        cluster_id_to_gold_node[cluster_id] = node_id
        cluster_id_to_sys_node[cluster_id] = node_id

        tid += 1

        if cluster_id > max_cluster_id:
            max_cluster_id = cluster_id

    # Some system mentions cannot be mapped to a gold mention, so it cannot be mapped to a gold cluster. Here we add
    # additional cluster nodes for these mentions.
    additional_cluster_id = max_cluster_id + 1
    for nugget in sys_nuggets:
        if nugget not in mapped_system_nuggets:
            node_id = "te%d" % tid
            cluster_nodes_in_sys.append(node_id)
            cluster_id_to_sys_node[additional_cluster_id] = node_id
            rewritten_lookup[nugget] = additional_cluster_id
            tid += 1

    return cluster_nodes_in_gold, cluster_nodes_in_sys, cluster_id_to_gold_node, cluster_id_to_sys_node, rewritten_lookup


def store_nugget_nodes(gold_nuggets, sys_nuggets, m_mapping):
    """
    Store nuggets as nodes.
    :param gold_nuggets:
    :param sys_nuggets:
    :param m_mapping:
    :return:
    """
    # Stores time ML nodes that actually exists in gold standard and system.
    gold_nodes = []
    sys_nodes = []

    # Store the mapping from nugget id to unified time ML node id.
    system_nugget_to_node = {}
    gold_nugget_to_node = {}

    mapped_system_mentions = set()

    tid = 0
    for gold_index, (system_index, _) in enumerate(m_mapping):
        node_id = "te%d" % tid
        tid += 1

        gold_script_instance_id = gold_nuggets[gold_index]
        gold_nugget_to_node[gold_script_instance_id] = node_id
        gold_nodes.append(node_id)

        if system_index != -1:
            system_nugget_id = sys_nuggets[system_index]
            system_nugget_to_node[system_nugget_id] = node_id
            sys_nodes.append(node_id)
            mapped_system_mentions.add(system_index)

    for system_index, system_nugget in enumerate(sys_nuggets):
        if system_index not in mapped_system_mentions:
            node_id = "te%d" % tid
            tid += 1

            system_nugget_to_node[system_nugget] = node_id
            sys_nodes.append(node_id)

    return gold_nodes, sys_nodes, gold_nugget_to_node, system_nugget_to_node


class TemporalEval:
    """
    This class help us converting the input into TLINK format and evaluate them using the script evaluation tools
    developed by UzZaman. We use the variation of their method that use both the reduced graph and rewarding
    un-inferable implicit relations.

    Reference:
    Interpreting the Temporal Aspects of Language, Naushad UzZaman, 2012
    """

    def __init__(self, g2s_mapping, gold_nugget_table, raw_gold_links, sys_nugget_table,
                 raw_sys_links, gold_corefs, sys_corefs):
        self.gold_nugget_table = gold_nugget_table
        self.sys_nugget_table = sys_nugget_table

        # Store how the event nugget ids.
        self.gold_nuggets = [nugget[2] for nugget in gold_nugget_table]
        self.sys_nuggets = [nugget[2] for nugget in sys_nugget_table]

        self.g2s_mapping = g2s_mapping

        self.gold_nodes, self.sys_nodes, self.gold_nugget_to_node, self.system_nugget_to_node = store_nugget_nodes(
            self.gold_nuggets, self.sys_nuggets, g2s_mapping)

        self.gold_clusters, self.gold_cluster_lookup = find_equivalent_sets(gold_corefs, self.gold_nuggets)
        self.sys_clusters, self.sys_cluster_lookup = find_equivalent_sets(sys_corefs, self.sys_nuggets)

        # Propagate mention level gold and system links.
        self.gold_links_by_type = propagate_through_equivalence(raw_gold_links, self.gold_clusters,
                                                                self.gold_cluster_lookup)
        self.sys_links_by_type = propagate_through_equivalence(raw_sys_links, self.sys_clusters,
                                                               self.sys_cluster_lookup)

    def validate_gold(self):
        return validate(set([nugget[2] for nugget in self.gold_nugget_table]), self.gold_links_by_type,
                        self.gold_cluster_lookup, self.gold_clusters)

    def validate_sys(self):
        return validate(set([nugget[2] for nugget in self.sys_nugget_table]), self.sys_links_by_type,
                        self.gold_cluster_lookup, self.gold_clusters)

    def write_time_ml(self, doc_id):
        """
        Write the TimeML file to disk.
        :return:
        """
        # Store another set of time ML nodes that represents clusters.
        gold_cluster_nodes, sys_cluster_nodes, gold_cluster_to_node, sys_cluster_to_node, rewritten_lookup \
            = store_cluster_nodes(self.gold_clusters, self.gold_cluster_lookup, self.gold_nuggets,
                                  self.sys_nuggets, self.g2s_mapping)

        gold_cluster_links = convert_to_cluster_links(self.gold_links_by_type, self.gold_cluster_lookup)
        sys_cluster_links = convert_to_cluster_links(self.sys_links_by_type, rewritten_lookup)

        gold_time_ml = self.make_all_time_ml(convert_links(self.gold_links_by_type), self.gold_nugget_to_node,
                                             self.gold_nodes)
        sys_time_ml = self.make_all_time_ml(convert_links(self.sys_links_by_type), self.system_nugget_to_node,
                                            self.sys_nodes)

        gold_cluster_time_ml = self.make_all_time_ml(convert_links(gold_cluster_links), gold_cluster_to_node,
                                                     gold_cluster_nodes)
        sys_cluster_time_ml = self.make_all_time_ml(convert_links(sys_cluster_links), sys_cluster_to_node,
                                                    sys_cluster_nodes)

        TemporalEval.write(gold_time_ml, Config.script_gold_dir, doc_id)
        TemporalEval.write(sys_time_ml, Config.script_sys_dir, doc_id)

        TemporalEval.write(gold_cluster_time_ml, Config.script_gold_dir + "_cluster", doc_id)
        TemporalEval.write(sys_cluster_time_ml, Config.script_sys_dir + "_cluster", doc_id)

    @staticmethod
    def write(time_ml_data, subdir, doc_id):
        """
        Write out time ml files into sub directories.
        :param time_ml_data:
        :param time_ml_dir:
        :param subdir:
        :return:
        """
        for name, time_ml in time_ml_data.iteritems():
            output_dir = os.path.join(Config.script_result_dir, name, subdir)
            utils.supermakedirs(output_dir)

            temp_file = open(os.path.join(output_dir, "%s.tml" % doc_id), 'w')
            temp_file.write(pretty_xml(time_ml))
            temp_file.close()

    @staticmethod
    def eval_time_ml():
        logger.info("Running TimeML scorer.")

        for link_type in Config.script_types + ["All"]:
            # Evaluate mention level links.
            run_eval(link_type, os.path.join(Config.script_result_dir, link_type, Config.script_out),
                     Config.script_gold_dir, Config.script_sys_dir)

            if Config.eval_cluster_level_links:
                # Evaluate cluster level links.
                run_eval(link_type, os.path.join(Config.script_result_dir, link_type, Config.script_out_cluster),
                         Config.script_gold_dir + "_cluster", Config.script_sys_dir + "_cluster")

    @staticmethod
    def get_eval_output():
        script_output = os.path.join(Config.script_result_dir, Config.script_out)
        with open(script_output, 'r') as f:
            score_line = False
            for l in f:
                if score_line:
                    prec, recall, f1 = [float(x) for x in l.strip().split("\t")]
                    return prec, recall, f1

                if l.startswith("Temporal Score"):
                    score_line = True

    def make_all_time_ml(self, links_by_name, normalized_nodes, nodes):
        all_time_ml = {}

        all_links = []

        for name in Config.script_types:
            if name in links_by_name:
                links = links_by_name[name]
                all_time_ml[name] = self.make_time_ml(links, normalized_nodes, nodes)
                all_links.extend(links)
            else:
                all_time_ml[name] = self.make_time_ml([], normalized_nodes, nodes)

        all_time_ml["All"] = self.make_time_ml(all_links, normalized_nodes, nodes)

        return all_time_ml

    def make_time_ml(self, links, normalized_nodes, nodes):
        # Create the root.
        time_ml = create_root()
        # Add TEXT.
        self.annotate_timeml_events(time_ml, nodes)

        # Add instances.
        self.create_instance(time_ml, nodes)
        self.create_tlinks(time_ml, links, normalized_nodes)

        return time_ml

    @staticmethod
    def create_instance(parent, nodes):
        for node in nodes:
            instance = SubElement(parent, "MAKEINSTANCE")
            instance.set("eiid", "instance_" + node)
            instance.set("eid", node)

    @staticmethod
    def annotate_timeml_events(parent, nodes):
        text = SubElement(parent, "TEXT")
        for tid in nodes:
            make_event(text, tid)

    @staticmethod
    def create_tlinks(time_ml, links, normalized_nodes):
        lid = 0

        unknown_nodes = set()

        for left, right, relation_type in links:
            if left not in normalized_nodes:
                unknown_nodes.add(left)
                continue

            if right not in normalized_nodes:
                unknown_nodes.add(right)
                continue

            normalized_left = normalized_nodes[left]
            normalized_right = normalized_nodes[right]

            link = SubElement(time_ml, "TLINK")
            link.set("lid", "l%d" % lid)
            link.set("relType", relation_type)
            link.set("eventInstanceID", normalized_left)
            link.set("relatedToEventInstance", normalized_right)

        for node in unknown_nodes:
            logger.error("Node %s is not a known node." % node)
