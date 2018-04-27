#!/usr/bin/python

"""
    Run the scorer through test cases

    This run the scorer in an external way and go through the scores in file to determine the correctness. Though
    calling the functions and checking the return values might be a better way, a standalone tester like this is easier
"""
import glob
import logging
import os
import subprocess


class Config:
    """
    Configuration variables
    """
    scorer_executable = "scorer_v1.8.py"
    test_temp = "test_tmp"
    test_base = "data/test_cases"
    test_log_output = "test.log"

    conll_scorer_executable = "./reference-coreference-scorers-8.01/scorer.pl"

    # Test types.
    mention_detection_tests = "mention_detection_tests"
    conll_tests = "conll_tests"
    wrong_format_tests = "wrong_format_tests"

    # Test cases for each type.
    detection_test_cases = os.path.join(test_base, mention_detection_tests)
    conll_test_cases = os.path.join(test_base, conll_tests)
    wrong_format_test_cases = os.path.join(test_base, wrong_format_tests)

    # Suffix of test cases.
    tbf_response_suffix = ".response.tbf"
    tbf_key_suffix = ".key.tbf"
    conll_response_suffix = ".response.conll"
    conll_key_suffix = ".key.conll"
    format_test_suffix = ".reason"
    mention_test_suffix = ".score"


def run_scorer(gold_path, system_path, token_path, result_out, coref_log):
    """
        Run the scorer script and provide arguments for output
    :param gold_path: The path to the gold standard file
    :param system_path: The path to the system file
    :param result_out: Path to output the scores
    :return:
    """

    cmd = ["python", Config.scorer_executable, '-g', gold_path, '-s', system_path, '-t', token_path]

    if coref_log:
        cmd.append('-c')
        cmd.append(coref_log)

    with open(result_out, 'wb', 0) as out_file:
        subprocess.call(cmd, stdout=out_file)
    return " ".join(cmd)


def extract_key_metrics(result_out, coref_log):
    pass


class ScorerTest:
    def __init__(self):
        # Prepare test temporary output.
        if not os.path.exists(Config.test_temp):
            os.mkdir(Config.test_temp)
        self.logger = logging.getLogger()
        test_out = self.prepare_temp_file(Config.test_log_output)
        test_result_output = open(test_out, 'w')
        stream_handler = logging.StreamHandler(test_result_output)
        stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)
        self.num_tests_finished = 0
        self.num_tests_passed = 0

        print "Please see %s for test logs." % test_out

    def record_pass(self):
        self.num_tests_finished += 1
        self.num_tests_passed += 1

    def record_fail(self, msg):
        self.num_tests_finished += 1
        self.logger.error("Test Failed : %s" % msg)

    @staticmethod
    def prepare_temp_file(*args):
        p = os.path.join(Config.test_temp, *args)
        parent = os.path.abspath(os.path.join(p, os.pardir))
        if not os.path.exists(parent):
            os.makedirs(parent)
        return p

    @staticmethod
    def get_tbf_key(test_dir, response_basename):
        real_basename = response_basename.rsplit("_", 1)[0]
        return os.path.join(test_dir, real_basename + Config.tbf_key_suffix)

    @staticmethod
    def get_conll_key(test_dir, response_basename):
        real_basename = response_basename.rsplit("_", 1)[0]
        return os.path.join(test_dir, real_basename + Config.conll_key_suffix)

    def run_mention_detection_tests(self, mention_test_dir):
        """
        Run through the test cases for mention detection. The result is verify by checking
        whether it give expected values.
        """
        self.logger.info("Running mention tests")
        print "Running mention tests"

    def conll_result_check(self, conll_out_reference, conll_out_test):
        return cmp(self.get_conll_scores(conll_out_reference), self.get_conll_scores(conll_out_test)) == 0

    @staticmethod
    def run_conll_script(gold_path, system_path, script_out):
        """
        Run the Conll script and output result to the path given
        :param gold_path:
        :param system_path:
        :param script_out: Path to output the scores
        :return:
        """
        with open(script_out, 'wb', 0) as out_file:
            subprocess.call(
                ["perl", Config.conll_scorer_executable, "all", gold_path, system_path],
                stdout=out_file)

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

    def run_conll_tests(self, conll_test_dir):
        """
        Run through the test cases for CoNLL scoring. The result is verify by comparing
        the results of the mention scorer on the test case and the results of the CoNLL
        reference scores on the expected test case. These two result should match.
        Hence, each test case contains 4 files, for two pair of results.
        :param conll_test_dir:
        :return:
        """
        self.logger.info("Running CoNLL tests.")
        print "Running CoNLL tests."
        token_path = os.path.join(conll_test_dir, "tkn")
        for f in glob.glob(os.path.join(conll_test_dir, "*" + Config.tbf_response_suffix)):
            basename = os.path.basename(f)[:-len(Config.tbf_response_suffix)]

            # Run our scorer.
            tbf_key_file = self.get_tbf_key(conll_test_dir, basename)

            scoring_out = self.prepare_temp_file(Config.conll_tests, basename + ".score_tmp")
            our_conll_out = self.prepare_temp_file(Config.conll_tests, basename + ".ours_conll_log")

            command_run = run_scorer(tbf_key_file, f, token_path, scoring_out, our_conll_out)

            self.logger.info("Test command is  : %s" % command_run)

            # Run conll version.
            conll_reference = os.path.join(conll_test_dir, basename + Config.conll_response_suffix)
            conll_key_file = self.get_conll_key(conll_test_dir, basename)
            reference_conll_out = self.prepare_temp_file(Config.conll_tests, basename + ".ref_conll_log")

            self.run_conll_script(conll_key_file, conll_reference, reference_conll_out)

            # Compare result of two versions.
            if self.conll_result_check(our_conll_out, reference_conll_out):
                self.record_pass()
            else:
                self.record_fail("Test [%s] is not passed, CoNLL score not matching expectation." % f)

    @staticmethod
    def check_reason_file(output_path, reason_path):
        with open(output_path, 'r', 0) as f:
            file_str = f.read()
            reason = open(reason_path).read().strip()
            if reason in file_str:
                return True

    def run_format_error_tests(self, wrong_format_test_dir):
        """
        Run through the test cases that should make scorer raise format exceptions,
        and that there are relevant error message in the output.
        :param wrong_format_test_dir:
        :return:
        """
        self.logger.info("Running format error test cases.")
        print "Running format error test cases."
        reference_gold = os.path.join(wrong_format_test_dir, "correct_example.key.tbf")
        token_path = os.path.join(wrong_format_test_dir, "tkn")

        for f in glob.glob(os.path.join(wrong_format_test_dir, "*" + Config.tbf_response_suffix)):
            if f != reference_gold:
                basename = os.path.basename(f)[:-len(Config.tbf_response_suffix)]
                scoring_out = self.prepare_temp_file(Config.wrong_format_tests, basename + ".score_tmp")
                conll_out = self.prepare_temp_file(Config.wrong_format_tests, basename + ".conll_log")

                # Reason file stores the reason why this tbf is wrong, must be matched to pass this test.
                reason_file = os.path.join(wrong_format_test_dir, basename + Config.format_test_suffix)
                if not os.path.exists(reason_file):
                    continue

                command_run = run_scorer(reference_gold, f, token_path, scoring_out, conll_out)
                self.logger.info("Test command is  : %s" % command_run)

                if self.check_reason_file(scoring_out, reason_file):
                    self.record_pass()
                else:
                    self.record_fail("Test [%s] is not passed, expected format error not found in output." % f)

    def run_all(self):
        self.logger.info("Start tests.")
        self.run_mention_detection_tests(Config.detection_test_cases)
        self.run_format_error_tests(Config.wrong_format_test_cases)
        self.run_conll_tests(Config.conll_test_cases)
        test_finish = self.test_finish_info()
        self.logger.info(test_finish)
        print test_finish

    def test_finish_info(self):
        return "All test finished.\nNumber of tests : %d, number of tests passed : %d, number of tests failed : %d\n" % (
            self.num_tests_finished, self.num_tests_passed,
            self.num_tests_finished - self.num_tests_passed)


def main():
    test = ScorerTest()
    test.run_all()


if __name__ == "__main__":
    main()
