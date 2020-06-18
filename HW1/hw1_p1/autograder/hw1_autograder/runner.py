import argparse
import json
import os
import re
import sys
import pytest
import six
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from problems import problems

class LoggerPlugin(object):
    def __init__(self):
        self.logs = {}

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            name = re.split('::', report.nodeid)[1]
            passed = 1 if report.outcome == 'passed' else 0
            self.logs[name] = passed


def main(argv):


    parser = argparse.ArgumentParser(description='Test Runner')
    default_module_path = '../solution/gold'
    parser.add_argument('--module-path', type=str, default=default_module_path,
                        help='output directory (default: {})'.format(default_module_path))
    args = parser.parse_args(argv[1:])
    sys.path.append(args.module_path)
    test_path = os.path.dirname(os.path.abspath(__file__))
    logger = LoggerPlugin()

    start = time.time()
    pytest.main(["-s","-qq", test_path], plugins=[logger])
    end = time.time()
    print("Run time: ", end - start)

    scores = {}
    total = 0
    for k, v in six.iteritems(logger.logs):
        problem, value = problems[k]
        if problem not in scores:
            scores[problem] = 0
        scores[problem] += value * v
        total += value * v
    print(f' {"_"*40}{"_"*11}')
    print(f'|{"TASK":<40}|{"SCORE":<10}|')
    print(f'|{"_"*40}|{"_"*10}|')
    for task, score in scores.items():
        print(f'|{task:<40}|{score:<10}|')
        print(f'|{"_"*40}|{"_"*10}|')
    print(f'|{"TOTAL SCORE":<40}|{total:<10}|')
    print(f'|{"_"*40}|{"_"*10}|')

if __name__ == '__main__':
    main(sys.argv)
