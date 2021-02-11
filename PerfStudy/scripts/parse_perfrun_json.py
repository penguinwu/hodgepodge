import argparse
import json
from tabulate import tabulate

def print_benchmark_stats(data):
    print_stats = ['min', 'max', 'mean', 'stddev', 'rounds', 'median']

    headers = ['name (time in s)'] + print_stats 
    rows = []
    for benchmark in data['benchmarks']:
        row = [benchmark['name']]
        row += [benchmark['stats'][k] for k in print_stats]
        rows.append(row)
    print(tabulate(rows, headers=headers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)
        print_benchmark_stats(data)
