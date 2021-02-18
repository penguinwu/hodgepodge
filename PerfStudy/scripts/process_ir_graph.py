import argparse
import json
from tabulate import tabulate
import subprocess
from collections import OrderedDict
from operator import itemgetter
import re

benchmarks = ['alexnet', 'BERT_pytorch', 'densenet121', 'fastNLP', 'LearningToPaint', 'mnasnet1_0', 'mobilenet_v2',
              'pytorch_mobilenet_v3', 'pytorch_stargan','pytorch_struct', 'resnet18', 'resnet50',
              'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_1', 'vgg16']
internal_benchmarks = ['adindexer-merge-net-ctr-mobilefeed', 'deep-and-wide']
ops_pattern = ['*', 'aten::', "prim::", "fb::", "quantized::", "block*()", "prim::If", "prim::Loop", "prim::CallMethod", "prim::SetAttr"]
op_name = "prim::If"

def sort_by_graph_size(stats_table):
    ordered_stats = OrderedDict(sorted(stats_table.items(), key=lambda x:x[1][0]))
    return ordered_stats

def print_benchmark_stats(stats_table):
    stats_table = sort_by_graph_size(stats_table)
    headers = ['name (occurances)'] + ops_pattern
    rows = []
    for bench_name in stats_table:
        row = [bench_name] + stats_table[bench_name]
        rows.append(row)
    print(tabulate(rows, headers=headers))

def get_log_file_name(bench_name):
    return f"{bench_name}.last_executed_graph.dump.log"

def collect_stat(bench_name, stats_table):
    log_file_name = get_log_file_name(bench_name)
    stat = []
    for grep_str in ops_pattern:
        cmd = f"egrep \"{grep_str}\" {log_file_name} | wc -l"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        stat.append(int(output))
    stats_table[bench_name] = stat

def process_all_stats(benchmarks):
    stats_table = {}
    for bench_name in benchmarks:
        collect_stat(bench_name, stats_table)
    print_benchmark_stats(stats_table)

def extract_source_from(line, source_dict):
    line = line.strip().decode('utf-8')
    result = re.split("\s+", line)
    no_source_ifs = 0
    if result[-2] != "#":
        # print(f"WARNING - no source info: {line}")
        no_source_ifs = 1
    else:
        result2 = re.split(":", result[-1])
        # take filename and lineno
        source = result2[0] + ":" + result2[1]
        if source in source_dict:
            source_dict[source] += 1
        else:
            source_dict[source] = 1
    return no_source_ifs

def analyze_bench(benchmark, source_dict={}, print_stat=False):
    log_file_name = get_log_file_name(benchmark)
    cmd = f"egrep \"{op_name}\" {log_file_name}"
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    no_source_count = 0
    while True:
        line = ps.stdout.readline()
        if not line:
            break
        no_source_count += extract_source_from(line, source_dict)
    if print_stat:
        print_source_count(benchmark, source_dict, no_source_count)
    return no_source_count

def print_source_count(benchmark, source_dict, no_source_count):
    print(f"\"{op_name}\" in [{benchmark}] (source:count): {len(source_dict)} total sources")
    for key in source_dict:
        print(f"  - {key}: {source_dict[key]}")
    if no_source_count > 0:
        print(f"Missing source file info for {no_source_count} occurances")

def analyze_all(benchmarks):
    source_dict = {}
    no_source_count = 0
    for bench in benchmarks:
        no_source_count += analyze_bench(bench, source_dict)
    print_source_count("all benchmarks", source_dict, no_source_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commands to process Graph-IR dump logs")
    parser.add_argument("-l", "--list",
                        action="store_true",
                        help="List benchmark names")
    parser.add_argument("--list_ir",
                        action="store_true",
                        help="List IR types")
    parser.add_argument("-a", "--all_stats",
                        action="store_true",
                        help="Dump stats for all benchmarks")
    parser.add_argument("--analyze",
                        help="Analyze graph ir for <benchmark_name>")
    parser.add_argument("--analyze_all",
                        action="store_true",
                        help="Analyze graph ir for all benchmarks")
    args = parser.parse_args()

    # TODO: check if logs for internal_benchmarks are available
    benchmarks += internal_benchmarks

    if args.list:
        for bench_name in benchmarks:
            print(bench_name)

    if args.list_ir:
        print("IR type names: ")
        for op in ops_pattern:
            if op != "*": print(f"   - {op}")

    if args.all_stats:
        process_all_stats(benchmarks)

    if args.analyze:
        analyze_bench(args.analyze, {}, True)

    if args.analyze_all:
        analyze_all(benchmarks)
