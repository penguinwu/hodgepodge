import argparse
import json
from tabulate import tabulate
import subprocess
from collections import OrderedDict
from operator import itemgetter

benchmarks = ['alexnet', 'BERT_pytorch', 'densenet121', 'fastNLP', 'LearningToPaint', 'mnasnet1_0', 'mobilenet_v2',
              'pytorch_mobilenet_v3', 'pytorch_stargan','pytorch_struct', 'resnet18', 'resnet50',
              'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_1', 'vgg16']
internal_benchmarks = ['adindexer-merge-net-ctr-mobilefeed', 'deep-and-wide']
#benchmarks = ['alexnet']
ops_pattern = ['*', 'aten::', "prim::", "fb::", "quantized::", "block*()", "prim::If", "prim::Loop", "prim::CallMethod", "prim::SetAttr"]
# ops_pattern = ['*', 'aten::', "prim::", "block*()", "prim::CallMethod", "prim::If", "prim::Loop", "prim::SetAttr", "prim::GetAttr"]

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

def collect_stat(bench_name, stats_table):
    log_file_name = f"{bench_name}.last_executed_graph.dump.log"
    stat = []
    for grep_str in ops_pattern:
        cmd = f"egrep \"{grep_str}\" {log_file_name} | wc -l"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        stat.append(int(output))
    stats_table[bench_name] = stat

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("json_file")
    # args = parser.parse_args()
    stats_table = {}
    benchmarks += internal_benchmarks
    for bench_name in benchmarks:
        collect_stat(bench_name, stats_table)

    print_benchmark_stats(stats_table)
