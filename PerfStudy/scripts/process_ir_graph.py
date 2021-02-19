import argparse
import json
from tabulate import tabulate
import subprocess
from collections import OrderedDict
from operator import itemgetter
import re
import os

ops_pattern = ['*', 'aten::', "prim::", "fb::", "quantized::", "block*()", "prim::If", "prim::Loop", "prim::CallMethod", "prim::SetAttr", "prim::GetAttr"]
op_name = "prim::If"

def sort_by_graph_size(stats_table):
    ordered_stats = OrderedDict(sorted(stats_table.items(), key=lambda x:x[1][0]))
    return ordered_stats

def sort_by_key(stats_table):
    return OrderedDict(sorted(stats_table.items()))

def print_benchmark_stats(stats_table):
    stats_table = sort_by_graph_size(stats_table)
    headers = ['Logfile (ir counts)'] + ops_pattern
    rows = []
    for bench_name in stats_table:
        row = [bench_name] + add_percentage_to_stat(stats_table[bench_name])
        rows.append(row)
    print(tabulate(rows, headers=headers))

def process_stat(logfile_name, stats_table):
    stat = []
    for grep_str in ops_pattern:
        cmd = f"egrep \"{grep_str}\" {logfile_name} | wc -l"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        count = int(output)
        stat.append(count)
    stats_table[logfile_name] = stat

def add_percentage_to_stat(stat):
    total_ops = stat[0]
    stat_with_percentage = []
    assert(total_ops > 0)
    for count in stat:
        if total_ops == count or count == 0:
            stat_with_percentage.append(count)
        else:
            ratio = count * 100/total_ops
            stat_with_percentage.append(f"{count} ({ratio:2.0f}%)")
    return stat_with_percentage

def process_all_stats(logfile_list):
    stats_table = {}
    for logfile in logfile_list:
        print(".", end="", flush=True)
        process_stat(logfile, stats_table)
    print("")
    print_benchmark_stats(stats_table)

# Since Predictor models are collected at different times
# the source info may contain different vm-id info, strip
# those so that we can consolidate identical source locations
def remove_devvm_prefix(filepath):
    #filepath ="/mnt/xarfuse/uid-30718/7f35743e-seed-f6439dc8-443c-4262-aba8-e81f46a0cf6f-ns-4026534173/pytext/data/bert_tensorizer.py"
    #filepath = "/mnt/xxxx/uid-xx/yy-seed-xxx-yy-ns-2-1xx/sdg.py"
    #filepath = "/mnt/xarfuse/uid-183475/c17bcd48-ns-4026531840/torch/nn/functional.py"
    result = re.match("\/mnt\/(?P<firstpart>\w+)\/uid-(?P<secondpart>[\w-]+)\/[\w-]+-ns-(?P<thirdpart>[\w-]+)\/(?P<filepath>[\w.\/]+)$", filepath)
    newpath = filepath
    if result:
        newpath = result.group('filepath')
        #print(f"Found match {newpath}")
    else:
         print(f"Did not find match {filepath}")
    return newpath


def extract_source_from(line, source_dict, filter_devvm_path):
    line = line.strip().decode('utf-8')
    result = re.split("\s+", line)
    no_source_ifs = 0
    if result[-2] != "#":
        # print(f"WARNING - no source info: {line}")
        no_source_ifs = 1
    else:
        long_source = result[-1]
        result2 = re.split(":", long_source)
        # take filename and lineno
        filename = result2[0]
        if filter_devvm_path:
            filename = remove_devvm_prefix(filename)
        lineno = result2[1]
        source = filename + ":" + lineno
        if source in source_dict:
            source_dict[source] += 1
        else:
            source_dict[source] = 1
    return no_source_ifs

def analyze_bench(log_file_name, source_dict, print_stat, filter_devvm_path):
    cmd = f"egrep \"{op_name}\" {log_file_name}"
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    no_source_count = 0
    while True:
        line = ps.stdout.readline()
        if not line:
            break
        no_source_count += extract_source_from(line, source_dict, filter_devvm_path)
    if print_stat:
        print_source_count(log_file_name, source_dict, no_source_count)
    return no_source_count

def print_source_count(benchmark, source_dict, no_source_count):
    print(f"\"{op_name}\" in [{benchmark}] (source: [count]): {len(source_dict)} total sources")
    source_dict = sort_by_key(source_dict)
    for key in source_dict:
        print(f"  - {key}: [{source_dict[key]}]")
    if no_source_count > 0:
        print(f"Missing source file info for {no_source_count} occurances")

def analyze_dir(logfile_list, filter_devvm_path):
    source_dict = {}
    no_source_count = 0
    for logfile_name in logfile_list:
        print(".", end="", flush=True)
        no_source_count += analyze_bench(logfile_name, source_dict, False, filter_devvm_path)
    print("")
    print_source_count("all logfiles", source_dict, no_source_count)

# scan files under the current directory and return ones whose filename
# matched filename_pattern
def collect_files(filename_ext):
    filelist = []
    with os.scandir('./') as entries:
        for entry in entries:
            t = re.split("\.", entry.name)
            if len(t) > 1 and t[-1] == filename_ext:
                filelist.append(entry.name)
    return filelist

if __name__ == "__main__":
    #remove_devvm_prefix("")

    parser = argparse.ArgumentParser(description="Commands to process Graph-IR dump logs")
    parser.add_argument("--list_ir",
                        action="store_true",
                        help="List IR types")
    parser.add_argument("--stats_dir",
                        help="Dump stats for files with extension <STATS_DIR> in current dir, e.g., \"--stats_dir txt\" ")
    parser.add_argument("--stats_file",
                        help="Dump stats for file <STATS_FILE>, e.g., \"--stats_file foo.txt\" ")
    parser.add_argument("--analyze_file",
                        help="Analyze ops pattern for file <ANALYZE_FILE>, e.g., \"--analyze_file foo.txt\" ")
    parser.add_argument("--analyze_dir",
                        help="Analyze ops pattern for files w/ extension <ANALYZE_DIR> in current dir, e.g., \"--analyze_dir txt\"")
    parser.add_argument("-f", "--filter_devvm_path",
                        action="store_true",
                        help="Filter out devvm generated path prefix in source info")

    args = parser.parse_args()

    if args.list_ir:
        print("IR type names: ")
        for op in ops_pattern:
            if op != "*": print(f"   - {op}")

    if args.stats_dir:
        logfile_list = collect_files(args.stats_dir)
        if logfile_list:
            process_all_stats(logfile_list)
        else:
            print(f"Found no graph dump files w \"{args.stats_dir}\" extension")

    if args.stats_file:
        stats_table = {}
        process_stats(args.stats_file, stats_table)
        print_benchmark_stats(stats_table)

    if args.analyze_file:
        analyze_bench(args.analyze_file, {}, True, args.filter_devvm_path)

    if args.analyze_dir:
        logfile_list = collect_files(args.analyze_dir)
        if logfile_list:
            analyze_dir(logfile_list, args.filter_devvm_path)
        else:
            print(f"Found no graph dump files w \"{args.analyze_dir}\" extension")
