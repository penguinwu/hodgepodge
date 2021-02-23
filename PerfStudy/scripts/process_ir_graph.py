import argparse
import json
from tabulate import tabulate
import subprocess
from collections import OrderedDict
from operator import itemgetter
import re
import os

block_pattern = "[[:space:]]+block*()"
block_pattern_name ="block*()"
block_return_pattern = "[[:space:]]+-> *"
block_return_pattern_name = "block return"
return_pattern = "[[:space:]]+return *"
return_pattern_name = "return"
graph_decl_pattern="^graph\("
graph_decl_pattern_name="graph decl"
graph_decl_leftover_pattern="^[[:space:]]+%\w+\.\w+ \: \w+\)\:$"
graph_decl_leftover_pattern_name="graph decl others"
emptyline_pattern="^$"
ops_pattern = ["*", "aten::", "prim::", "fb::", "quantized::", block_pattern, block_return_pattern, "prim::If", "prim::Loop", "prim::CallMethod",
               "prim::SetAttr", "prim::GetAttr", return_pattern, "internal::", "_caffe2::", graph_decl_pattern, graph_decl_leftover_pattern]
ops_pattern_header = ["*", "aten::", "prim::", "fb::", "quantized::", block_pattern_name, block_return_pattern_name, "prim::If", "prim::Loop", "prim::CallMethod",
               "prim::SetAttr", "prim::GetAttr", return_pattern_name, "internal::", "_caffe2::", graph_decl_pattern_name, graph_decl_leftover_pattern_name]
reported_ops_pattern = ["*", "aten::", "prim::", "fb::", "quantized::", "internal::", block_pattern, block_return_pattern, "prim::If",
               "prim::Loop", "prim::CallMethod", "prim::SetAttr", "prim::GetAttr", "_caffe2::"]
ops_category = {"aten::", "prim::", "fb::", "quantized::", "internal::", "_caffe2::", block_pattern, block_return_pattern,
                return_pattern, graph_decl_pattern, graph_decl_leftover_pattern}
op_name = "prim::If"

def sort_by_graph_size(stats_table):
    ordered_stats = OrderedDict(sorted(stats_table.items(), key=lambda x:x[1][0]))
    return ordered_stats

def sort_by_key(stats_table):
    return OrderedDict(sorted(stats_table.items()))

def print_benchmark_stats(stats_table):
    stats_table = sort_by_graph_size(stats_table)
    headers = ['Logfile (ir counts)'] + reported_ops_pattern + ['Others', 'Uncounted']
    rows = []
    for bench_name in stats_table:
        row = [bench_name] + postprocess_stats(stats_table[bench_name])
        rows.append(row)
    print(tabulate(rows, headers=headers))

# check if there are lines not captured by current pattern
def check_ir_coverage(logfile_name):
    tmp_source = "inter1.tmp"
    tmp_output = "inter2.tmp"
    cp_cmd = f"cp {logfile_name} {tmp_source}"
    subprocess.run(cp_cmd, shell=True, check=True)
    for grep_str in ops_pattern:
        if grep_str == "*":
            continue
        grep_cmd = f"egrep -v \"{grep_str}\" {tmp_source} > {tmp_output}"
        subprocess.run(grep_cmd, shell=True, check=False)
        mv_cmd = f"mv {tmp_output} {tmp_source}"
        subprocess.run(mv_cmd, shell=True, check=True)
    file1 = open(tmp_source)
    print(file1.read())
    file1.close()

def process_stats(logfile_name, stats_table):
    stat = []
    for grep_str in ops_pattern:
        cmd = f"egrep \"{grep_str}\" {logfile_name} | wc -l"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        count = int(output)
        stat.append(count)
    stats_table[logfile_name] = stat

def postprocess_stats(stat):
    total_ops = stat[0]
    stat_with_percentage = []
    uncounted_ops = total_ops
    other_ops = 0
    assert(total_ops > 0)
    for index, count in enumerate(stat):
        if ops_pattern[index] in ops_category:
            # print(f"found {ops_pattern[index]} count={count}")
            uncounted_ops -= count
        if ops_pattern[index] not in reported_ops_pattern:
            other_ops += count
            continue
        if index == 0:
            assert(total_ops == count)
            stat_with_percentage.append(count)
        elif count == 0:
            stat_with_percentage.append(count)
        else:
            ratio = count * 100/total_ops
            stat_with_percentage.append(f"{count} ({ratio:2.0f}%)")
    if other_ops > 0:
        ratio = other_ops * 100/total_ops
        stat_with_percentage.append(f"{other_ops} ({ratio:2.0f}%)")
    else:
        stat_with_percentage.append(f"{other_ops}")
    if uncounted_ops > 0:
        ratio = uncounted_ops * 100/total_ops
        stat_with_percentage.append(f"{uncounted_ops} ({ratio:2.0f}%)")
    else:
        assert(uncounted_ops == 0)
        stat_with_percentage.append(f"{uncounted_ops}")
    return stat_with_percentage

def process_all_stats(logfile_list):
    stats_table = {}
    for logfile in logfile_list:
        print(".", end="", flush=True)
        process_stats(logfile, stats_table)
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
    parser.add_argument("--ir_coverage_file",
                        help="Check recognized IR coverage for file <IR_COVERAGE_FILE>, e.g., \"--ir_coverage_file foo.txt\" ")
    parser.add_argument("--ir_coverage_dir",
                        help="Check recognized IR coverage for file with extebsuib <IR_COVERAGE_DIR> in current dir, e.g., \"--ir_coverage_file foo.txt\" ")
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

    if args.ir_coverage_file:
        check_ir_coverage(args.ir_coverage_file)

    if args.ir_coverage_dir:
        logfile_list = collect_files(args.ir_coverage_dir)
        for logfile in logfile_list:
            check_ir_coverage(logfile)
        if logfile_list is None:
            print(f"Found no graph dump files w \"{args.ir_coverage_dir}\" extension")

    if args.analyze_file:
        analyze_bench(args.analyze_file, {}, True, args.filter_devvm_path)

    if args.analyze_dir:
        logfile_list = collect_files(args.analyze_dir)
        if logfile_list:
            analyze_dir(logfile_list, args.filter_devvm_path)
        else:
            print(f"Found no graph dump files w \"{args.analyze_dir}\" extension")
