import argparse
import json
from tabulate import tabulate
import subprocess
from collections import OrderedDict
from operator import itemgetter
import re
import os

all_pattern_name = "all"
block_op_name ="block*()"
block_return_name = "block-ret"
return_op_name = "return"
graph_decl_op_name="graph"
graph_decl_sig_name="graph-others"
aten_op_name = "aten::*"
prim_op_name = "prim::*"
fb_op_name = "fb::*"
quantized_op_name = "quantized::*"
internal_op_name = "internal::*"
caffe2_op_name = "caffe2::*"
if_op_name = "if"
loop_op_name = "loop"
call_op_name = "call"
setattr_op_name = "setattr"
getattr_op_name = "getattr"

op_name = "if"
ops_pattern = {}
reported_ops = [aten_op_name, prim_op_name, fb_op_name, quantized_op_name, internal_op_name, caffe2_op_name,
                if_op_name, loop_op_name, call_op_name, setattr_op_name, getattr_op_name]
ops_category = {aten_op_name, prim_op_name, fb_op_name, quantized_op_name, internal_op_name, caffe2_op_name,
                block_op_name, block_return_name, return_op_name, graph_decl_op_name, graph_decl_sig_name}

def initialize_ops_pattern():
    # ops category
    ops_pattern[all_pattern_name] = "*"
    ops_pattern[aten_op_name] = "aten::"
    ops_pattern[prim_op_name] = "prim::"
    ops_pattern[fb_op_name] = "fb::"
    ops_pattern[quantized_op_name] = "quantized::"
    ops_pattern[block_op_name] = "[[:space:]]+block*()"
    ops_pattern[block_return_name] = "[[:space:]]+-> *"
    ops_pattern[internal_op_name] = "internal::"
    ops_pattern[caffe2_op_name] = "caffe2::"
    ops_pattern[return_op_name] = "[[:space:]]+return *"
    ops_pattern[graph_decl_op_name] = "^graph\("
    ops_pattern[graph_decl_sig_name] = "^[[:space:]]+%\w+\.\w+ \: \w+\)\:$"

    # ops pattern
    ops_pattern[if_op_name] = "prim::If"
    ops_pattern[loop_op_name] = "prim::Loop"
    ops_pattern[call_op_name] = "prim::CallMethod"
    ops_pattern[setattr_op_name] = "prim::SetAttr"
    ops_pattern[getattr_op_name] = "prim::GetAttr"

def sort_by_graph_size(stats_table):
    ordered_stats = OrderedDict(sorted(stats_table.items(), key=lambda x:x[1][all_pattern_name]))
    return ordered_stats

def sort_by_key(stats_table):
    return OrderedDict(sorted(stats_table.items()))

def get_ignored_ops():
    ignored_ops = ""
    for op_name in ops_pattern:
        if op_name != all_pattern_name and op_name not in reported_ops:
            ignored_ops += f"{op_name} "
    return ignored_ops

def print_benchmark_stats(stats_table):
    stats_table = sort_by_graph_size(stats_table)
    ignored_ops = "graph-decl/others, return, block*, block-ret"
    print("=======================================================================================================")
    print(f"Note: \"others\" is the sum of unreported ops ({get_ignored_ops()})")
    print("=======================================================================================================")
    headers = ['Logfile (ir counts)'] + reported_ops + [f"{all_pattern_name} [others]"]
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
    for ir_name, grep_str in ops_pattern.items():
        if grep_str == "*":
            continue
        grep_cmd = f"egrep -v \"{grep_str}\" {tmp_source} > {tmp_output}"
        subprocess.run(grep_cmd, shell=True, check=False)
        mv_cmd = f"mv {tmp_output} {tmp_source}"
        subprocess.run(mv_cmd, shell=True, check=True)
    if os.path.getsize(tmp_source) != 0:
        print(f"{logfile_name} has lines with undetected patterns:")
        file1 = open(tmp_source)
        print(file1.read())
        file1.close()

def process_stats(logfile_name, stats_table):
    stat = {}
    for ir_name, grep_str in ops_pattern.items():
        cmd = f"egrep \"{grep_str}\" {logfile_name} | wc -l"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        count = int(output)
        stat[ir_name] = count
    stats_table[logfile_name] = stat

def count_nonzero_ratio(count, total_ops):
    if count == 0:
        return f"-"
    else:
        ratio = count * 100/total_ops
        return f"{count} ({ratio:2.0f}%)"

def postprocess_stats(stat):
    stat_with_percentage = []
    total_ops = stat[all_pattern_name]
    assert(total_ops > 0)

    # print "reported_ops"
    for op_name in reported_ops:
        count = stat[op_name]
        stat_with_percentage.append(count_nonzero_ratio(count, total_ops))

    # print "total_ops" and "other_ops"
    other_ops = 0
    for op_name, count in stat.items():
        if op_name != all_pattern_name and op_name not in reported_ops:
            other_ops += count
    stat_with_percentage.append(f"{total_ops} [{other_ops}]")
    #stat_with_percentage.append(count_nonzero_ratio(other_ops, total_ops))

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

    initialize_ops_pattern()

    if args.list_ir:
        print("IR type names: ")
        for op, pattern in ops_pattern.items():
            if op != "all": print(f"   - {op} (egrep=\"{pattern}\")")

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
