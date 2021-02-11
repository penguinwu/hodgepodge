#!/bin/bash

#benchmarks_not_yet_collected="attention_is_all_you_need demucs dlrm maml yolov3"

#benchmarks_mod_init="BERT_pytorch pytorch_stargan"
#benchmarks2_dump_in_test_py="mobilenet_v2 mobilenet_v2 LearningToPaint alexnet densenet121 fastNLP mnasnet1_0 pytorch_struct resnet18 resnet50 resnext50_32x4d shufflenet_v2_x1_0 squeezenet1_1 vgg16"
#benchmarks_no_jit="Background_Matting moco pytorch_CycleGAN_and_pix2pix Super_SloMo tacotron2"

benchmarks="pytorch_mobilenet_v3"

core="13"

for prog in ${benchmarks}; do
   test_name="test_${prog}_example_cpu"
   dump_filename="${prog}.last_executed_graph.dump.log"
   #cmd_line="taskset -c ${core} python test.py -k \"${test_name}\" "
   echo "taskset -c ${core} python test.py -k "${test_name}" > ${dump_filename}"
   taskset -c ${core} python test.py -k "${test_name}" > ${dump_filename}
done
