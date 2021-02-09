#!/bin/bash

#benchmarks="BERT_pytorch Background_Matting LearningToPaint Super_SloMo alexnet attention_is_all_you_need demucs densenet121 dlrm fastNLP maml mnasnet1_0 mobilenet_v2 moco pytorch_CycleGAN_and_pix2pix pytorch_stargan pytorch_struct resnet18 resnet50 resnext50_32x4d shufflenet_v2_x1_0 squeezenet1_1 tacotron2 vgg16 yolov3"
benchmarks="pytorch_stargan"
core="13"

for prog in ${benchmarks}; do
   test_name="test_${prog}_example_cpu"
   dump_filename="${prog}.last_executed_graph.dump.log"
   cmd_line="taskset -c ${core} python test.py -k \"${test_name}\""
   echo "${cmd_line} > ${dump_filename}"
   echo ${cmd_line}
   ${cmd_line} > ${dump_filename}
done
