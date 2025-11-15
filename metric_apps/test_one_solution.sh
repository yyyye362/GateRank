#!/bin/bash
###
 # @Author: sssszh 376003349@qq.com
 # @Date: 2023-07-02 10:12:15
 # @LastEditors: sssszh 376003349@qq.com
 # @LastEditTime: 2023-09-12 09:11:12
 # @FilePath: /codet5_APPS/metric_apps/test_one_solution.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 


ulimit -S -s unlimited
ulimit -c unlimited

code_path=/home/yzj/CodeRL/outputs1/3B_epoch5_sas1
output_path=/home/yzj/CodeRL/outputs1/3B_epoch5_sas1_results
test_path=/home/yzj/CodeRL/data/test
example_tests=0 # 0: run hidden unit tests; 1: run example unit tests 
start=0
end=5000

# 1300
threads=1

if [ ! -d $output_path ] 
then
    echo "Directory DOES NOT exists." 
    mkdir $output_path
fi

index=0
for (( i=start;i<end;i++ )) ; do 
    echo 'testing sample index #' ${i}
    ((index++))   
    (
    python  test_one_solution.py\
        --code_path ${code_path} \
        --output_path ${output_path} \
        --test_path $test_path \
        --example_tests $example_tests \
        --i $i 
    ) &        
    if (( index % threads == 0 )); then wait; fi 
done 

wait 

