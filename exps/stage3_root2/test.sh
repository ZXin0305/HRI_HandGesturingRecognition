export PROJECT_HOME='/path/to/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python test.py -p "/home/zx/code2020/SMAP-master/pretrained" \
-t run_inference \
-d test \
-rp "/home/zx/code2020/SMAP-master/pretrained" \
--batch_size 16 \
--do_flip 1 \
--dataset_path "/path/to/custom/image_dir"
