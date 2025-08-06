#!/bin/bash

src_dir="/home/saul/Downloads/archive/celeba_hq_256"
dst_dir="/home/saul/code/rknn_python/img_celeba_256"
output_txt="/home/saul/code/rknn_python/dataset.txt"

mkdir -p "$dst_dir"
> "$output_txt"

# 复制前 80 张图片并记录路径
find "$src_dir" -type f | sort | head -n 80 | while read -r filepath; do
  filename=$(basename "$filepath")
  cp "$filepath" "$dst_dir/$filename"
  echo "$dst_dir/$filename" >> "$output_txt"
done
