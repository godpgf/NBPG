# Navigability-Balanced Proximity Graph (NBPG) for Efficient Approximate Nearest Neighbor Search

## 一、下载数据集

首先，下载1百万规模的BIGANN数据集。

```shell
mkdir -p data/ods && cd data/ods
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
mv sift sift1m
```

## 二、编译所有工具

```shell
mkdir build
cd build
make
```

## 三、数据处理

对下载的数据集进行格式转换
```shell
./FmtConV vec2bin --ifile ../data/ods/sift1m/sift_query.fvecs --ofile ../data/ods/sift/query.fbin
./FmtConV vec2bin --ifile ../data/ods/sift/sift_base.fvecs --ofile ../data/ods/sift/base.fbin
```

如果数据集中所有值都在0~255，可以将数据类型转成uint8
```shell
./FmtConV fbin2u8bin --ifile ../data/ods/sift/query.fbin --ofile ../data/ods/sift/query.u8bin
./FmtConV fbin2u8bin --ifile ../data/ods/sift/base.fbin --ofile ../data/ods/sift/base.u8bin
```

如果数据集中的取值分布不在0~255，可以对向量进行压缩
```shell
# 计算压缩矩阵
cd ../py_data_tools
python pca_process.py ../data/ods/sift/sift_base.fbin ../data/dw/sift/sift_base_64.mat 64
# [方式1]：压缩成无符号整数
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.u8bin --matFile ../data/dw/sift/sift_base_64.mat
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.u8bin --matFile ../data/dw/sift/sift_base_64.mat
# [方式2]：压缩成带符号的整数
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.i8bin --matFile ../data/dw/sift/sift_base_64.mat --data_type int8
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.i8bin --matFile ../data/dw/sift/sift_base_64.mat --data_type int8
# [方式3]：将chunk_size个维度压缩成一个chunk，存储在一个无符号整数中
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.u4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.u4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2
# [方式4]：将chunk_size个维度压缩成一个chunk，存储在一个带符号整数中
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.i4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2 --data_type int8
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.i4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2 --data_type int8
```

为每个查询向量找到最近的k个底库向量
```shell
./CompGT --base_path ../data/ods/sift/sift_base.u8bin --query_path ../data/ods/sift/sift_query.u8bin --gt_path ../data/ods/sift/sift.gt
```

测试经过压缩后的向量的召回率
```shell
# [方式1]
./PCATest --base_path ../data/dw/sift/sift_base_64.u8bin --query_path ../data/dw/sift/sift_query_64.u8bin --gt_path ../data/ods/sift/sift.gt --q_num 100
# [方式2]
./PCATest --base_path ../data/dw/sift/sift_base_64.i8bin --query_path ../data/dw/sift/sift_query_64.i8bin --gt_path ../data/ods/sift/sift.gt --q_num 100 --data_type int8
# [方式3]
./PCATest --base_path ../data/dw/sift/sift_base_64.u4bin --query_path ../data/dw/sift/sift_query_64.u4bin --gt_path ../data/ods/sift/sift.gt --q_num 100 --chunk_size 2
# [方式4]
./PCATest --base_path ../data/dw/sift/sift_base_64.i4bin --query_path ../data/dw/sift/sift_query_64.i4bin --gt_path ../data/ods/sift/sift.gt --q_num 100 --chunk_size 2 --data_type int8
```

推荐使用“方式3”或“方式4”，将取得效率与精度的完美平衡。

## 三、对比实验

对比其他方案
### 1、sift1m
```shell
# vamana
./Vamana --quant_base_path ../data/ods/sift1m/base.u8bin --quant_query_path ../data/ods/sift1m/query.u8bin --gt_path ../data/ods/sift1m/groundtruth.gt --index_path ../data/dw/sift1m/vamana_index --res_path ../data/dw/sift1m/vamana.csv --num_split 0 --num_passes 1 --rebuild

# hnsw
./HNSW --quant_base_path ../data/ods/sift1m/base.u8bin --quant_query_path ../data/ods/sift1m/query.u8bin --gt_path ../data/ods/sift1m/groundtruth.gt --index_path ../data/dw/sift1m/hnsw_index --res_path ../data/dw/sift1m/hnsw.csv --rebuild

# nsg
./nnDescent --quant_base_path ../data/ods/sift1m/base.u8bin --quant_query_path ../data/ods/sift1m/query.u8bin --gt_path ../data/ods/sift1m/groundtruth.gt --index_path ../data/dw/sift1m/nndescent_index --res_path ../data/dw/sift1m/nndescent.csv --L 50 --R 50 --alpha 0 --rebuild
./Vamana --quant_base_path ../data/ods/sift1m/base.u8bin --quant_query_path ../data/ods/sift1m/query.u8bin --gt_path ../data/ods/sift1m/groundtruth.gt --index_path ../data/dw/sift1m/nsg_index --res_path ../data/dw/sift1m/nsg.csv --num_split 0 --num_passes 1 --rebuild --pre_index_path ../data/dw/sift1m/nndescent_index --rand_sp

# NBPG
./Vamana --quant_base_path ../data/ods/sift1m/base.u8bin --quant_query_path ../data/ods/sift1m/query.u8bin --gt_path ../data/ods/sift1m/groundtruth.gt --index_path ../data/dw/sift1m/ceg_index --num_passes 2 --dir_bias_scale 0 --num_split 4 --res_path ../data/dw/sift1m/nbpg.csv --dynamic_prune --rebuild --min_alpha 1.05

# FGIM
./FGIM --quant_base_path ../data/ods/sift1m/base.u8bin --quant_query_path ../data/ods/sift1m/query.u8bin --gt_path ../data/ods/sift1m/groundtruth.gt --index_path ../data/dw/sift1m/figm_index --num_passes 2 --dir_bias_scale 0 --num_split 4 --res_path ../data/dw/sift1m/fgim.csv --rebuild
```
