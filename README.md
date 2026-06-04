# Navigability-Balanced Proximity Graph (NBPG) for Efficient Approximate Nearest Neighbor Search

## 1. Download Dataset

First, download the 1-million-scale BIGANN dataset.

```shell
mkdir -p data/ods && cd data/ods
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
mv sift sift1m
```

## 2. Build All Tools

```shell
mkdir build
cd build
make
```

## 3. Data Processing

Convert the downloaded dataset format

```shell
./FmtConV vec2bin --ifile ../data/ods/sift1m/sift_query.fvecs --ofile ../data/ods/sift/query.fbin
./FmtConV vec2bin --ifile ../data/ods/sift/sift_base.fvecs --ofile ../data/ods/sift/base.fbin
```

If all values in the dataset are in the range 0~255, convert the data type to uint8

```shell
./FmtConV fbin2u8bin --ifile ../data/ods/sift/query.fbin --ofile ../data/ods/sift/query.u8bin
./FmtConV fbin2u8bin --ifile ../data/ods/sift/base.fbin --ofile ../data/ods/sift/base.u8bin
```

If the value distribution in the dataset is not within 0~255, compress the vectors

```shell
# Compute the compression matrix
cd ../py_data_tools
python pca_process.py ../data/ods/sift/sift_base.fbin ../data/dw/sift/sift_base_64.mat 64
# [Method 1]: Compress to unsigned integers
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.u8bin --matFile ../data/dw/sift/sift_base_64.mat
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.u8bin --matFile ../data/dw/sift/sift_base_64.mat
# [Method 2]: Compress to signed integers
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.i8bin --matFile ../data/dw/sift/sift_base_64.mat --data_type int8
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.i8bin --matFile ../data/dw/sift/sift_base_64.mat --data_type int8
# [Method 3]: Compress `chunk_size` dimensions into one chunk and store in an unsigned integer
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.u4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.u4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2
# [Method 4]: Compress `chunk_size` dimensions into one chunk and store in a signed integer
./PCACompress --ifile ../data/ods/sift/sift_base.fbin --ofile ../data/dw/sift/sift_base_64.i4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2 --data_type int8
./PCACompress --ifile ../data/ods/sift/sift_query.fbin --ofile ../data/dw/sift/sift_query_64.i4bin --matFile ../data/dw/sift/sift_base_64.mat --chunk_size 2 --data_type int8
```

Find the nearest k base vectors for each query vector

```shell
./CompGT --base_path ../data/ods/sift/sift_base.u8bin --query_path ../data/ods/sift/sift_query.u8bin --gt_path ../data/ods/sift/sift.gt
```

Test the recall of compressed vectors

```shell
# [Method 1]
./PCATest --base_path ../data/dw/sift/sift_base_64.u8bin --query_path ../data/dw/sift/sift_query_64.u8bin --gt_path ../data/ods/sift/sift.gt --q_num 100
# [Method 2]
./PCATest --base_path ../data/dw/sift/sift_base_64.i8bin --query_path ../data/dw/sift/sift_query_64.i8bin --gt_path ../data/ods/sift/sift.gt --q_num 100 --data_type int8
# [Method 3]
./PCATest --base_path ../data/dw/sift/sift_base_64.u4bin --query_path ../data/dw/sift/sift_query_64.u4bin --gt_path ../data/ods/sift/sift.gt --q_num 100 --chunk_size 2
# [Method 4]
./PCATest --base_path ../data/dw/sift/sift_base_64.i4bin --query_path ../data/dw/sift/sift_query_64.i4bin --gt_path ../data/ods/sift/sift.gt --q_num 100 --chunk_size 2 --data_type int8
```

It is recommended to use "Method 3" or "Method 4" to achieve the perfect balance between efficiency and accuracy.

## 3. Comparative Experiments

Compare with other methods


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
