#pragma once
#include <cstdint>
#include <fstream>
#include <cassert>
#include <utility>
#include <sys/mman.h>
#include <omp.h>
#include <iostream>
#include "utils/Point/point.h"
#include "utils/mmap.h"

namespace ant
{
    // 读取向量二进制文件头，返回 {向量数量, 维度}
    template <typename IndexType>
    std::pair<IndexType, IndexType> read_vector_file_head(const char *filename)
    {
        std::ifstream reader(filename, std::ios::in | std::ios::binary);
        assert(reader.is_open());
        IndexType num_vectors, dims;
        reader.read(reinterpret_cast<char *>(&num_vectors), sizeof(IndexType));
        reader.read(reinterpret_cast<char *>(&dims), sizeof(IndexType));
        reader.close();
        return std::make_pair(num_vectors, dims);
    }

    // 基于 mmap 的磁盘向量集，按需访问，不做 64 字节对齐填充
    template <class Point_>
    struct DiskPointRange
    {
        using Point = Point_;
        using Parameters = typename Point::Parameters;
        using byte = uint8_t;

        uint32_t dimension() const { return params.dims; }
        size_t capacity() const { return num_vectors; }

        DiskPointRange(const char *filename, size_t num_vectors, uint32_t dims, size_t head_offset, const Parameters &params)
            : num_vectors(num_vectors), filename(filename), head_offset(head_offset), file_ptr(nullptr), file_length(0), params(params)
        {
            (void)dims;
            stride_bytes = params.num_bytes();
        }

        DiskPointRange(const char *filename, size_t num_vectors, uint32_t dims, size_t head_offset, DistanceMetric distance_metric = DistanceMetric::EUCLIDEAN)
            : DiskPointRange(filename, num_vectors, dims, head_offset, Parameters(dims, distance_metric))
        {
        }

        // 将向量文件映射到内存；查询前必须调用
        void init_cache()
        {
            std::tie(file_ptr, file_length) = mmapStringFromFile(filename);
        }

        void clear_cache()
        {
            if (file_length > 0)
            {
                munmap(file_ptr, file_length);
                file_ptr = nullptr;
                file_length = 0;
            }
        }

        size_t size() const { return num_vectors; }

        // 异步预读第 i 个向量所在页
        void preload(size_t i)
        {
            prefetch_async(file_ptr + head_offset + i * stride_bytes, stride_bytes);
        }

        Point operator[](size_t i) const
        {
            if (i >= num_vectors)
            {
                std::cout << "ERROR: point index out of range: " << i << " from range [0, " << num_vectors << ")" << std::endl;
                abort();
            }
            return Point(reinterpret_cast<byte *>(file_ptr + head_offset + i * stride_bytes), params);
        }

        Parameters params;

    protected:
        size_t stride_bytes;   // 磁盘上单个向量的字节跨度
        size_t num_vectors;
        const char *filename;
        size_t head_offset;    // 向量数据区相对文件开头的偏移
        char *file_ptr;
        size_t file_length;
    };
}
