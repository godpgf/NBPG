#pragma once
#include <cstdint>
#include <sys/mman.h> // madvise, MADV_HUGEPAGE
#include <omp.h>
#include <iostream>
#include "utils/mmap.h"

namespace ant
{
    template<typename indexType>
    std::pair<indexType, indexType> read_head(const char* filename){
        std::ifstream reader(filename, std::ios::in | std::ios::binary);
        assert(reader.is_open());
        indexType n, dims;
        reader.read((char*)&n, sizeof(indexType));
        reader.read((char*)&dims, sizeof(indexType));
        reader.close();
        return std::make_pair(n, dims);
    }

    // 不支持
    template <class Point_>
    struct DiskPointRange
    {
        using Point = Point_;
        using Parameters = typename Point::Parameters;
        using byte = uint8_t;
        uint32_t dimension() const { return params.dims; }
        size_t capacity() { return n; }

        DiskPointRange(const char *filename, size_t num_points, uint32_t d, size_t headOffset, const Parameters p) : n(num_points), filename(filename), headOffset(headOffset), filelength(0), params(p)
        {
            aligned_bytes = params.num_bytes();
        }

        DiskPointRange(const char *filename, size_t num_points, uint32_t d, size_t headOffset, DistanceMetric distance_metric=DistanceMetric::EUCLIDIAN) : DiskPointRange(filename, num_points, d, headOffset, Parameters(d, distance_metric)) {
        }

        void init_cache()
        {
            std::tie(fileptr, filelength) = mmapStringFromFile(filename);
        }

        void clear_cache()
        {
            if (filelength > 0)
            {
                munmap(fileptr, filelength);
            }
        }

        size_t size() { 
            return n;
        }

        void preload(size_t i){
            prefetch_async(fileptr + headOffset + i * aligned_bytes, aligned_bytes);
        }

        Point operator[](size_t i) const
        {
            if (i >= n || i < 0)
            {
                std::cout << "ERROR: point index out of range: " << i << " from range [" << 0 << ", " << n << ")" << std::endl;
                abort();
            }
            return Point((uint8_t*)(fileptr + headOffset + i * aligned_bytes), params);
        }

        Parameters params;

    protected:
        size_t aligned_bytes;
        size_t n;
        const char *filename;
        size_t headOffset;
        char *fileptr;
        size_t filelength;
    };
}