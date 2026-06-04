#pragma once
#include <fstream>
#include <cstdint>
#include <omp.h>
#include "utils/Point/point.h"
#include "utils/Point/point_range.h"
#include "utils/utilities.h"

namespace ant
{

    template<typename indexType, typename PR>
    void loadFloatPoint(PointRange<Point<float>> &to_points, PR &form_quant_points, indexType* inserts){
        assert(to_points.params.distance_metric == DistanceMetric::MIPS);
        using dataType = typename PR::T;
        auto dims = form_quant_points.dimension();
        #pragma omp parallel for
            for (auto j = 0; j < to_points.size(); ++j)
            {
                float *ptr = to_points[j].data();
                dataType *from_ptr = form_quant_points[inserts[j]].data();
                std::transform(from_ptr, from_ptr + dims, ptr, [](dataType v)
                                   { return (float)v; });
            }
    }

    void loadFloatPoint(std::string iFile, std::ifstream &reader, PointRange<Point<float>> &from_points, PointRange<Point<uint8_t>> &form_quant_points)
    {
        assert(to_points.params.distance_metric == DistanceMetric::MIPS);
        assert(form_quant_points.params.distance_metric == DistanceMetric::MIPS);
        auto dims = from_points.dimension();
        if (ends_with(iFile, ".fbin") || ends_with(iFile, ".bin"))
        {
            from_points.load(reader);
        }
        else if (ends_with(iFile, ".u8bin") || ends_with(iFile, ".i8bin"))
        {
            form_quant_points.load(reader);
            bool is_u8 = ends_with(iFile, ".u8bin");
#pragma omp parallel for
            for (auto j = 0; j < form_quant_points.size(); ++j)
            {
                float *ptr = from_points[j].data();
                uint8_t *from_ptr = form_quant_points[j].data();
                if (is_u8)
                {
                    std::transform(from_ptr, from_ptr + dims, ptr, [](uint8_t v)
                                   { return (float)v; });
                }
                else
                {
                    std::transform(from_ptr, from_ptr + dims, ptr, [](uint8_t v)
                                   { return (float)((int8_t)v); });
                }
            }
        }
    }

    std::pair<PointRange<Point<float>>, std::vector<float>> loadMatrix(std::string matFile)
    {
        PointRange<Point<float>> mat;
        std::vector<float> bias;

        std::ifstream reader(matFile, std::ios::in | std::ios::binary);
        assert(reader.is_open());
        uint32_t d_out, d_in;
        bool has_bias;
        reader.read((char *)&d_out, sizeof(uint32_t));
        reader.read((char *)&d_in, sizeof(uint32_t));
        reader.read((char *)&has_bias, sizeof(bool));

        using Parameters = typename PointRange<Point<float>>::Parameters;
        mat = PointRange<Point<float>>(d_out, Parameters(d_in, DistanceMetric::MIPS));
        mat.load(reader);
        bias = std::vector<float>(d_out, 0);
        if (has_bias)
        {
            reader.read((char *)bias.data(), d_out * sizeof(float));
        }
        reader.close();

        return std::pair<PointRange<Point<float>>, std::vector<float>>(mat, bias);
    }


    void matmul(PointRange<Point<float>> &from_points, PointRange<Point<float>> &to_points, PointRange<Point<float>> &mat, std::vector<float> &bias)
    {
#pragma omp parallel for
        for (auto j = 0; j < from_points.size(); ++j)
        {
            float *values = to_points[j].data();
            for (auto jj = 0; jj < bias.size(); ++jj)
            {
                values[jj] = bias[jj] - mat[jj].distance(from_points[j]);
            }
        }
    }

    void compress(PointRange<Point<float>> &to_points, PointRange<Point<uint8_t>> &compress_points, std::string data_type, uint32_t chunk_size)
    {
        auto dims = to_points.dimension();
        auto out_dims = compress_points.dimension();
        int max_num = ((1 << 8) - 1);
        int offset = 8 - (8 / chunk_size);
        int delta = (data_type == "uint") ? 0 : (max_num + 1) >> 1;
        int mask = (1 << (8 / chunk_size)) - 1;
#pragma omp parallel for
        for (auto pi = 0; pi < to_points.size(); ++pi)
        {
            float *values = to_points[pi].data();
            uint8_t *to_values = compress_points[pi].data();

            for (int i = 0; i < out_dims; ++i)
            {
                uint8_t res = 0;
                for (int j = 0; j < chunk_size; ++j)
                {
                    float v = values[(i + 1) * chunk_size - 1 - j];
                    res = (res << (8 / chunk_size));
                    int ti = (v + 1) * 0.5 * max_num;
                    ti = ((std::clamp(ti, 0, max_num) - delta) >> offset);
                    res += static_cast<uint8_t>(ti & mask);
                }
                to_values[i] = res;
            }
        }
    }
}