// compute groundtruth
#include <fstream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <limits> // max float
#include "CLI11.hpp"
#include "log.h"
#include "utils/Point/point.h"
#include "utils/Point/point_range.h"
#include "utils/utilities.h"
using namespace ant;

struct Args
{
    // 输入输出文件
    std::string base_path;
    std::string query_path;
    std::string gt_path;

    bool use_mips = false;

    // std::string data_type = "uint8";
    // std::string dist_func = "euclidian";
    // std::string index_type = "uint32";
    int k = 100;
};

template <typename PR, typename indexType>
void compute_groundtruth(Args args)
{
    using id_dist = std::pair<indexType, float>;
    int k = args.k;
    indexType n, dims;
    PR query_points;
    {
        std::ifstream reader(args.query_path, std::ios::in | std::ios::binary);
        assert(reader.is_open());
        reader.read((char *)&n, sizeof(indexType));
        reader.read((char *)&dims, sizeof(indexType));
        query_points = PR(n, dims);
        query_points.load(reader);
    }

    std::vector<std::vector<id_dist>> answers(query_points.size());

    std::vector<id_dist> top_cache(query_points.size(), std::make_pair(0, -std::numeric_limits<float>::max()));

    uint32_t BLOCK_SIZE = 100000;
    using Parameters = typename PR::Parameters;
    PR base_points = PR(BLOCK_SIZE, Parameters(query_points.dimension(), args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDIAN));
    std::ifstream reader(args.base_path, std::ios::in | std::ios::binary);
    assert(reader.is_open());
    reader.read((char *)&n, sizeof(indexType));
    reader.read((char *)&dims, sizeof(indexType));
    for (size_t i = 0; i < n; i += BLOCK_SIZE)
    {
        auto batch_size = (i + BLOCK_SIZE) > n ? n - i : BLOCK_SIZE;
        base_points.resize(batch_size);
        base_points.load(reader);

#pragma omp parallel for
        for (auto j = 0; j < query_points.size(); ++j)
        {
            auto &topk = answers[j];
            auto [toppos, topdist] = top_cache[j];
            for (size_t ii = 0; ii < batch_size; ii++)
            {

                float dist = base_points[ii].distance(query_points[j]);
                if (topk.size() < k)
                {
                    if (dist > topdist)
                    {
                        topdist = dist;
                        toppos = topk.size();
                    }
                    topk.push_back(std::make_pair((indexType)i + ii, dist));
                }
                else if (dist < topdist)
                {
                    float new_topdist = topk[0].second;
                    indexType new_toppos = 0;
                    topk[toppos] = std::make_pair((indexType)i + ii, dist);
                    for (size_t l = 1; l < topk.size(); l++)
                    {
                        if (topk[l].second > new_topdist)
                        {
                            new_topdist = topk[l].second;
                            new_toppos = l;
                        }
                    }
                    topdist = new_topdist;
                    toppos = new_toppos;
                }
                top_cache[j] = std::make_pair(toppos, topdist);
            }
        }

        printProgressBar((i + batch_size) / (double)n);
    }

    // save answers-------------------------------------------------------------------------------------------
    std::vector<indexType> flat_ids = std::vector<indexType>(query_points.size() * k);
    std::vector<float> flat_dist = std::vector<float>(query_points.size() * k);

    auto less = [&](id_dist a, id_dist b)
    {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
    };

#pragma omp parallel for
    for (auto j = 0; j < query_points.size(); ++j)
    {
        auto *flat_ids_ptr = flat_ids.data() + j * k;
        auto *flat_dist_ptr = flat_dist.data() + j * k;
        auto &topk = answers[j];
        std::sort(topk.begin(), topk.end(), less);

        std::transform(topk.data(), topk.data() + k, flat_ids_ptr, [](id_dist v)
                       { return v.first; });
        std::transform(topk.data(), topk.data() + k, flat_dist_ptr, [](id_dist v)
                       { return v.second; });
    }

    std::ofstream writer(args.gt_path, std::ios::out | std::ios::binary);
    n = query_points.size();
    dims = k;
    writer.write((char *)&n, sizeof(indexType));
    writer.write((char *)&dims, sizeof(indexType));
    writer.write((char *)flat_ids.data(), sizeof(indexType) * flat_ids.size());
    writer.write((char *)flat_dist.data(), sizeof(float) * flat_dist.size());
    writer.close();
}

int main(int argc, char *argv[])
{
    Args args;
    CLI::App app{"CompGT"};

    app.add_option("--base_path", args.base_path, "File path for input vectors");
    app.add_option("--query_path", args.query_path, "File path for query vectors");
    app.add_option("--gt_path", args.gt_path, "File path for output groundtruth");
    // app.add_option("--data_type", args.data_type, "data type");
    // app.add_option("--dist_func", args.dist_func, "dist func");
    // app.add_option("--index_type", args.index_type, "index type");
    app.add_flag("--use_mips", args.use_mips, "verbose");
    app.add_option("--k", args.k, "top k");

    CLI11_PARSE(app, argc, argv);

    if (ends_with(args.base_path, ".u8bin"))
    {
        compute_groundtruth<PointRange<Point<uint8_t>>, uint32_t>(args);
    }
    else if (ends_with(args.base_path, ".i8bin"))
    {
        compute_groundtruth<PointRange<Point<int8_t>>, uint32_t>(args);
    }
    else if (ends_with(args.base_path, ".fbin") || ends_with(args.base_path, ".bin"))
    {
        compute_groundtruth<PointRange<Point<float>>, uint32_t>(args);
    }

    return 0;
}