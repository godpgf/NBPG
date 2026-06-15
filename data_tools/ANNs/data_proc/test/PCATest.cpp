#include <fstream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <limits>
#include "CLI11.hpp"
#include "log.h"
#include "utils/Point/point.h"
#include "utils/Point/point_range.h"
#include "utils/stats/get_time.h"
#include "base/ground_truth.h"
#include "utils/utilities.h"
using namespace ant;

struct Args
{
    // 输入输出文件
    std::string base_path;
    std::string query_path;
    std::string gt_path;
    // std::string data_type = "uint8";
    // std::string dist_func = "euclidean";
    std::string index_type = "uint32";
    bool use_mips = false;
    // uint32_t chunk_size = 1;
    int k = 100;
    int recall_num = 320;
    int q_num = 0;
};

template <typename indexType>
float cmpRecall(const GroundTruth<indexType> &GT, std::vector<std::vector<std::pair<indexType, float>>> &res_recall, long k = 100)
{
    if (k > GT.dim)
    {
        std::cout << "Set topk = " << GT.dim << std::endl;
        k = GT.dim;
    }
    float recall = 0;
    if (GT.size() > 0)
    {
        size_t n = res_recall.size();

        int numCorrect = 0;
        for (auto i = 0; i < n; i++)
        {
            std::vector<size_t> results_with_ties;
            for (auto l = 0; l < k; l++)
                results_with_ties.push_back((size_t)GT.coordinates(i, l));

            std::set<size_t> reported_nbhs;
            for (auto l : res_recall[i])
                reported_nbhs.insert(l.first);
            for (auto l = 0; l < results_with_ties.size(); l++)
            {
                size_t t = results_with_ties[l];
                if (reported_nbhs.find(t) != reported_nbhs.end())
                {
                    numCorrect += 1;
                }
            }
        }
        recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
    }
    return recall;
}

template <typename PR, typename QyPR, typename indexType>
void test(Args args)
{
    using id_dist = std::pair<indexType, float>;
    int k = args.k;

    auto query_points = load_point_range<indexType, QyPR>(args.query_path.c_str(), args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN);
    if (args.q_num > 0)
    {
        query_points.resize(args.q_num);
    }

    indexType q_num = query_points.size();
    std::vector<std::vector<id_dist>> pq_recall = std::vector<std::vector<id_dist>>(q_num);
    id_dist tmp = std::make_pair((indexType)0, -std::numeric_limits<float>::max());
    std::vector<id_dist> top_dist_and_pos = std::vector<id_dist>(q_num, tmp);

    // std::fill_n(top_dist_and_pos, q_num, tmp);
    // memset(top_dist_and_pos.data(), 0, sizeof(id_dist) * q_num);

    uint32_t BLOCK_SIZE = 100000;
    indexType n, dims;
    std::ifstream reader(args.base_path, std::ios::in | std::ios::binary);
    assert(reader.is_open());
    reader.read((char *)&n, sizeof(indexType));
    reader.read((char *)&dims, sizeof(indexType));

    using Parameters = typename PR::Parameters;
    PR base_points = PR(BLOCK_SIZE, Parameters(dims, args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN));

    // 计算查询向量与所有当前读出向量的距离

    Timer t_infer("TestPCA");
    std::vector<float> res_buffer = std::vector<float>(q_num * BLOCK_SIZE);

    for (size_t i = 0; i < n; i += BLOCK_SIZE)
    {
        auto batch_size = (i + BLOCK_SIZE) > n ? n - i : BLOCK_SIZE;
        base_points.resize(batch_size);
        base_points.load(reader);

// 计算距离
#pragma omp parallel for
        for (size_t j = 0; j < q_num; ++j)
        {
            float *cur_res_buffer = res_buffer.data() + j * batch_size;

            auto cq = query_points[j];
            for (size_t ii = 0; ii < batch_size; ++ii)
            {
                cur_res_buffer[ii] = base_points[ii].distance(cq);
            }
        }

// fill res
#pragma omp parallel for
        for (size_t qi = 0; qi < q_num; ++qi)
        {
            float &topdist = top_dist_and_pos.data()[qi].second;
            indexType &toppos = top_dist_and_pos.data()[qi].first;
            auto &topk = pq_recall[qi];
            for (size_t vi = 0; vi < batch_size; ++vi)
            {
                float dist = res_buffer[qi * batch_size + vi];
                indexType index = i + vi;

                if (topk.size() < args.recall_num)
                {
                    if (dist > topdist)
                    {
                        topdist = dist;
                        toppos = topk.size();
                    }
                    topk.push_back(std::make_pair(index, dist));
                }
                else if (dist < topdist || (dist == topdist && index < topk[toppos].first))
                {
                    float new_topdist = -std::numeric_limits<float>::max();
                    indexType new_toppos = 0;
                    topk[toppos] = std::make_pair(index, dist);
                    for (size_t l = 0; l < topk.size(); l++)
                    {
                        if (topk[l].second > new_topdist)
                        {
                            new_topdist = topk[l].second;
                            new_toppos = (indexType)l;
                        }
                    }
                    topdist = new_topdist;
                    toppos = new_toppos;
                }
            }
        }

        printProgressBar((i + batch_size) / (n + 1e-5));
    }

    // auto less = [&](id_dist a, id_dist b)
    // {
    // return a.second < b.second || (a.second == b.second && a.first < b.first);
    // };
    // std::sort(pq_recall[0].begin(), pq_recall[0].end(), less);

    std::cout << "[Timer 1]:test pca:" << t_infer.get_next() << std::endl;
    GroundTruth<indexType> GT = GroundTruth<indexType>(args.gt_path.c_str(), false);
    // std::cout<<GT.coordinates(0, 0)<<" "<<GT.coordinates(0, 1)<<" "<<GT.coordinates(0, 2)<<std::endl;
    // std::cout<<GT.distances(0, 0)<<" "<<GT.distances(0, 1)<<" "<<GT.distances(0, 2)<<std::endl;

    std::cout << "recall " << args.k << "@" << args.recall_num << " = " << cmpRecall(GT, pq_recall, args.k) << std::endl;
}

int main(int argc, char *argv[])
{
    Args args;
    CLI::App app{"PCATest"};

    app.add_option("--base_path", args.base_path, "File path for input vectors");
    app.add_option("--query_path", args.query_path, "File path for query vectors");
    app.add_option("--gt_path", args.gt_path, "File path for output groundtruth");
    app.add_option("--index_type", args.index_type, "index type");
    app.add_option("--k", args.k, "top k");
    app.add_option("--recall_num", args.recall_num, "recall num");
    app.add_option("--q_num", args.q_num, "query num");
    app.add_flag("--use_mips", args.use_mips, "verbose");
    CLI11_PARSE(app, argc, argv);

    if (ends_with(args.base_path, ".u8bin"))
    {
        using PR = PointRange<Point<uint8_t>>;
        using QyPR = PR;
        test<PR, QyPR, uint32_t>(args);
    }
    else if (ends_with(args.base_path, ".u4bin"))
    {
        using PR = PointRange<Point<uint4_2>>;
        using QyPR = PR;
        test<PR, QyPR, uint32_t>(args);
    }
    else if (ends_with(args.base_path, ".i8bin"))
    {
        using PR = PointRange<Point<int8_t>>;
        using QyPR = PR;
        test<PR, QyPR, uint32_t>(args);
    }
    else if (ends_with(args.base_path, ".i4bin"))
    {
        using PR = PointRange<Point<int4_2>>;
        using QyPR = PR;
        test<PR, QyPR, uint32_t>(args);
    }
    else
    {
        using PR = PointRange<Point<float>>;
        test<PR, PR, uint32_t>(args);
    }

    return 0;
}