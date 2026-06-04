#pragma once
#include <unordered_set>
#include "nsw.h"
#include "utils/Graph/beamSearch.h"
#include "utils/Graph/prune.h"
#include "utils/Graph/merge_edges.h"
#include "utils/stats/get_time.h"
#include "utils/Graph/graph.h"
#include "utils/utilities.h"

namespace ant
{
    template <typename indexType>
    struct SPANNIndex
    {
    public:
        using QNode = QueryNode<indexType>;
        using IDist = IdDist<indexType>;

        SPANNIndex(uint32_t bin_size) : bin_size(bin_size) {}

        void init_c2v_ids(size_t pointSize, size_t centroid_size){
            c2v_ids.resize(pointSize);
            std::iota(c2v_ids.begin(), c2v_ids.end(), 0);
            std::shuffle(c2v_ids.begin(), c2v_ids.end(), std::mt19937(0));
            c2v_ids.resize(centroid_size);
            c2v_ids.shrink_to_fit();
        }

        void init_v2c_ids(size_t pointSize){
            v2c_ids.resize(pointSize);
            std::fill_n(v2c_ids.data(), vector_num, -1);
            for (uint32_t i = 0; i < c2v_ids.size(); ++i)
            {
                v2c_ids[c2v_ids[i]] = i;
            }            
        }

        template <typename PR>
        size_t build_centroid_graph(PR &Points, unsigned maxDeg, uint32_t L, uint32_t VL,
                        uint32_t top_degree, uint32_t num_split, uint32_t num_passes, 
                        float dir_bias_scale, double min_alpha, double max_alpha, double min_cos_angle, double max_cos_angle,
                        bool dynamic_prune, bool rand_sp, bool repair_isolate, QueryStatic &BuildStats){
            size_t n = v2c_ids.size();
            centroidGraph = Graph<uint32_t>(maxDeg, n);
            RefPointRange<PR, indexType> centroids = RefPointRange<PR, indexType>(c2v_ids.data(), c2v_ids.size(), &Points);
            size_t limit = centroids.size();

            return build_vamana(centroids, centroidGraph, L, VL,
                        top_degree, num_split, num_passes, limit,
                        dir_bias_scale, min_alpha, max_alpha, min_cos_angle, max_cos_angle,
                        dynamic_prune, rand_sp, repair_isolate, BuildStats)            
        }

        template <typename PR>
        void build_ivf(PR &Points, uint32_t beamSize, size_t limit, double ivf_alpha, uint32_t ivf_R, indexType max_ivf_size, size_t max_visited_size, bool congestionAwareAssignment, QueryStatic &BuildStats, bool keep_order = false, size_t max_cache_size = 10000)
        {
            RefPointRange<PR, indexType> centroids = RefPointRange<PR, indexType>(c2v_ids.data(), c2v_ids.size(), &Points);

            if(max_cache_size < Points.size() * 0.001f){
                max_cache_size = 100000;
            }

            ivf_index.resize(bin_size * centroids.size());
            for (size_t i = 0; i < ivf_index.size(); ++i)
            {
                ivf_index[i].resize(0);
            }

            auto less = [](id_dist a, id_dist b)
            {
                return id_dist_less(a, b);
            };

            // 申请beamSearch需要的空间
            auto num_workers = omp_get_max_threads();
            auto bsTable = BeamSearchMemoryTable<indexType>(max_cache_size, num_workers, beamSize, max_visited_size, centroidGraph.max_degree());
            std::vector<BeamSearchMemoryCell<indexType>> new_out_(max_cache_size);
            for (size_t sid = 0; sid < Points.size(); sid += max_cache_size)
            {
                size_t cur_batch_size = sid + max_cache_size > Points.size() ? Points.size() - sid : max_cache_size;

#pragma omp parallel for
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    uint worker_id = omp_get_thread_num();
                    auto bsCell = bsTable.getCell(i, worker_id, beamSize, max_visited_size);
                    // searchNghAndPrune(sid + i, bsCell);
                    indexType index = sid + i;
                    auto qp = Points[index];
                    // 搜索最近质心
                    ivf_searchNgh(index, bsCell,
                                  qp, centroids, beamSize,
                                  limit, BuildStats);

                    if (bsCell.visited_size > beamSize)
                        bsCell.visited_size = beamSize;

                    // 删除拥挤的质心
                    size_t new_visited_size = 0;
                    size_t candidate_id = 0;
                    while (candidate_id < bsCell.visited_size)
                    {
                        auto cnt = get_ivf_size(ivf_index, bsCell.visited[candidate_id++].first, bin_size);
                        if (cnt > max_ivf_size)
                            continue;
                        bsCell.visited[new_visited_size] = bsCell.visited[candidate_id - 1];
                        if (congestionAwareAssignment)
                            bsCell.visited[new_visited_size].second *= (1 + log(cnt + 1));
                        new_visited_size++;
                    }
                    bsCell.visited_size = new_visited_size;

                    if (congestionAwareAssignment)
                        std::sort(bsCell.visited, bsCell.visited + new_visited_size, less);

                    if (bsCell.visited_size > ivf_R)
                        bsCell.visited_size = ivf_R;

                    new_out_[i] = bsCell;
                }

                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    indexType index = sid + i;
                    uint32_t bin_id = index % bin_size;
                    uint32_t iid = index / bin_size;
                    for (size_t j = 0; j < new_out_[i].visited_size; ++j)
                    {
                        auto [centroid_id, dist] = new_out_[i].visited[j];
                        auto ivf_id = centroid_id * bin_size + bin_id;
                        ivf_index[ivf_id].push_back(iid);
                        
                        if (keep_order)
                        {
                            auto *ptr = ivf_index[ivf_id].data();
                            auto ptr_size = ivf_index[ivf_id].size();
                            // 找到第一个大于current的元素位置
                            auto position_id = std::upper_bound(ptr, ptr + (ptr_size - 1), iid) - ptr;
                            if (position_id < ptr_size - 1)
                            {
                                std::rotate(ptr + position_id, ptr + (ptr_size - 1), ptr + ptr_size);
                            }
                        }
                    }
                }

                printProgressBar((sid + cur_batch_size) / (double)Points.size());
            }
        }

        // 记录向量对应的质心id和质心对应的向量id
        std::vector<indexType> c2v_ids;
        std::vector<uint32_t> v2c_ids;
        // 质心图
        Graph<uint32_t> centroidGraph;

        // 记录质心对应的向量列表
        uint32_t bin_size;
        std::vector<std::vector<uint32_t>> ivf_index;

    protected:
        // 搜索一个向量的邻居质心
        template <typename CPR>
        void ivf_searchNgh(indexType index, BeamSearchMemoryCell<indexType> &bsCell,
                           typename CPR::Point &qp, CPR &centroids,
                           uint32_t beamSize, size_t limit, QueryStatic &BuildStats)
        {
            if (v2c_ids[index] != -1)
            {
                // 直接插入对应质心的ivf就行
                bsCell.visited[0].first = v2c_ids[index];
                bsCell.visited[0].second = distType(0.0f);
                bsCell.visited_size = 1;
            }
            else
            {
                QueryParams QP(0, beamSize, 0.0, limit, centroidGraph.max_degree());

                // 搜索邻居
                indexType sp = 0;
                uint sp_num = 1;

                // 搜索邻居
                bsCell.init_starting_points(qp, centroids, &sp, sp_num);

                size_t bs_distance_comps = beam_search(bsCell, centroidGraph, qp, centroids, QP);
                BuildStats.increment_dist(index, bs_distance_comps);
                BuildStats.increment_visited(index, bsCell.num_visited);
            }
        }
    };
}
