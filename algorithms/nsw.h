#pragma once
#include <cmath>
#include "utils/Graph/beamSearch.h"
#include "utils/Graph/prune.h"
#include "utils/Graph/merge_edges.h"
#include "utils/stats/get_time.h"
#include "utils/Graph/graph.h"
#include "utils/Graph/dfs.h"
#include "utils/utilities.h"

namespace ant
{

    enum class NodePopular : int
    {
        FixedMinPopular = 1,
        FixedMaxPopular,
        CalPopular,
        PopularPrune
    };

    // 指数退火 + 固定窗口的批量插入区间
    struct BatchInsertRange
    {
        size_t floor;
        size_t ceiling;
        size_t next_count;
    };

    inline BatchInsertRange next_batch_range(size_t inc, size_t count, size_t total,
                                             size_t max_batch_size, double base = 2.0)
    {
        BatchInsertRange r;
        if (std::pow(base, static_cast<double>(inc)) <= static_cast<double>(max_batch_size))
        {
            r.floor = static_cast<size_t>(std::pow(base, static_cast<double>(inc))) - 1;
            r.ceiling = std::min(static_cast<size_t>(std::pow(base, static_cast<double>(inc + 1))) - 1, total);
            r.next_count = r.ceiling;
        }
        else
        {
            r.floor = count;
            r.ceiling = std::min(count + max_batch_size, total);
            r.next_count = count + (r.ceiling - r.floor);
        }
        return r;
    }

    // 批量插入进度输出（约每 10% 打印一次）
    inline void report_batch_progress(size_t floor, size_t ceiling, size_t total,
                                      float &frac, float progress_inc = 0.1f)
    {
        const auto ind = static_cast<size_t>(frac * static_cast<float>(total));
        if (floor <= ind && ceiling > ind)
        {
            frac += progress_inc;
            std::cout << "Pass " << static_cast<int>(100 * frac) << "% complete" << std::endl;
        }
    }

    template <typename indexType>
    class NSWIndex
    {
    public:
        using QNode = QueryNode<indexType>;
        using IDist = IdDist<indexType>;

        using SearchFun = std::function<void(indexType, indexType *, uint32_t, BeamSearchMemoryCell<indexType> &)>;
        using UpdateFun = std::function<void(indexType, IDist *, size_t, size_t &)>;
        using FillStartFun = std::function<void(std::vector<std::vector<indexType>> &, size_t, size_t)>;

        struct SearchResult
        {
            QNode *prune_res;
            QNode *vit_res;
            size_t prune_size;
            size_t vit_size;
            indexType index;
        };

        NSWIndex(Prune prune,
                 uint32_t maxBeamSize,
                 uint32_t max_visited_size,
                 uint32_t num_workers,
                 uint32_t max_degree,
                 size_t max_batch_size,
                 uint32_t top_degree,
                 float minAlpha,
                 float maxAlpha,
                 float minCosAngle,
                 float maxCosAngle) : prune(prune),
                                      maxBeamSize(maxBeamSize),
                                      max_visited_size(max_visited_size),
                                      num_workers(num_workers),
                                      max_degree(max_degree),
                                      minAlpha(minAlpha),
                                      maxAlpha(maxAlpha),
                                      minCosAngle(minCosAngle),
                                      maxCosAngle(maxCosAngle),
                                      max_batch_size(max_batch_size),
                                      maxPopular(1.0f),
                                      minPopular(0.0f),
                                      top_degree(top_degree == 0 ? max_degree : top_degree),
                                      bsTable(max_batch_size, num_workers, maxBeamSize, max_visited_size, max_degree),
                                      merge_edges(max_batch_size * max_degree, max_visited_size, num_workers, max_degree)

        {
        }

        template <typename PR>
        void batch_insert(indexType *inserts, size_t insert_size, Graph<indexType> &G, PR &Points,
                          uint32_t beamSize, size_t limit, QueryStatic *BuildStats, FillStartFun fillStartPoints,
                          NodePopular nodePopular = NodePopular::FixedMaxPopular, float dir_bias_scale = 0,
                          double base = 2, double max_fraction = .02, bool print = true)
        {

            auto less = [&](const QNode &a, const QNode &b)
            {
                return id_dist_less(a, b);
            };

            auto searchNghAndPrune = [&](indexType index, indexType *starting_points, uint32_t sp_num, BeamSearchMemoryCell<indexType> &bsCell)
            {
                auto [alpha, cos_angle] = getAlphaAndCosAngle(index, nodePopular);

                assert(index >= 0 && index < G.size());
                QueryParams QP(0, beamSize, 0.0, limit, max_degree);
                if (sp_num > beamSize)
                {
                    std::cout << "sp_num=" << sp_num << " beamSize=" << beamSize << " sp_num must less then beamSize!" << std::endl;
                    abort();
                }

                // 搜索邻居
                bsCell.init_starting_points(Points[index], Points, starting_points, sp_num);

                size_t bs_distance_comps = 0;

                auto curQP = QP;

                bs_distance_comps = beam_search(bsCell, G, Points[index], Points, curQP);

                if (BuildStats != nullptr)
                {
                    BuildStats->increment_dist(index, bs_distance_comps);
                    BuildStats->increment_visited(index, bsCell.num_visited);
                }

                // 将 beam search 结果与图上已有邻居合并，作为 RobustPrune 候选池
                auto ngh = G[index];
                bsCell.frontier_size = 0;
                size_t rp_distance_comps = 0;
                for (size_t i = 0; i < ngh.size(); ++i)
                {
                    if (!bsCell.has_been_seen(ngh[i]))
                    {
                        bsCell.frontier[bsCell.frontier_size++] =
                            QNode(ngh[i], Points[ngh[i]].distance(Points[index]));
                        ++rp_distance_comps;
                    }
                }
                bsCell.frontier_size = merge_union<QNode>(
                    bsCell.frontier, bsCell.frontier_size,
                    bsCell.visited, bsCell.visited_size,
                    bsCell.copy_visited, bsTable.get_max_visited_size(), less);
                prune.sort_candidates(index, G, bsCell.copy_visited, bsCell.frontier_size, dir_bias_scale);

                // RobustPrune 剪枝
                size_t start_prune_id = 1;
                rp_distance_comps += prune.robustPrune(index, bsCell.copy_visited, bsCell.frontier_size, Points, alpha, cos_angle, start_prune_id);
                assert(bsCell.frontier_size <= prune.R);

                if (BuildStats != nullptr)
                    BuildStats->increment_dist(index, rp_distance_comps);
            };

            auto updateDist = [&](indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size)
            {
                updateNeighborsForGraph(Points, index, candidates, pre_candidate_size, candidate_size, BuildStats);
            };
            // ---------------------------------------------------------------------------------

            size_t m = insert_size;
            size_t inc = 0;
            size_t count = 0;
            float frac = 0.0f;
            const size_t effective_batch_size = std::min(getMaxBatchSize(G, max_fraction), max_batch_size);

            Timer t_beam("beam search time");
            Timer t_bidirect("bidirect time");
            Timer t_prune("prune time");

            t_beam.stop();
            t_bidirect.stop();
            t_prune.stop();

            // cache start points for current batch
            std::vector<std::vector<indexType>> sp_cache(effective_batch_size);

            if (nodePopular == NodePopular::CalPopular)
            {
                popular.resize(G.size());
                memset(popular.data(), 0, G.size() * sizeof(float));
            }

            while (count < m)
            {
                const auto batch = next_batch_range(inc, count, m, effective_batch_size, base);
                const size_t floor = batch.floor;
                const size_t ceiling = batch.ceiling;
                count = batch.next_count;

                std::vector<SearchResult> new_out_(ceiling - floor);
                for (size_t i = 0; i < ceiling - floor; ++i)
                    sp_cache[i].resize(0);
                fillStartPoints(sp_cache, floor, ceiling);

                t_beam.start();
                batch_search(new_out_, inserts, floor, ceiling, bsTable, sp_cache, searchNghAndPrune);
                if (nodePopular == NodePopular::CalPopular)
                {
                    // 统计流行度
                    for (SearchResult res : new_out_)
                    {
                        for (size_t i = 0; i < res.vit_size; ++i)
                        {
                            popular[res.vit_res[i].index] += 1.0f;
                        }
                    }
                }

                // 更新图结构
                {
                    batch_update_graph(G, new_out_);

                    t_beam.stop();

                    // 反转边
                    t_bidirect.start();
                    inv_edges(merge_edges, new_out_);
                    t_bidirect.stop();

                    // 对反转边进行剪枝
                    t_prune.start();
                    batch_update_graph(G, merge_edges, updateDist, max_degree);
                    t_prune.stop();
                }

                if (print)
                    report_batch_progress(floor, ceiling, m, frac);
                inc += 1;
            }


            if (nodePopular == NodePopular::CalPopular)
            {
                // 累加popular
                {
                    std::vector<float> popular_tmp(G.size());
                    for (size_t i = 0; i < G.size(); ++i)
                    {
                        popular_tmp[i] = popular[i];
                        for(size_t j = 0; j < G[i].size(); ++j)
                        {
                            popular_tmp[G[i][j]] += popular_tmp[i];
                        }
                    }
                    memcpy(popular.data(), popular_tmp.data(), G.size() * sizeof(float));
                }

                
                std::transform(popular.begin(), popular.end(), popular.begin(), [](float a)
                               { return log(a + 1); });
                auto [min_it, max_it] = std::minmax_element(popular.begin(), popular.end());
                minPopular = (*min_it);
                maxPopular = (*max_it);
                std::cout << minPopular << "," << maxPopular << std::endl;
            }

            if (print)
            {
                t_beam.total();
                t_bidirect.total();
                t_prune.total();
            }
        }

        static size_t getMaxBatchSize(Graph<indexType> &G, double max_fraction = .02)
        {
            size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(G.size())),
                                             1000000ul);
            if (max_batch_size == 0)
                max_batch_size = 1;
            return max_batch_size;
        }

        std::pair<float, float> getAlphaAndCosAngle(indexType index, NodePopular nodePopular) const
        {

            float alpha, cos_angle;
            switch (nodePopular)
            {
            case NodePopular::FixedMinPopular:
            {
                alpha = maxAlpha;
                cos_angle = maxCosAngle;
            }
            break;
            case NodePopular::PopularPrune:
            {
                float cof = (popular[index] - maxPopular) / (minPopular - maxPopular);
                alpha = minAlpha + cof * (maxAlpha - minAlpha);
                cos_angle = minCosAngle + cof * (maxCosAngle - minCosAngle);
            }
            break;
            default:
            {
                alpha = 1;
                cos_angle = minCosAngle;
            }
            break;
            }
            return std::pair<float, float>(alpha, cos_angle);
        }

    public:
        // 快速估计新加入的邻居的距离，这些新邻居是有序的，以此性质来估计距离
        template <typename PR>
        static size_t eval_dist_of_new_neighbors(PR &Points, indexType index, IDist *candidates, size_t pre_candidate_size, size_t candidate_size, uint32_t top_degree)
        {
            size_t rp_distance_comps = 0;
            // 填写头距离
            size_t end_id = std::min(pre_candidate_size + top_degree, candidate_size);
            for (auto j = pre_candidate_size; j < end_id; ++j)
            {
                candidates[j].dist = Points[candidates[j].index].distance(Points[index]);
                rp_distance_comps++;
            }
            // 填写尾距离
            if (end_id < candidate_size)
            {
                auto a = candidates[end_id - 1].dist;
                auto b = Points[candidates[candidate_size - 1].index].distance(Points[index]);
                candidates[candidate_size - 1].dist = b;
                rp_distance_comps++;
                auto deltaDis = (a - b) / (candidate_size - end_id);

                // 填写中间距离
                for (auto j = end_id; j < candidate_size - 1; ++j)
                {
                    candidates[j].dist = a + (j - end_id + 1) * deltaDis;
                }
            }
            return rp_distance_comps;
        }

        template <typename rangeType>
        static void add_neighbors_without_repeats(const rangeType &ngh, IDist *candidates, size_t &candidate_size,
                                                  size_t max_capacity)
        {
            // 候选规模通常 <= max_degree，线性查重比 unordered_set 更省分配
            for (size_t i = 0; i < ngh.size(); ++i)
            {
                const indexType nb = ngh[i];
                bool exists = false;
                for (size_t j = 0; j < candidate_size; ++j)
                {
                    if (candidates[j].index == nb)
                    {
                        exists = true;
                        break;
                    }
                }
                if (exists || candidate_size >= max_capacity)
                    continue;
                candidates[candidate_size++].index = nb;
            }
        }

        static void batch_search(std::vector<SearchResult> &new_out_, indexType *inserts, size_t floor, size_t ceiling, BeamSearchMemoryTable<indexType> &bsTable, std::vector<std::vector<indexType>> &sp_cache, SearchFun searchNghAndPrune)
        {
            if (floor == ceiling)
                return;
#pragma omp parallel for
            for (size_t i = floor; i < ceiling; ++i)
            {
                indexType index = inserts[i];

                uint32_t cell_id = i - floor;
                auto &res = new_out_.data()[cell_id];

                uint32_t worker_id = omp_get_thread_num();

                // 找到候选邻居集合
                auto bsCell = bsTable.getCell(cell_id, worker_id);

                if (res.vit_size > 0)
                {
                    // 当前是修复
                    sp_cache[i - floor].clear();
                    sp_cache[i - floor].push_back(index);
                }

                searchNghAndPrune(index, sp_cache[i - floor].data(), sp_cache[i - floor].size(), bsCell);

                // 保存剪枝后的结果
                res.prune_res = bsCell.copy_visited;
                res.prune_size = bsCell.frontier_size;
                assert(bsCell.frontier_size <= bsTable.max_degree);
                res.vit_res = bsCell.visited;
                res.vit_size = bsCell.visited_size;
                res.index = index;
            }
        }

        static void batch_update_graph(Graph<indexType> &G, const std::vector<SearchResult> &new_out_)
        {
// 更新图结构
#pragma omp parallel for schedule(dynamic, 64) // 每个线程一次处理64个迭代
            for (size_t i = 0; i < new_out_.size(); ++i)
            {
                auto &cur_visited = new_out_[i];
                auto index = cur_visited.index;
                assert(index >= 0 && index < G.size());

                auto edges = G[index];
                edges.update_neighbors(cur_visited.prune_res, cur_visited.prune_size);
            }
        }

        static void inv_edges(MergeEdges<indexType> &merge_edges, const std::vector<SearchResult> &new_out_)
        {
            merge_edges.clear();
            for (const auto &bsCell : new_out_)
            {
                const indexType source = bsCell.index;
                for (size_t j = 0; j < bsCell.prune_size; ++j)
                {
                    const auto &edge = bsCell.prune_res[j];
                    merge_edges.insert_edge(edge.index, IDist(source, edge.dist));
                }
            }
            merge_edges.build_groups();
        }

        // HNSW 分层插入：每个节点可能在多层产生出边
        static void inv_edges(MergeEdges<indexType> &merge_edges,
                              const std::vector<std::vector<std::pair<indexType, std::vector<IDist>>>> &layer_inserts)
        {
            merge_edges.clear();
            for (const auto &node_layers : layer_inserts)
            {
                for (const auto &layer : node_layers)
                {
                    const indexType source = layer.first;
                    for (const auto &edge : layer.second)
                        merge_edges.insert_edge(edge.index, IDist(source, edge.dist));
                }
            }
            merge_edges.build_groups();
        }

        static void batch_update_graph(Graph<indexType> &G, MergeEdges<indexType> &merge_edges, UpdateFun updateDist, uint32_t max_degree)
        {
            if (merge_edges.num_groups() == 0)
                return;
            std::vector<std::vector<std::pair<indexType, float>>> res(merge_edges.num_groups());
            std::vector<indexType> all_index(merge_edges.num_groups());
#pragma omp parallel for
            for (size_t i = 0; i < merge_edges.num_groups(); ++i)
            {
                auto worker_id = omp_get_thread_num();
                indexType index = merge_edges.group_source(i);
                all_index[i] = index;
                assert(index >= 0 && index < G.size());
                auto [candidates, candidate_size] = merge_edges.group_neighbors(i, worker_id);
                auto pre_candidate_size = candidate_size;

                auto edges = G[index];
                add_neighbors_without_repeats(edges, candidates, candidate_size, merge_edges.buffer_capacity());

                updateDist(index, candidates, pre_candidate_size, candidate_size);
                edges.update_neighbors(candidates, candidate_size);
            }
        }

        // 反向边合并后：补全距离、排序/剪枝，再写回图
        template <typename PR>
        static size_t update_neighbors_for_graph(PR &Points, indexType index, IDist *candidates,
                                                 size_t pre_candidate_size, size_t &candidate_size,
                                                 Prune &prune, uint32_t max_degree, uint32_t top_degree)
        {
            const auto less = [](const IDist &a, const IDist &b)
            { return id_dist_less(a, b); };

            size_t rp_distance_comps = eval_dist_of_new_neighbors(
                Points, index, candidates, pre_candidate_size, candidate_size, top_degree);

            if (candidate_size <= max_degree)
            {
                std::sort(candidates, candidates + candidate_size, less);
            }
            else
            {
                // 候选过多时仅按距离粗剪，避免完整 RobustPrune 的开销
                prune.sort_candidates(candidates, candidate_size);
                if (candidate_size > prune.R)
                    candidate_size = prune.R;
            }
            return rp_distance_comps;
        }

        template <typename PR>
        void updateNeighborsForGraph(PR &Points, indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size, QueryStatic *BuildStats)
        {
            const size_t rp_distance_comps = update_neighbors_for_graph(
                Points, index, candidates, pre_candidate_size, candidate_size, prune, max_degree, top_degree);
            if (BuildStats != nullptr)
                BuildStats->increment_dist(index, rp_distance_comps);
        }

    protected:
        Prune prune;

        uint32_t max_visited_size;
        uint32_t num_workers;

        // 只对top_degree个邻居和最后一个邻居进行距离计算，其他邻居的距离用插值法
        uint32_t top_degree;
        uint32_t max_degree;

        // 统计
        MergeEdges<indexType> merge_edges;

    public:
        std::vector<float> popular;
        float maxPopular, minPopular;
        float minAlpha, maxAlpha;
        float minCosAngle, maxCosAngle;

        const size_t max_batch_size;
        uint32_t maxBeamSize;
        // 申请beamSearch所需要使用的空间
        BeamSearchMemoryTable<indexType> bsTable;
    };

    template <typename PR, typename indexType>
    size_t build_vamana(const PR &QPoints, Graph<indexType> &G, uint32_t L, uint32_t VL,
                        uint32_t top_degree, uint32_t num_split, uint32_t num_passes, size_t limit,
                        float dir_bias_scale, double min_alpha, double max_alpha, double min_cos_angle, double max_cos_angle,
                        bool dynamic_prune, bool rand_sp, bool repair_isolate, QueryStatic &BuildStats)
    {
        using NSWType = NSWIndex<indexType>;
        using FillStartFun = std::function<void(std::vector<std::vector<indexType>> &, size_t, size_t)>;

        auto prune = Prune(G.max_degree(), omp_get_max_threads());
        auto nsw = NSWType(prune, L, VL, omp_get_max_threads(), G.max_degree(), NSWType::getMaxBatchSize(G), top_degree,
                           min_alpha, max_alpha, min_cos_angle, max_cos_angle);

        std::vector<indexType> inserts = std::vector<indexType>(QPoints.size());
        std::iota(inserts.begin(), inserts.end(), 0);

        NodePopular nodePopular = NodePopular::FixedMaxPopular;
        if (num_split >= 1)
        {

            FillStartFun fillSP = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
            {
                for (uint32_t j = 0; j < ceiling - floor; ++j)
                {
                    indexType index = inserts[floor + j];
                    sp_cache[j].push_back(index % num_split);
                }
            };

            
            for (int i = 0; i < 2; ++i)
            {
                std::shuffle(inserts.begin(), inserts.end(), std::mt19937(i));
                nsw.batch_insert(inserts.data(), inserts.size(), G, QPoints, L, limit, &BuildStats, fillSP, nodePopular, dir_bias_scale, 2, .02);
            }
        }

        FillStartFun fillMergeSP = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
        {
            for (uint32_t j = 0; j < ceiling - floor; ++j)
            {
                indexType index = inserts[floor + j];
                for (auto s = 0; s < num_split; ++s)
                {
                    indexType rid = randomInt<indexType>(G.size()) / num_split * num_split;
                    indexType sp = rid + s;
                    if (sp >= G.size())
                        sp -= num_split;

                    sp_cache[j].push_back(sp);
                }
            }
        };

        FillStartFun fillRandSP = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
        {
            for (uint32_t j = 0; j < ceiling - floor; ++j)
            {
                indexType index = inserts[floor + j];
                sp_cache[j].push_back(inserts.size() - index - 1);
            }
        };

        FillStartFun fillDefaultSP = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
        {
            for (uint32_t j = 0; j < ceiling - floor; ++j)
            {
                sp_cache[j].push_back(0);
            }
        };

        for (int i = 0; i < num_passes; ++i)
        {
            std::shuffle(inserts.begin(), inserts.end(), std::mt19937(i));

            if (i == num_passes - 2 && dynamic_prune)
            {
                // 计算流行度
                nodePopular = NodePopular::CalPopular;
            }
            else if (i == num_passes - 1)
            {
                if (dynamic_prune && num_passes > 1)
                {
                    nodePopular = NodePopular::PopularPrune;
                }
                else
                {
                    nodePopular = NodePopular::FixedMinPopular;
                }
            }

            if (i != num_passes - 1)
            {
                nsw.batch_insert(inserts.data(), inserts.size(), G, QPoints, L, limit, &BuildStats, num_split > 1 ? fillMergeSP : fillDefaultSP, nodePopular, dir_bias_scale, 2, .02);
            }
            else
            {
                if (rand_sp)
                {
                    nsw.batch_insert(inserts.data(), inserts.size(), G, QPoints, L, limit, &BuildStats, fillRandSP, nodePopular, dir_bias_scale, 2, .02);
                }
                else
                {
                    nsw.batch_insert(inserts.data(), inserts.size(), G, QPoints, L, limit, &BuildStats, fillDefaultSP, nodePopular, dir_bias_scale, 2, .02);
                }
            }
        }

        size_t isolate_node_num = 0;
        if (repair_isolate)
        {
            isolate_node_num = DFS::repair_isolate_node(G, static_cast<indexType>(0));
            std::cout << "isolate_node_num=" << isolate_node_num << std::endl;
        }
        return isolate_node_num;
    }
}