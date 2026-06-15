#pragma once
#include <random>
#include "nsw.h"
#include "hier_utils.h"

namespace ant
{

    template <typename indexType>
    class HNSWIndex
    {
    public:
        using QNode = QueryNode<indexType>;
        using IDist = IdDist<indexType>;
        // 单个节点在某一层插入后产生的出边：(源节点 id, 邻居列表)
        using LayerInsert = std::pair<indexType, std::vector<IDist>>;
        using BatchLayerInserts = std::vector<std::vector<LayerInsert>>;

        HNSWIndex(Prune prune,
                  uint32_t beamSize,
                  uint32_t hier_beamSize,
                  uint32_t max_visited_size,
                  uint32_t num_workers,
                  uint32_t max_degree,
                  size_t max_batch_size,
                  uint32_t top_degree,
                  double cos_angle = 0) : prune(prune),
                                          beamSize(beamSize),
                                          hier_beamSize(hier_beamSize),
                                          max_visited_size(max_visited_size),
                                          num_workers(num_workers),
                                          max_degree(max_degree),
                                          max_batch_size(max_batch_size),
                                          top_degree(top_degree == 0 ? max_degree : top_degree),
                                          cos_angle(cos_angle),
                                          bsTable(max_batch_size, num_workers, beamSize, max_visited_size, max_degree),
                                          merge_edges(max_batch_size * max_degree, max_visited_size, num_workers, max_degree)
        {
        }

        template <typename PR>
        void batch_insert(Graph<indexType> &G, HIdMapping &hidMapping, PR &Points, double alpha, double m_l, QueryStatic *BuildStats,
                          double base = 2, double max_fraction = .02, bool print = true)
        {
            auto HierPoints = HierPointRange<PR>(&hidMapping, &Points);
            const auto less = [](const QNode &a, const QNode &b)
            { return id_dist_less(a, b); };

            // 单层：beam search → 合并 visited → RobustPrune
            auto searchNghAndPrune = [&](indexType index, int level, indexType *spPtr, bool is_insert,
                                         BeamSearchMemoryCell<indexType> &bsCell)
            {
                const size_t limit = Points.size();
                QueryParams QP(0, is_insert ? beamSize : hier_beamSize, 0.0, limit, G.max_degree());
                bsCell.max_visited_size = QP.beamSize * 2;

                const uint32_t sp_num = count_valid_entry_points(spPtr, hier_beamSize);
                bsCell.init_starting_points(HierPoints[index], HierPoints, spPtr, sp_num);

                const size_t bs_distance_comps = beam_search(bsCell, G, HierPoints[index], HierPoints, QP);
                if (BuildStats != nullptr)
                {
                    BuildStats->increment_dist(index, bs_distance_comps);
                    BuildStats->increment_visited(index, bsCell.num_visited);
                }

                bsCell.frontier_size = merge_union<QNode>(
                    bsCell.visited, static_cast<size_t>(0),
                    bsCell.visited, bsCell.visited_size,
                    bsCell.copy_visited, bsTable.get_max_visited_size(), less);

                const double prune_alpha = (level == 0) ? alpha : 1.0;
                const size_t rp_distance_comps = prune.robustPrune(
                    index, bsCell.copy_visited, bsCell.frontier_size, HierPoints, prune_alpha, cos_angle, 1);
                assert(bsCell.frontier_size <= prune.R);

                if (BuildStats != nullptr)
                    BuildStats->increment_dist(index, rp_distance_comps);
            };

            auto updateDist = [&](indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size)
            {
                const size_t rp = NSWIndex<indexType>::update_neighbors_for_graph(
                    HierPoints, index, candidates, pre_candidate_size, candidate_size,
                    prune, max_degree, top_degree);
                if (BuildStats != nullptr)
                    BuildStats->increment_dist(index, rp);
            };

            const size_t m = Points.size();
            size_t inc = 0;
            size_t count = 0;
            float frac = 0.0f;
            const size_t effective_batch_size = std::min(NSWIndex<indexType>::getMaxBatchSize(G, max_fraction), max_batch_size);

            // 每批工作区
            std::vector<indexType> all_enter_points(max_batch_size * hier_beamSize);
            std::vector<indexType> inserts(max_batch_size);
            std::vector<uint> levels(max_batch_size);

            uint max_level = 0;
            uint64_t ep = 0;

            Timer t_beam("beam search time");
            Timer t_bidirect("bidirect time");
            Timer t_prune("prune time");
            t_beam.stop();
            t_bidirect.stop();
            t_prune.stop();

            while (count < m)
            {
                const auto batch = next_batch_range(inc, count, m, effective_batch_size, base);
                const size_t floor = batch.floor;
                const size_t ceiling = batch.ceiling;
                count = batch.next_count;
                const size_t cur_batch_size = ceiling - floor;

                // 1) 采样层级并初始化入口点
                srand(static_cast<unsigned>(ceiling));
                std::fill_n(all_enter_points.data(), cur_batch_size * hier_beamSize, static_cast<indexType>(-1));
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    indexType *spPtr = all_enter_points.data() + i * hier_beamSize;
                    spPtr[0] = static_cast<indexType>(ep);
                    const double r = static_cast<double>(rand()) / RAND_MAX;
                    levels[i] = std::min(static_cast<uint>(-std::log(r) * m_l), max_level + 1);
                }

                // 2) 分配分层节点 id，并建立 level 链
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    size_t &cur_node_num = hidMapping.cur_node_num;
                    int l = static_cast<int>(levels[i]);
                    int64_t node_id = static_cast<int64_t>(cur_node_num);
                    inserts[i] = static_cast<indexType>(node_id);
                    int64_t pre_id;
                    while (l >= 0)
                    {
                        ++cur_node_num;
                        assert(cur_node_num <= hidMapping.max_node_num);
                        if (l == 0)
                        {
                            pre_id = static_cast<int64_t>(-1) - static_cast<int64_t>(floor + i);
                            assert(-pre_id - 1 < static_cast<int64_t>(m));
                        }
                        else
                        {
                            pre_id = static_cast<int64_t>(cur_node_num);
                        }
                        hidMapping.set_id_in_pre_level(node_id, pre_id);
                        node_id = pre_id;
                        --l;
                    }
                }

                // 3) 若本批出现新顶层，先随机连边保证连通，搜索从 max_level 层开始
                std::vector<size_t> new_level_nodes;
                new_level_nodes.reserve(cur_batch_size);
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    if (levels[i] > max_level)
                    {
                        new_level_nodes.push_back(inserts[i]);
                        inserts[i] = static_cast<indexType>(hidMapping.get_id_in_pre_level(inserts[i]));
                        --levels[i];
                    }
                }
                if (!new_level_nodes.empty())
                {
                    const size_t link_cnt = std::min(new_level_nodes.size(), static_cast<size_t>(G.max_degree()));
                    for (size_t i = 0; i < new_level_nodes.size(); ++i)
                    {
                        for (size_t j = 1; j < link_cnt; ++j)
                        {
                            const size_t nb_id = new_level_nodes[(i + j) % new_level_nodes.size()];
                            G[new_level_nodes[i]].append_neighbor(static_cast<indexType>(nb_id));
                        }
                    }
                }

                // 4) 自顶向下逐层 beam search + 插入
                BatchLayerInserts layer_inserts(cur_batch_size);
                t_beam.start();
#pragma omp parallel for
                for (size_t i = floor; i < ceiling; ++i)
                {
                    const uint32_t worker_id = omp_get_thread_num();
                    const uint32_t cell_id = static_cast<uint32_t>(i - floor);
                    indexType *spPtr = all_enter_points.data() + cell_id * hier_beamSize;
                    auto bsCell = bsTable.getCell(cell_id, worker_id, beamSize, max_visited_size);

                    for (int lc = static_cast<int>(max_level); lc >= 0; --lc)
                    {
                        bsCell.frontier_size = 0;
                        bsCell.visited_size = 0;
                        bsCell.num_visited = 0;

                        const indexType index = inserts[cell_id];
                        const bool is_insert = (lc <= static_cast<int>(levels[cell_id]));

                        searchNghAndPrune(index, lc, spPtr, is_insert, bsCell);

                        if (is_insert)
                        {
                            const auto node_id = inserts[cell_id];
                            G[node_id].update_neighbors(bsCell.copy_visited, bsCell.frontier_size);

                            LayerInsert layer;
                            layer.first = index;
                            layer.second.resize(bsCell.frontier_size);
                            std::transform(bsCell.copy_visited,
                                           bsCell.copy_visited + bsCell.frontier_size,
                                           layer.second.begin(),
                                           [](const QNode &d)
                                           { return IDist(d.index, d.dist); });
                            layer_inserts[cell_id].push_back(std::move(layer));

                            assert(node_id < hidMapping.cur_node_num);
                            inserts[cell_id] = static_cast<indexType>(hidMapping.get_id_in_pre_level(node_id));
                        }

                        if (lc > 0)
                        {
                            fill_next_level_entry_points(
                                spPtr, hier_beamSize, bsCell.visited, bsCell.visited_size, hidMapping, lc);
                        }
                    }
                }
                t_beam.stop();

                // 5) 反向边合并 + 剪枝写回
                t_bidirect.start();
                NSWIndex<indexType>::inv_edges(merge_edges, layer_inserts);
                t_bidirect.stop();

                t_prune.start();
                NSWIndex<indexType>::batch_update_graph(G, merge_edges, updateDist, G.max_degree());
                t_prune.stop();

                if (!new_level_nodes.empty())
                {
                    ep = new_level_nodes[0];
                    assert(hidMapping.get_level(ep) == max_level + 1);
                    ++max_level;
                }

                if (print)
                    report_batch_progress(floor, ceiling, m, frac);
                ++inc;
            }

            hidMapping.set_sp(ep);
            G.resize(hidMapping.cur_node_num);
            if (print)
            {
                t_beam.total();
                t_bidirect.total();
                t_prune.total();
            }
        }

    protected:
        Prune prune;

        uint32_t beamSize;
        uint32_t hier_beamSize;
        uint32_t num_workers;
        uint32_t max_visited_size;
        size_t max_batch_size;
        uint32_t top_degree;
        uint32_t max_degree;
        double cos_angle;

        BeamSearchMemoryTable<indexType> bsTable;
        MergeEdges<indexType> merge_edges;
    };
}
