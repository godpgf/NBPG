#pragma once
#include <unordered_set>
#include <random>
#include "nsw.h"

namespace ant
{

    template <typename indexType>
    class HNSWIndex
    {
    public:
        using QNode = QueryNode<indexType>;
        using IDist = IdDist<indexType>;

        HNSWIndex(Prune prune,
                  uint32_t beamSize,
                  uint32_t hier_beamSize,
                  uint32_t max_visited_size,
                  uint32_t num_workers,
                  uint32_t max_degree,
                  size_t max_batch_size,
                  uint32_t top_degree) : prune(prune),
                                         beamSize(beamSize),
                                         hier_beamSize(hier_beamSize),
                                         max_visited_size(max_visited_size),
                                         num_workers(num_workers),
                                         max_degree(max_degree),
                                         max_batch_size(max_batch_size),
                                         top_degree(top_degree == 0 ? max_degree : top_degree),
                                         bsTable(max_batch_size, num_workers, beamSize, max_visited_size, max_degree),
                                         merge_edges(max_batch_size * max_degree, max_visited_size, num_workers, max_degree)

        {
        }

        template <typename PR>
        void batch_insert(Graph<indexType> &G, HIdMapping &hidMapping, PR &Points, double alpha, double m_l, QueryStatic *BuildStats,
                                  double base = 2, double max_fraction = .02, bool print = true)
        {
            auto HierPoints = HierPointRange<PR>(&hidMapping, &Points);
            auto less = [&](const QNode &a, const QNode &b)
            {
                return id_dist_less(a, b);
            };

            auto searchNghAndPrune = [&](indexType index, int level, indexType* spPtr, bool is_insert, BeamSearchMemoryCell<indexType> &bsCell)
            {
                size_t limit = Points.size();
                QueryParams QP(0, is_insert ? beamSize : hier_beamSize, 0.0, limit, G.max_degree());
                bsCell.max_visited_size = QP.beamSize * 2;
                uint32_t sp_num = 1;
                while (sp_num < hier_beamSize)
                {
                    if(spPtr[sp_num] != -1){
                        sp_num++;
                    }
                    else{
                        break;
                    }
                }
                
                bsCell.init_starting_points(HierPoints[index], HierPoints, spPtr, sp_num);
                size_t bs_distance_comps = 0;
                bs_distance_comps = beam_search(bsCell, G, HierPoints[index], HierPoints, QP);

                if (BuildStats != nullptr)
                {
                    BuildStats->increment_dist(index, bs_distance_comps);
                    BuildStats->increment_visited(index, bsCell.num_visited);
                }

                bsCell.frontier_size = merge_union<QNode>(bsCell.visited, (size_t)0, bsCell.visited, bsCell.visited_size, bsCell.copy_visited, bsTable.get_max_visited_size(), less);

                auto rp_distance_comps = prune.robustPrune(index, bsCell.copy_visited, bsCell.frontier_size, HierPoints, level == 0 ? alpha : 1, 0, 1);
                assert(bsCell.frontier_size <= prune.R);

                if (BuildStats != nullptr)
                    BuildStats->increment_dist(index, rp_distance_comps);
            };

            auto updateDist = [&](indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size)
            {
                updateNeighborsForGraph(HierPoints, index, candidates, pre_candidate_size, candidate_size, BuildStats);
            };
            // ---------------------------------------------------------------------------------

            size_t n = G.size();
            size_t m = Points.size();
            size_t inc = 0;
            size_t count = 0;
            float frac = 0.0;
            float progress_inc = .1;
            size_t max_batch_size = std::min(NSWIndex<indexType>::getMaxBatchSize(G, max_fraction), this->max_batch_size);

            // hnsw 中间参数
            // 记录当前插入点的起点
            auto all_enter_points = std::vector<indexType>(max_batch_size * hier_beamSize);
            // 当前最大等级
            uint max_level = 0;
            uint64_t ep = 0;
            // 记录当前插入点的真实id
            auto inserts = std::vector<indexType>(max_batch_size);
            // 记录当前插入点的插入级别
            auto levels = std::vector<uint>(max_batch_size);

            Timer t_beam("beam search time");
            Timer t_bidirect("bidirect time");
            Timer t_prune("prune time");

            t_beam.stop();
            t_bidirect.stop();
            t_prune.stop();

            while (count < m)
            {
                size_t floor;
                size_t ceiling;
                if (pow(base, inc) <= max_batch_size)
                {
                    floor = static_cast<size_t>(pow(base, inc)) - 1;
                    ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
                    count = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
                }
                else
                {
                    floor = count;
                    ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
                    count += (ceiling - floor);
                }

                // set seed
                srand(ceiling);
                // 初始化每个待插入节点的level
                size_t cur_batch_size = ceiling - floor;
                std::fill_n(all_enter_points.data(), cur_batch_size * hier_beamSize, -1);
                
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    indexType* spPtr = all_enter_points.data() + i * hier_beamSize;
                    spPtr[0] = ep;
                    double r = static_cast<double>(rand()) / RAND_MAX;
                    levels[i] = std::min((uint)(-log(r) * m_l), max_level + 1);
                }

                // 申请图上的节点，并将所插入的向量id转化为节点id
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    size_t &cur_node_num = hidMapping.cur_node_num;
                    int l = levels[i];
                    int64_t node_id = cur_node_num;
                    // 当前层l对应的节点id
                    inserts[i] = node_id;
                    int64_t pre_id;
                    while (l >= 0)
                    {
                        cur_node_num++;
                        assert(cur_node_num <= hidMapping.max_node_num);
                        if (l == 0)
                        {
                            pre_id = ((int64_t)-1) - (floor + i);
                            assert(-pre_id - 1 < m);
                        }
                        else
                        {
                            pre_id = cur_node_num;
                        }
                        hidMapping.set_id_in_pre_level(node_id, pre_id);
                        node_id = pre_id;
                        l--;
                    }
                }

                // 顶层之上如果又加了一层，在这一层随意插入当前的节点，并保证连通性，但是暂时先不对这一层进行搜索
                std::vector<size_t> new_level_nodes;
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    if (levels[i] > max_level)
                    {
                        new_level_nodes.push_back(inserts[i]);
                        inserts[i] = hidMapping.get_id_in_pre_level(inserts[i]);
                        --levels[i];
                    }
                }
                for (size_t i = 0; i < new_level_nodes.size(); ++i)
                {
                    for (size_t j = 1; j < std::min(new_level_nodes.size(), (size_t)G.max_degree()); ++j)
                    {
                        size_t nb_id = new_level_nodes[(i + j) % new_level_nodes.size()];
                        size_t node_id = new_level_nodes[i];
                        G[node_id].append_neighbor(nb_id);
                    }
                }

                // hnsw核心代码-----------------------------------------------------------------------

                // 用来存每一层子图的反向边
                std::vector<std::vector<std::pair<indexType, std::vector<IDist>>>> new_out_(ceiling - floor);
                t_beam.start();
#pragma omp parallel for
                for (size_t i = floor; i < ceiling; ++i)
                {
                    uint32_t worker_id = omp_get_thread_num();
                    uint32_t cell_id = i - floor;
                    indexType* spPtr = all_enter_points.data() + cell_id * hier_beamSize;
                    auto bsCell = bsTable.getCell(cell_id, worker_id, beamSize, max_visited_size);
                    for (int lc = max_level; lc >= 0; lc--)
                    {
                        bsCell.frontier_size = 0;
                        bsCell.visited_size = 0;
                        bsCell.num_visited = 0;

                        // 用来查询的向量id
                        indexType index = inserts[cell_id];
                        // size_t sp = all_enter_points[cell_id];

                        bool is_insert = (lc <= levels[cell_id]);

                        // 找到候选邻居集合
                        searchNghAndPrune(index, lc, spPtr, is_insert, bsCell);
                        if (is_insert)
                        {
                            auto node_id = inserts[cell_id];
                            G[node_id].update_neighbors(bsCell.copy_visited, bsCell.frontier_size);

                            std::pair<indexType, std::vector<IDist>> res;
                            res.first = index;
                            res.second.resize(bsCell.frontier_size);
                            std::transform(bsCell.copy_visited, bsCell.copy_visited + bsCell.frontier_size, res.second.data(), [](QNode d){return IDist(d.index, d.dist);});
                            new_out_[cell_id].push_back(std::move(res));


                            assert(node_id < hidMapping.cur_node_num);
                            inserts[cell_id] = hidMapping.get_id_in_pre_level(node_id);
                            if (lc > 0)
                            {
                                node_id = inserts[cell_id];
                                assert(node_id < hidMapping.cur_node_num);
                            }
                        }

                        // 设置下一个层级的起点
                        // auto near_node_id = bsCell.visited[0].first;
                        if (lc > 0)
                        {
                            std::fill_n(spPtr, hier_beamSize, -1);
                            for(auto j = 0; j < std::min((size_t)hier_beamSize, bsCell.visited_size); ++j){
                                auto near_node_id = bsCell.visited[j].index;
                                auto next_ep = hidMapping.get_id_in_pre_level(near_node_id);
                                assert(next_ep < G.size() && next_ep >= 0);
                                spPtr[j] = next_ep;
                            }
                        }
                    }
                }
                t_beam.stop();

                // 反转边
                // make each edge bidirectional by first adding each new edge
                //(i,j) to a sequence, then semisorting the sequence by key values
                t_bidirect.start();
                merge_edges.clear();
                for (size_t i = floor; i < ceiling; ++i)
                {
                    uint32_t cell_id = i - floor;
                    for (auto res : new_out_[cell_id])
                    {
                        auto index = res.first;
                        for (auto vit : res.second)
                        {
                            auto target_index = vit.index;
                            merge_edges.insert_edge(target_index, IDist(index, vit.dist));
                        }
                    }
                }
                merge_edges.merge_edges();
                t_bidirect.stop();

                // 对反转边进行剪枝
                t_prune.start();
                NSWIndex<indexType>::batch_update_graph(G, merge_edges, updateDist, G.max_degree());
                t_prune.stop();

                if (new_level_nodes.size() > 0)
                {
                    // new level
                    ep = new_level_nodes[0];
                    auto ep_level = hidMapping.get_level(ep);
                    assert(ep_level == max_level + 1);
                    ++max_level;
                }

                if (print)
                {
                    auto ind = frac * m;
                    if (floor <= ind && ceiling > ind)
                    {
                        frac += progress_inc;
                        std::cout << "Pass " << (int)(100 * frac) << "% complete"
                                  << std::endl;
                    }
                }
                inc += 1;
            }

            hidMapping.set_sp(ep);
            G.resize(hidMapping.cur_node_num);
            t_beam.total();
            t_bidirect.total();
            t_prune.total();
        }



        template <typename PR>
        void updateNeighborsForGraph(PR &Points, indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size, QueryStatic *BuildStats)
        {
            auto less = [&](const IDist &a, const IDist &b)
            {
                return id_dist_less(a, b);
            };

            size_t rp_distance_comps;
            if (candidate_size <= max_degree)
            {
                // 如果效率慢，可以注释掉下面这行
                // gist1m在98%精度下的搜索效率慢约2%，但建图效率影响不大
                rp_distance_comps = NSWIndex<indexType>::eval_dist_of_new_neighbors(Points, index, candidates, pre_candidate_size, candidate_size, top_degree);
                std::sort(candidates, candidates + candidate_size, less);
            }
            else
            {
                rp_distance_comps = NSWIndex<indexType>::eval_dist_of_new_neighbors(Points, index, candidates, pre_candidate_size, candidate_size, top_degree);

                // 为了提高速度，反向边直接使用基于距离的剪枝
                prune.sort_candidates(candidates, candidate_size);
                if (candidate_size > prune.R)
                    candidate_size = prune.R;
                // rp_distance_comps += prune.robustPrune(index, candidates, candidate_size, Points, 0, 0);
            }

            if (BuildStats != nullptr)
            {
                BuildStats->increment_dist(index, rp_distance_comps);
            }
        }


    protected:
        Prune prune;

        uint32_t beamSize;
        uint32_t hier_beamSize;
        uint32_t num_workers;
        uint32_t max_visited_size;
        size_t max_batch_size;
        // 只对top_degree个邻居和最后一个邻居进行距离计算，其他邻居的距离用插值法
        uint32_t top_degree;
        uint32_t max_degree;

        // 申请beamSearch所需要使用的空间
        BeamSearchMemoryTable<indexType> bsTable;
        // 申请合并边的空间
        MergeEdges<indexType> merge_edges;
    };
}