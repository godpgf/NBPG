#pragma once
#include "utils/Graph/beamSearch.h"
#include "hier_idmapping.h"

namespace ant
{
    template <typename indexType, typename QyPR, typename QyQPR, typename PR, typename QPR, class GraphType>
    std::vector<std::vector<indexType>>
    hier_qsearchAll(const QyPR &Query_Points,
                    const QyQPR &Q_Query_Points,
                    const GraphType &G,
                    HIdMapping &hidMapping,
                    PR &Base_Points,
                    const QPR &Q_Base_Points,
                    QueryStatic &QueryStats,
                    std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                    const QueryParams &QP,
                    uint32_t hier_beamSize = 1)
    {
        // 有两种查询模式，rerank和filtering。
        // rerank是用压缩向量得到结果集合后再计算真实距离并重新排序（适合真实向量存放在内存的情况）。
        // filtering是在查询的过程中直接选出最好的，计算真实距离（适合真实向量存放在硬盘的情况）。
        if (QP.k > QP.beamSize)
        {
            std::cout << "Error: beam search parameter Q = " << QP.beamSize
                      << " same size or smaller than k = " << QP.k << std::endl;
            abort();
        }

        // 申请beamSearch所需要使用的空间----------------------------
        uint32_t num_workers = omp_get_max_threads();
        auto bsTable = BeamSearchMemoryTable<indexType>(Query_Points.size(),
                                                        num_workers,
                                                        QP.beamSize,
                                                        QP.beamSize * 8,
                                                        G.max_degree());

        std::vector<std::vector<indexType>> all_neighbors(Query_Points.size());

        auto Q_HierPoints = HierPointRange<QPR>(&hidMapping, &Q_Base_Points);
        auto HierPoints = HierPointRange<PR>(&hidMapping, &Base_Points);

        std::vector<indexType> all_enter_points = std::vector<indexType>(Query_Points.size() * hier_beamSize, -1);

#pragma omp parallel for
        for (size_t i = 0; i < Query_Points.size(); ++i)
        {
            uint32_t worker_id = omp_get_thread_num();
            // 开始分层查询
            auto [starting_points, sp_num] = get_sp(i);
            indexType sp = starting_points[0];
            indexType *spPtr = all_enter_points.data() + i * hier_beamSize;
            spPtr[0] = sp;
            while (true)
            {
                QueryParams cur_QP = QP;
                auto sp_level = hidMapping.get_level(spPtr[0]);
                if (sp_level > 0)
                {
                    cur_QP.beamSize = hier_beamSize;
                    // cur_QP.k = 1;
                }
                auto bsCell = bsTable.getCell(i, worker_id, QP.beamSize, QP.beamSize * 8);

                uint32_t sp_num = 1;
                while (sp_num < hier_beamSize)
                {
                    if (spPtr[sp_num] != -1)
                    {
                        sp_num++;
                    }
                    else
                    {
                        break;
                    }
                }
                assert(spPtr[0] < G.size());
                bsCell.init_starting_points(Q_Query_Points[i], Q_HierPoints, spPtr, sp_num);

                if (sp_level > 0)
                {
                    beam_search(bsCell, G, Q_Query_Points[i], Q_HierPoints, cur_QP);
                    std::fill_n(spPtr, hier_beamSize, -1);
                    for (size_t j = 0; j < std::min((size_t)hier_beamSize, bsCell.visited_size); ++j)
                    {
                        auto near_id = bsCell.visited[j].index;
                        auto pre_node_id = hidMapping.get_id_in_pre_level(near_id);
                        assert(hidMapping.get_level(pre_node_id) == sp_level - 1);
                        spPtr[j] = pre_node_id;
                    }

                    // sp = pre_node_id;
                }
                else
                {
                    auto ngh_dist = beam_search_rerank(bsCell, Query_Points[i], Q_Query_Points[i],
                                                       G, HierPoints, Q_HierPoints,
                                                       QueryStats, QP, (indexType)i);
                    std::vector<indexType> ngh(ngh_dist.size());
                    std::transform(ngh_dist.begin(), ngh_dist.end(), ngh.data(), [&](std::pair<indexType, float> d)
                                   { return hidMapping.get_root_id(d.first); });
                    all_neighbors[i] = ngh;
                    break;
                }
            }
        }
        return all_neighbors;
    }

    

}