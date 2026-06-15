#pragma once
#include "utils/Graph/beamSearch.h"
#include "hier_idmapping.h"

namespace ant
{
    // -------------------------------------------------------------------------
    // HNSW 分层构图 / 查询共用的入口点工具
    // -------------------------------------------------------------------------

    // 入口点数组以 -1 为哨兵；至少返回 1（sp[0] 始终有效）
    template <typename indexType>
    inline uint32_t count_valid_entry_points(const indexType *sp, uint32_t max_sp)
    {
        uint32_t sp_num = 1;
        while (sp_num < max_sp && sp[sp_num] != static_cast<indexType>(-1))
            ++sp_num;
        return sp_num;
    }

    // 根据当前层 beam search 的 visited 列表，填充下一层搜索入口
    template <typename indexType>
    inline void fill_next_level_entry_points(indexType *spPtr,
                                             uint32_t hier_beamSize,
                                             const QueryNode<indexType> *visited,
                                             size_t visited_size,
                                             const HIdMapping &hidMapping,
                                             int current_level)
    {
        std::fill_n(spPtr, hier_beamSize, static_cast<indexType>(-1));
        const size_t n = std::min(static_cast<size_t>(hier_beamSize), visited_size);
        for (size_t j = 0; j < n; ++j)
        {
            const indexType near_id = visited[j].index;
            const uint near_level = hidMapping.get_level(near_id);
            // level-0 节点的 pre_id 为负（向量 id 编码），需直接使用 near_id
            if (near_level >= static_cast<uint>(current_level))
            {
                const auto next_ep = hidMapping.get_id_in_pre_level(near_id);
                assert(next_ep >= 0 && static_cast<size_t>(next_ep) < hidMapping.cur_node_num);
                spPtr[j] = static_cast<indexType>(next_ep);
            }
            else
            {
                spPtr[j] = near_id;
            }
        }
    }

} // namespace ant
