#pragma once
#include <omp.h>
#include <functional>
#include <cassert>
#include "query_params.h"

namespace ant
{
    // -------------------------------------------------------------------------
    // Beam search 核心
    //
    // 模板参数约定：
    //   Point / PointRange     — 原始（全精度）向量及其存储；距离慢但准确，适合 rerank / filtering 终判
    //   QPoint / QPointRange   — 量化向量及其存储；距离快，用于图遍历中的邻居扩展与 beam 排序
    //
    // 当 PointRange 与 QPointRange 的 num_bytes() 相同时，二者退化为同一套向量（构建索引或无 rerank 场景）。
    // -------------------------------------------------------------------------

    namespace detail
    {
        // 是否存在独立的原始向量存储（量化字节数 < 原始字节数）
        template <typename PointRange, typename QPointRange>
        inline bool has_exact_point_storage(const PointRange &Points, const QPointRange &Q_Points)
        {
            return Points.params.num_bytes() != Q_Points.params.num_bytes();
        }

        // rerank_factor == 0 且存在原始向量 → SymphonyQG filtering：遍历用量化距，pop 时用原始距
        inline bool use_filtering_mode(uint32_t rerank_factor, bool has_exact_storage)
        {
            return rerank_factor == 0 && has_exact_storage;
        }

        // rerank_factor > 0 且存在原始向量 → 量化搜索 + 对 frontier 子集做原始距 rerank
        inline bool use_rerank_mode(uint32_t rerank_factor, bool has_exact_storage)
        {
            return rerank_factor > 0 && has_exact_storage;
        }
    }

    // 支持量化/原始双轨距离的 beam search（filtering 模式内核）。
    // 邻居扩展始终用 QPointRange::distance；若 has_exact_storage，pop 节点时额外计算 PointRange 精确距离。
    template <typename indexType, typename Point, typename PointRange, typename QPoint, typename QPointRange, class GraphType>
    size_t filtered_beam_search(BeamSearchMemoryCell<indexType> &bsCell, const GraphType &G,
                                const Point p, PointRange &Points,
                                const QPoint qp, QPointRange &Q_Points,
                                const QueryParams &QP)
    {
        const bool track_exact_dist = detail::has_exact_point_storage(Points, Q_Points);
        using QNode = QueryNode<indexType>;

        const auto beamSize = QP.beamSize;
        const auto limit = QP.limit;

        // init_starting_points 中已对 frontier 起点做过距离计算，计入统计
        size_t quant_dist_cmps = bsCell.frontier_size;
        size_t exact_dist_cmps = 0;

        int remain = bsCell.frontier_size;
        size_t &num_visited = bsCell.num_visited;

        // unvisited_frontier 延迟更新：用 offset 顺序扫描，避免每步对 frontier 做 set_difference
        int offset = 0;
        size_t candidates_size = 0;

        while (remain > offset && num_visited < limit)
        {
            QNode &current = bsCell.unvisited_frontier[offset];
            G[current.index].prefetch();

            // 1) 收集未访问邻居 id（不做距离计算）
            const long num_elts = std::min<long>(G[current.index].size(), QP.degree_limit);
            size_t filtered_size = 0;
            for (indexType i = 0; i < num_elts; i++)
            {
                assert(G[current.index].size() == num_elts);
                auto a = G[current.index][i];
                assert(a < G.size());
                if (bsCell.has_been_seen(a) || Points[a].same_as(p))
                    continue;
                Q_Points[a].prefetch();
                bsCell.filtered[filtered_size++] = a;
            }

            // 2) 对邻居做量化距离，写入 candidates（延迟合并 frontier，减少排序次数）
            for (size_t i = 0; i < filtered_size; ++i)
            {
                indexType a = bsCell.filtered[i];
                assert(a < G.size());
                float dist = Q_Points[a].distance(qp);
                quant_dist_cmps++;
                bsCell.candidates[candidates_size++] = QNode(a, dist, num_visited + 1);
            }

            // 3) 将 current 按距离插入 visited（保持有序）
            auto position_id = std::upper_bound(bsCell.visited, bsCell.visited + bsCell.visited_size, current, id_dist_less<indexType>) - bsCell.visited;
            num_visited++;

            if (bsCell.visited_size < bsCell.max_visited_size)
                bsCell.visited_size++;

            if (position_id < bsCell.visited_size - 1)
            {
                std::rotate(bsCell.visited + position_id, bsCell.visited + (bsCell.visited_size - 1), bsCell.visited + bsCell.visited_size);
                if (track_exact_dist)
                    std::rotate(bsCell.tmp_visited + position_id, bsCell.tmp_visited + (bsCell.visited_size - 1), bsCell.tmp_visited + bsCell.visited_size);
            }
            if (position_id <= bsCell.visited_size - 1)
            {
                bsCell.visited[position_id] = current;
                if (track_exact_dist)
                {
                    // filtering：pop 时用原始向量重算精确距离（SymphonyQG）
                    bsCell.tmp_visited[position_id] = Points[current.index].distance(p);
                    exact_dist_cmps++;
                }
            }

            // 4) 候选不足时继续 pop，积累到 beamSize/8 再批量合并（摊销排序开销）
            if (candidates_size == 0 ||
                (limit >= 2 * beamSize && candidates_size < beamSize / 8 && offset + 1 < remain))
            {
                offset++;
                continue;
            }
            offset = 0;

            std::sort(bsCell.candidates, bsCell.candidates + candidates_size, id_dist_less<indexType>);
            auto candidates_end = std::unique(bsCell.candidates, bsCell.candidates + candidates_size,
                                              [](const QNode &a, const QNode &b) { return a.index == b.index; });

            auto new_frontier_size =
                std::set_union(bsCell.frontier, bsCell.frontier + bsCell.frontier_size, bsCell.candidates,
                               candidates_end, bsCell.new_frontier, id_dist_less<indexType>) -
                bsCell.new_frontier;
            candidates_size = 0;

            new_frontier_size = std::min<size_t>(beamSize, new_frontier_size);

            // 查询阶段：按第 k 近候选 × cut 剪枝过远节点
            if (QP.k > 0 && new_frontier_size > QP.k)
            {
                new_frontier_size = std::max<indexType>(
                    (std::upper_bound(bsCell.new_frontier,
                                      bsCell.new_frontier + new_frontier_size,
                                      QNode(0, QP.cut * bsCell.new_frontier[QP.k].dist), id_dist_less<indexType>) -
                     bsCell.new_frontier),
                    bsCell.frontier_size);
            }

            memcpy(bsCell.frontier, bsCell.new_frontier, new_frontier_size * sizeof(QNode));
            bsCell.frontier_size = new_frontier_size;

            // 5) frontier \ visited → unvisited_frontier
            remain = (std::set_difference(bsCell.frontier,
                                          bsCell.frontier + std::min<long>(bsCell.frontier_size, QP.beamSize),
                                          bsCell.visited,
                                          bsCell.visited + bsCell.visited_size,
                                          bsCell.unvisited_frontier, id_dist_less<indexType>) -
                      bsCell.unvisited_frontier);
        }

        if (track_exact_dist)
        {
            for (auto i = 0; i < bsCell.visited_size; ++i)
                bsCell.visited[i].dist = bsCell.tmp_visited[i];
            std::sort(bsCell.visited, bsCell.visited + bsCell.visited_size, id_dist_less<indexType>);
        }

        return quant_dist_cmps + exact_dist_cmps;
    }

    // 单向量版本：Point 与 QPoint 为同一类型，全程用一套距离
    template <typename indexType, typename Point, typename PointRange, class GraphType>
    size_t beam_search(BeamSearchMemoryCell<indexType> &bsCell, const GraphType &G,
                       const Point p, PointRange &Points,
                       const QueryParams &QP)
    {
        return filtered_beam_search(bsCell, G, p, Points, p, Points, QP);
    }

    // 查询入口：根据 rerank_factor 与是否具备原始向量，选择搜索模式并返回 top-k
    template <typename indexType, typename Point, typename QPoint, typename PointRange, typename QPointRange, class GraphType>
    std::vector<std::pair<indexType, float>> beam_search_rerank(BeamSearchMemoryCell<indexType> &bsCell,
                                                                const Point &p,
                                                                const QPoint &qp,
                                                                const GraphType &G,
                                                                PointRange &Base_Points,
                                                                QPointRange &Q_Base_Points,
                                                                QueryStatic &QueryStats,
                                                                const QueryParams &QP,
                                                                indexType stat_pid = 0,
                                                                bool stats = true)
    {
        using PID = std::pair<indexType, float>;
        using QNode = QueryNode<indexType>;

        const bool has_exact_storage = detail::has_exact_point_storage(Base_Points, Q_Base_Points);
        const bool filtering_mode = detail::use_filtering_mode(QP.rerank_factor, has_exact_storage);
        const bool rerank_mode = detail::use_rerank_mode(QP.rerank_factor, has_exact_storage);

        size_t dist_cmps = 0;
        if (filtering_mode)
        {
            dist_cmps = filtered_beam_search(bsCell, G, p, Base_Points, qp, Q_Base_Points, QP);
        }
        else
        {
            // 纯量化或 rerank 前置搜索：全程使用 QPointRange
            dist_cmps = beam_search(bsCell, G, qp, Q_Base_Points, QP);
        }

        if (stats)
        {
            QueryStats.increment_visited(stat_pid, bsCell.num_visited);
            QueryStats.increment_dist(stat_pid, dist_cmps);
        }

        bsCell.visited_size = std::min(bsCell.frontier_size, bsCell.visited_size);

        if (rerank_mode)
        {
            const int num_check = std::min<int>(QP.k * QP.rerank_factor, bsCell.frontier_size);
            std::vector<PID> pts;
            pts.reserve(num_check);
            for (int i = 0; i < num_check; i++)
            {
                auto j = bsCell.frontier[i].index;
                pts.emplace_back(j, Base_Points[j].distance(p));
            }
            std::sort(pts.begin(), pts.end(), [](const PID &a, const PID &b)
                      { return a.second < b.second || (a.second == b.second && a.first < b.first); });

            const size_t k = std::min(static_cast<size_t>(QP.k), pts.size());
            pts.resize(k);
            return pts;
        }

        if (filtering_mode)
        {
            // filtering：visited 已按原始向量精确距离排序
            const size_t k = std::min(static_cast<size_t>(QP.k), bsCell.visited_size);
            std::vector<PID> pts(k);
            std::transform(bsCell.visited, bsCell.visited + k, pts.begin(), [](const QNode &a)
                           { return PID(a.index, a.dist); });
            return pts;
        }

        // 纯量化：frontier 距离即为搜索结果
        const size_t k = std::min(static_cast<size_t>(QP.k), bsCell.frontier_size);
        std::vector<PID> pts(k);
        std::transform(bsCell.frontier, bsCell.frontier + k, pts.begin(), [](const QNode &a)
                       { return PID(a.index, a.dist); });
        return pts;
    }

    template <typename indexType, typename QyPR, typename QyQPR, typename PR, typename QPR, class GraphType>
    std::vector<std::vector<indexType>>
    qsearchAll(const QyPR &Query_Points,
               const QyQPR &Q_Query_Points,
               const GraphType &G,
               PR &Base_Points,
               const QPR &Q_Base_Points,
               QueryStatic &QueryStats,
               std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
               const QueryParams &QP)
    {
        if (QP.k > QP.beamSize)
        {
            std::cout << "Error: beam search parameter Q = " << QP.beamSize
                      << " same size or smaller than k = " << QP.k << std::endl;
            abort();
        }

        const uint32_t num_workers = omp_get_max_threads();
        auto bsTable = BeamSearchMemoryTable<indexType>(Query_Points.size(),
                                                        num_workers,
                                                        QP.beamSize,
                                                        QP.beamSize * 8,
                                                        G.max_degree());

        std::vector<std::vector<indexType>> all_neighbors(Query_Points.size());

#pragma omp parallel for
        for (size_t i = 0; i < Query_Points.size(); ++i)
        {
            uint32_t worker_id = omp_get_thread_num();

            auto [starting_points, sp_num] = get_sp(i);
            auto bsCell = bsTable.getCell(i, worker_id, QP.beamSize, QP.beamSize * 8);
            // 起点距离用量化 query × 量化 base 计算
            bsCell.init_starting_points(Q_Query_Points[i], Q_Base_Points, starting_points, sp_num);
            auto ngh_dist = beam_search_rerank(bsCell, Query_Points[i], Q_Query_Points[i],
                                                 G, Base_Points, Q_Base_Points,
                                                 QueryStats, QP, (indexType)i);
            std::vector<indexType> ngh(ngh_dist.size());
            std::transform(ngh_dist.begin(), ngh_dist.end(), ngh.data(), [](const std::pair<indexType, float> &d)
                           { return d.first; });
            all_neighbors[i] = std::move(ngh);
        }

        return all_neighbors;
    }
}
