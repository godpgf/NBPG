#pragma once
#include <omp.h>
#include <functional>
#include <random>
#include <cassert>
#include "query_params.h"

namespace ant
{
    // main beam search
    template <typename indexType, typename Point, typename PointRange, typename QPoint, typename QPointRange, class GraphType>
    size_t filtered_beam_search(BeamSearchMemoryCell<indexType> &bsCell, const GraphType &G,
                                const Point p, PointRange &Points,
                                const QPoint qp, QPointRange &Q_Points,
                                const QueryParams &QP)
    {
        bool use_filtering = (Points.params.num_bytes() != Q_Points.params.num_bytes());
        using QNode = QueryNode<indexType>;

        using CmpFun = std::function<bool(const QNode& a, const QNode& b)>;
        CmpFun less = [&](const QNode& a, const QNode& b)
        {
            return id_dist_less(a, b);
        };

        auto beamSize = QP.beamSize;

        // 初始化起点时，有多少个起点就需要做多少次距离计算
        size_t dist_cmps = 0;                         // 压缩向量距离计算次数（暂时没用）
        size_t full_dist_cmps = bsCell.frontier_size; // 原始向量距离计算次数
        int remain = bsCell.frontier_size;
        size_t &num_visited = bsCell.num_visited;

        // 记录在“unvisited_frontier”队列中，下一个要访问的元素。
        // 如果实时更新“unvisited_frontier”，其实就不需要记录下一个要访问的元素，直接拿第一个就可以
        // 但是为了效率，所以不实时更新“unvisited_frontier”
        int offset = 0;

        size_t candidates_size = 0;

        // The main loop.  Terminate beam search when the entire frontier
        // has been visited or have reached max_visit.
        auto limit = QP.limit;
        
        while (remain > offset && num_visited < limit)
        {
            // the next node to visit is the unvisited frontier node that is closest to p
            auto unvisited_id = offset;
            QNode &current = bsCell.unvisited_frontier[unvisited_id];
            G[current.index].prefetch();

            // 先选出之前没有见过的元素并加入候选集---------------------------------------------------------------------------------------------
            long num_elts = std::min<long>(G[current.index].size(), QP.degree_limit);
            size_t filtered_size = 0;
            for (indexType i = 0; i < num_elts; i++)
            {
                // if(QP.limit_bin_size > 0 && i % QP.limit_bin_size != QP.cur_bin){
                //     continue;
                // }
                auto a = G[current.index][i];
                assert(a < G.size());
                if (bsCell.has_been_seen(a) || Points[a].same_as(p))
                    continue; // skip if already seen
                Q_Points[a].prefetch();
                bsCell.filtered[filtered_size++] = a;
            }

            // 精确计算距离，并得到candidates。最多向candidates中加入max_degree个元素
            // 为什么不直接更新到“unvisited_frontier”？就是考虑排序的效率！所以才先放入“candidates”中，以后再对“unvisited_frontier”延迟更新
            current.isLeaf = true;
            for (size_t i = 0; i < filtered_size; ++i)
            {
                indexType a = bsCell.filtered[i];
                assert(a < G.size());
                float dist = Q_Points[a].distance(qp);
                full_dist_cmps++;
                bsCell.candidates[candidates_size++] = std::move(QNode(a, dist, num_visited + 1));
                if(dist < current.dist){
                    current.isLeaf = false;
                }
            }

            // 找到第一个大于current的元素位置，将当前访问的元素插入到visited队列-------------------------------------------------------------------------------
            auto position_id = std::upper_bound(bsCell.visited, bsCell.visited + bsCell.visited_size, current, less) - bsCell.visited;
            num_visited++;

            // 插入访问过的元素
            if (bsCell.visited_size < bsCell.max_visited_size)
                bsCell.visited_size++;

            if (position_id < bsCell.visited_size - 1)
            {
                std::rotate(bsCell.visited + position_id, bsCell.visited + (bsCell.visited_size - 1), bsCell.visited + bsCell.visited_size);
                if (use_filtering)
                    std::rotate(bsCell.tmp_visited + position_id, bsCell.tmp_visited + (bsCell.visited_size - 1), bsCell.tmp_visited + bsCell.visited_size);
            }
            if (position_id <= bsCell.visited_size - 1)
            {
                // 将当前访问的元素插入到“访问队列”。
                bsCell.visited[position_id] = current;
                if (use_filtering)
                {
                    // “unvisited_frontier”队列中存放的是压缩向量的距离，并不准确，更新成真实距离
                    // 参考论文：SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search
                    // auto future_dist = std::async(std::launch::async, [&]()
                    //   { return Points[current.index].distance(p); });
                    // bsCell.tmp_visited[position_id] = std::move(future_dist);
                    bsCell.tmp_visited[position_id] = Points[current.index].distance(p);
                }
            }

            // 将候选集插入到frontier队列(同时包含已经访问的元素和尚未访问的元素)---------------------------------------------------------------
            if (candidates_size == 0 ||
                (limit >= 2 * beamSize && candidates_size < beamSize / 8 && offset + 1 < remain))
            {
                // 当candidates中元素很少（小于 beamSize/8）时，继续查看没有访问过的元素，并填写到candidates中
                // 这样可以避免后面步骤的排序，等到candidates积累足够多的元素后再排序，节省算力
                // 所以，可以推测出，candidates最多包含max_degree+beamSize/8个元素
                offset++;
                continue;
            }
            // 准备刷新unvisited_frontier，offset表示当前访问到的unvisited_frontier下标，所以要清零
            offset = 0;

            // 给candidates排序并去重
            std::sort(bsCell.candidates, bsCell.candidates + candidates_size, less);
            auto candidates_end = std::unique(bsCell.candidates, bsCell.candidates + candidates_size,
                                              [](auto a, auto b)
                                              { return a.index == b.index; });

            // 将candidates和frontier合并后存入new_frontier，清空candidates
            // frontier的最大数量是beamSize,candidates的最大数量是max_degree+beamSize/8
            // 所以，new_frontier的最大数量是beamSize+max_degree+beamSize/8
            auto new_frontier_size =
                std::set_union(bsCell.frontier, bsCell.frontier + bsCell.frontier_size, bsCell.candidates,
                               candidates_end, bsCell.new_frontier, less) -
                bsCell.new_frontier;
            candidates_size = 0;

            // 合并后的候选邻居集合，仅保留其中的前beamSize个
            new_frontier_size = std::min<size_t>(beamSize, new_frontier_size);

            // if a k is given (i.e. k != 0) then trim off entries that have a
            // distance greater than cut * current-kth-smallest-distance.
            // Only used during query and not during build.
            if (QP.k > 0 && new_frontier_size > QP.k)
            {
                // 按照距离剪枝，探测第topK个邻居的距离，并用QP.cut来缩放，大于dist(topk)*cut的都剪枝
                new_frontier_size = std::max<indexType>(
                    (std::upper_bound(bsCell.new_frontier,
                                      bsCell.new_frontier + new_frontier_size,
                                      QNode(0, QP.cut * bsCell.new_frontier[QP.k].dist), less) -
                     bsCell.new_frontier),
                    bsCell.frontier_size);
            }

            memcpy(bsCell.frontier, bsCell.new_frontier, new_frontier_size * sizeof(QNode));
            bsCell.frontier_size = new_frontier_size;

            // 找到frontier和visited中不同的元素，用来刷新unvisited_frontier
            remain = (std::set_difference(bsCell.frontier,
                                          bsCell.frontier + std::min<long>(bsCell.frontier_size, QP.beamSize),
                                          bsCell.visited,
                                          bsCell.visited + bsCell.visited_size,
                                          bsCell.unvisited_frontier, less) -
                      bsCell.unvisited_frontier);
        }

        if (use_filtering)
        {
            // 要想挑选出那些没有访问过的元素，必须使用量化距离排序
            for (auto i = 0; i < bsCell.visited_size; ++i)
                bsCell.visited[i].dist = bsCell.tmp_visited[i];
            std::sort(bsCell.visited, bsCell.visited + bsCell.visited_size, less);
        }

        // 返回距离计算次数
        return full_dist_cmps;
    }

    template <typename indexType, typename Point, typename PointRange, class GraphType>
    size_t beam_search(BeamSearchMemoryCell<indexType> &bsCell, const GraphType &G,
                       const Point p, PointRange &Points,
                       const QueryParams &QP)
    {
        return filtered_beam_search(bsCell, G, p, Points, p, Points, QP);
    }

    // 最高层的封装
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

        // 传入三个压缩级别的向量数据
        // 当第二级别的压缩向量和原始向量不同，将使用重排序
        bool use_rerank = false;
        bool use_filtering = (QP.rerank_factor == 0);
        if (Base_Points.params.num_bytes() == Q_Base_Points.params.num_bytes())
        {
            use_filtering = false;
        }
        else
        {
            use_rerank = !use_filtering;
        }
        size_t dist_cmps = 0;
        if (use_filtering == false)
        {
            dist_cmps = beam_search(bsCell, G, qp, Q_Base_Points, QP);
        }
        else
        {
            dist_cmps = filtered_beam_search(bsCell, G, p, Base_Points, qp, Q_Base_Points, QP);
        }

        if (stats)
        {
            QueryStats.increment_visited(stat_pid, bsCell.num_visited);
            QueryStats.increment_dist(stat_pid, dist_cmps);
        }

        bsCell.visited_size = std::min(bsCell.frontier_size, bsCell.visited_size);

        if (use_rerank)
        {
            // recalculate distances with non-quantized points and sort
            int num_check = std::min<int>(QP.k * QP.rerank_factor, bsCell.frontier_size);
            std::vector<PID> pts;
            for (int i = 0; i < num_check; i++)
            {
                auto j = bsCell.frontier[i].index;
                pts.push_back(PID(j, Base_Points[j].distance(p)));
            }
            auto less = [](PID a, PID b)
            {
                return a.second < b.second || (a.second == b.second && a.first < b.first);
            };
            std::sort(pts.begin(), pts.end(), less);

            // keep first k
            size_t k = std::min((size_t)QP.k, pts.size());
            pts.resize(k);
            return pts;
        }
        else
        {
            // return beamElts;
            size_t k = std::min((size_t)QP.k, bsCell.frontier_size);
            std::vector<PID> pts(k);
            std::transform(bsCell.frontier, bsCell.frontier + k, pts.data(), [](QueryNode<indexType> a)
                           { return PID(a.index, a.dist); });
            return pts;
        }
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

#pragma omp parallel for
        for (size_t i = 0; i < Query_Points.size(); ++i)
        {
            uint32_t worker_id = omp_get_thread_num();

            // 找到候选邻居集合
            auto [starting_points, sp_num] = get_sp(i);
            auto bsCell = bsTable.getCell(i, worker_id, QP.beamSize, QP.beamSize * 8);
            bsCell.init_starting_points(Q_Query_Points[i], Q_Base_Points, starting_points, sp_num);
            auto ngh_dist = beam_search_rerank(bsCell, Query_Points[i], Q_Query_Points[i],
                                               G, Base_Points, Q_Base_Points,
                                               QueryStats, QP, (indexType)i);
            std::vector<indexType> ngh(ngh_dist.size());
            std::transform(ngh_dist.begin(), ngh_dist.end(), ngh.data(), [](std::pair<indexType, float> d)
                           { return d.first; });
            all_neighbors[i] = ngh;
        }

        return all_neighbors;
    }
}