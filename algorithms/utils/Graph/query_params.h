#pragma once
#include <algorithm>
#include <utility>
#include "utils/utilities.h"

namespace ant
{

    // -------------------------------------------------------------------------
    // QueryParams — beam search 查询参数
    // -------------------------------------------------------------------------
    struct QueryParams
    {
        // k > 0 时：以 frontier 中第 k 近候选的距离 × cut 作为剪枝阈值（仅查询阶段使用）
        uint32_t k = 0;
        double cut = 0.0;

        // beam 宽度：frontier 最多保留的候选数
        uint32_t beamSize = 0;

        // 最多从 frontier 中 pop 并扩展的节点数（通常设为 |V|）
        size_t limit = 0;

        // 每个图节点最多扩展的邻居数
        uint32_t degree_limit = 0;

        // rerank 模式：先用量化向量搜索，再对 frontier 前 k×rerank_factor 个结果
        // 用原始向量重算距离并重排。rerank_factor == 0 且存在独立原始向量时启用 filtering 模式。
        uint32_t rerank_factor = 40;

        // IVF 搜索中对质心 beam 结果做 rerank 时的扩展因子
        uint32_t ivf_rerank_factor = 15;

        QueryParams(uint32_t k, uint32_t Q, double cut, size_t limit, uint32_t dg,
                    uint32_t rerank_factor = 100, uint32_t ivf_rerank_factor = 40)
            : k(k), beamSize(Q), cut(cut), limit(limit), degree_limit(dg),
              rerank_factor(rerank_factor), ivf_rerank_factor(ivf_rerank_factor)
        {
        }

        QueryParams() = default;
    };

    // -------------------------------------------------------------------------
    // QueryStatic — 查询统计（每 query 的访问数 / 距离计算次数）
    // -------------------------------------------------------------------------
    struct QueryStatic
    {
        QueryStatic(size_t n) : visited(n, 0), distances(n, 0) {}

        void increment_dist(size_t i, uint32_t j) { distances[i] += j; }
        void increment_visited(size_t i, uint32_t j) { visited[i] += j; }

        std::pair<uint32_t, uint32_t> visited_stats() { return statistics(visited); }
        uint32_t visited_std() { return (uint32_t)std(visited); }
        std::pair<uint32_t, uint32_t> dist_stats() { return statistics(distances); }
        uint32_t dist_std() { return (uint32_t)std(distances); }

        void clear()
        {
            size_t n = visited.size();
            memset(visited.data(), 0, n * sizeof(uint32_t));
            memset(distances.data(), 0, n * sizeof(uint32_t));
        }

        std::pair<uint32_t, uint32_t> statistics(std::vector<uint32_t> &s)
        {
            std::sort(s.begin(), s.end());
            double sum = 0;
            for (auto si : s)
                sum += si;
            uint32_t avg = sum / s.size();
            size_t tail_index = .99 * ((float)s.size());
            uint32_t tail = s[tail_index];
            return std::pair<uint32_t, uint32_t>(avg, tail);
        }

        float std(std::vector<uint32_t> &s)
        {
            double sum_2 = 0;
            double sum = 0;
            for (auto si : s)
            {
                sum += si;
                sum_2 += si * si;
            }
            sum /= s.size();
            sum_2 /= s.size();
            return sqrt(sum_2 - sum * sum);
        }

        std::vector<uint32_t> visited;
        std::vector<uint32_t> distances;
    };

    // (index, distance) 对，按 dist 升序、index 升序打破平局
    template <typename indexType>
    struct IdDist
    {
        IdDist(indexType index, float dist) : index(index), dist(dist) {}
        IdDist() = default;
        indexType index;
        float dist;
    };

    template <typename indexType>
    struct QueryNode : public IdDist<indexType>
    {
        QueryNode(indexType index, float dist, float log_popular = 0)
            : IdDist<indexType>(index, dist), log_popular(log_popular) {}
        QueryNode() = default;
        float log_popular = 0;
    };

    template <typename indexType>
    bool id_dist_less(const IdDist<indexType> &a, const IdDist<indexType> &b)
    {
        return a.dist < b.dist || (a.dist == b.dist && a.index < b.index);
    }

    // -------------------------------------------------------------------------
    // BeamSearchMemoryCell — 单次 beam search 的工作区（栈上指针，内存由 Table 预分配）
    //
    // frontier        : 当前最优 beam，按距离升序，大小 ≤ beamSize
    // unvisited_frontier : frontier 中尚未 pop 扩展的节点（延迟更新，避免每步全量重排）
    // visited         : 已 pop 并扩展过的节点
    // tmp_visited     : filtering 模式下 visited 对应节点的原始向量精确距离
    // candidates      : 本轮新发现、待并入 frontier 的邻居
    // hash_filter     : 已见节点哈希过滤（允许假阴性）
    // -------------------------------------------------------------------------
    template <typename indexType>
    struct BeamSearchMemoryCell
    {
        using QNode = QueryNode<indexType>;

        BeamSearchMemoryCell() = default;

        BeamSearchMemoryCell(const BeamSearchMemoryCell<indexType> &bsCell)
        {
            beamSize = bsCell.beamSize;
            visited = bsCell.visited;
            frontier = bsCell.frontier;
            visited_size = bsCell.visited_size;
            frontier_size = bsCell.frontier_size;
        }

        BeamSearchMemoryCell(uint32_t beamSize,
                             QNode *frontier,
                             QNode *new_frontier,
                             QNode *unvisited_frontier,
                             indexType *hash_filter,
                             indexType *filtered,
                             QNode *candidates,
                             QNode *visited,
                             QNode *copy_visited,
                             float *tmp_visited,
                             const size_t max_visited_size)
            : beamSize(beamSize),
              frontier(frontier),
              new_frontier(new_frontier),
              unvisited_frontier(unvisited_frontier),
              hash_filter(hash_filter),
              filtered(filtered),
              candidates(candidates),
              visited(visited),
              copy_visited(copy_visited),
              tmp_visited(tmp_visited),
              max_visited_size(max_visited_size)
        {
            clear();
        }

        void clear()
        {
            hash_filter_size = get_max_hash_filter_size(beamSize);
            frontier_size = 0;
            visited_size = 0;
            num_visited = 0;
            std::fill_n(hash_filter, hash_filter_size, -1);
        }

        // 起点距离已由调用方算好（常用于量化 query × 量化 base）
        void init_starting_points(QNode *starting_points, uint32_t sp_num)
        {
            if (beamSize < sp_num)
            {
                std::cout << "beamSize < sp_num " << beamSize << " < " << sp_num << std::endl;
                abort();
            }

            for (auto sid = 0; sid < sp_num; ++sid)
            {
                auto q = starting_points[sid];
                frontier[sid] = q;
                has_been_seen(q.index);
            }
            frontier_size = sp_num;
            std::sort(frontier, frontier + sp_num, id_dist_less<indexType>);
            memcpy(unvisited_frontier, frontier, frontier_size * sizeof(QNode));
        }

        // 用 PointRange 对 starting_points 逐点算距并初始化 frontier
        template <typename Point, typename PointRange, typename spType>
        void init_starting_points(const Point p, PointRange &Points, spType *starting_points, uint32_t sp_num)
        {
            if (beamSize < sp_num)
            {
                std::cout << "beamSize < sp_num " << beamSize << " < " << sp_num << std::endl;
                abort();
            }

            for (auto sid = 0; sid < sp_num; ++sid)
            {
                auto q = starting_points[sid];
                frontier[sid] = QNode(static_cast<indexType>(q), Points[q].distance(p));
                has_been_seen(q);
            }
            frontier_size = sp_num;
            std::sort(frontier, frontier + frontier_size, id_dist_less<indexType>);
            memcpy(unvisited_frontier, frontier, frontier_size * sizeof(QNode));
        }

        bool has_been_seen(indexType a)
        {
            int loc = hash64_2(a) & (hash_filter_size - 1);
            if (hash_filter[loc] == a)
                return true;
            hash_filter[loc] = a;
            return false;
        }

        uint32_t beamSize;

        QNode *frontier;
        size_t frontier_size;

        QNode *new_frontier;

        QNode *unvisited_frontier;

        indexType *hash_filter;
        size_t hash_filter_size;
        static size_t get_max_hash_filter_size(uint32_t beamSize)
        {
            int bits = std::max<int>(10, std::ceil(std::log2(beamSize * beamSize)) - 2);
            return (1 << bits);
        }

        indexType *filtered;

        QNode *candidates;

        QNode *visited;
        QNode *copy_visited;
        float *tmp_visited;
        size_t visited_size;
        size_t num_visited;
        // 剪枝时会复用 copy_visited，需预留 max_degree 个额外槽位
        size_t max_visited_size;
    };

    // 预分配多 query × 多线程的 beam search 缓冲区，避免热路径上动态分配
    template <typename indexType>
    struct BeamSearchMemoryTable
    {
        using QNode = QueryNode<indexType>;

        BeamSearchMemoryTable(uint32_t num_cells,
                              uint32_t num_workers,
                              uint32_t maxBeamSize,
                              size_t max_visited_size,
                              uint32_t max_degree)
            : num_cells(num_cells),
              num_workers(num_workers),
              maxBeamSize(maxBeamSize),
              max_visited_size(max_visited_size),
              max_degree(max_degree),
              all_frontier(num_cells * get_max_frontier_size(maxBeamSize)),
              all_new_frontier(num_workers * get_max_new_frontier_size(maxBeamSize, max_degree)),
              all_unvisited_frontier(num_workers * get_max_frontier_size(maxBeamSize)),
              all_hash_filter(num_workers * BeamSearchMemoryCell<indexType>::get_max_hash_filter_size(maxBeamSize)),
              all_filtered(num_workers * get_max_filtered_size(max_degree)),
              all_candidates(num_workers * get_max_candidate_size(maxBeamSize, max_degree)),
              all_visited(num_cells * max_visited_size),
              all_copy_visited(num_cells * max_visited_size),
              all_tmp_visited(num_cells * max_visited_size)
        {
        }

        BeamSearchMemoryCell<indexType> getCell(uint32_t cell_id, uint32_t worker_id, uint32_t beamSize = 0, size_t max_visited_size = 0)
        {
            if (beamSize == 0)
                beamSize = maxBeamSize;
            if (max_visited_size == 0)
                max_visited_size = this->max_visited_size;
            if (cell_id >= num_cells || worker_id >= num_workers || beamSize > this->maxBeamSize || max_visited_size > this->max_visited_size)
            {
                std::cout << "getCell ERROR!" << std::endl;
                abort();
            }
            QNode *frontier = all_frontier.data() + cell_id * get_max_frontier_size(beamSize);
            QNode *new_frontier = all_new_frontier.data() + worker_id * get_max_new_frontier_size(beamSize, max_degree);
            QNode *unvisited_frontier = all_unvisited_frontier.data() + worker_id * get_max_frontier_size(beamSize);
            indexType *hash_filter = all_hash_filter.data() + worker_id * BeamSearchMemoryCell<indexType>::get_max_hash_filter_size(beamSize);
            indexType *filtered = all_filtered.data() + worker_id * get_max_filtered_size(max_degree);
            QNode *candidates = all_candidates.data() + worker_id * get_max_candidate_size(beamSize, max_degree);
            QNode *visited = all_visited.data() + cell_id * max_visited_size;
            auto *copy_visited = all_copy_visited.data() + cell_id * max_visited_size;
            float *tmp_visited = all_tmp_visited.data() + cell_id * max_visited_size;
            return BeamSearchMemoryCell<indexType>(beamSize,
                                                   frontier,
                                                   new_frontier,
                                                   unvisited_frontier,
                                                   hash_filter,
                                                   filtered,
                                                   candidates,
                                                   visited,
                                                   copy_visited,
                                                   tmp_visited,
                                                   max_visited_size);
        }

        size_t get_max_visited_size() { return max_visited_size; }
        const uint32_t max_degree;

    protected:
        std::vector<QNode> all_frontier;
        std::vector<QNode> all_new_frontier;
        std::vector<QNode> all_unvisited_frontier;
        std::vector<indexType> all_hash_filter;
        std::vector<indexType> all_filtered;
        std::vector<QNode> all_candidates;
        std::vector<QNode> all_visited;
        std::vector<QNode> all_copy_visited;
        std::vector<float> all_tmp_visited;
        uint32_t num_cells;
        uint32_t num_workers;
        uint32_t maxBeamSize;
        size_t max_visited_size;

        static size_t get_max_frontier_size(uint32_t beamSize) { return beamSize; }
        static size_t get_max_new_frontier_size(uint32_t beamSize, uint32_t max_degree) { return 2 * beamSize + max_degree; }
        static uint32_t get_max_filtered_size(uint32_t max_degree) { return max_degree; }
        // 延迟合并：最多累积约 beamSize/8 批邻居后再排序，上界取 beamSize + max_degree
        static uint32_t get_max_candidate_size(uint32_t beamSize, uint32_t max_degree) { return beamSize + max_degree; }
    };
}
