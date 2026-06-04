#pragma once
#include <algorithm>
#include <utility> // 包含 std::pair 的头文件
#include "utils/utilities.h"

namespace ant
{

    // 搜索参数
    struct QueryParams
    {
        // k > 0 表示使用第topk个候选邻居到查询向量的距离，
        // 经过cut缩放后，作为阈值去剪枝其他候选邻居
        uint32_t k;
        double cut;

        // 候选邻居大小
        uint32_t beamSize;

        // 限制总的跳转次数，可以设置为数据集的长度
        size_t limit;

        // 限制邻居数量
        uint32_t degree_limit;

        // 利用压缩向量计算距离，然后对最近的s个结果加载真实向量并重新排序，s=k*rerank_factor
        uint32_t rerank_factor = 40;

        // 用于对ivf进行重排序
        uint32_t ivf_rerank_factor = 15;


        QueryParams(uint32_t k, uint32_t Q, double cut, size_t limit, uint32_t dg, uint32_t rerank_factor = 100, uint32_t ivf_rerank_factor = 40) : k(k), beamSize(Q), cut(cut), limit(limit), degree_limit(dg), rerank_factor(rerank_factor), ivf_rerank_factor(ivf_rerank_factor) {}

        QueryParams() {}
    };

    // 搜索统计
    struct QueryStatic
    {
        QueryStatic(size_t n) : visited(n, 0), distances(n, 0)
        {
        }

        void increment_dist(size_t i, uint32_t j)
        {
            distances[i] += j;
        }
        void increment_visited(size_t i, uint32_t j)
        {
            visited[i] += j;
        }

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
            {
                sum += si;
            }
            uint32_t avg = sum / s.size();
            size_t tail_index = .99 * ((float)s.size());
            uint32_t tail = s[tail_index];
            // 返回均值和99分位的极大值
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

    template <typename indexType>
    struct IdDist {
        IdDist(indexType index, float dist) : index(index), dist(dist) {}
        IdDist() = default;
        indexType index;
        float dist;
    };

    template <typename indexType>
    struct QueryNode : public IdDist<indexType> {
        QueryNode(indexType index, float dist, uint32_t depth = 0, bool isLeaf = false)
            : IdDist<indexType>(index, dist), depth(depth), isLeaf(isLeaf) {}
        QueryNode() = default;
        uint32_t depth = 0;
        bool isLeaf = false;
    };

    template <typename indexType>
    bool id_dist_less(const IdDist<indexType>& a, const IdDist<indexType>& b){
        return a.dist < b.dist || (a.dist == b.dist && a.index < b.index);
    }

    // 缓存搜索过程中的各种数据
    template <typename indexType>
    struct BeamSearchMemoryCell
    {
        using QNode = QueryNode<indexType>;
        BeamSearchMemoryCell() {}

        BeamSearchMemoryCell(const BeamSearchMemoryCell<indexType> &bsCell)
        {
            this->beamSize = bsCell.beamSize;
            this->visited = bsCell.visited;
            this->frontier = bsCell.frontier;
            this->visited_size = bsCell.visited_size;
            this->frontier_size = bsCell.frontier_size;
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
                             const size_t max_visited_size) : beamSize(beamSize),
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
            this->hash_filter_size = get_max_hash_filter_size(beamSize);
            this->frontier_size = 0;
            this->visited_size = 0;
            this->num_visited = 0;
            std::fill_n(hash_filter, hash_filter_size, -1);
        }

        void init_starting_points(QNode *starting_points, uint32_t sp_num)
        {
            if (beamSize < sp_num)
            {
                std::cout << "beamSize < sp_num " << beamSize << " < " << sp_num << std::endl;
                abort();
            }

            // 插入frontier
            for (auto sid = 0; sid < sp_num; ++sid)
            {
                auto q = starting_points[sid];
                frontier[sid] = q;
                has_been_seen(q.first);
            }
            frontier_size = sp_num;

            auto less = [&](const QNode& a, const QNode& b)
            {
                return query_node_less(a, b);
            };
            std::sort(frontier, frontier + sp_num, less);

            // 更新unvisited_frontier
            memcpy(unvisited_frontier, frontier, frontier_size * sizeof(QNode));
        }

        template <typename Point, typename PointRange, typename spType>
        void init_starting_points(const Point p, PointRange &Points, spType *starting_points, uint32_t sp_num)
        {
            if (beamSize < sp_num)
            {
                std::cout << "beamSize < sp_num " << beamSize << " < " << sp_num << std::endl;
                abort();
            }

            // 插入frontier
            for (auto sid = 0; sid < sp_num; ++sid)
            {
                auto q = starting_points[sid];
                frontier[sid] = std::move(QueryNode(static_cast<indexType>(q), Points[q].distance(p)));
                has_been_seen(q);
            }
            frontier_size = sp_num;

            auto less = [&](const QNode& a, const QNode& b)
            {
                return id_dist_less(a, b);
            };
            std::sort(frontier, frontier + frontier_size, less);

            // 更新unvisited_frontier
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

        // Frontier maintains the closest points found so far and its size
        // is always at most beamSize.  Each entry is a (id,distance) pair.
        // Initialized with starting points and kept sorted by distance.
        QNode *frontier;
        size_t frontier_size;

        // used as temporaries in the loop
        // new_frontier.size() == 2 * beamSize + G.max_degree()
        QNode *new_frontier;

        // 记录frontier中没有访问过的元素，最大长度是beamSize
        QNode *unvisited_frontier;

        // used as a hash filter (can give false negative -- i.e. can say
        // not in table when it is)
        indexType *hash_filter;
        size_t hash_filter_size;
        static size_t get_max_hash_filter_size(uint32_t beamSize)
        {
            int bits = std::max<int>(10, std::ceil(std::log2(beamSize * beamSize)) - 2);
            return (1 << bits);
        }

        // 当前遍历到的，准备插入candidates的元素
        indexType *filtered;

        // 当前遍历到的，准备插入frontier的元素
        QNode *candidates;

        // 记录以及访问过的元素
        QNode *visited;
        QNode *copy_visited; // 用于剪枝
        // 零时内存，用于存储访问过的元素的量化距离和真实距离
        float *tmp_visited;
        size_t visited_size;
        size_t num_visited;
        // 【潜规则】：剪枝时会利用这部分空间，添加已有邻居都访问列表。所以，需要多预留max_degree个位置
        size_t max_visited_size;
    };

    // 预先一次性申请多个beamSearch算法需要的所有空间，避免动态申请空间产生的内存碎片
    template <typename indexType>
    struct BeamSearchMemoryTable
    {
        using QNode = QueryNode<indexType>;

        BeamSearchMemoryTable(uint32_t num_cells,
                              uint32_t num_workers,
                              uint32_t maxBeamSize,
                              size_t max_visited_size,
                              uint32_t max_degree) : num_cells(num_cells),
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
        static uint32_t get_max_candidate_size(uint32_t beamSize, uint32_t max_degree) { return beamSize + max_degree; }
    };
}