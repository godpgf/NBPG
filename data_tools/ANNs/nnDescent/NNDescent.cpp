#include "CLI11.hpp"
#include "base/parse_results.h"
#include "base/ground_truth.h"
#include "base/check_nn_recall.h"
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include "utils/Point/disk_point_range.h"
#include "utils/Graph/query_params.h"
#include "utils/Graph/dfs.h"
#include "utils/Graph/prune.h"
#include "clusterNN.h"
#include "nnDescent.h"
using namespace ant;

// 构图阶段命令行参数
class BuildArgs
{
public:
    // 量化底库向量路径
    std::string quant_base_path = "";
    // 索引输出路径前缀（实际写入 index_path.graph）
    std::string indexFile = "";

    // 图最大出度
    uint32_t R = 50;
    // 构图 beam 宽度
    uint32_t L = 70;

    // RobustPrune 的 alpha 阈值
    double alpha = 0;
    // 角度剪枝阈值；0 表示不启用
    double cos_angle = 0;

    // NN-Descent 收敛判定阈值
    double delta = 0.05;
    // NN-Descent 最大迭代轮数
    int max_rounds = 64;
    // 初始近邻树分簇数量
    size_t num_clusters = 32;
    // 每个分簇的目标规模
    size_t cluster_size = 1000;

    bool use_mips = false;
    // 为 true 时忽略已有图文件并重新构图
    bool rebuild = false;

    virtual void parse(CLI::App &app)
    {
        app.add_option("--quant_base_path", quant_base_path, "Path to quantized base vectors");
        app.add_option("--index_path", indexFile, "Path prefix for saving or loading the graph index");
        app.add_option("--R", R, "Maximum graph degree");
        app.add_option("--L", L, "Beam width during graph construction");

        app.add_option("--alpha", alpha, "Robust prune alpha threshold");
        app.add_option("--cos_angle", cos_angle, "Cosine angle threshold for prune; 0 disables angle prune");

        app.add_option("--delta", delta, "Convergence threshold for NN-Descent iterations");
        app.add_option("--max_rounds", max_rounds, "Maximum NN-Descent iteration rounds");
        app.add_option("--num_clusters", num_clusters, "Number of clusters for initial neighbor trees");
        app.add_option("--cluster_size", cluster_size, "Target size of each initial neighbor cluster");

        app.add_flag("--use_mips", use_mips, "Use MIPS distance instead of L2");
        app.add_flag("--rebuild", rebuild, "Rebuild the index even if graph files already exist");
    }
};

// 查询评测阶段命令行参数
class QueryArgs
{
public:
    // 原始底库路径，用于 rerank；为空则复用量化底库
    std::string base_path = "";
    // 原始查询向量路径
    std::string query_path = "";
    // 量化查询向量路径
    std::string quant_query_path = "";
    std::string gtFile = "";
    std::string rFile = "";

    uint32_t k = 0;
    bool verbose = false;
    // 固定 beam 宽度；0 表示按 rerank_factor 自动搜索
    uint32_t Q = 0;
    uint32_t rerank_factor = 100;

    // 为 true 时将原始底库加载到内存，否则使用磁盘映射
    bool cache_raw = false;
    // 限制参与评测的查询数量；0 表示使用全部查询
    uint32_t q_num = 0;

    virtual void parse(CLI::App &app)
    {
        app.add_flag("--cache_raw", cache_raw, "Load raw base vectors into memory instead of disk mapping");
        app.add_option("--base_path", base_path, "Path to raw base vectors for reranking");
        app.add_option("--query_path", query_path, "Path to raw query vectors");
        app.add_option("--quant_query_path", quant_query_path, "Path to quantized query vectors");
        app.add_option("--gt_path", gtFile, "Path to ground truth file");
        app.add_option("--res_path", rFile, "Path to save recall benchmark CSV");
        app.add_option("--k", k, "Top-k for recall evaluation");
        app.add_flag("--verbose", verbose, "Print detailed recall statistics");
        app.add_option("--Q", Q, "Fixed beam width; 0 enables automatic beam search");
        app.add_option("--rerank_factor", rerank_factor, "Initial beam width multiplier for rerank search");
        app.add_option("--q_num", q_num, "Number of queries to evaluate; 0 uses all queries");
    }
};

template <typename indexType>
Graph_ stats_graph(Graph<indexType> &G, QueryStatic &BuildStats, const BuildArgs &build_args, double idx_time = 0, size_t isolated_cnt = 0)
{
    std::ostringstream params_ss;
    params_ss << std::fixed << std::setprecision(2)
              << "R = " << build_args.R
              << ", L = " << build_args.L
              << ", alpha = " << build_args.alpha
              << ", cos_angle = " << build_args.cos_angle
              << ", delta = " << build_args.delta
              << ", num_clusters = " << build_args.num_clusters
              << ", cluster_size = " << build_args.cluster_size
              << ", max_rounds = " << build_args.max_rounds;

    auto [avg_deg, max_deg] = graph_stats_(G);
    auto [avg_visited, tail_visited] = BuildStats.visited_stats();
    std::cout << "Average visited: " << avg_visited << ", Tail visited: " << tail_visited
              << std::endl;

    Graph_ G_("NNDescent", params_ss.str(), G.size(), avg_deg, max_deg, idx_time, isolated_cnt);
    G_.print();
    return G_;
}

template <typename dataType, typename indexType>
Graph_ build(BuildArgs build_args, QueryArgs query_args, bool repair_isolate = true)
{
    using PR = PointRange<Point<dataType>>;

    auto [pointSize, dims] = read_vector_file_head<indexType>(build_args.quant_base_path.c_str());
    QueryStatic BuildStats = QueryStatic(pointSize);

    Graph_ G_;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
    const std::string graph_path = build_args.indexFile + ".graph";

    if (file_exists(graph_path) && !build_args.rebuild)
    {
        Graph<indexType> G(graph_path.c_str());
        G_ = stats_graph(G, BuildStats, build_args);
    }
    else
    {
        Graph<indexType> G(build_args.R, pointSize);
        PR QPoints = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

        Timer t("ANN");
        auto prune = Prune(G.max_degree(), omp_get_max_threads());
        std::vector<std::vector<IdDist<indexType>>> old_neighbors;
        ClusterPID<PR, indexType>::multiple_clustertrees(
            QPoints, build_args.cluster_size, build_args.num_clusters, G.max_degree(),
            old_neighbors, BuildStats);

        NNDescent<PR, indexType> nn_descent(build_args.L, build_args.delta, build_args.max_rounds);
        nn_descent.nn_descent_wrapper(QPoints, old_neighbors, BuildStats);
        nn_descent.undirect_and_prune(G, prune, QPoints, build_args.alpha, build_args.cos_angle, old_neighbors, BuildStats);

        size_t isolate_node_num = 0;
        if (repair_isolate)
        {
            isolate_node_num = DFS::repair_isolate_node(G, static_cast<indexType>(0));
            std::cout << "isolate_node_num=" << isolate_node_num << std::endl;
        }
        G.save(graph_path.c_str());

        G_ = stats_graph(G, BuildStats, build_args, t.get_next(), isolate_node_num);
    }
    return G_;
}

template <typename PR, typename QyPR, typename QPoint_, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           PR &Points, QyPR &Query_Points,
           PointRange<QPoint_> &QBase_Points, PointRange<QPoint_> &QQuery_Points,
           Graph<indexType> &G, Graph_ &G_)
{
    if (query_args.q_num > 0 && query_args.q_num < Query_Points.size())
    {
        Query_Points.resize(query_args.q_num);
    }
    GroundTruth<indexType> GT(query_args.gtFile.c_str(), false);

    search_and_parse(G_, G, Points, Query_Points, QBase_Points, QQuery_Points, GT,
                     query_args.rFile.c_str(), query_args.k, query_args.verbose,
                     query_args.Q, query_args.rerank_factor);
}

template <typename QPR, typename dataType, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           QPR &QBase_Points, QPR &QQuery_Points,
           Graph<indexType> &G, Graph_ &G_)
{
    using Point_ = Point<dataType>;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
    auto Query_Points = load_point_range<indexType, PointRange<Point_>>(query_args.query_path.c_str(), distance_metric);
    if (query_args.cache_raw)
    {
        auto Points = load_point_range<indexType, PointRange<Point_>>(query_args.base_path.c_str(), distance_metric);
        _test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G, G_);
    }
    else
    {
        indexType n, dims;
        std::tie(n, dims) = read_vector_file_head<indexType>(query_args.base_path.c_str());
        DiskPointRange<Point_> Points(query_args.base_path.c_str(), n, dims, 2 * sizeof(indexType), distance_metric);
        _test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G, G_);
    }
}

template <typename dataType, typename indexType>
void test(BuildArgs build_args, QueryArgs query_args, Graph_ G_)
{
    if (query_args.quant_query_path.empty() || query_args.gtFile.empty())
        return;

    using PR = PointRange<Point<dataType>>;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    auto QQuery_Points = load_point_range<indexType, PR>(query_args.quant_query_path.c_str(), distance_metric);
    if (query_args.q_num > 0 && query_args.q_num < QQuery_Points.size())
    {
        QQuery_Points.resize(query_args.q_num);
    }

    const std::string graph_path = build_args.indexFile + ".graph";
    Graph<indexType> G(graph_path.c_str());
    auto QBase_Points = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

    if (!query_args.base_path.empty())
    {
        if (ends_with(query_args.base_path, ".u8bin"))
        {
            _test<PR, uint8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".u4bin"))
        {
            _test<PR, uint4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".i8bin"))
        {
            _test<PR, int8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".i4bin"))
        {
            _test<PR, int4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".fbin") || ends_with(query_args.base_path, ".bin"))
        {
            _test<PR, float, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
    }
    else
    {
        _test(build_args, query_args, QBase_Points, QQuery_Points, QBase_Points, QQuery_Points, G, G_);
    }
}

int main(int argc, char *argv[])
{
    BuildArgs build_args;
    QueryArgs query_args;

    CLI::App app{"nnDescent"};
    build_args.parse(app);
    query_args.parse(app);
    CLI11_PARSE(app, argc, argv);

    if (ends_with(build_args.quant_base_path, ".u8bin"))
    {
        auto G_ = build<uint8_t, uint32_t>(build_args, query_args);
        test<uint8_t, uint32_t>(build_args, query_args, G_);
    }
    else if (ends_with(build_args.quant_base_path, ".u4bin"))
    {
        auto G_ = build<uint4_2, uint32_t>(build_args, query_args);
        test<uint4_2, uint32_t>(build_args, query_args, G_);
    }
    else if (ends_with(build_args.quant_base_path, ".i8bin"))
    {
        auto G_ = build<int8_t, uint32_t>(build_args, query_args);
        test<int8_t, uint32_t>(build_args, query_args, G_);
    }
    else if (ends_with(build_args.quant_base_path, ".i4bin"))
    {
        auto G_ = build<int4_2, uint32_t>(build_args, query_args);
        test<int4_2, uint32_t>(build_args, query_args, G_);
    }

    return 0;
}
