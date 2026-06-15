#include "CLI11.hpp"
#include "base/parse_results.h"
#include "base/ground_truth.h"
#include "base/check_nn_recall.h"
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include "utils/Point/disk_point_range.h"
#include "utils/Graph/query_params.h"
#include "utils/Graph/prune.h"
#include "nsw.h"
using namespace ant;

// 构图阶段命令行参数
class BuildArgs
{
public:
    // 量化底库向量路径
    std::string quant_base_path = "";
    // 预加载图路径前缀；非空时在其基础上继续构图
    std::string preIndexFile = "";
    // 索引输出路径前缀（实际写入 index_path.graph）
    std::string indexFile = "";

    // 图最大出度
    uint32_t R = 50;
    // 构图 beam 宽度
    uint32_t L = 70;
    // 构图阶段单次搜索最多访问的节点数；0 表示使用底库规模
    size_t limit = 0;

    // beam search 访问列表上限
    uint32_t VL = 140;
    // 动态 alpha 下界
    double min_alpha = 1;
    // 动态 alpha 上界
    double max_alpha = 1.1;
    // 动态 cos_angle 下界
    double min_cos_angle = 0;
    // 动态 cos_angle 上界
    double max_cos_angle = 0;

    // 更新边时精确计算距离的邻居数；0 表示全部重新计算
    uint32_t top_degree = 0;
    // Vamana 构图轮数
    uint32_t num_passes = 2;
    // 分片数量；>=1 时启用分片插入
    uint32_t num_split = 4;

    // 数据集存在方向偏差时可调大该参数
    float dir_bias_scale = 0;
    // 是否在最后一轮前启用动态剪枝
    bool dynamic_prune = false;
    bool use_mips = false;
    // 为 true 时忽略已有图文件并重新构图
    bool rebuild = false;
    // 为 true 时使用随机起点而非默认起点
    bool rand_sp = false;

    virtual void parse(CLI::App &app)
    {
        app.add_option("--quant_base_path", quant_base_path, "Path to quantized base vectors");
        app.add_option("--pre_index_path", preIndexFile, "Path prefix of a pre-built graph to continue from");
        app.add_option("--index_path", indexFile, "Path prefix for saving or loading the graph index");

        app.add_option("--R", R, "Maximum graph degree");
        app.add_option("--L", L, "Beam width during graph construction");
        app.add_option("--limit", limit, "Maximum nodes visited per construction search; 0 uses dataset size");

        app.add_option("--VL", VL, "Maximum visited list size during beam search");
        app.add_option("--top_degree", top_degree, "Number of neighbors for exact distance computation; 0 means all");
        app.add_option("--num_passes", num_passes, "Number of Vamana construction passes");
        app.add_option("--num_split", num_split, "Number of graph splits for staged insertion; 0 disables split stage");

        app.add_option("--dir_bias_scale", dir_bias_scale, "Direction bias scale for datasets with directional skew");
        app.add_flag("--dynamic_prune", dynamic_prune, "Enable dynamic prune before the final pass");
        app.add_flag("--use_mips", use_mips, "Use MIPS distance instead of L2");
        app.add_flag("--rebuild", rebuild, "Rebuild the index even if graph files already exist");
        app.add_flag("--rand_sp", rand_sp, "Use random start points instead of the default start point");

        app.add_option("--max_alpha", max_alpha, "Upper bound of dynamic alpha during prune");
        app.add_option("--min_alpha", min_alpha, "Lower bound of dynamic alpha during prune");
        app.add_option("--max_cos_angle", max_cos_angle, "Upper bound of dynamic cosine angle threshold during prune");
        app.add_option("--min_cos_angle", min_cos_angle, "Lower bound of dynamic cosine angle threshold during prune");
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
Graph_ stats_graph(Graph<indexType> &G, QueryStatic &BuildStats, const BuildArgs &args, double idx_time = 0, size_t isolated_cnt = 0)
{
    std::string name = "Vamana";

    std::stringstream min_alpha_s, max_alpha_s;
    min_alpha_s << std::fixed << std::setprecision(2) << args.min_alpha;
    max_alpha_s << std::fixed << std::setprecision(2) << args.max_alpha;

    std::string params =
        "R = " + std::to_string(args.R) + ", L = " + std::to_string(args.L) + ", alpha = (" + min_alpha_s.str() + "," + max_alpha_s.str() +
        "), use_mips = " + std::to_string(args.use_mips);
    auto [avg_deg, max_deg] = graph_stats_(G);
    auto [avg_visited, tail_visited] = BuildStats.visited_stats();
    std::cout << "Average visited: " << avg_visited << ", Tail visited: " << tail_visited
              << std::endl;

    Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time, isolated_cnt);
    G_.print();
    return G_;
}

template <typename dataType, typename indexType>
Graph_ build(BuildArgs build_args, QueryArgs query_args, bool repair_isolate = true)
{
    using PR = PointRange<Point<dataType>>;

    auto [pointSize, dims] = read_vector_file_head<indexType>(build_args.quant_base_path.c_str());
    if (build_args.limit == 0)
    {
        build_args.limit = pointSize;
    }

    QueryStatic BuildStats = QueryStatic(pointSize);

    Graph_ G_;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    if (file_exists(build_args.indexFile + ".graph") && build_args.rebuild == false)
    {
        Graph<indexType> G((build_args.indexFile + ".graph").c_str());
        G_ = stats_graph(G, BuildStats, build_args);
    }
    else
    {
        PR QPoints = load_point_range<indexType, PointRange<Point<dataType>>>(build_args.quant_base_path.c_str(), distance_metric);
        Graph<indexType> G;
        if (build_args.preIndexFile != "")
        {
            G = Graph<indexType>((build_args.preIndexFile + ".graph").c_str());
        }
        else
        {
            G = Graph<indexType>(build_args.R, QPoints.size());
        }

        Timer t("ANN");

        size_t isolate_node_num = build_vamana(QPoints, G, build_args.L, build_args.VL, build_args.top_degree, build_args.num_split, build_args.num_passes, build_args.limit,
                                               build_args.dir_bias_scale, build_args.min_alpha, build_args.max_alpha, build_args.min_cos_angle, build_args.max_cos_angle,
                                               build_args.dynamic_prune, build_args.rand_sp, repair_isolate, BuildStats);

        G.save((build_args.indexFile + ".graph").c_str());

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
    GroundTruth<indexType> GT = GroundTruth<indexType>(query_args.gtFile.c_str(), false);

    search_and_parse(G_, G, Points, Query_Points, QBase_Points, QQuery_Points, GT, (const char *)query_args.rFile.c_str(), query_args.k, query_args.verbose, query_args.Q, query_args.rerank_factor);
}

template <typename QPR, typename dataType, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           QPR &QBase_Points, QPR &QQuery_Points,
           Graph<indexType> &G, Graph_ &G_)
{
    using Point_ = Point<dataType>;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
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
    if (query_args.quant_query_path == "" || query_args.gtFile == "")
        return;
    using PR = PointRange<Point<dataType>>;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    auto QQuery_Points = load_point_range<indexType, PR>(query_args.quant_query_path.c_str(), distance_metric);
    if (query_args.q_num > 0 && query_args.q_num < QQuery_Points.size())
    {
        QQuery_Points.resize(query_args.q_num);
    }

    auto gfile = build_args.indexFile + ".graph";
    Graph<indexType> G(gfile.c_str());

    auto QBase_Points = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

    if (query_args.base_path != "")
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

    CLI::App app{"Vamana"};
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
