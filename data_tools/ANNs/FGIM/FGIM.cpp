#include "CLI11.hpp"
#include "base/parse_results.h"
#include "base/ground_truth.h"
#include "base/check_nn_recall.h"
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include "utils/Point/disk_point_range.h"
#include "utils/Graph/query_params.h"
#include "utils/Graph/prune.h"
#include "utils/Graph/dfs.h"
#include "FGIM_nsw.h"
using namespace ant;

namespace
{
    NodePopular node_popular_for_pass(int pass, int num_passes, bool dynamic_prune)
    {
        if (pass == num_passes - 2 && dynamic_prune)
        {
            return NodePopular::CalPopular;
        }
        if (pass == num_passes - 1)
        {
            if (dynamic_prune && num_passes > 1)
            {
                return NodePopular::PopularPrune;
            }
            return NodePopular::FixedMinPopular;
        }
        return NodePopular::FixedMaxPopular;
    }
}

// 构图阶段命令行参数
class BuildArgs
{
public:
    std::string quant_base_path = "";
    std::string preIndexFile = "";
    std::string indexFile = "";

    uint32_t R = 50;
    uint32_t L = 70;
    size_t limit = 0;

    uint32_t VL = 140;
    double min_alpha = 1;
    double max_alpha = 1.1;
    double min_cos_angle = 0;
    double max_cos_angle = 0;

    // 更新边时精确计算距离的邻居数；0 表示全部重新计算
    uint32_t top_degree = 0;
    uint32_t num_passes = 2;
    // 分块数；>1 时先做分块 warm-up，再在 pair 阶段用多起始点搜索
    uint32_t num_split = 4;

    // 数据集存在方向偏差时可调大
    float dir_bias_scale = 0;
    bool dynamic_prune = false;
    bool max_angle_prune = false;
    bool use_mips = false;
    bool rebuild = false;
    // 最后一轮是否用随机起始点
    bool rand_sp = false;

    virtual void parse(CLI::App &app)
    {
        app.add_option("--quant_base_path", quant_base_path, "Path to quantized base vectors");
        app.add_option("--pre_index_path", preIndexFile, "Path prefix for a pre-built graph to warm-start from");
        app.add_option("--index_path", indexFile, "Path prefix for saving or loading the graph index");

        app.add_option("--R", R, "Maximum graph degree");
        app.add_option("--L", L, "Beam width during graph construction");
        app.add_option("--limit", limit, "Maximum nodes visited per construction search; 0 uses dataset size");

        app.add_option("--VL", VL, "Maximum visited list size during beam search");
        app.add_option("--top_degree", top_degree, "Number of neighbors for exact distance computation; 0 means all");
        app.add_option("--num_passes", num_passes, "Number of full-graph construction passes");
        app.add_option("--num_split", num_split, "Number of split partitions for warm-up and multi-start search");

        app.add_option("--dir_bias_scale", dir_bias_scale, "Direction bias scale for directional datasets");
        app.add_flag("--dynamic_prune", dynamic_prune, "Enable popularity-aware dynamic pruning on the last passes");
        app.add_flag("--use_mips", use_mips, "Use MIPS distance instead of L2");
        app.add_flag("--rebuild", rebuild, "Rebuild the index even if graph files already exist");
        app.add_flag("--rand_sp", rand_sp, "Use random starting points on the final pass");

        app.add_option("--max_alpha", max_alpha, "Maximum RobustPrune alpha on the final pass");
        app.add_option("--min_alpha", min_alpha, "Minimum RobustPrune alpha under dynamic pruning");
    }

    void fixed_args(size_t pointSize)
    {
        if (limit == 0)
        {
            limit = pointSize;
        }
    }
};

// 查询评测阶段命令行参数
class QueryArgs
{
public:
    std::string base_path = "";
    std::string query_path = "";
    std::string quant_query_path = "";
    std::string gtFile = "";
    std::string rFile = "";

    uint32_t k = 0;
    bool verbose = false;
    uint32_t Q = 0;
    uint32_t rerank_factor = 100;

    bool cache_raw = false;
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
    const std::string name = "FIGM";

    std::stringstream min_alpha_s, max_alpha_s;
    min_alpha_s << std::fixed << std::setprecision(2) << args.min_alpha;
    max_alpha_s << std::fixed << std::setprecision(2) << args.max_alpha;

    const std::string params =
        "R = " + std::to_string(args.R) + ", L = " + std::to_string(args.L) +
        ", alpha = (" + min_alpha_s.str() + "," + max_alpha_s.str() +
        "), use_mips = " + std::to_string(args.use_mips);
    const auto [avg_deg, max_deg] = graph_stats_(G);
    const auto [avg_visited, tail_visited] = BuildStats.visited_stats();
    std::cout << "Average visited: " << avg_visited << ", Tail visited: " << tail_visited << std::endl;

    Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time, isolated_cnt);
    G_.print();
    return G_;
}

template <typename dataType, typename indexType>
Graph_ build(BuildArgs build_args, QueryArgs query_args, bool repair_isolate = true)
{
    using PR = PointRange<Point<dataType>>;

    const auto [pointSize, dims] = read_vector_file_head<indexType>(build_args.quant_base_path.c_str());
    (void)dims;
    build_args.fixed_args(pointSize);

    QueryStatic BuildStats = QueryStatic(pointSize);
    Graph_ G_;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    if (file_exists(build_args.indexFile + ".graph") && !build_args.rebuild)
    {
        Graph<indexType> G((build_args.indexFile + ".graph").c_str());
        G_ = stats_graph(G, BuildStats, build_args);
    }
    else
    {
        PR QPoints = load_point_range<indexType, PointRange<Point<dataType>>>(build_args.quant_base_path.c_str(), distance_metric);
        Graph<indexType> G;
        if (!build_args.preIndexFile.empty())
        {
            G = Graph<indexType>((build_args.preIndexFile + ".graph").c_str());
        }
        else
        {
            G = Graph<indexType>(build_args.R, QPoints.size());
        }
        Graph<indexType> toG(build_args.R, QPoints.size());

        Timer t("ANN");

        using NSWType = FIGM_NSWIndex<indexType>;
        using FillStartFun = std::function<void(std::vector<std::vector<indexType>> &, size_t, size_t)>;

        const auto prune = Prune(G.max_degree(), omp_get_max_threads());
        auto nsw = NSWType(prune, build_args.L, build_args.VL, omp_get_max_threads(), G.max_degree(),
                           NSWType::getMaxBatchSize(G), build_args.top_degree,
                           build_args.min_alpha, build_args.max_alpha, build_args.min_cos_angle, build_args.max_cos_angle);

        std::vector<indexType> inserts(QPoints.size());
        std::iota(inserts.begin(), inserts.end(), 0);

        NodePopular nodePopular = NodePopular::FixedMaxPopular;
        if (build_args.num_split > 1)
        {
            FillStartFun fill_split_start = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
            {
                for (size_t j = 0; j < ceiling - floor; ++j)
                {
                    const indexType index = inserts[floor + j];
                    sp_cache[j].push_back(index % build_args.num_split);
                }
            };

            for (int i = 0; i < 2; ++i)
            {
                std::shuffle(inserts.begin(), inserts.end(), std::mt19937(i));
                nsw.batch_insert(inserts.data(), inserts.size(), G, QPoints, build_args.L, build_args.limit,
                                 &BuildStats, fill_split_start, nodePopular, build_args.dir_bias_scale, 2, .02);
            }
        }

        FillStartFun fill_reverse_start = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
        {
            for (size_t j = 0; j < ceiling - floor; ++j)
            {
                const indexType index = inserts[floor + j];
                sp_cache[j].push_back(inserts.size() - index - 1);
            }
        };

        FillStartFun fill_default_start = [&](std::vector<std::vector<indexType>> &sp_cache, size_t floor, size_t ceiling)
        {
            for (size_t j = 0; j < ceiling - floor; ++j)
            {
                sp_cache[j].push_back(0);
            }
        };

        for (int pass = 0; pass < static_cast<int>(build_args.num_passes); ++pass)
        {
            std::shuffle(inserts.begin(), inserts.end(), std::mt19937(pass));
            nodePopular = node_popular_for_pass(pass, static_cast<int>(build_args.num_passes), build_args.dynamic_prune);

            if (pass != static_cast<int>(build_args.num_passes) - 1)
            {
                nsw.pair_batch_insert(inserts.data(), inserts.size(), G, toG, QPoints, build_args.L, build_args.limit,
                                      &BuildStats, build_args.num_split, nodePopular, build_args.dir_bias_scale, 2, .02);
            }
            else
            {
                const FillStartFun &final_fill = build_args.rand_sp ? fill_reverse_start : fill_default_start;
                nsw.batch_insert(inserts.data(), inserts.size(), toG, QPoints, build_args.L, build_args.limit,
                                 &BuildStats, final_fill, nodePopular, build_args.dir_bias_scale, 2, .02, false);
            }
        }

        size_t isolate_node_num = 0;
        if (repair_isolate)
        {
            isolate_node_num = DFS::repair_isolate_node(toG, static_cast<indexType>(0));
            std::cout << "isolate_node_num=" << isolate_node_num << std::endl;
        }
        toG.save((build_args.indexFile + ".graph").c_str());

        G_ = stats_graph(toG, BuildStats, build_args, t.get_next(), isolate_node_num);
    }

    return G_;
}

template <typename PR, typename QyPR, typename QPoint_, typename indexType = uint32_t>
void run_recall_test(const BuildArgs &build_args, QueryArgs query_args,
                     PR &Points, QyPR &Query_Points,
                     PointRange<QPoint_> &QBase_Points, PointRange<QPoint_> &QQuery_Points,
                     Graph<indexType> &G, Graph_ &G_)
{
    if (query_args.q_num > 0 && query_args.q_num < Query_Points.size())
    {
        Query_Points.resize(query_args.q_num);
    }
    GroundTruth<indexType> GT = GroundTruth<indexType>(query_args.gtFile.c_str(), false);

    search_and_parse(G_, G, Points, Query_Points, QBase_Points, QQuery_Points, GT,
                     query_args.rFile.c_str(), query_args.k, query_args.verbose, query_args.Q, query_args.rerank_factor);
}

template <typename QPR, typename dataType, typename indexType = uint32_t>
void run_recall_test(const BuildArgs &build_args, QueryArgs query_args,
                     QPR &QBase_Points, QPR &QQuery_Points,
                     Graph<indexType> &G, Graph_ &G_)
{
    using Point_ = Point<dataType>;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
    auto Query_Points = load_point_range<indexType, PointRange<Point_>>(query_args.query_path.c_str(), distance_metric);
    if (query_args.cache_raw)
    {
        auto Points = load_point_range<indexType, PointRange<Point_>>(query_args.base_path.c_str(), distance_metric);
        run_recall_test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G, G_);
    }
    else
    {
        indexType n, dims;
        std::tie(n, dims) = read_vector_file_head<indexType>(query_args.base_path.c_str());
        DiskPointRange<Point_> Points(query_args.base_path.c_str(), n, dims, 2 * sizeof(indexType), distance_metric);
        run_recall_test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G, G_);
    }
}

template <typename dataType, typename indexType>
void test(BuildArgs build_args, QueryArgs query_args, Graph_ G_)
{
    if (query_args.quant_query_path.empty() || query_args.gtFile.empty())
    {
        return;
    }

    using PR = PointRange<Point<dataType>>;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    auto QQuery_Points = load_point_range<indexType, PR>(query_args.quant_query_path.c_str(), distance_metric);
    if (query_args.q_num > 0 && query_args.q_num < QQuery_Points.size())
    {
        QQuery_Points.resize(query_args.q_num);
    }

    const auto gfile = build_args.indexFile + ".graph";
    Graph<indexType> G(gfile.c_str());
    auto QBase_Points = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

    if (!query_args.base_path.empty())
    {
        if (ends_with(query_args.base_path, ".u8bin"))
        {
            run_recall_test<PR, uint8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".u4bin"))
        {
            run_recall_test<PR, uint4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".i8bin"))
        {
            run_recall_test<PR, int8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".i4bin"))
        {
            run_recall_test<PR, int4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
        else if (ends_with(query_args.base_path, ".fbin") || ends_with(query_args.base_path, ".bin"))
        {
            run_recall_test<PR, float, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, G_);
        }
    }
    else
    {
        run_recall_test(build_args, query_args, QBase_Points, QQuery_Points, QBase_Points, QQuery_Points, G, G_);
    }
}

int main(int argc, char *argv[])
{
    BuildArgs build_args;
    QueryArgs query_args;

    CLI::App app{"FIGM"};
    build_args.parse(app);
    query_args.parse(app);
    CLI11_PARSE(app, argc, argv);

    if (ends_with(build_args.quant_base_path, ".u8bin"))
    {
        const auto G_ = build<uint8_t, uint32_t>(build_args, query_args);
        test<uint8_t, uint32_t>(build_args, query_args, G_);
    }
    else if (ends_with(build_args.quant_base_path, ".u4bin"))
    {
        const auto G_ = build<uint4_2, uint32_t>(build_args, query_args);
        test<uint4_2, uint32_t>(build_args, query_args, G_);
    }
    else if (ends_with(build_args.quant_base_path, ".i8bin"))
    {
        const auto G_ = build<int8_t, uint32_t>(build_args, query_args);
        test<int8_t, uint32_t>(build_args, query_args, G_);
    }
    else if (ends_with(build_args.quant_base_path, ".i4bin"))
    {
        const auto G_ = build<int4_2, uint32_t>(build_args, query_args);
        test<int4_2, uint32_t>(build_args, query_args, G_);
    }

    return 0;
}
