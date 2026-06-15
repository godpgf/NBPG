#include <filesystem>
#include "CLI11.hpp"
#include "cluster/hierarchical_cluster.h"
#include "spann.h"
#include "base/parse_results.h"
#include "base/ground_truth.h"
#include "base/check_nn_recall.h"

#include "utils/Point/point.h"
#include "utils/Point/disk_point_range.h"
#include "utils/Graph/query_params.h"
#include "utils/Graph/prune.h"
#include "ivf_search/ivf_check_nn_recall.h"
#include "ivf_search/disk_ivf_opt.h"
using namespace ant;

// Build-phase CLI parameters.
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

    // Exact distance recomputation count during edge update; 0 means all neighbors.
    uint32_t top_degree = 0;
    uint32_t num_passes = 2;
    uint32_t num_split = 4;

    float dir_bias_scale = 0;
    bool dynamic_prune = false;
    bool use_mips = false;
    bool rebuild = false;
    bool rand_sp = false;

    float centroid_percent = 0.16f;
    size_t centroid_size = 0;
    uint32_t bin_size = 1;

    double ivf_alpha = 0;
    size_t ivf_limit = 100;
    uint32_t ivf_R = 16;
    uint32_t ivf_L = 70;
    uint32_t ivf_VL = 140;
    uint32_t max_ivf_size = 1024;
    bool disk_ivf_and_pos = false;

    bool rand_cluster = false;
    uint32_t cluster_k = 16;
    uint32_t max_cell_size = 8;
    uint32_t max_reps = 8;
    uint32_t init_vec_num = 256;

    virtual void parse(CLI::App &app)
    {
        app.add_option("--quant_base_path", quant_base_path, "Path to quantized base vectors");
        app.add_option("--pre_index_path", preIndexFile, "Path prefix for a pre-built graph to warm-start from");
        app.add_option("--index_path", indexFile, "Path prefix for saving or loading the SPANN index");

        app.add_option("--R", R, "Maximum centroid graph degree");
        app.add_option("--L", L, "Beam width during centroid graph construction");
        app.add_option("--limit", limit, "Maximum nodes visited per construction search; 0 uses centroid count");

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

        app.add_flag("--rand_cluster", rand_cluster, "Use random centroid selection instead of hierarchical clustering");
        app.add_option("--centroid_percent", centroid_percent, "Fraction of base vectors used as centroids");
        app.add_option("--bin_size", bin_size, "Number of sub-quantizer bins for IVF packing");

        app.add_option("--ivf_alpha", ivf_alpha, "IVF assignment alpha threshold");
        app.add_option("--ivf_limit", ivf_limit, "Maximum nodes visited during IVF centroid search; 0 uses base size");
        app.add_option("--ivf_R", ivf_R, "Maximum centroids assigned to each base vector");
        app.add_option("--ivf_L", ivf_L, "Beam width during IVF assignment");
        app.add_option("--ivf_VL", ivf_VL, "Maximum visited list size during IVF assignment search");
        app.add_option("--max_ivf_size", max_ivf_size, "Skip centroids whose inverted list already exceeds this size");
        app.add_flag("--disk_ivf_and_pos", disk_ivf_and_pos, "Store IVF lists and vector payloads on disk");
    }

    void fixed_args(size_t point_size)
    {
        centroid_size = static_cast<size_t>(point_size * centroid_percent);
        if (centroid_size == 0 && point_size > 0)
        {
            centroid_size = 1;
        }
        if (ivf_limit == 0)
        {
            ivf_limit = point_size;
        }
    }
};

// Query-phase CLI parameters.
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

    uint32_t ivf_rerank_factor = 30;

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
        app.add_option("--ivf_rerank_factor", ivf_rerank_factor, "Candidate pool multiplier after IVF centroid search");
    }
};

template <typename indexType>
Graph_ stats_graph(Graph<indexType> &G, QueryStatic &BuildStats, const BuildArgs &args,
                   double idx_time = 0, size_t isolated_cnt = 0)
{
    const std::string name = "SPANN";

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
Graph_ build(BuildArgs build_args, QueryArgs /*query_args*/)
{
    using PR = PointRange<Point<dataType>>;

    const auto [point_size, dims] = read_vector_file_head<indexType>(build_args.quant_base_path.c_str());
    (void)dims;
    build_args.fixed_args(point_size);

    QueryStatic BuildStats = QueryStatic(point_size);
    Graph_ G_;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    if (file_exists(build_args.indexFile + ".graph") && !build_args.rebuild)
    {
        Graph<indexType> G((build_args.indexFile + ".graph").c_str());
        G_ = stats_graph(G, BuildStats, build_args);
        return G_;
    }

    PR QPoints = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);
    SPANNIndex<indexType> spann(build_args.bin_size);

    if (build_args.rand_cluster)
    {
        spann.init_c2v_ids(point_size, build_args.centroid_size);
    }
    else
    {
        HierCluster<indexType> hierCluster(build_args.centroid_size + build_args.cluster_k,
                                           build_args.max_cell_size, build_args.cluster_k, build_args.max_reps);
        std::vector<indexType> indices(point_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(0));
        hierCluster.clear();
        hierCluster.init_hier_cluster(indices, QPoints, build_args.init_vec_num);
        hierCluster.build_hier_cluster(QPoints);

        spann.c2v_ids.reserve(hierCluster.cls_tree.size());
        for (const auto &node : hierCluster.cls_tree)
        {
            if (node.numElement == 0)
            {
                continue;
            }
            assert(node.childs.empty());
            spann.c2v_ids.push_back(node.centroidId);
        }
    }

    build_args.centroid_size = spann.c2v_ids.size();
    std::cout << "centroid count=" << build_args.centroid_size << std::endl;

    spann.init_v2c_ids(point_size);

    size_t search_limit = build_args.limit;
    if (search_limit == 0)
    {
        search_limit = build_args.centroid_size;
    }

    Timer t("ANN");
    const size_t isolate_node_num = spann.build_centroid_graph(
        QPoints, build_args.R, build_args.L, build_args.VL,
        build_args.top_degree, build_args.num_split, build_args.num_passes, search_limit,
        build_args.dir_bias_scale, build_args.min_alpha, build_args.max_alpha,
        build_args.min_cos_angle, build_args.max_cos_angle,
        build_args.dynamic_prune, build_args.rand_sp, true, BuildStats);
    std::cout << "isolate_node_num=" << isolate_node_num << std::endl;

    saveIds(build_args.indexFile + ".c2v", spann.c2v_ids);

    spann.build_ivf(QPoints, build_args.ivf_L, build_args.ivf_limit, build_args.ivf_alpha,
                    build_args.ivf_R, build_args.max_ivf_size, build_args.ivf_VL, false, BuildStats);
    if (build_args.disk_ivf_and_pos)
    {
        saveIVFBinary(spann.ivf_index, QPoints.size(), build_args.indexFile + ".ivf");
        InvIVF::copy_point2ivf<PR, indexType>(build_args.indexFile + ".ivf", QPoints,
                                              build_args.indexFile + ".ivf_pos", build_args.ivf_R);
    }
    else
    {
        saveVectorBinary(spann.ivf_index, build_args.indexFile + ".ivf");
    }

    spann.centroidGraph.save((build_args.indexFile + ".graph").c_str());
    G_ = stats_graph(spann.centroidGraph, BuildStats, build_args, t.get_next(), isolate_node_num);
    return G_;
}

template <typename PR, typename QyPR, typename QPoint_, typename indexType = uint32_t>
void run_recall_test(const BuildArgs &build_args, QueryArgs query_args,
                     PR &Points, QyPR &Query_Points,
                     PointRange<QPoint_> &QBase_Points, PointRange<QPoint_> &QQuery_Points,
                     Graph_ &G_)
{
    if (query_args.q_num > 0 && query_args.q_num < Query_Points.size())
    {
        Query_Points.resize(query_args.q_num);
    }

    GroundTruth<indexType> GT(query_args.gtFile.c_str(), false);
    Graph<indexType> centroidGraph((build_args.indexFile + ".graph").c_str());

    std::vector<indexType> c2v_ids;
    loadIds(build_args.indexFile + ".c2v", c2v_ids);

    indexType sp = 0;
    using GetSpFn = std::function<std::pair<indexType *, uint32_t>(indexType)>;
    GetSpFn get_sp = [&](indexType) -> std::pair<indexType *, uint32_t>
    {
        return std::make_pair(&sp, static_cast<uint32_t>(1));
    };

    if (build_args.disk_ivf_and_pos)
    {
        using QPR = PointRange<QPoint_>;
        auto Q_Centroids = RefPointRange<QPR, indexType>(c2v_ids.data(), c2v_ids.size(), &QBase_Points);
        ivf_search_and_parse(G_, centroidGraph, build_args.indexFile + ".ivf_pos",
                             Points, Query_Points, Q_Centroids, c2v_ids, QQuery_Points, GT,
                             query_args.rFile.c_str(), query_args.k, get_sp,
                             query_args.verbose, query_args.Q,
                             query_args.rerank_factor, query_args.ivf_rerank_factor);
    }
    else
    {
        std::vector<std::vector<uint32_t>> ivf_index;
        loadVectorBinary(ivf_index, build_args.indexFile + ".ivf");
        ivf_search_and_parse(G_, centroidGraph, ivf_index, build_args.bin_size,
                             QBase_Points, Points, Query_Points, c2v_ids, QQuery_Points, GT,
                             query_args.rFile.c_str(), query_args.k, get_sp,
                             query_args.verbose, query_args.Q,
                             query_args.rerank_factor, query_args.ivf_rerank_factor);
    }
}

template <typename QPR, typename dataType, typename indexType = uint32_t>
void run_recall_test(const BuildArgs &build_args, QueryArgs query_args,
                     QPR &QBase_Points, QPR &QQuery_Points, Graph_ &G_)
{
    using Point_ = Point<dataType>;
    const DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
    auto Query_Points = load_point_range<indexType, PointRange<Point_>>(query_args.query_path.c_str(), distance_metric);
    if (query_args.cache_raw)
    {
        auto Points = load_point_range<indexType, PointRange<Point_>>(query_args.base_path.c_str(), distance_metric);
        run_recall_test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G_);
    }
    else
    {
        indexType n, dims;
        std::tie(n, dims) = read_vector_file_head<indexType>(query_args.base_path.c_str());
        DiskPointRange<Point_> Points(query_args.base_path.c_str(), n, dims, 2 * sizeof(indexType), distance_metric);
        run_recall_test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G_);
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

    auto QBase_Points = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

    if (!query_args.base_path.empty())
    {
        if (ends_with(query_args.base_path, ".u8bin"))
        {
            run_recall_test<PR, uint8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".u4bin"))
        {
            run_recall_test<PR, uint4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".i8bin"))
        {
            run_recall_test<PR, int8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".i4bin"))
        {
            run_recall_test<PR, int4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".fbin") || ends_with(query_args.base_path, ".bin"))
        {
            run_recall_test<PR, float, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
    }
    else
    {
        run_recall_test(build_args, query_args, QBase_Points, QQuery_Points, QBase_Points, QQuery_Points, G_);
    }
}

int main(int argc, char *argv[])
{
    BuildArgs build_args;
    QueryArgs query_args;

    CLI::App app{"SPANN"};
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
    else if (ends_with(build_args.quant_base_path, ".fbin") || ends_with(build_args.quant_base_path, ".bin"))
    {
        const auto G_ = build<float, uint32_t>(build_args, query_args);
        test<float, uint32_t>(build_args, query_args, G_);
    }

    return 0;
}
