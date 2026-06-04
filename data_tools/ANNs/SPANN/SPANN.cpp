#include <filesystem>
#include "CLI11.hpp"
#include "cluster/hierarchical_cluster.h"
#include "spann.h"
using namespace ant;

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

    // 更新边需要重新计算每个邻居的距离，可以只计算top_degree个邻居的距离，其他插值算出来。
    // 如果等于0将会重新计算所有距离
    uint32_t top_degree = 0;
    uint32_t num_passes = 2;
    uint32_t num_split = 4;

    // 数据集存在方向偏差，就调大这个参数
    float dir_bias_scale = 0;
    // 是否动态剪枝
    bool dynamic_prune = false;
    bool max_angle_prune = false;
    bool use_mips = false;
    bool rebuild = false;
    bool rand_sp = false;

    // SPANN 特殊的参数-------------------------------------------------------------
    bool rand_cluster = false;
    float centroid_percent = 0.16f;
    uint32_t bin_size = 1;

    virtual void parse(CLI::App &app)
    {
        app.add_option("--quant_base_path", quant_base_path, "File path for input quant vectors");
        app.add_option("--pre_index_path", preIndexFile, "File path for pre load graph");
        app.add_option("--index_path", indexFile, "File path for load graph");

        app.add_option("--R", R, "max degree of graph");
        app.add_option("--L", L, "candidate neighbors size");
        app.add_option("--limit", limit, "max search nodes num");

        app.add_option("--VL", VL, "max visited size");
        app.add_option("--top_degree", top_degree, "top degree to cal dist");
        app.add_option("--num_passes", num_passes, "num passes");
        app.add_option("--num_split", num_split, "num split");

        app.add_option("--dir_bias_scale", dir_bias_scale, "dir_bias_scale");
        app.add_flag("--dynamic_prune", dynamic_prune, "dynamic prune");
        app.add_flag("--use_mips", use_mips, "use mips");
        app.add_flag("--rebuild", rebuild, "rebuild");
        app.add_flag("--rand_sp", rand_sp, "rand_sp");

        app.add_option("--max_alpha", max_alpha, "max alpha");
        app.add_option("--min_alpha", min_alpha, "min alpha");

        // --------------------------------------------------------------
        app.add_flag("--rand_cluster", rand_cluster, "use rand cluster");
        app.add_option("--centroid_percent", centroid_percent, "centroid percent");
        app.add_option("--bin_size", bin_size, "ben_size");
    }

    void fixed_args(size_t pointSize)
    {
        centroid_size = pointSize * centroid_percent;
    }
};

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
    // 设置固定的候选邻居集合长度
    uint32_t Q = 0;
    uint32_t rerank_factor = 100;

    bool cache_raw = false;
    // 用来减小查询向量的数量，以便加速实验
    uint32_t q_num = 0;

    virtual void parse(CLI::App &app)
    {
        app.add_flag("--cache_raw", cache_raw, "cache raw vectors");
        app.add_option("--base_path", base_path, "File path for input vectors");
        app.add_option("--query_path", query_path, "query vectors path");
        app.add_option("--quant_query_path", quant_query_path, "query quant vectors path");
        app.add_option("--gt_path", gtFile, "gt path");
        app.add_option("--res_path", rFile, "save res file");
        app.add_option("--k", k, "topk");
        app.add_flag("--verbose", verbose, "verbose");
        app.add_option("--Q", Q, "fixed beam width");
        app.add_option("--rerank_factor", rerank_factor, "rerank factor");
        app.add_option("--q_num", q_num, "query num");
    }
};

template <typename indexType>
Graph_ stats_graph(Graph<indexType> &G, QueryStatic &BuildStats, BuildArgs &args, double idx_time = 0, size_t isolated_cnt = 0)
{

    std::string name = "NSW";

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
    auto [pointSize, dims] = read_head<indexType>(build_args.quant_base_path.c_str());

    Timer t("ANN");
    SPANNIndex<indexType> spann(build_args.bin_size);

    if (build_args.rand_cluster)
    {
        spann.init_c2v_ids(pointSize, build_args.centroid_size);
    }
    else
    {
        HierCluster<indexType> hierCluster = HierCluster<indexType>(build_args.centroid_size + args.cluster_k, args.max_cell_size, args.cluster_k, args.max_reps);
        std::vector<indexType> indices = std::vector<indexType>(pointSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(0));
        hierCluster.clear();
        hierCluster.init_hier_cluster(indices, QPoints, build_args.init_vec_num);
        hierCluster.build_hier_cluster(QPoints);
        for (auto node : hierCluster.cls_tree)
        {
            if (node.numElement == 0)
            {
                continue;
            }
            assert(node.childs.size() == 0);
            auto centroidId = node.centroidId;
            spann.c2v_ids.push_back(centroidId);
        }
    }
    std::cout << "c2v_ids.size()=" << c2v_ids.size() << std::endl;
    build_args.centroid_size = c2v_ids.size();
    spann.init_v2c_ids(pointSize);
    QueryStatic BuildStats = QueryStatic(QPoints.size());
    size_t isolate_node_num = spann.build_centroid_graph(QPoints, build_args.maxDeg, build_args.L, build_args.VL,
                                                         build_args.top_degree, build_args.num_split, build_args.num_passes,
                                                         build_args.dir_bias_scale, build_args.min_alpha, build_args.max_alpha,
                                                         build_args.min_cos_angle, build_args.max_cos_angle,
                                                         build_args.dynamic_prune, build_args.rand_sp, True, BuildStats);
}