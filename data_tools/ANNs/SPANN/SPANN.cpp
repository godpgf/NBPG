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

    // 分层聚类
    bool rand_cluster = false;
    uint32_t cluster_k = 16;
    uint32_t max_cell_size = 8;
    uint32_t max_reps = 8;
    uint32_t init_vec_num = 256;

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

        app.add_option("--ivf_alpha", ivf_alpha, "ivf_alpha");
        app.add_option("--ivf_limit", ivf_limit, "ivf_limit");
        app.add_option("--ivf_R", ivf_R, "ivf_R");
        app.add_option("--ivf_L", ivf_L, "ivf_L");
        app.add_option("--ivf_VL", ivf_VL, "ivf_VL");
        app.add_option("--max_ivf_size", max_ivf_size, "max ivf size");
        app.add_flag("--disk_ivf_and_pos", disk_ivf_and_pos, "是否将ivf和pos存在硬盘上");
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

    uint32_t ivf_rerank_factor = 30;

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

        app.add_option("--ivf_rerank_factor", ivf_rerank_factor, "ivf rerank factor");
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
    auto [pointSize, dims] = read_vector_file_head<indexType>(build_args.quant_base_path.c_str());
    build_args.fixed_args(pointSize);

    Timer t("ANN");

    QueryStatic BuildStats = QueryStatic(pointSize);

    Graph_ G_;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    if (file_exists(build_args.indexFile + ".graph") && build_args.rebuild == false)
    {
        // 如果质心图已经存在，说明索引已经构建过
        Graph<indexType> G((build_args.indexFile + ".graph").c_str());
        G_ = stats_graph(G, BuildStats, build_args);
    }
    else
    {
        PR QPoints = load_point_range<indexType, PointRange<Point<dataType>>>(build_args.quant_base_path.c_str(), distance_metric);
        SPANNIndex<indexType> spann(build_args.bin_size);
        if (build_args.rand_cluster)
        {
            spann.init_c2v_ids(pointSize, build_args.centroid_size);
        }
        else
        {
            HierCluster<indexType> hierCluster = HierCluster<indexType>(build_args.centroid_size + build_args.cluster_k, build_args.max_cell_size, build_args.cluster_k, build_args.max_reps);
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
        std::cout << "c2v_ids.size()=" << spann.c2v_ids.size() << std::endl;
        build_args.centroid_size = spann.c2v_ids.size();
        spann.init_v2c_ids(pointSize);

        size_t isolate_node_num = spann.build_centroid_graph(QPoints, build_args.R, build_args.L, build_args.VL,
                                                             build_args.top_degree, build_args.num_split, build_args.num_passes,
                                                             build_args.dir_bias_scale, build_args.min_alpha, build_args.max_alpha,
                                                             build_args.min_cos_angle, build_args.max_cos_angle,
                                                             build_args.dynamic_prune, build_args.rand_sp, true, BuildStats);
        std::cout << "isolate_node_num:" << isolate_node_num << std::endl;

        // 保存质心到向量id的映射
        saveIds(build_args.indexFile + ".c2v", spann.c2v_ids);

        // 保存ivf
        spann.build_ivf(QPoints, build_args.ivf_L, build_args.ivf_limit, build_args.ivf_alpha, build_args.ivf_R, build_args.max_ivf_size, build_args.ivf_VL, false, BuildStats);
        if(build_args.disk_ivf_and_pos){
            saveIVFBinary(spann.ivf_index, QPoints.size(), build_args.indexFile + ".ivf");
            InvIVF::copy_point2ivf<PointRange<Point<dataType>>, indexType>(build_args.indexFile + ".ivf", QPoints, build_args.indexFile + ".ivf_pos", (size_t)build_args.ivf_R);
        } else {
            saveVectorBinary(spann.ivf_index, build_args.indexFile + ".ivf");
        }
        
        // 保存质心图
        spann.centroidGraph.save((build_args.indexFile + ".graph").c_str());
        G_ = stats_graph(spann.centroidGraph, BuildStats, build_args, t.get_next(), isolate_node_num);
    }

    return G_;
}

template <typename PR, typename QyPR, typename QPoint_, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           PR &Points, QyPR &Query_Points,
           PointRange<QPoint_> &QBase_Points, PointRange<QPoint_> &QQuery_Points,
        Graph_ &G_)
{
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
    if (query_args.q_num > 0 && query_args.q_num < Query_Points.size())
    {
        Query_Points.resize(query_args.q_num);
    }
    GroundTruth<indexType> GT = GroundTruth<indexType>(query_args.gtFile.c_str(), false);


    // 加载图
    auto gfile = build_args.indexFile + ".graph";
    Graph<uint32_t> centroidGraph(gfile.c_str());

    // 加载质心对应的向量id
    std::vector<indexType> c2v_ids;
    loadIds(build_args.indexFile + ".c2v", c2v_ids);



    indexType sp = 0;
    
    using FType = std::function<std::pair<indexType *, uint32_t>(indexType)>;
    FType getSP = [&](indexType index) -> std::pair<indexType *, uint32_t>
    {
        return std::make_pair(&sp, (uint32_t)1);
    };

    if(build_args.disk_ivf_and_pos){
        // todo 以后实现
        std::cout<<"以后实现"<<std::endl;
    } else {
        // 加载ivf
        std::vector<std::vector<uint32_t>> ivf_index;
        loadVectorBinary(ivf_index, build_args.indexFile + ".ivf");
        ivf_search_and_parse(G_,
                            centroidGraph,
                            ivf_index,
                            build_args.bin_size,
                            QBase_Points,
                            Points,
                            Query_Points,
                            c2v_ids,
                            QQuery_Points,
                            GT,
                            (const char *)query_args.rFile.c_str(),
                            query_args.k,
                            getSP,
                            query_args.verbose, query_args.Q,
                            query_args.rerank_factor,
                            query_args.ivf_rerank_factor);
    }


}

template <typename QPR, typename dataType, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           QPR &QBase_Points, QPR &QQuery_Points,
           Graph_ &G_)
{
    using Point_ = Point<dataType>;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;
    auto Query_Points = load_point_range<indexType, PointRange<Point_>>(query_args.query_path.c_str(), distance_metric);
    if (query_args.cache_raw)
    {
        auto Points = load_point_range<indexType, PointRange<Point_>>(query_args.base_path.c_str(), distance_metric);
        _test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G_);
    }
    else
    {
        indexType n, dims;
        std::tie(n, dims) = read_vector_file_head<indexType>(query_args.base_path.c_str());
        DiskPointRange<Point_> Points(query_args.base_path.c_str(), n, dims, 2 * sizeof(indexType), distance_metric);
        _test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G_);
    }
}

template <typename dataType, typename indexType>
void test(BuildArgs build_args, QueryArgs query_args, Graph_ G_)
{
    if (query_args.quant_query_path == "" || query_args.gtFile == "")
        return;
    using PR = PointRange<Point<dataType>>;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDEAN;

    // 加载查询量化向量
    auto QQuery_Points = load_point_range<indexType, PR>(query_args.quant_query_path.c_str(), distance_metric);
    if (query_args.q_num > 0 && query_args.q_num < QQuery_Points.size())
    {
        QQuery_Points.resize(query_args.q_num);
    }


    // 加载量化质心或者底库量化向量
    auto QBase_Points = load_point_range<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

    if (query_args.base_path != "")
    {
        indexType n, dims;
        std::tie(n, dims) = read_vector_file_head<indexType>(query_args.base_path.c_str());

        if (ends_with(query_args.base_path, ".u8bin"))
        {
            _test<PR, uint8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".u4bin"))
        {
            _test<PR, uint4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".i8bin"))
        {
            _test<PR, int8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".i4bin"))
        {
            _test<PR, int4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
        else if (ends_with(query_args.base_path, ".fbin") || ends_with(query_args.base_path, ".bin"))
        {
            _test<PR, float, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G_);
        }
    }
    else
    {
        _test(build_args, query_args, QBase_Points, QQuery_Points, QBase_Points, QQuery_Points, G_);
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