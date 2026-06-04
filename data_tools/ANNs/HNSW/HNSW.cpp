#include <filesystem>
#include "CLI11.hpp"
#include "base/hier_dfs.h"
#include "base/hnsw.h"
#include "base/hier_check_nn_Recall.h"
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include "utils/Point/disk_point_range.h"
using namespace ant;

class BuildArgs
{
public:
    // 基本构图向量路径
    std::string quant_base_path = "";

    std::string indexFile = "";

    
    uint32_t R = 50;
    uint32_t PR = 0;
    uint32_t L = 70;
    size_t limit = 0;

    uint32_t VL = 140;
    double alpha = 1.1;

    double cos_angle = 0;

    double m_l = 0;
    size_t max_node_num = 0;
    // 更新边需要重新计算每个邻居的距离，可以只计算top_degree个邻居的距离，其他插值算出来。
    // 如果等于0将会重新计算所有距离
    uint32_t top_degree = 0;

    uint32_t hier_beamSize = 1;

    bool use_mips = false;
    bool rebuild = false;

    virtual void parse(CLI::App &app)
    {
        app.add_option("--quant_base_path", quant_base_path, "File path for input quant vectors");
        app.add_option("--index_path", indexFile, "File path for load graph");


        
        app.add_option("--R", R, "max degree of graph");
        app.add_option("--PR", PR, "rerank size for prune enhance");
        app.add_option("--L", L, "candidate neighbors size");
        app.add_option("--limit", limit, "max search nodes num");

        app.add_option("--VL", VL, "max visited size");
        app.add_option("--alpha", alpha, "alpha");

        app.add_option("--cos_angle", cos_angle, "cos_angle");

        app.add_option("--m_l", m_l, "m_l");
        app.add_option("--max_node_num", max_node_num, "max node num");
        app.add_option("--top_degree", top_degree, "top degree to cal dist");
        app.add_option("--hier_beamSize", hier_beamSize, "hier beamSize");

        app.add_flag("--use_mips", use_mips, "use mips");
        app.add_flag("--rebuild", rebuild, "Should the currently loaded graph structure be rebuilt");
    }

    void fixed_args(size_t pointSize)
    {
        if (m_l < 1e-5)
            m_l = 1.0f / log(R);
        if (max_node_num == 0)
            max_node_num = (exp(-1.0f / m_l) * 2 + 1) * pointSize;

        if (limit == 0)
        {
            limit = pointSize;
        }
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
Graph_ stats_graph(Graph<indexType> &G, QueryStatic &BuildStats, BuildArgs build_args, double idx_time=0, size_t isolated_cnt=0)
{

    std::string name = "HNSW";

    std::stringstream alpha_s;
    alpha_s << std::fixed << std::setprecision(2) << build_args.alpha;

    std::stringstream m_l_s;
    m_l_s << std::fixed << std::setprecision(4) << build_args.m_l;

    std::string params =
        "R = " + std::to_string(build_args.R) + ", L = " + std::to_string(build_args.L) +
        ", alpha = " + alpha_s.str() + ", m_l = " + m_l_s.str();
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
    build_args.fixed_args(pointSize);
    QueryStatic BuildStats = QueryStatic(build_args.max_node_num);

    Graph_ G_;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDIAN;

    if (file_exists(build_args.indexFile + ".graph") && build_args.rebuild == false)
    {
        // 如果质心图已经存在，说明索引已经构建过
        Graph<indexType> G((build_args.indexFile + ".graph").c_str());
        G_ = stats_graph(G, BuildStats, build_args);
    }
    else
    {
        using HNSWType = HNSWIndex<indexType>;
        PR QPoints = loadPointRange<indexType, PointRange<Point<dataType>>>(build_args.quant_base_path.c_str(), distance_metric);
        Graph<indexType> G(build_args.R, build_args.max_node_num);
        auto hidMapping = HIdMapping(build_args.max_node_num);

        Timer t("ANN");
        auto prune = Prune(G.max_degree(), omp_get_max_threads());
        auto hnsw = HNSWType(prune, build_args.L, build_args.hier_beamSize, build_args.VL, omp_get_max_threads(), G.max_degree(), NSWIndex<indexType>::getMaxBatchSize(G), build_args.top_degree);
        hnsw.batch_insert(G, hidMapping, QPoints, build_args.alpha, build_args.m_l, &BuildStats);

        size_t isolate_node_num = 0;
        if (repair_isolate)
        {
            auto start_point = hidMapping.get_sp();
            isolate_node_num = HierDFS::repair_isolate_node(G, start_point, 1000, &hidMapping);
            std::cout << "isolate_node_num=" << isolate_node_num << std::endl;
        }

        G.save((build_args.indexFile + ".graph").c_str());
        hidMapping.save((build_args.indexFile + ".hid").c_str());

        G_ = stats_graph(G, BuildStats, build_args, t.get_next(), isolate_node_num);
    }
    return G_;
}

template <typename PR, typename QyPR, typename QPoint_, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           PR &Points, QyPR &Query_Points,
           PointRange<QPoint_> &QBase_Points, PointRange<QPoint_> &QQuery_Points,
           Graph<indexType> &G, HIdMapping &hidMapping, Graph_ &G_)
{
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDIAN;
    if (query_args.q_num > 0 && query_args.q_num < Query_Points.size())
    {
        Query_Points.resize(query_args.q_num);
    }
    GroundTruth<indexType> GT = GroundTruth<indexType>(query_args.gtFile.c_str(), false);

    indexType sp = 0;
    using FType = std::function<std::pair<indexType *, indexType>(indexType)>;
    FType getSP = [&](indexType index) -> std::pair<indexType *, uint32_t>
    {
        return std::make_pair(&sp, (uint32_t)1);
    };

    hier_search_and_parse(G_, G, hidMapping, Points, Query_Points, QBase_Points, QQuery_Points, GT, (const char *)query_args.rFile.c_str(), query_args.k, query_args.verbose, query_args.Q, query_args.rerank_factor, build_args.hier_beamSize);
}

template <typename QPR, typename dataType, typename indexType = uint32_t>
void _test(BuildArgs build_args, QueryArgs query_args,
           QPR &QBase_Points, QPR &QQuery_Points,
           Graph<indexType> &G, HIdMapping &hidMapping, Graph_ &G_)
{
    using Point_ = Point<dataType>;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDIAN;
    auto Query_Points = loadPointRange<indexType, PointRange<Point_>>(query_args.query_path.c_str(), distance_metric);
    if (query_args.cache_raw)
    {
        auto Points = loadPointRange<indexType, PointRange<Point_>>(query_args.base_path.c_str(), distance_metric);
        _test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G, hidMapping, G_);
    }
    else
    {
        indexType n, dims;
        std::tie(n, dims) = read_head<indexType>(query_args.base_path.c_str());
        DiskPointRange<Point_> Points(query_args.base_path.c_str(), n, dims, 2 * sizeof(indexType), distance_metric);
        _test(build_args, query_args, Points, Query_Points, QBase_Points, QQuery_Points, G, hidMapping, G_);
    }
}

template <typename dataType, typename indexType>
void test(BuildArgs build_args, QueryArgs query_args, Graph_ G_)
{
    if (query_args.quant_query_path == "" || query_args.gtFile == "")
        return;
    using PR = PointRange<Point<dataType>>;
    DistanceMetric distance_metric = build_args.use_mips ? DistanceMetric::MIPS : DistanceMetric::EUCLIDIAN;

    // 加载查询量化向量
    auto QQuery_Points = loadPointRange<indexType, PR>(query_args.quant_query_path.c_str(), distance_metric);
    if (query_args.q_num > 0 && query_args.q_num < QQuery_Points.size())
    {
        QQuery_Points.resize(query_args.q_num);
    }

    // 加载图
    auto gfile = build_args.indexFile + ".graph";
    Graph<indexType> G(gfile.c_str());
    auto hfile = build_args.indexFile + ".hid";
    HIdMapping hidMapping(hfile.c_str());

    // 加载量化质心或者底库量化向量
    auto QBase_Points = loadPointRange<indexType, PR>(build_args.quant_base_path.c_str(), distance_metric);

    if (query_args.base_path != "")
    {
        indexType n, dims;
        std::tie(n, dims) = read_head<indexType>(query_args.base_path.c_str());

        if (ends_with(query_args.base_path, ".u8bin"))
        {
            _test<PR, uint8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, hidMapping, G_);
        }
        else if (ends_with(query_args.base_path, ".u4bin"))
        {
            _test<PR, uint4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, hidMapping, G_);
        }
        else if (ends_with(query_args.base_path, ".i8bin"))
        {
            _test<PR, int8_t, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, hidMapping, G_);
        }
        else if (ends_with(query_args.base_path, ".i4bin"))
        {
            _test<PR, int4_2, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, hidMapping, G_);
        }
        else if (ends_with(query_args.base_path, ".fbin") || ends_with(query_args.base_path, ".bin"))
        {
            _test<PR, float, indexType>(build_args, query_args, QBase_Points, QQuery_Points, G, hidMapping, G_);
        }
    }
    else
    {
        _test(build_args, query_args, QBase_Points, QQuery_Points, QBase_Points, QQuery_Points, G, hidMapping, G_);
    }
}

int main(int argc, char *argv[])
{
    BuildArgs build_args;
    QueryArgs query_args;

    CLI::App app{"HNSE"};
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