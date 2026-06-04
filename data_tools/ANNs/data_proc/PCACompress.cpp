// pca compress vector
#include <fstream>
#include <cassert>
#include <algorithm>
#include "CLI11.hpp"
#include "log.h"
#include "utils/stats/get_time.h"
#include "base/matrix.h"
#include "utils/utilities.h"
using namespace ant;

struct Args
{
    // 输入输出文件
    std::string iFile;
    std::string oFile;
    std::string matFile;
    std::string index_type = "uint32";
};

template <typename indexType>
void compress(Args args, PointRange<Point<float>> &mat, std::vector<float> &bias)
{

    uint32_t chunk_size = 1;
    if (ends_with(args.oFile, ".u4bin") || ends_with(args.oFile, ".i4bin"))
    {
        chunk_size = 2;
    }
    indexType out_dims = bias.size() / chunk_size;

    std::string data_type = "float";
    if (ends_with(args.oFile, ".u4bin") || ends_with(args.oFile, ".u8bin")){
        data_type = "uint";
    } else if(ends_with(args.oFile, ".i4bin") || ends_with(args.oFile, ".i8bin")){
        data_type = "int";
    }

    indexType n, dims;
    std::ifstream reader(args.iFile, std::ios::in | std::ios::binary);
    assert(reader.is_open());
    reader.read((char *)&n, sizeof(indexType));
    reader.read((char *)&dims, sizeof(indexType));

    std::ofstream writer(args.oFile, std::ios::out | std::ios::binary);
    writer.write((char *)&n, sizeof(indexType));
    writer.write((char *)&out_dims, sizeof(indexType));

    uint32_t BLOCK_SIZE = 100000;

    auto from_points = PointRange<Point<float>>(BLOCK_SIZE, PointRange<Point<float>>::Parameters(dims, DistanceMetric::MIPS));
    auto form_quant_points = PointRange<Point<uint8_t>>(BLOCK_SIZE, PointRange<Point<uint8_t>>::Parameters(dims, DistanceMetric::MIPS));
    auto to_points = PointRange<Point<float>>(BLOCK_SIZE, bias.size());
    auto compress_points = PointRange<Point<uint8_t>>(BLOCK_SIZE, out_dims);

    Timer t_infer("PCA compress");
    for (auto i = 0; i < n; i += BLOCK_SIZE)
    {
        auto batch_size = (i + BLOCK_SIZE) > n ? n - i : BLOCK_SIZE;
        from_points.resize(batch_size);
        to_points.resize(batch_size);
        compress_points.resize(batch_size);
        loadFloatPoint(args.iFile, reader, from_points, form_quant_points);
        matmul(from_points, to_points, mat, bias);

        if (data_type == "float")
        {
            to_points.save(writer);
        } else {
            compress(to_points, compress_points, data_type, chunk_size);
            compress_points.save(writer);
        }

        printProgressBar((i + batch_size) / (double)n);
    }
    std::cout << "[Timer 1]:compress pca:" << t_infer.get_next() << std::endl;
    reader.close();
    writer.close();
}

int main(int argc, char *argv[])
{
    Args args;
    CLI::App app{"PCACompress"};

    app.add_option("--ifile", args.iFile, "File path for input vectors");
    app.add_option("--ofile", args.oFile, "File path for compress vectors");
    app.add_option("--matFile", args.matFile, "File path for loat mat");
    app.add_option("--index_type", args.index_type, "index type");

    CLI11_PARSE(app, argc, argv);



    PointRange<Point<float>> mat;
    std::vector<float> bias;
    std::tie(mat, bias) = loadMatrix(args.matFile);

    if (args.index_type == "uint64")
    {
        compress<uint64_t>(args, mat, bias);
    }
    else if (args.index_type == "uint32")
    {
        compress<uint32_t>(args, mat, bias);
    }

    return 0;
}