// 格式转换
#include <fstream>
#include <cassert>
#include <omp.h>
#include "CLI11.hpp"
#include "log.h"
#include "utils/Point/point.h"
#include "utils/Point/point_range.h"
using namespace ant;

struct Args
{
    // 输入输出文件
    std::string iFile;
    std::string oFile;
};

void vec2bin(Args args)
{
    std::streamoff start = 0;
    std::streamoff end = 0;
    // std::ios::ate 初始读写位置定位到文件末尾
    std::ifstream reader(args.iFile, std::ios::in | std::ios::binary | std::ios::ate);
    assert(reader.is_open());
    auto length = static_cast<std::streamoff>(reader.tellg());
    start = (std::min)(start, length);
    if (end == 0)
        end = length;
    else
        end = (std::min)(end, length);
    reader.seekg(start, std::ios::beg);

    uint32_t dims;
    reader.read((char*)&dims, sizeof(uint32_t));
    reader.seekg(0);
    uint32_t n = (end - start) / (4 * dims + 4);

    std::ofstream writer(args.oFile, std::ios::out | std::ios::binary);
    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&dims, sizeof(uint32_t));

    uint32_t BLOCK_SIZE = 100000;
    auto points = PointRange<Point<float>>(BLOCK_SIZE, dims);
    for(auto i = 0; i < n; i+=BLOCK_SIZE){
        auto batch_size = (i + BLOCK_SIZE) > n ? n - i : BLOCK_SIZE;
        points.resize(batch_size);
        for(auto j = 0; j < batch_size; ++j){
            auto p = points[j];
            reader.read((char*)&dims, sizeof(uint32_t));
            reader.read((char*)p.data(), sizeof(float) * dims);
        }
        points.save(writer);
        printProgressBar((i + batch_size) / (double)n);
    }
    reader.close();
    writer.close();

}

void fbin2u8bin(Args args){
    std::ifstream reader(args.iFile, std::ios::in | std::ios::binary);
    assert(reader.is_open());
    uint32_t n, dims;
    reader.read((char*)&n, sizeof(uint32_t));
    reader.read((char*)&dims, sizeof(uint32_t));

    std::ofstream writer(args.oFile, std::ios::out | std::ios::binary);
    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&dims, sizeof(uint32_t));

    uint32_t BLOCK_SIZE = 100000;
    auto from_points = PointRange<Point<float>>(BLOCK_SIZE, dims);
    auto to_points = PointRange<Point<uint8_t>>(BLOCK_SIZE, dims);

    for(auto i = 0; i < n; i+=BLOCK_SIZE){
        auto batch_size = (i + BLOCK_SIZE) > n ? n - i : BLOCK_SIZE;
        from_points.resize(batch_size);
        to_points.resize(batch_size);
        from_points.load(reader);

        #pragma omp parallel for
        for(auto j = 0; j < batch_size; ++j){
            auto fp = from_points[j];
            auto tp = to_points[j];
            std::transform(fp.data(), fp.data() + dims, tp.data(), [](float v){
                assert(v >= 0 && v < 256);
                return static_cast<unsigned char>(v);
            });
        }

        to_points.save(writer);
        printProgressBar((i + batch_size) / (double)n);
    }

}

void u8bin2u4bin(Args args){
    std::ifstream reader(args.iFile, std::ios::in | std::ios::binary);
    assert(reader.is_open());
    uint32_t n, dims, half_dims;
    reader.read((char*)&n, sizeof(uint32_t));
    reader.read((char*)&dims, sizeof(uint32_t));
    half_dims = (dims >> 1);

    std::ofstream writer(args.oFile, std::ios::out | std::ios::binary);
    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&half_dims, sizeof(uint32_t));

    uint32_t BLOCK_SIZE = 100000;
    auto from_points = PointRange<Point<uint8_t>>(BLOCK_SIZE, dims);
    auto to_points = PointRange<Point<uint8_t>>(BLOCK_SIZE, half_dims);

    for(auto i = 0; i < n; i+=BLOCK_SIZE){
        auto batch_size = (i + BLOCK_SIZE) > n ? n - i : BLOCK_SIZE;
        from_points.resize(batch_size);
        to_points.resize(batch_size);
        from_points.load(reader);

        #pragma omp parallel for
        for(auto j = 0; j < batch_size; ++j){
            auto* fp = from_points[j].data();
            auto* tp = to_points[j].data();
            for(auto jj = 0; jj < half_dims; ++jj){
                uint8_t v = fp[jj * 2 + 1] & 0xF0;
                v += (fp[jj * 2] >> 4);
                tp[jj] = v;
            }
        }

        to_points.save(writer);
        printProgressBar((i + batch_size) / (double)n);
    }

}

int main(int argc, char *argv[])
{
    Args args;
    CLI::App app{"FmtConv"};

    auto vec2bin_cmd = app.add_subcommand("vec2bin", "vec2bin");
    vec2bin_cmd->add_option("--ifile", args.iFile, "File path for input");
    vec2bin_cmd->add_option("--ofile", args.oFile, "File path for output");

    auto fbin2u8bin_cmd = app.add_subcommand("fbin2u8bin", "fbin2u8bin");
    fbin2u8bin_cmd->add_option("--ifile", args.iFile, "File path for input");
    fbin2u8bin_cmd->add_option("--ofile", args.oFile, "File path for output");

    auto u8bin2u4bin_cmd = app.add_subcommand("u8bin2u4bin", "u8bin2u4bin");
    u8bin2u4bin_cmd->add_option("--ifile", args.iFile, "File path for input");
    u8bin2u4bin_cmd->add_option("--ofile", args.oFile, "File path for output");

    CLI11_PARSE(app, argc, argv);

    if (*vec2bin_cmd)
    {
        vec2bin(args);
    } else if(*fbin2u8bin_cmd){
        fbin2u8bin(args);
    } else if(*u8bin2u4bin_cmd){
        u8bin2u4bin(args);
    }
    return 0;
}