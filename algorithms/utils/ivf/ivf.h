#pragma once
#include <vector>

namespace ant
{
    struct IVF
    {
        template <typename indexType>
        static void make_ivf(const std::vector<indexType> &id2cls, std::vector<size_t> &nextPos, std::vector<indexType> &ivf)
        {
            if (nextPos.size() > 0 || ivf.size() > 0)
            {
                std::cout << "make_ivf:ERROR nextPos or ivf list must empty!" << std::endl;
                abort();
            }

            // 申请内存
            auto max_it = std::max_element(id2cls.begin(), id2cls.end());
            auto cls_num = (*max_it) + 1;
            nextPos.resize(cls_num);
            memset(nextPos.data(), 0, cls_num * sizeof(size_t));
            ivf.resize(id2cls.size());

            // 统计每个分类中元素的数量
            std::vector<size_t> &cls_size = nextPos;
            for (auto cid : id2cls)
                cls_size.data()[cid]++;

            // 计算每个聚类的下一个偏移
            for (size_t cid = 1; cid < cls_num; ++cid)
            {
                nextPos[cid] = nextPos[cid] + nextPos[cid - 1];
            }
            assert(nextPos[cls_num - 1] == id2cls.size());

            // 开始填写归属每个节点的向量id，记录当前填写到第几个
            std::vector<indexType> node_offset(cls_num, 0);
            for (indexType i = 0; i < id2cls.size(); ++i)
            {
                auto cid = id2cls[i];
                size_t pre_offset = (cid == 0 ? 0 : nextPos[cid - 1]);
                ivf[pre_offset + node_offset[cid]] = i;
                node_offset[cid] += 1;
            }
        }

        template <typename indexType>
        static void make_ivf(const std::vector<size_t> &fromNextPos, const std::vector<indexType> &fromIvf, std::vector<size_t> &toNextPos, std::vector<indexType> &toIvf)
        {
            if (toNextPos.size() > 0 || toIvf.size() > 0)
            {
                std::cout << "make_ivf:ERROR nextPos or ivf list must empty!" << std::endl;
                abort();
            }

            // 申请内存
            auto max_it = std::max_element(fromIvf.begin(), fromIvf.end());
            auto cls_num = (*max_it) + 1;
            toNextPos.resize(cls_num);
            toIvf.resize(fromIvf.size());

            // 统计每个分类中元素的数量
            std::vector<size_t> &cls_size = toNextPos;
            for (auto cid : fromIvf)
                cls_size.data()[cid]++;

            // 计算每个聚类的下一个偏移
            for (size_t cid = 1; cid < cls_num; ++cid)
            {
                toNextPos[cid] = toNextPos[cid] + toNextPos[cid - 1];
            }
            assert(toNextPos[cls_num - 1] == fromIvf.size());

            // 开始填写归属每个节点的向量id，记录当前填写到第几个
            std::vector<indexType> node_offset(cls_num, 0);
            for (indexType i = 0; i < fromNextPos.size(); ++i)
            {
                size_t fid = (i == 0) ? 0 : fromNextPos[i - 1];
                size_t eid = fromNextPos[i];
                for (size_t j = fid; j < eid; ++j)
                {
                    indexType cid = fromIvf[j];
                    size_t pre_offset = (cid == 0 ? 0 : toNextPos[cid - 1]);
                    toIvf[pre_offset + node_offset[cid]] = i;
                    node_offset[cid] += 1;
                }
            }
        }
    };

    size_t get_ivf_size(std::vector<std::vector<uint32_t>> &index, size_t centroid_id, uint32_t bin_size)
    {
        size_t ivf_size = 0;
        for (size_t i = centroid_id * bin_size; i < ((centroid_id + 1) * bin_size); ++i)
        {
            ivf_size += index[i].size();
        }
        return ivf_size;
    }

    template<typename indexType>
    void get_ivf(std::vector<indexType>& vec_id_list, std::vector<std::vector<uint32_t>> &index, size_t centroid_id, uint32_t bin_size){
        vec_id_list.resize(0);
        for (auto bin_id = 0; bin_id < bin_size; ++bin_id)
        {
            auto &ivf_data = index[centroid_id * (size_t)bin_size + bin_id];
            for (uint ii = 0; ii < ivf_data.size(); ++ii)
            {
                indexType index = ivf_data[ii] * bin_size + bin_id;
                vec_id_list.push_back(index);
            }
        }
    }

    template <typename indexType>
    void sortIVF(std::vector<std::vector<indexType>> &data)
    {
        #pragma omp parallel for
        for(size_t i = 0; i < data.size(); ++i){
            indexType* ptr = data[i].data();
            std::sort(ptr, ptr + data[i].size());
        }
    }

    // 保存到文件（二进制格式）
    template <typename indexType>
    bool saveVectorBinary(const std::vector<std::vector<indexType>> &data, const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        // 写入外层vector的大小
        size_t outerSize = data.size();
        file.write(reinterpret_cast<const char *>(&outerSize), sizeof(size_t));

        // 写入每个内层vector
        for (const auto &innerVec : data)
        {
            // 写入内层vector的大小
            size_t innerSize = innerVec.size();
            file.write(reinterpret_cast<const char *>(&innerSize), sizeof(size_t));

            // 写入内层vector的数据
            if (!innerVec.empty())
            {
                file.write(reinterpret_cast<const char *>(innerVec.data()),
                           innerSize * sizeof(indexType));
            }
        }

        file.close();
        return true;
    }

    template <typename indexType>
    bool saveIVFBinary(const std::vector<std::vector<indexType>> &ivf_index, size_t vec_num, const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        // 写入外层vector的大小
        size_t outerSize = ivf_index.size();
        file.write(reinterpret_cast<const char *>(&outerSize), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(&vec_num), sizeof(size_t));
        {
            std::vector<size_t> nextPos = std::vector<size_t>(outerSize);
            size_t pre_pos = 0;
            for(size_t i = 0; i < outerSize; ++i){
                nextPos[i] = ivf_index[i].size() + pre_pos;
                pre_pos = nextPos[i];
            }
            file.write((char*)nextPos.data(), outerSize * sizeof(size_t));
        }

        // 写入每个内层vector
        for (const auto &innerVec : ivf_index)
        {
            // 写入内层vector的大小
            size_t innerSize = innerVec.size();
            // 写入内层vector的数据
            if (!innerVec.empty())
            {
                file.write(reinterpret_cast<const char *>(innerVec.data()),
                           innerSize * sizeof(indexType));
            }
        }

        file.close();
        return true;
    }

    template <typename indexType>
    bool loadIVFBinary(std::vector<std::vector<indexType>> &ivf_index, const std::string &filename)
    {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        // 读出外层vector的大小
        size_t outerSize, vec_num;
        file.read((char*)(&outerSize), sizeof(size_t));
        file.read((char*)(&vec_num), sizeof(size_t));
        ivf_index.resize(outerSize);
        for(size_t i = 0; i < outerSize; ++i){
            ivf_index[i].resize(0);
        }


        {
            std::vector<size_t> nextPos = std::vector<size_t>(outerSize);
            file.read((char*)nextPos.data(), outerSize * sizeof(size_t));

            size_t pre_pos = 0;
            for(size_t i = 0; i < outerSize; ++i){
                ivf_index[i].resize(nextPos[i] - pre_pos);
                pre_pos = nextPos[i];
            }
        }

        // 读出每个内层vector
        for (const auto &innerVec : ivf_index)
        {
            // 读出内层vector的大小
            size_t innerSize = innerVec.size();
            // 读出内层vector的数据
            if (!innerVec.empty())
            {
                file.read((char*)(innerVec.data()),
                           innerSize * sizeof(indexType));
            }
        }

        file.close();
        return true;
    }

    // 从文件读取（二进制格式）
    template <typename indexType>
    bool loadVectorBinary(std::vector<std::vector<indexType>> &data, const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        // 清空原有数据
        data.clear();

        // 读取外层vector的大小
        size_t outerSize;
        file.read(reinterpret_cast<char *>(&outerSize), sizeof(size_t));

        // 读取每个内层vector
        for (size_t i = 0; i < outerSize; ++i)
        {
            // 读取内层vector的大小
            size_t innerSize;
            file.read(reinterpret_cast<char *>(&innerSize), sizeof(size_t));

            // 创建内层vector并读取数据
            std::vector<indexType> innerVec(innerSize);
            if (innerSize > 0)
            {
                file.read(reinterpret_cast<char *>(innerVec.data()),
                          innerSize * sizeof(indexType));
            }

            data.push_back(std::move(innerVec));
        }

        file.close();
        return true;
    }

    // 保存质心对应的向量id
    template <typename indexType>
    void saveIds(std::string idsFile, std::vector<indexType> &ids)
    {
        std::ofstream writer;
        writer.open(idsFile.c_str(), std::ios::out | std::ios::binary);
        if (!writer.is_open())
        {
            std::cout << "Data file " << idsFile << " not found" << std::endl;
            std::abort();
        }

        size_t n, d;
        n = ids.size();
        d = 1;
        writer.write((char *)&n, sizeof(size_t));
        writer.write((char *)&d, sizeof(size_t));
        writer.write((char *)ids.data(), sizeof(indexType) * n);
        writer.close();
    }

    template <typename indexType>
    void loadIds(std::string idsFile, std::vector<indexType> &ids)
    {
        std::ifstream reader;
        reader.open(idsFile.c_str(), std::ios::in | std::ios::binary);
        if (!reader.is_open())
        {
            std::cout << "Data file " << idsFile << " not found" << std::endl;
            std::abort();
        }

        size_t n, d;
        reader.read((char *)&n, sizeof(size_t));
        reader.read((char *)&d, sizeof(size_t));
        ids.resize(n);
        reader.read((char *)ids.data(), sizeof(indexType) * n);
        reader.close();
    }
}