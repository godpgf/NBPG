#pragma once
#include <random>
#include "disk_ivf.h"
#include "utils/ivf/ivf.h"

namespace ant
{

    struct InvIVF
    {
        // 读取某些质心的反转ivf
        template <typename indexType>
        static void read_inv_index_ivf(std::string ivf_path, std::vector<indexType> &ids, std::function<bool(indexType)> filter)
        {
            if (ivf_path == "")
                return;
            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t keyNum;
            size_t valueNum;
            ivf_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_reader.read((char *)&valueNum, sizeof(size_t));
            std::vector<size_t> nextPos = std::vector<size_t>(keyNum);
            ivf_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));

            size_t head_offset = (2 + keyNum) * sizeof(size_t);
            // 读取所有数据
            ivf_reader.seekg(head_offset);
            // #pragma omp parallel for
            for (auto vid = 0; vid < nextPos.size(); ++vid)
            {
                size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                std::vector<indexType> cache_data(nextPos[vid] - pre_pos);
                // memcpy((char *)cache_data.data(), fileptr + head_offset + pre_pos * sizeof(indexType), cache_data.size() * sizeof(indexType));
                ivf_reader.read((char *)cache_data.data(), cache_data.size() * sizeof(indexType));
                // 筛选出内容在范围内的数据
                for (auto cid : cache_data)
                {
                    if (filter(cid))
                    {
                        ids.push_back(vid);
                    }
                }
            }
            ivf_reader.close();
        }

        // 反转ivf，保存数据格式是：[质心数量][节点数量][[质心1的节点数],[质心2的节点数+之前的节点数],...][节点id1，节点id2,...]
        template <typename indexType>
        static void inv_index_ivf(std::string ivf_path, std::string inv_ivf_path, size_t epoch_num = 1)
        {
            // v2c转c2v
            if (ivf_path == "" || inv_ivf_path == "")
                return;
            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t keyNum;
            size_t valueNum;
            ivf_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_reader.read((char *)&valueNum, sizeof(size_t));
            std::vector<size_t> nextPos = std::vector<size_t>(keyNum);
            ivf_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));

            size_t head_offset = (2 + keyNum) * sizeof(size_t);
            size_t max_batch_size = valueNum / epoch_num;

            std::ofstream inv_writer;
            std::vector<size_t> inv_nextPos = std::vector<size_t>(valueNum);
            // 写入尺寸
            inv_writer.open(inv_ivf_path.c_str(), std::ios::binary | std::ios::out);
            inv_writer.write((char *)&valueNum, sizeof(size_t));
            inv_writer.write((char *)&keyNum, sizeof(size_t));
            inv_writer.write((char *)inv_nextPos.data(), valueNum * sizeof(size_t));

            // 开始写入
            for (size_t si = 0; si < valueNum; si += max_batch_size)
            {
                size_t cur_batch_size = (si + max_batch_size > valueNum) ? valueNum - si : max_batch_size;
                std::vector<std::vector<indexType>> new_c2v_(cur_batch_size);

                // 读取所有数据
                ivf_reader.seekg(head_offset);
                // #pragma omp parallel for
                for (auto vid = 0; vid < nextPos.size(); ++vid)
                {
                    size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                    std::vector<indexType> cache_data(nextPos[vid] - pre_pos);
                    // memcpy((char *)cache_data.data(), fileptr + head_offset + pre_pos * sizeof(indexType), cache_data.size() * sizeof(indexType));
                    ivf_reader.read((char *)cache_data.data(), cache_data.size() * sizeof(indexType));
                    // 筛选出内容在范围内的数据
                    for (auto cid : cache_data)
                    {
                        if (cid >= si && cid < si + cur_batch_size)
                        {
                            new_c2v_[cid - si].push_back(vid);
                        }
                    }
                }

                // 写入ivf
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    auto &cur_out = new_c2v_[i];
                    inv_writer.write((char *)cur_out.data(), cur_out.size() * sizeof(indexType));
                    auto cur_index = si + i;
                    inv_nextPos[cur_index] = cur_out.size() + ((cur_index == 0) ? 0 : inv_nextPos[cur_index - 1]);
                }
                std::cout << "Pass " << (int)(100 * (si + cur_batch_size) / (double)valueNum) << "% complete" << std::endl;
            }
            inv_writer.close();
            ivf_reader.close();

            // 重新写入
            inv_writer.open(inv_ivf_path.c_str(), std::ios::binary | std::ios::out | std::ios::in);
            inv_writer.seekp(sizeof(size_t) * 2, std::ios::beg);
            inv_writer.write((char *)inv_nextPos.data(), inv_nextPos.size() * sizeof(size_t));
            inv_writer.close();
        }

        /*template <typename dataType, typename indexType>
        static void inv_index_ivf_and_pos(std::string ivf_path, std::string inv_ivf_path, std::string pos_path, size_t max_ivf_size, size_t epoch_num = 20)
        {
            // v2c转c2v
            if (ivf_path == "" || inv_ivf_path == "")
                return;

            indexType n, dims;
            std::ifstream pos_reader;
            pos_reader.open(pos_path.c_str(), std::ios::binary | std::ios::in);
            pos_reader.read((char *)&n, sizeof(indexType));
            pos_reader.read((char *)&dims, sizeof(indexType));
            size_t pos_offset = 2 * sizeof(indexType);

            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t keyNum;
            size_t valueNum;
            ivf_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_reader.read((char *)&valueNum, sizeof(size_t));
            std::vector<size_t> nextPos = std::vector<size_t>(keyNum);
            ivf_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));

            size_t head_offset = (2 + keyNum) * sizeof(size_t);
            size_t max_batch_size = valueNum / epoch_num;

            std::ofstream inv_writer;
            std::vector<size_t> inv_nextPos = std::vector<size_t>(valueNum);
            // 写入尺寸
            inv_writer.open(inv_ivf_path.c_str(), std::ios::binary | std::ios::out);
            inv_writer.write((char *)&valueNum, sizeof(size_t));
            inv_writer.write((char *)&keyNum, sizeof(size_t));
            size_t d = dims;
            inv_writer.write((char *)&d, sizeof(size_t));
            inv_writer.write((char *)inv_nextPos.data(), valueNum * sizeof(size_t));

            // 开始写入
            for (size_t si = 0; si < valueNum; si += max_batch_size)
            {
                size_t cur_batch_size = (si + max_batch_size > valueNum) ? valueNum - si : max_batch_size;
                std::vector<std::vector<indexType>> new_c2v_(cur_batch_size);
                std::vector<std::vector<dataType>> new_c2v_pos_(cur_batch_size);
                std::vector<dataType> tmp_pos(dims);

                for(size_t i = 0; i < cur_batch_size; ++i){
                    new_c2v_pos_[i].resize(2 * max_ivf_size * dims);
                }

                // 读取所有数据
                ivf_reader.seekg(head_offset);
                pos_reader.seekg(pos_offset);
                // #pragma omp parallel for
                for (auto vid = 0; vid < nextPos.size(); ++vid)
                {
                    size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                    std::vector<indexType> cache_data(nextPos[vid] - pre_pos);
                    // memcpy((char *)cache_data.data(), fileptr + head_offset + pre_pos * sizeof(indexType), cache_data.size() * sizeof(indexType));
                    ivf_reader.read((char *)cache_data.data(), cache_data.size() * sizeof(indexType));
                    pos_reader.read((char *)tmp_pos.data(), dims * sizeof(dataType));
                    // 筛选出内容在范围内的数据
                    for (auto cid : cache_data)
                    {
                        if (cid >= si && cid < si + cur_batch_size)
                        {
                            auto cur_offset = new_c2v_[cid - si].size();
                            new_c2v_[cid - si].push_back(vid);
                            if((cur_offset + 1) * dims > new_c2v_pos_[cid - si].size()){
                                new_c2v_pos_[cid - si].resize((cur_offset + 1) * dims);
                            }
                            memcpy((char*)new_c2v_pos_[cid - si].data() + cur_offset * dims, tmp_pos.data(), dims * sizeof(dataType));
                        }
                    }
                }

                // 写入ivf
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    auto &cur_out = new_c2v_[i];
                    auto &cur_pos_out = new_c2v_pos_[i];
                    inv_writer.write((char *)cur_out.data(), cur_out.size() * sizeof(indexType));
                    inv_writer.write((char*)cur_pos_out.data(), cur_out.size() * dims * sizeof(dataType));
                    auto cur_index = si + i;
                    inv_nextPos[cur_index] = cur_out.size() + ((cur_index == 0) ? 0 : inv_nextPos[cur_index - 1]);
                }
                std::cout << "Pass " << (int)(100 * (si + cur_batch_size) / (double)valueNum) << "% complete" << std::endl;
            }
            inv_writer.close();
            ivf_reader.close();

            // 重新写入
            inv_writer.open(inv_ivf_path.c_str(), std::ios::binary | std::ios::out | std::ios::in);
            inv_writer.seekp(sizeof(size_t) * 3, std::ios::beg);
            inv_writer.write((char *)inv_nextPos.data(), inv_nextPos.size() * sizeof(size_t));
            inv_writer.close();
        }*/

        template <typename PR, typename indexType>
        static void copy_point2ivf(std::string ivf_path, PR& Points, std::string ivf_pos_path, size_t ivf_R)
        {
            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t keyNum;
            size_t valueNum;
            ivf_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_reader.read((char *)&valueNum, sizeof(size_t));
            std::vector<size_t> nextPos = std::vector<size_t>(keyNum);
            ivf_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));

            std::ofstream ivf_writer;
            ivf_writer.open(ivf_pos_path.c_str(), std::ios::binary | std::ios::out);
            ivf_writer.write((char *)&keyNum, sizeof(size_t));
            ivf_writer.write((char *)&valueNum, sizeof(size_t));

            indexType n, dims;
            n = Points.size();
            dims = Points.dimension();
            size_t num_bytes = Points.params.num_bytes();

            size_t d = dims;
            ivf_writer.write((char *)&d, sizeof(size_t));
            ivf_writer.write((char *)nextPos.data(), keyNum * sizeof(size_t));

            // 统计
            size_t max_ivf_size = 0;
            size_t pre_offset = 0;
            for (auto vid = 0; vid < nextPos.size(); ++vid)
            {
                size_t ivf_size = nextPos[vid] - pre_offset;
                if (ivf_size > max_ivf_size)
                    max_ivf_size = ivf_size;
                pre_offset = nextPos[vid];
            }
            std::cout << "max ivf size:" << max_ivf_size << std::endl;

            std::vector<char> cache = std::vector<char>(max_ivf_size * (num_bytes + sizeof(indexType)));

            // 先写入ivf数据
            char *cache_ptr = cache.data();
            for(auto vid = 0; vid < nextPos.size(); ++vid){
                size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                size_t cur_pos_num = nextPos[vid] - pre_pos;
                indexType *ivf_cache_ptr = (indexType *)cache_ptr;
                char *pos_cache_ptr = (char *)(cache_ptr + cur_pos_num * sizeof(indexType));
                ivf_reader.read((char *)ivf_cache_ptr, cur_pos_num * sizeof(indexType));
                for(size_t i = 0; i < cur_pos_num; ++i){
                    char *cur_pos_cache_ptr = pos_cache_ptr + i * num_bytes;
                    memcpy(cur_pos_cache_ptr, Points[vid].data(), num_bytes);
                }
                ivf_writer.write(cache.data(), cur_pos_num * (num_bytes + sizeof(indexType)));
            }
  
            ivf_writer.close();
        }

        template <typename dataType, typename indexType>
        static void copy_point2ivf(std::string ivf_path, std::string pos_path, std::string ivf_pos_path, size_t ivf_R, size_t max_batch_size = 10000, size_t epoch_num = 4)
        {
            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t keyNum;
            size_t valueNum;
            ivf_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_reader.read((char *)&valueNum, sizeof(size_t));
            std::vector<size_t> nextPos = std::vector<size_t>(keyNum);
            ivf_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));

            std::ofstream ivf_writer;
            ivf_writer.open(ivf_pos_path.c_str(), std::ios::binary | std::ios::out);
            ivf_writer.write((char *)&keyNum, sizeof(size_t));
            ivf_writer.write((char *)&valueNum, sizeof(size_t));

            indexType n, dims;
            std::ifstream pos_reader;
            pos_reader.open(pos_path.c_str(), std::ios::binary | std::ios::in);
            pos_reader.read((char *)&n, sizeof(indexType));
            pos_reader.read((char *)&dims, sizeof(indexType));
            size_t num_bytes = dims * sizeof(dataType);

            size_t d = dims;
            ivf_writer.write((char *)&d, sizeof(size_t));
            ivf_writer.write((char *)nextPos.data(), keyNum * sizeof(size_t));

            // 统计
            size_t max_ivf_size = 0;
            size_t pre_offset = 0;
            for (auto vid = 0; vid < nextPos.size(); ++vid)
            {
                size_t ivf_size = nextPos[vid] - pre_offset;
                if (ivf_size > max_ivf_size)
                    max_ivf_size = ivf_size;
                pre_offset = nextPos[vid];
            }
            std::cout << "max ivf size:" << max_ivf_size << std::endl;

            std::vector<char> cache = std::vector<char>(max_ivf_size * max_batch_size * (num_bytes + sizeof(indexType)));

            // 先写入ivf数据
            for (auto sid = 0; sid < nextPos.size(); sid += max_batch_size)
            {
                size_t cur_batch_size = (sid + max_batch_size > nextPos.size()) ? nextPos.size() - sid : max_batch_size;
                char *cache_ptr = cache.data();
                for (size_t i = 0; i < cur_batch_size; ++i)
                {
                    auto vid = sid + i;
                    size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                    size_t cur_pos_num = nextPos[vid] - pre_pos;
                    indexType *ivf_cache_ptr = (indexType *)cache_ptr;
                    dataType *pos_cache_ptr = (dataType *)(cache_ptr + cur_pos_num * sizeof(indexType));
                    ivf_reader.read((char *)ivf_cache_ptr, cur_pos_num * sizeof(indexType));
                    cache_ptr = ((char *)pos_cache_ptr) + cur_pos_num * num_bytes;
                }
                ivf_writer.write(cache.data(), cache_ptr - cache.data());
            }
            ivf_writer.close();
            ivf_writer.open(ivf_pos_path.c_str(), std::ios::binary | std::ios::out | std::ios::in);

            // 再写入ivf包含的向量坐标
            max_batch_size = n / epoch_num;
            std::vector<size_t> pos_offset = std::vector<size_t>(max_batch_size * ivf_R);

            for (size_t si = 0; si < n; si += max_batch_size)
            {
                // 先读取每个向量写入到硬盘的偏移位置
                size_t cur_batch_size = (si + max_batch_size > n) ? n - si : max_batch_size;
                std::fill_n(pos_offset.data(), cur_batch_size * ivf_R, -1);
                ivf_reader.seekg((keyNum + 2) * sizeof(size_t));
                indexType *ivf_cache_ptr = (indexType *)cache.data();
                for (auto vid = 0; vid < nextPos.size(); ++vid)
                {
                    size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                    size_t cur_pos_num = nextPos[vid] - pre_pos;
                    ivf_reader.read((char *)ivf_cache_ptr, cur_pos_num * sizeof(indexType));
                    for (size_t j = 0; j < cur_pos_num; ++j)
                    {
                        auto index = ivf_cache_ptr[j];
                        if (index >= si && index < si + cur_batch_size)
                        {
                            size_t offset = (keyNum + 3) * sizeof(size_t) + nextPos[vid] * sizeof(indexType) + (pre_pos + j) * num_bytes;
                            size_t *ptr = pos_offset.data() + (index - si) * ivf_R;
                            for (auto jj = 0; jj < ivf_R; ++jj)
                            {
                                if (ptr[jj] == -1)
                                {
                                    ptr[jj] = offset;
                                    break;
                                }
                            }
                        }
                    }
                }

                // 开始写入
                dataType *pos_cache_ptr = (dataType *)cache.data();
                pos_reader.seekg(2 * sizeof(indexType) + si * num_bytes);
                pos_reader.read((char *)pos_cache_ptr, num_bytes * cur_batch_size);

                std::vector<std::pair<size_t, char*>> write_cache;
                for (auto i = 0; i < cur_batch_size; ++i)
                {
                    size_t *ptr = pos_offset.data() + i * ivf_R;
                    char* pos = (char *)pos_cache_ptr + i * num_bytes;
                    for (auto jj = 0; jj < ivf_R; ++jj)
                    {
                        if (ptr[jj] != -1)
                        {
                            write_cache.push_back(std::pair(ptr[jj], pos));
                            // ivf_writer.seekp(ptr[jj]);
                            // ivf_writer.write((char *)pos_cache_ptr, num_bytes);
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                std::sort(write_cache.begin(), write_cache.end(), [](auto a, auto b){return a.first < b.first;});
                for(auto w : write_cache){
                    ivf_writer.seekp(w.first);
                    ivf_writer.write(w.second, num_bytes);
                }

                std::cout << "Pass " << (si + cur_batch_size) * 100 / n << "% complete"
                                        << std::endl;
            }

            ivf_writer.close();
            ivf_reader.close();
            pos_reader.close();
        }

        /*template <typename dataType, typename indexType>
        static void copy_point2ivf(std::string ivf_path, std::string pos_path, std::string ivf_pos_path, size_t ivf_R, size_t max_batch_size=10000)
        {
            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t keyNum;
            size_t valueNum;
            ivf_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_reader.read((char *)&valueNum, sizeof(size_t));
            std::vector<size_t> nextPos = std::vector<size_t>(keyNum);
            ivf_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));

            std::ofstream ivf_writer;
            ivf_writer.open(ivf_pos_path.c_str(), std::ios::binary | std::ios::out);
            ivf_writer.write((char *)&keyNum, sizeof(size_t));
            ivf_writer.write((char *)&valueNum, sizeof(size_t));

            indexType n, dims;
            std::ifstream pos_reader;
            pos_reader.open(pos_path.c_str(), std::ios::binary | std::ios::in);
            pos_reader.read((char *)&n, sizeof(indexType));
            pos_reader.read((char *)&dims, sizeof(indexType));
            size_t pos_offset = 2 * sizeof(indexType);
            size_t num_bytes = dims * sizeof(dataType);

            size_t d = dims;
            ivf_writer.write((char *)&d, sizeof(size_t));
            ivf_writer.write((char *)nextPos.data(), keyNum * sizeof(size_t));

            // 统计
            size_t max_ivf_size = 0;
            size_t pre_offset = 0;
            for(auto vid = 0; vid < nextPos.size(); ++vid){
                size_t ivf_size = nextPos[vid] - pre_offset;
                if(ivf_size > max_ivf_size)
                    max_ivf_size = ivf_size;
                pre_offset = nextPos[vid];
            }
            std::cout<<"max ivf size:"<<max_ivf_size<<std::endl;

            std::vector<char> cache = std::vector<char>(max_ivf_size * max_batch_size * (num_bytes + sizeof(indexType)));


            int pre_p = 0;
            for (auto sid = 0; sid < nextPos.size(); sid += max_batch_size){
                size_t cur_batch_size = (sid + max_batch_size > nextPos.size()) ? nextPos.size() - sid : max_batch_size;
                char* cache_ptr = cache.data();
                for(size_t i = 0; i < cur_batch_size; ++i){
                    auto vid = sid + i;
                    size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                    size_t cur_pos_num = nextPos[vid] - pre_pos;
                    indexType* ivf_cache_ptr = (indexType*)cache_ptr;
                    dataType* pos_cache_ptr = (dataType*)(cache_ptr + cur_pos_num * sizeof(indexType));
                    ivf_reader.read((char *)ivf_cache_ptr, cur_pos_num * sizeof(indexType));
                    for (size_t j = 0; j < cur_pos_num; ++j)
                    {
                        size_t index = ivf_cache_ptr[j];
                        dataType *ptr = pos_cache_ptr + j * dims;
                        pos_reader.seekg(pos_offset + index * num_bytes);
                        pos_reader.read((char *)ptr, num_bytes);
                    }
                    cache_ptr = ((char*)pos_cache_ptr) + cur_pos_num * num_bytes;

                    {
                        // log
                        int p = (int)((vid + 1) / (float)nextPos.size() * 100);
                        if (p > pre_p)
                        {
                            if (p % 10 == 0)
                            {
                                std::cout << "Pass " << p << "% complete"
                                        << std::endl;
                            }

                            pre_p = p;
                        }
                    }
                }
                ivf_writer.write(cache.data(), cache_ptr - cache.data());
            }

            ivf_writer.close();
            ivf_reader.close();
            pos_reader.close();
        }*/
    };

}