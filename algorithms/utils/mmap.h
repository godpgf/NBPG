#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ant
{

  // 异步预加载
  void prefetch_async(void* addr, size_t length){
    madvise(addr, length, MADV_WILLNEED);
  }

  // returns a pointer and a length
  std::pair<char *, size_t> mmapStringFromFile(const char *filename)
  {
    // struct stat sb;
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("open failed");
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("fstat failed");
    }

    if (!S_ISREG(sb.st_mode)) {
        close(fd);
        throw std::runtime_error("not a regular file");
    }

    // 关键修复：MAP_SHARED 替代 MAP_PRIVATE
    char *p = static_cast<char *>(mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0));
    if (p == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap failed");
    }

    // 优化提示：顺序访问 1TB 大文件
    #ifdef __linux__
    madvise(p, sb.st_size, MADV_SEQUENTIAL);
    #endif

    close(fd);  // fd 可以立即关闭，不影响已建立的映射
    size_t n = sb.st_size;
    return std::make_pair(p, n);
  }

}