#pragma once
#include <iostream>
#include <iomanip>

// RAII guard 类
class PrecisionGuard
{
public:
    explicit PrecisionGuard(std::ostream &os)
        : os_(os), original_precision_(os.precision()) {}

    ~PrecisionGuard()
    {
        os_.precision(original_precision_); // 自动恢复
    }

    // 禁止拷贝
    PrecisionGuard(const PrecisionGuard &) = delete;
    PrecisionGuard &operator=(const PrecisionGuard &) = delete;

private:
    std::ostream &os_;
    std::streamsize original_precision_;
};

void printProgressBar(double progress, int width = 50)
{
    PrecisionGuard guard(std::cout); // 保存状态
    std::cout << "[";
    int pos = width * progress;
    for (int i = 0; i < width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << " %\r";
    std::cout.flush();
}

void printProgressBar(double progress, std::string attach_str, int width = 50)
{
    PrecisionGuard guard(std::cout); // 保存状态
    std::cout << "[";
    int pos = width * progress;
    for (int i = 0; i < width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << attach_str<< "\t" << progress * 100.0 << " %\r";
    std::cout.flush();
}