#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>


// 命名空间为 GraphOptimizor
namespace GraphOptimizor {
    class Timer {
    private:
        std::chrono::time_point<std::chrono::system_clock> start, end;

    public:
        Timer() {
            Start();
        }

        void Start() {
            start = std::chrono::system_clock::now();
        }

        double Stop() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000;
        }
    };
}