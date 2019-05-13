#include <vector>
#include <chrono>
#include <iostream>

#include <uncertainties/ureal.hpp>

using uncertainties::udouble;

int main() {
    std::vector<udouble> v;
    const int n = 10000000;
    v.reserve(n);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        v.emplace_back(1, 1);
    }
    const auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << time.count() << "\n";
}
