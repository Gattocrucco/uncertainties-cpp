#include <chrono>
#include <iostream>

#include <uncertainties/ureal.hpp>

using uncertainties::udouble;

int main() {
    const auto start = std::chrono::steady_clock::now();
    const int n = 1000000;
    udouble x;
    for (int i = 0; i < n; ++i) {
        x += udouble {1, 1};
    }
    const auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << time.count() << "\n";
}
