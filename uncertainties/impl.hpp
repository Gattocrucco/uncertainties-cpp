// impl.hpp
//
// Copyright (c) Giacomo Petrillo 2019
//
// This file is part of uncertainties-cpp.
//
// uncertainties-cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// uncertainties-cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with uncertainties-cpp.  If not, see <http://www.gnu.org/licenses/>.

#ifndef UNCERTAINTIES_IMPL_HPP_8AC76B25
#define UNCERTAINTIES_IMPL_HPP_8AC76B25

/*! \file
\brief Import this header in one (and only one) source file of your choice.

This header contains non-template code to be compiled once. 
*/

#include <string>
#include <cstdlib>
#include <atomic>
#include <random>
#include <vector>
#include <utility>
#include <stdexcept>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        std::atomic<Id> last_id {0}; // must be >= 0

        void insert_dot(std::string *s, int n, int e) {
            e += s->size() - n;
            n = s->size();
            if (e >= n - 1) {
                // no dot at end of mantissa
            } else if (e >= 0) {
                s->insert(1 + e, 1, '.');
            } else if (e <= -1) {
                s->insert(0, -e, '0');
                s->insert(1, 1, '.');
            }
        }
        
        std::string format_exp(const int e) {
            return (e > 0 ? "+" : "-") + std::to_string(std::abs(e));
        }

        Lazy::Token Lazy::push(const bool enable) {
            thread_local static std::random_device r;
            thread_local static std::minstd_rand source(r());
            Token t;
            t.token = std::uniform_int_distribution<int>(0)(source);
            this->stack.push_back({t, enable});
            return t;
        }
        
        void Lazy::pop(Token t) {
            if (this->stack.size() == 1) {
                throw std::runtime_error("uncertainties::Lazy::pop: can not pop first entry");
            }
            if (this->stack.back().first.token != t.token) {
                throw std::invalid_argument("uncertainties::Lazy::pop: wrong token");
            }
            this->stack.pop_back();
        }
        
        bool Lazy::read() const noexcept {
            return this->stack.back().second;
        }
        
        thread_local Lazy lazy;
    }
}

#endif /* end of include guard: UNCERTAINTIES_IMPL_HPP_8AC76B25 */
