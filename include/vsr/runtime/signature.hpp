#pragma once

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vsr::runtime {

/**
 * A Clifford algebra Cl(p,q,r): generators square to +1, -1 or 0.
 *
 * This is the runtime counterpart of the compile-time algebra in
 * vsr::algebra.  The immutable metric, product-sign table, dual signs and
 * blade square signs live in one shared allocation, so copying a Signature or
 * a Multivector never copies the tables.
 */
class Signature {
public:
    Signature() : Signature(std::vector<int>{}, 0) {}

    explicit Signature(std::vector<int> metric, int name_offset = 0) {
        if (metric.size() > 14) {
            throw std::invalid_argument(
                "Clifford algebras above 14 generators are not supported");
        }
        for (int m : metric) {
            if (m < -1 || m > 1) {
                throw std::invalid_argument("metric entries must be -1, 0 or 1");
            }
        }

        auto data = std::make_shared<Data>();
        data->metric = std::move(metric);
        data->n = static_cast<int>(data->metric.size());
        data->size = 1 << data->n;
        data->full = data->size - 1;
        data->name_offset = name_offset;
        build_aux_tables(*data);
        if (data->n <= 10) {
            build_sign_table(*data);
        }
        data_ = std::move(data);
    }

    static Signature of(std::initializer_list<int> metric) {
        return Signature(std::vector<int>(metric), 0);
    }

    static Signature of(std::vector<int> metric) {
        return Signature(std::move(metric), 0);
    }

    /** Cl(p,q): first p generators square to +1, next q to -1. */
    static Signature cl(int p, int q) {
        if (p < 0 || q < 0) {
            throw std::invalid_argument("Cl(p,q) needs p,q >= 0");
        }
        const int n = p + q;
        if (n > 14) {
            throw std::invalid_argument(
                "Clifford algebras above 14 generators are not supported");
        }
        std::vector<int> metric(static_cast<std::size_t>(n), -1);
        for (int i = 0; i < p; ++i) {
            metric[static_cast<std::size_t>(i)] = 1;
        }
        return Signature(std::move(metric), 0);
    }

    static Signature euclidean(int n) {
        if (n < 0 || n > 14) {
            throw std::invalid_argument("euclidean(n) needs 0 <= n <= 14");
        }
        return Signature(std::vector<int>(static_cast<std::size_t>(n), 1), 1);
    }

    /** 3D projective GA: e0^2 = 0, e1^2 = e2^2 = e3^2 = 1. */
    static Signature pga3() { return Signature({0, 1, 1, 1}, 0); }

    /** 3D Euclidean GA (quaternions). */
    static Signature e3() { return Signature({1, 1, 1}, 1); }

    /** Spacetime algebra Cl(1,3); e0 is timelike. */
    static Signature sta() { return Signature({1, -1, -1, -1}, 0); }

    /** 3D conformal model: e1..e4 spacelike, e5 timelike. */
    static Signature cga3() { return Signature({1, 1, 1, 1, -1}, 1); }

    [[nodiscard]] int n() const noexcept { return data_->n; }
    [[nodiscard]] int size() const noexcept { return data_->size; }
    [[nodiscard]] int full_mask() const noexcept { return data_->full; }
    [[nodiscard]] int metric(int i) const {
        return data_->metric.at(static_cast<std::size_t>(i));
    }
    [[nodiscard]] std::vector<int> metric_copy() const { return data_->metric; }

    [[nodiscard]] int grade(int mask) const noexcept {
        return std::popcount(static_cast<unsigned>(mask));
    }

    [[nodiscard]] static int reverse_sign(int grade) noexcept {
        return ((grade * (grade - 1) / 2) & 1) == 0 ? 1 : -1;
    }

    [[nodiscard]] static int conjugate_sign(int grade) noexcept {
        return ((grade * (grade + 1) / 2) & 1) == 0 ? 1 : -1;
    }

    [[nodiscard]] static int involution_sign(int grade) noexcept {
        return (grade & 1) == 0 ? 1 : -1;
    }

    /** Metric factor of a basis blade squared. */
    [[nodiscard]] int blade_square_sign(int mask) const noexcept {
        return data_->square[static_cast<std::size_t>(mask)];
    }

    /** Sign of the Poincare complement used by Multivector::dual(). */
    [[nodiscard]] int dual_sign(int mask) const noexcept {
        return data_->dual[static_cast<std::size_t>(mask)];
    }

    [[nodiscard]] bool is_scalar(int mask) const noexcept { return mask == 0; }

    /** Name of a basis blade, for example "e0e1". */
    [[nodiscard]] std::string blade_name(int mask) const {
        if (mask == 0) {
            return "1";
        }
        std::string out;
        for (int i = 0; i < data_->n; ++i) {
            if ((mask & (1 << i)) != 0) {
                out += 'e';
                out += std::to_string(i + data_->name_offset);
            }
        }
        return out;
    }

    /**
     * Sign and metric factor of the geometric product of two basis blades.
     * Returns 0 when a null generator would be squared.
     */
    [[nodiscard]] int blade_product_sign(int a, int b) const noexcept {
        if (!data_->sign.empty()) {
            return data_->sign[static_cast<std::size_t>(a) *
                                   static_cast<std::size_t>(data_->size) +
                               static_cast<std::size_t>(b)];
        }
        return compute_blade_product_sign(a, b);
    }

    [[nodiscard]] int compute_blade_product_sign(int a, int b) const noexcept {
        int sign = 1;
        for (int j = 0; j < data_->n; ++j) {
            if ((b & (1 << j)) == 0) {
                continue;
            }
            if ((std::popcount(static_cast<unsigned>(a >> (j + 1))) & 1) != 0) {
                sign = -sign;
            }
            if ((a & (1 << j)) != 0) {
                const int m = data_->metric[static_cast<std::size_t>(j)];
                if (m == 0) {
                    return 0;
                }
                sign *= m;
            }
        }
        return sign;
    }

    /** Flat row-major sign table for n <= 10, or nullptr. */
    [[nodiscard]] const std::int8_t* sign_table() const noexcept {
        return data_->sign.empty() ? nullptr : data_->sign.data();
    }

    [[nodiscard]] bool operator==(const Signature& other) const noexcept {
        return data_->n == other.data_->n &&
               data_->name_offset == other.data_->name_offset &&
               data_->metric == other.data_->metric;
    }

    [[nodiscard]] bool operator!=(const Signature& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] std::string to_string() const {
        std::string out = "Cl([";
        for (std::size_t i = 0; i < data_->metric.size(); ++i) {
            if (i != 0) {
                out += ", ";
            }
            out += std::to_string(data_->metric[i]);
        }
        out += "])";
        return out;
    }

private:
    struct Data {
        std::vector<int> metric;
        int n = 0;
        int size = 1;
        int full = 0;
        int name_offset = 0;
        std::vector<std::int8_t> sign;
        std::vector<std::int8_t> dual;
        std::vector<std::int8_t> square;
    };

    std::shared_ptr<const Data> data_;

    static void build_aux_tables(Data& data) {
        data.dual.resize(static_cast<std::size_t>(data.size));
        data.square.resize(static_cast<std::size_t>(data.size));
        for (int mask = 0; mask < data.size; ++mask) {
            const int comp = data.full ^ mask;
            int swaps = 0;
            for (int i = 0; i < data.n; ++i) {
                if ((mask & (1 << i)) != 0) {
                    swaps += std::popcount(
                        static_cast<unsigned>(comp & ((1 << i) - 1)));
                }
            }
            data.dual[static_cast<std::size_t>(mask)] =
                static_cast<std::int8_t>((swaps & 1) == 0 ? 1 : -1);

            int sign = 1;
            for (int i = 0; i < data.n; ++i) {
                if ((mask & (1 << i)) == 0) {
                    continue;
                }
                const int m = data.metric[static_cast<std::size_t>(i)];
                if (m == 0) {
                    sign = 0;
                    break;
                }
                sign *= m;
            }
            data.square[static_cast<std::size_t>(mask)] =
                static_cast<std::int8_t>(sign);
        }
    }

    static void build_sign_table(Data& data) {
        data.sign.resize(static_cast<std::size_t>(data.size) *
                         static_cast<std::size_t>(data.size));
        for (int a = 0; a < data.size; ++a) {
            for (int b = 0; b < data.size; ++b) {
                int sign = 1;
                for (int j = 0; j < data.n; ++j) {
                    if ((b & (1 << j)) == 0) {
                        continue;
                    }
                    if ((std::popcount(static_cast<unsigned>(a >> (j + 1))) & 1) !=
                        0) {
                        sign = -sign;
                    }
                    if ((a & (1 << j)) != 0) {
                        const int m = data.metric[static_cast<std::size_t>(j)];
                        if (m == 0) {
                            sign = 0;
                            break;
                        }
                        sign *= m;
                    }
                }
                data.sign[static_cast<std::size_t>(a) *
                              static_cast<std::size_t>(data.size) +
                          static_cast<std::size_t>(b)] =
                    static_cast<std::int8_t>(sign);
            }
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const Signature& sig) {
    return os << sig.to_string();
}

} // namespace vsr::runtime
