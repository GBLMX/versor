#pragma once

#include <vsr/runtime/multivector.hpp>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vsr::runtime {

/**
 * Sparse multivector in COO form (sorted masks + values).
 *
 * Low-density products use a hash accumulator; dense-ish inputs switch to a
 * dense accumulator.  Masks are always kept sorted for deterministic output.
 */
class SparseMultivector {
public:
    SparseMultivector() = default;

    [[nodiscard]] static SparseMultivector zero(const Signature& sig) {
        return SparseMultivector(sig, {}, {});
    }

    [[nodiscard]] static SparseMultivector from_dense(const Multivector& m) {
        std::vector<int> masks;
        std::vector<double> values;
        for (int mask = 0; mask < m.size(); ++mask) {
            const double v = m.get(mask);
            if (v != 0.0) {
                masks.push_back(mask);
                values.push_back(v);
            }
        }
        return SparseMultivector(m.signature(), std::move(masks), std::move(values));
    }

    [[nodiscard]] static SparseMultivector of(const Signature& sig,
                                              const std::vector<int>& masks,
                                              const std::vector<double>& values) {
        if (masks.size() != values.size()) {
            throw std::invalid_argument("mask/value length mismatch");
        }
        std::map<int, double> merged;
        for (std::size_t i = 0; i < masks.size(); ++i) {
            if (masks[i] < 0 || masks[i] >= sig.size()) {
                throw std::invalid_argument("mask outside signature");
            }
            if (values[i] == 0.0) {
                continue;
            }
            merged[masks[i]] += values[i];
        }
        std::vector<int> m;
        std::vector<double> v;
        for (const auto& [mask, value] : merged) {
            if (value != 0.0) {
                m.push_back(mask);
                v.push_back(value);
            }
        }
        return SparseMultivector(sig, std::move(m), std::move(v));
    }

    [[nodiscard]] const Signature& signature() const noexcept { return sig_; }
    [[nodiscard]] int nnz() const noexcept { return static_cast<int>(masks_.size()); }
    [[nodiscard]] int mask(int i) const { return masks_.at(static_cast<std::size_t>(i)); }
    [[nodiscard]] double value(int i) const {
        return values_.at(static_cast<std::size_t>(i));
    }

    [[nodiscard]] Multivector to_dense() const {
        std::vector<double> c(static_cast<std::size_t>(sig_.size()), 0.0);
        for (std::size_t i = 0; i < masks_.size(); ++i) {
            c[static_cast<std::size_t>(masks_[i])] = values_[i];
        }
        return Multivector(sig_, std::move(c));
    }

    [[nodiscard]] SparseMultivector gp(const SparseMultivector& o) const {
        check(o);
        const long long pairs =
            static_cast<long long>(masks_.size()) * static_cast<long long>(o.masks_.size());
        const int hash_factor = sig_.n() <= 10 ? 4 : 16;
        if (pairs * hash_factor < sig_.size()) {
            return gp_hashed(o);
        }
        return gp_dense(o);
    }

    [[nodiscard]] SparseMultivector inverse() const {
        const double n2 = norm2();
        if (n2 == 0.0) {
            throw std::domain_error("element is not invertible");
        }
        return reverse().scale(1.0 / n2);
    }

    [[nodiscard]] SparseMultivector sandwich(const SparseMultivector& x) const {
        return gp(x).gp(reverse());
    }

    [[nodiscard]] SparseMultivector reverse() const {
        std::vector<double> v(values_.size());
        for (std::size_t i = 0; i < masks_.size(); ++i) {
            v[i] = Signature::reverse_sign(sig_.grade(masks_[i])) * values_[i];
        }
        return SparseMultivector(sig_, masks_, std::move(v));
    }

    [[nodiscard]] SparseMultivector add(const SparseMultivector& o) const {
        check(o);
        std::map<int, double> merged;
        for (std::size_t i = 0; i < masks_.size(); ++i) {
            merged[masks_[i]] += values_[i];
        }
        for (std::size_t i = 0; i < o.masks_.size(); ++i) {
            merged[o.masks_[i]] += o.values_[i];
        }
        std::vector<int> m;
        std::vector<double> v;
        for (const auto& [mask, value] : merged) {
            if (value != 0.0) {
                m.push_back(mask);
                v.push_back(value);
            }
        }
        return SparseMultivector(sig_, std::move(m), std::move(v));
    }

    [[nodiscard]] SparseMultivector scale(double s) const {
        std::vector<double> v(values_.size());
        for (std::size_t i = 0; i < values_.size(); ++i) {
            v[i] = values_[i] * s;
        }
        return SparseMultivector(sig_, masks_, std::move(v));
    }

    [[nodiscard]] double norm2() const {
        double sum = 0.0;
        for (std::size_t i = 0; i < masks_.size(); ++i) {
            sum += sig_.blade_square_sign(masks_[i]) * values_[i] * values_[i];
        }
        return sum;
    }

    [[nodiscard]] double max_abs() const {
        double max = 0.0;
        for (double v : values_) {
            max = std::max(max, std::abs(v));
        }
        return max;
    }

    [[nodiscard]] double distance_to(const SparseMultivector& o) const {
        check(o);
        return to_dense().distance_to(o.to_dense());
    }

private:
    Signature sig_;
    std::vector<int> masks_;
    std::vector<double> values_;

    SparseMultivector(Signature sig, std::vector<int> masks, std::vector<double> values)
        : sig_(std::move(sig)), masks_(std::move(masks)), values_(std::move(values)) {
        if (masks_.size() != values_.size()) {
            throw std::invalid_argument("sparse mask/value length mismatch");
        }
    }

    [[nodiscard]] SparseMultivector gp_hashed(const SparseMultivector& o) const {
        const std::size_t pairs = masks_.size() * o.masks_.size();
        std::size_t capacity = 16;
        while (capacity < std::max<std::size_t>(16, pairs * 2)) {
            capacity <<= 1;
        }
        std::vector<int> keys(capacity, -1);
        std::vector<double> acc(capacity, 0.0);
        const std::size_t slot_mask = capacity - 1;
        const std::int8_t* table = sig_.sign_table();
        for (std::size_t i = 0; i < masks_.size(); ++i) {
            const int ma = masks_[i];
            const double va = values_[i];
            for (std::size_t j = 0; j < o.masks_.size(); ++j) {
                const int mb = o.masks_[j];
                const int sign = table
                    ? table[static_cast<std::size_t>(ma) * sig_.size() + mb]
                    : sig_.compute_blade_product_sign(ma, mb);
                if (sign == 0) {
                    continue;
                }
                const int dest = ma ^ mb;
                std::size_t slot = mix(static_cast<std::uint32_t>(dest)) & slot_mask;
                while (keys[slot] != -1 && keys[slot] != dest) {
                    slot = (slot + 1) & slot_mask;
                }
                keys[slot] = dest;
                acc[slot] += static_cast<double>(sign) * va * o.values_[j];
            }
        }
        std::vector<int> masks;
        std::vector<double> values;
        masks.reserve(pairs);
        values.reserve(pairs);
        for (std::size_t slot = 0; slot < capacity; ++slot) {
            if (keys[slot] != -1 && acc[slot] != 0.0) {
                masks.push_back(keys[slot]);
                values.push_back(acc[slot]);
            }
        }
        std::vector<std::size_t> order(masks.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(),
                  [&masks](std::size_t a, std::size_t b) { return masks[a] < masks[b]; });
        std::vector<int> sorted_masks;
        std::vector<double> sorted_values;
        sorted_masks.reserve(order.size());
        sorted_values.reserve(order.size());
        for (std::size_t i : order) {
            sorted_masks.push_back(masks[i]);
            sorted_values.push_back(values[i]);
        }
        return SparseMultivector(sig_, std::move(sorted_masks), std::move(sorted_values));
    }

    static std::uint32_t mix(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        return x ^ (x >> 16);
    }

    [[nodiscard]] SparseMultivector gp_dense(const SparseMultivector& o) const {
        const int size = sig_.size();
        std::vector<double> acc(static_cast<std::size_t>(size), 0.0);
        std::vector<unsigned char> seen(static_cast<std::size_t>(size), 0);
        std::vector<int> touched;
        touched.reserve(
            std::min<std::size_t>(static_cast<std::size_t>(size),
                                  masks_.size() * o.masks_.size()));
        const std::int8_t* table = sig_.sign_table();
        for (std::size_t i = 0; i < masks_.size(); ++i) {
            const int ma = masks_[i];
            const double va = values_[i];
            for (std::size_t j = 0; j < o.masks_.size(); ++j) {
                const int mb = o.masks_[j];
                const int sign = table
                    ? table[static_cast<std::size_t>(ma) * size + mb]
                    : sig_.compute_blade_product_sign(ma, mb);
                if (sign == 0) {
                    continue;
                }
                const int dest = ma ^ mb;
                if (!seen[static_cast<std::size_t>(dest)]) {
                    seen[static_cast<std::size_t>(dest)] = 1;
                    touched.push_back(dest);
                }
                acc[static_cast<std::size_t>(dest)] +=
                    static_cast<double>(sign) * va * o.values_[j];
            }
        }
        std::sort(touched.begin(), touched.end());
        std::vector<int> masks;
        std::vector<double> values;
        masks.reserve(touched.size());
        values.reserve(touched.size());
        for (int mask : touched) {
            const double value = acc[static_cast<std::size_t>(mask)];
            if (value != 0.0) {
                masks.push_back(mask);
                values.push_back(value);
            }
        }
        return SparseMultivector(sig_, std::move(masks), std::move(values));
    }

    void check(const SparseMultivector& o) const {
        if (!(sig_ == o.sig_)) {
            throw std::invalid_argument("different algebras");
        }
    }
};

/**
 * Structure-of-arrays batch of multivectors sharing one signature.
 *
 * blades_[mask][i] is coefficient mask of element i.  Element i of any result
 * depends only on element i of the inputs.
 */
class MultivectorBatch {
public:
    MultivectorBatch(Signature sig, int count)
        : sig_(std::move(sig)), count_(count) {
        if (count_ < 0) {
            throw std::invalid_argument("negative count");
        }
        blades_.resize(static_cast<std::size_t>(sig_.size()),
                       std::vector<double>(static_cast<std::size_t>(count_), 0.0));
    }

    [[nodiscard]] static MultivectorBatch of(std::vector<Multivector> values) {
        if (values.empty()) {
            throw std::invalid_argument("empty batch");
        }
        const Signature sig = values[0].signature();
        MultivectorBatch batch(sig, static_cast<int>(values.size()));
        for (int i = 0; i < static_cast<int>(values.size()); ++i) {
            batch.set(i, values[static_cast<std::size_t>(i)]);
        }
        return batch;
    }

    [[nodiscard]] static MultivectorBatch of(
        std::initializer_list<Multivector> values) {
        return of(std::vector<Multivector>(values));
    }

    [[nodiscard]] int size() const noexcept { return count_; }
    [[nodiscard]] const Signature& signature() const noexcept { return sig_; }

    [[nodiscard]] Multivector get(int i) const {
        check_index(i);
        std::vector<double> c(static_cast<std::size_t>(sig_.size()), 0.0);
        for (int m = 0; m < sig_.size(); ++m) {
            c[static_cast<std::size_t>(m)] =
                blades_[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)];
        }
        return Multivector(sig_, std::move(c));
    }

    void set(int i, const Multivector& value) {
        check_index(i);
        check_signature(value);
        for (int m = 0; m < sig_.size(); ++m) {
            blades_[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] =
                value.get(m);
        }
    }

    [[nodiscard]] double get(int mask, int i) const {
        check_index(i);
        return blades_.at(static_cast<std::size_t>(mask)).at(static_cast<std::size_t>(i));
    }

    void set(int mask, int i, double value) {
        check_index(i);
        blades_.at(static_cast<std::size_t>(mask)).at(static_cast<std::size_t>(i)) = value;
    }

    [[nodiscard]] MultivectorBatch slice(int from, int to) const {
        if (from < 0 || to > count_ || from > to) {
            throw std::out_of_range("slice outside batch");
        }
        MultivectorBatch out(sig_, to - from);
        for (int m = 0; m < sig_.size(); ++m) {
            std::copy(blades_[static_cast<std::size_t>(m)].begin() + from,
                      blades_[static_cast<std::size_t>(m)].begin() + to,
                      out.blades_[static_cast<std::size_t>(m)].begin());
        }
        return out;
    }

    [[nodiscard]] MultivectorBatch copy() const { return *this; }

    [[nodiscard]] MultivectorBatch gp(const Multivector& left) const {
        check_signature(left);
        const int size = sig_.size();
        ActiveMask active;
        for (int b = 0; b < size; ++b) {
            active.set(static_cast<std::size_t>(b), has_nonzero(blades_[static_cast<std::size_t>(b)]));
        }
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(size),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        const auto& la = left.coefficients();
        const std::int8_t* table = sig_.sign_table();
        for (int a = 0; a < size; ++a) {
            const double ca = la[static_cast<std::size_t>(a)];
            if (ca == 0.0) {
                continue;
            }
            for (int b = 0; b < size; ++b) {
                if (!active[static_cast<std::size_t>(b)]) {
                    continue;
                }
                const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                       : sig_.compute_blade_product_sign(a, b);
                if (sign == 0) {
                    continue;
                }
                add_scaled(r[static_cast<std::size_t>(a ^ b)],
                           blades_[static_cast<std::size_t>(b)],
                           ca * static_cast<double>(sign));
            }
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    [[nodiscard]] MultivectorBatch gp_right(const Multivector& right) const {
        check_signature(right);
        const int size = sig_.size();
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(size),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        const auto& ra = right.coefficients();
        const std::int8_t* table = sig_.sign_table();
        for (int a = 0; a < size; ++a) {
            const auto& src = blades_[static_cast<std::size_t>(a)];
            if (!has_nonzero(src)) {
                continue;
            }
            for (int b = 0; b < size; ++b) {
                const double cb = ra[static_cast<std::size_t>(b)];
                if (cb == 0.0) {
                    continue;
                }
                const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                       : sig_.compute_blade_product_sign(a, b);
                if (sign == 0) {
                    continue;
                }
                add_scaled(r[static_cast<std::size_t>(a ^ b)], src,
                           cb * static_cast<double>(sign));
            }
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    [[nodiscard]] MultivectorBatch sandwich(const Multivector& versor) const {
        check_signature(versor);
        MultivectorBatch out(sig_, count_);
        return sandwich(versor, out);
    }

    [[nodiscard]] MultivectorBatch& sandwich(const Multivector& versor,
                                              MultivectorBatch& out) const;
    [[nodiscard]] MultivectorBatch op(const Multivector& left) const {
        return product(left, true, true);
    }
    [[nodiscard]] MultivectorBatch op_right(const Multivector& right) const {
        return product(right, true, false);
    }
    [[nodiscard]] MultivectorBatch ip(const Multivector& left) const {
        return product(left, false, true);
    }
    [[nodiscard]] MultivectorBatch ip_right(const Multivector& right) const {
        return product(right, false, false);
    }

    [[nodiscard]] MultivectorBatch dual() const {
        const int full = sig_.full_mask();
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(sig_.size()),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        for (int m = 0; m < sig_.size(); ++m) {
            if (!has_nonzero(blades_[static_cast<std::size_t>(m)])) {
                continue;
            }
            add_scaled(r[static_cast<std::size_t>(full ^ m)],
                       blades_[static_cast<std::size_t>(m)],
                       static_cast<double>(sig_.dual_sign(m)));
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    [[nodiscard]] MultivectorBatch undual() const {
        const int full = sig_.full_mask();
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(sig_.size()),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        for (int m = 0; m < sig_.size(); ++m) {
            const int comp = full ^ m;
            if (!has_nonzero(blades_[static_cast<std::size_t>(comp)])) {
                continue;
            }
            const int k = sig_.grade(m);
            const int sign = sig_.dual_sign(comp) *
                             (((k * (sig_.n() - k)) & 1) == 0 ? 1 : -1);
            add_scaled(r[static_cast<std::size_t>(m)],
                       blades_[static_cast<std::size_t>(comp)],
                       static_cast<double>(sign));
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    void clear() {
        for (auto& row : blades_) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }

    [[nodiscard]] MultivectorBatch add(const MultivectorBatch& o) const {
        check_same(o);
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(sig_.size()),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        for (int m = 0; m < sig_.size(); ++m) {
            for (int i = 0; i < count_; ++i) {
                r[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] =
                    blades_[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] +
                    o.blades_[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)];
            }
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    [[nodiscard]] MultivectorBatch scale(double s) const {
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(sig_.size()),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        for (int m = 0; m < sig_.size(); ++m) {
            for (int i = 0; i < count_; ++i) {
                r[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] =
                    blades_[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] * s;
            }
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    [[nodiscard]] MultivectorBatch reverse() const {
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(sig_.size()),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        for (int m = 0; m < sig_.size(); ++m) {
            const double s = Signature::reverse_sign(sig_.grade(m));
            for (int i = 0; i < count_; ++i) {
                r[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] =
                    blades_[static_cast<std::size_t>(m)][static_cast<std::size_t>(i)] * s;
            }
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    [[nodiscard]] std::vector<Multivector> to_multivectors() const {
        std::vector<Multivector> out;
        out.reserve(static_cast<std::size_t>(count_));
        for (int i = 0; i < count_; ++i) {
            out.push_back(get(i));
        }
        return out;
    }

private:
    using ActiveMask = std::bitset<1U << 14>;

    Signature sig_;
    int count_ = 0;
    std::vector<std::vector<double>> blades_;

    MultivectorBatch(Signature sig, int count,
                     std::vector<std::vector<double>> blades)
        : sig_(std::move(sig)), count_(count), blades_(std::move(blades)) {}

    [[nodiscard]] bool is_e3() const {
        return sig_.n() == 3 && sig_.metric(0) == 1 && sig_.metric(1) == 1 &&
               sig_.metric(2) == 1;
    }

    [[nodiscard]] bool is_pga3() const {
        return sig_.n() == 4 && sig_.metric(0) == 0 && sig_.metric(1) == 1 &&
               sig_.metric(2) == 1 && sig_.metric(3) == 1;
    }

    [[nodiscard]] bool only_blades(int m0, int m1, int m2) const {
        for (int m = 0; m < sig_.size(); ++m) {
            if (m != m0 && m != m1 && m != m2 && !all_zero(m)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool only_blades(int m0, int m1, int m2, int m3) const {
        for (int m = 0; m < sig_.size(); ++m) {
            if (m != m0 && m != m1 && m != m2 && m != m3 && !all_zero(m)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool all_zero(int mask) const {
        return !has_nonzero(blades_[static_cast<std::size_t>(mask)]);
    }

    static bool even_only(const Multivector& m) {
        const auto& c = m.coefficients();
        for (int mask = 1; mask < m.size(); ++mask) {
            if (c[static_cast<std::size_t>(mask)] != 0.0 &&
                (m.signature().grade(mask) & 1) != 0) {
                return false;
            }
        }
        return true;
    }

    void sandwich_e3(const Multivector& versor,
                     std::vector<std::vector<double>>& r) const {
        const double s = versor.get(0);
        const double b1 = versor.get(3);
        const double b2 = versor.get(5);
        const double b3 = versor.get(6);
        const auto& x = blades_[1];
        const auto& y = blades_[2];
        const auto& z = blades_[4];
        for (int i = 0; i < count_; ++i) {
            const double vx = x[static_cast<std::size_t>(i)];
            const double vy = y[static_cast<std::size_t>(i)];
            const double vz = z[static_cast<std::size_t>(i)];
            r[1][static_cast<std::size_t>(i)] =
                (s * s - b1 * b1 - b2 * b2 + b3 * b3) * vx +
                2.0 * (b1 * (s * vy + b3 * vz) + b2 * (s * vz - b3 * vy));
            r[2][static_cast<std::size_t>(i)] =
                (s * s - b1 * b1 + b2 * b2 - b3 * b3) * vy +
                2.0 * (-vx * (b1 * s + b2 * b3) + vz * (b3 * s - b1 * b2));
            r[4][static_cast<std::size_t>(i)] =
                (s * s + b1 * b1 - b2 * b2 - b3 * b3) * vz +
                2.0 * (vx * (b1 * b3 - b2 * s) - vy * (b1 * b2 + b3 * s));
        }
    }

    void sandwich_pga(const Multivector& versor,
                      std::vector<std::vector<double>>& r) const {
        const double s = versor.get(0);
        const double b01 = versor.get(3);
        const double b02 = versor.get(5);
        const double b12 = versor.get(6);
        const double b03 = versor.get(9);
        const double b13 = versor.get(10);
        const double b23 = versor.get(12);
        const double ps = versor.get(15);
        const auto& p7 = blades_[7];
        const auto& p11 = blades_[11];
        const auto& p13 = blades_[13];
        const auto& p14 = blades_[14];
        for (int i = 0; i < count_; ++i) {
            const double a7 = p7[static_cast<std::size_t>(i)];
            const double a11 = p11[static_cast<std::size_t>(i)];
            const double a13 = p13[static_cast<std::size_t>(i)];
            const double a14 = p14[static_cast<std::size_t>(i)];
            r[7][static_cast<std::size_t>(i)] =
                (s * s + b12 * b12 - b13 * b13 - b23 * b23) * a7 +
                2.0 * (b12 * (b13 * a11 + b23 * a13) +
                       s * (b23 * a11 - b13 * a13) +
                       a14 * (ps * b12 - b01 * b13 - b02 * b23 + b03 * s));
            r[11][static_cast<std::size_t>(i)] =
                (s * s - b12 * b12 + b13 * b13 - b23 * b23) * a11 +
                2.0 * (b12 * (b13 * a7 + s * a13) + b13 * b23 * a13 -
                       s * b23 * a7 +
                       a14 * (ps * b13 + b01 * b12 - b02 * s - b03 * b23));
            r[13][static_cast<std::size_t>(i)] =
                (s * s - b12 * b12 - b13 * b13 + b23 * b23) * a13 +
                2.0 * (b12 * (b23 * a7 - s * a11) +
                       b13 * (b23 * a11 + s * a7) +
                       a14 * (ps * b23 + b01 * s + b02 * b12 + b03 * b13));
            r[14][static_cast<std::size_t>(i)] =
                (s * s + b12 * b12 + b13 * b13 + b23 * b23) * a14;
        }
    }

    [[nodiscard]] MultivectorBatch product(const Multivector& fixed, bool outer,
                                            bool fixed_left) const {
        check_signature(fixed);
        const int size = sig_.size();
        ActiveMask active;
        for (int b = 0; b < size; ++b) {
            active.set(static_cast<std::size_t>(b), has_nonzero(blades_[static_cast<std::size_t>(b)]));
        }
        std::vector<std::vector<double>> r(
            static_cast<std::size_t>(size),
            std::vector<double>(static_cast<std::size_t>(count_), 0.0));
        const auto& f = fixed.coefficients();
        const std::int8_t* table = sig_.sign_table();
        for (int a = 0; a < size; ++a) {
            if (fixed_left) {
                if (f[static_cast<std::size_t>(a)] == 0.0) {
                    continue;
                }
                for (int b = 0; b < size; ++b) {
                    if (!active[static_cast<std::size_t>(b)]) {
                        continue;
                    }
                    if (outer ? (a & b) != 0 : (a & b) != a) {
                        continue;
                    }
                    const int sign = table
                        ? table[static_cast<std::size_t>(a) * size + b]
                        : sig_.compute_blade_product_sign(a, b);
                    if (sign == 0) {
                        continue;
                    }
                    add_scaled(r[static_cast<std::size_t>(a ^ b)],
                               blades_[static_cast<std::size_t>(b)],
                               f[static_cast<std::size_t>(a)] *
                                   static_cast<double>(sign));
                }
            } else {
                if (!active[static_cast<std::size_t>(a)]) {
                    continue;
                }
                for (int b = 0; b < size; ++b) {
                    if (f[static_cast<std::size_t>(b)] == 0.0) {
                        continue;
                    }
                    if (outer ? (a & b) != 0 : (a & b) != a) {
                        continue;
                    }
                    const int sign = table
                        ? table[static_cast<std::size_t>(a) * size + b]
                        : sig_.compute_blade_product_sign(a, b);
                    if (sign == 0) {
                        continue;
                    }
                    add_scaled(r[static_cast<std::size_t>(a ^ b)],
                               blades_[static_cast<std::size_t>(a)],
                               f[static_cast<std::size_t>(b)] *
                                   static_cast<double>(sign));
                }
            }
        }
        return MultivectorBatch(sig_, count_, std::move(r));
    }

    static void add_scaled(std::vector<double>& dst, const std::vector<double>& src,
                           double factor) {
        for (std::size_t i = 0; i < dst.size(); ++i) {
            dst[i] += factor * src[i];
        }
    }

    static bool has_nonzero(const std::vector<double>& values) {
        for (double v : values) {
            if (v != 0.0) {
                return true;
            }
        }
        return false;
    }

    void check_index(int i) const {
        if (i < 0 || i >= count_) {
            throw std::out_of_range("batch index outside range");
        }
    }

    void check_signature(const Multivector& value) const {
        if (value.signature() != sig_) {
            throw std::invalid_argument("different algebras");
        }
    }

    void check_same(const MultivectorBatch& o) const {
        if (o.sig_ != sig_ || o.count_ != count_) {
            throw std::invalid_argument("batch shape mismatch");
        }
    }
};

inline MultivectorBatch& MultivectorBatch::sandwich(const Multivector& versor,
                                                    MultivectorBatch& out) const {
    check_signature(versor);
    if (out.sig_ != sig_ || out.count_ != count_) {
        throw std::invalid_argument("batch shape mismatch");
    }
    out.clear();
    if (is_e3() && even_only(versor) && only_blades(1, 2, 4)) {
        sandwich_e3(versor, out.blades_);
        return out;
    }
    if (is_pga3() && even_only(versor) && only_blades(7, 11, 13, 14)) {
        sandwich_pga(versor, out.blades_);
        return out;
    }
    const auto& m = versor.coefficients();
    const Multivector inverse = versor.reverse();
    const auto& mr = inverse.coefficients();
    const int size = sig_.size();
    const std::int8_t* table = sig_.sign_table();
    ActiveMask active;
    for (int b = 0; b < size; ++b) {
        active.set(static_cast<std::size_t>(b), has_nonzero(blades_[static_cast<std::size_t>(b)]));
    }
    for (int a = 0; a < size; ++a) {
        if (m[static_cast<std::size_t>(a)] == 0.0) {
            continue;
        }
        for (int c = 0; c < size; ++c) {
            if (mr[static_cast<std::size_t>(c)] == 0.0) {
                continue;
            }
            for (int b = 0; b < size; ++b) {
                if (!active[static_cast<std::size_t>(b)]) {
                    continue;
                }
                const int mid = a ^ b;
                const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                       : sig_.compute_blade_product_sign(a, b);
                if (sign == 0) {
                    continue;
                }
                const int sign2 = table ? table[static_cast<std::size_t>(mid) * size + c]
                                        : sig_.compute_blade_product_sign(mid, c);
                if (sign2 == 0) {
                    continue;
                }
                const double factor = m[static_cast<std::size_t>(a)] *
                                      mr[static_cast<std::size_t>(c)] *
                                      static_cast<double>(sign * sign2);
                add_scaled(out.blades_[static_cast<std::size_t>(mid ^ c)],
                           blades_[static_cast<std::size_t>(b)], factor);
            }
        }
    }
    return out;
}

/** Dense matrix of multivectors; multiplication is the geometric product. */
class MultivectorMatrix {
public:
    MultivectorMatrix(Signature sig, int rows, int cols)
        : sig_(std::move(sig)), rows_(rows), cols_(cols) {
        if (rows_ < 0 || cols_ < 0) {
            throw std::invalid_argument("negative dimension");
        }
        entries_.resize(static_cast<std::size_t>(rows_),
                        std::vector<Multivector>(
                            static_cast<std::size_t>(cols_),
                            Multivector::zero(sig_)));
    }

    [[nodiscard]] static MultivectorMatrix identity(const Signature& sig, int n) {
        MultivectorMatrix m(sig, n, n);
        for (int i = 0; i < n; ++i) {
            m.entries_[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] =
                Multivector::scalar(sig, 1.0);
        }
        return m;
    }

    [[nodiscard]] const Signature& signature() const noexcept { return sig_; }
    [[nodiscard]] int rows() const noexcept { return rows_; }
    [[nodiscard]] int cols() const noexcept { return cols_; }

    [[nodiscard]] const Multivector& get(int r, int c) const {
        return entries_.at(static_cast<std::size_t>(r))
            .at(static_cast<std::size_t>(c));
    }

    void set(int r, int c, const Multivector& value) {
        check_signature(value);
        entries_.at(static_cast<std::size_t>(r)).at(static_cast<std::size_t>(c)) = value;
    }

    [[nodiscard]] MultivectorMatrix copy() const { return *this; }

    [[nodiscard]] MultivectorMatrix add(const MultivectorMatrix& o) const {
        check_shape(o);
        MultivectorMatrix m(sig_, rows_, cols_);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                m.entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                        .add(o.entries_[static_cast<std::size_t>(r)]
                                   [static_cast<std::size_t>(c)]);
            }
        }
        return m;
    }

    [[nodiscard]] MultivectorMatrix sub(const MultivectorMatrix& o) const {
        check_shape(o);
        MultivectorMatrix m(sig_, rows_, cols_);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                m.entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                        .sub(o.entries_[static_cast<std::size_t>(r)]
                                   [static_cast<std::size_t>(c)]);
            }
        }
        return m;
    }

    [[nodiscard]] MultivectorMatrix scale(double s) const {
        MultivectorMatrix m(sig_, rows_, cols_);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                m.entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                        .scale(s);
            }
        }
        return m;
    }

    [[nodiscard]] MultivectorMatrix multiply(const MultivectorMatrix& o) const {
        if (o.sig_ != sig_ || cols_ != o.rows_) {
            throw std::invalid_argument("matrix shape mismatch");
        }
        const int size = sig_.size();
        std::vector<double> acc(
            static_cast<std::size_t>(rows_) * static_cast<std::size_t>(o.cols_) *
                static_cast<std::size_t>(size),
            0.0);
        for (int r = 0; r < rows_; ++r) {
            for (int k = 0; k < cols_; ++k) {
                const Multivector& left =
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)];
                if (left.is_zero(0.0)) {
                    continue;
                }
                for (int c = 0; c < o.cols_; ++c) {
                    const std::size_t offset =
                        (static_cast<std::size_t>(r) * static_cast<std::size_t>(o.cols_) +
                         static_cast<std::size_t>(c)) *
                        static_cast<std::size_t>(size);
                    left.gp_accumulate(
                        o.entries_[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)],
                        std::span<double>(acc.data() + offset,
                                          static_cast<std::size_t>(size)));
                }
            }
        }
        MultivectorMatrix m(sig_, rows_, o.cols_);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < o.cols_; ++c) {
                const std::size_t offset =
                    (static_cast<std::size_t>(r) * static_cast<std::size_t>(o.cols_) +
                     static_cast<std::size_t>(c)) *
                    static_cast<std::size_t>(size);
                m.entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                    Multivector(sig_, std::vector<double>(
                                          acc.begin() + static_cast<std::ptrdiff_t>(offset),
                                          acc.begin() + static_cast<std::ptrdiff_t>(offset) +
                                              size));
            }
        }
        return m;
    }

    [[nodiscard]] MultivectorMatrix transpose() const {
        MultivectorMatrix m(sig_, cols_, rows_);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                m.entries_[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] =
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
        }
        return m;
    }

    [[nodiscard]] MultivectorMatrix inverse() const {
        if (rows_ != cols_) {
            throw std::logic_error("matrix is not square");
        }
        const int n = rows_;
        std::vector<std::vector<Multivector>> aug(
            static_cast<std::size_t>(n),
            std::vector<Multivector>(static_cast<std::size_t>(2 * n),
                                     Multivector::zero(sig_)));
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
            aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(n + r)] =
                Multivector::scalar(sig_, 1.0);
        }
        for (int col = 0; col < n; ++col) {
            int pivot = -1;
            Multivector inverse_value;
            std::vector<unsigned char> tried(static_cast<std::size_t>(n), 0);
            for (int attempt = col; attempt < n && pivot < 0; ++attempt) {
                int candidate = -1;
                double best = -1.0;
                for (int r = col; r < n; ++r) {
                    if (tried[static_cast<std::size_t>(r)]) {
                        continue;
                    }
                    const double score =
                        aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)]
                            .max_abs();
                    if (score > best) {
                        best = score;
                        candidate = r;
                    }
                }
                if (candidate < 0 || best == 0.0) {
                    break;
                }
                tried[static_cast<std::size_t>(candidate)] = 1;
                try {
                    const Multivector& pivot_value =
                        aug[static_cast<std::size_t>(candidate)]
                           [static_cast<std::size_t>(col)];
                    inverse_value = pivot_value.is_versor()
                        ? pivot_value.inverse()
                        : pivot_value.inverse_general().value;
                    pivot = candidate;
                } catch (const std::exception&) {
                    // Non-invertible candidate; try the next row.
                }
            }
            if (pivot < 0) {
                throw std::domain_error("matrix is singular");
            }
            std::swap(aug[static_cast<std::size_t>(col)],
                      aug[static_cast<std::size_t>(pivot)]);
            for (int c = 0; c < 2 * n; ++c) {
                aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(c)] =
                    inverse_value.gp(
                        aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(c)]);
            }
            for (int r = 0; r < n; ++r) {
                if (r == col) {
                    continue;
                }
                const Multivector factor =
                    aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)];
                if (factor.is_zero(0.0)) {
                    continue;
                }
                for (int c = 0; c < 2 * n; ++c) {
                    std::vector<double> dst =
                        aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                            .coefficients();
                    factor.gp_accumulate(
                        aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(c)],
                        dst, -1.0);
                    aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                        Multivector(sig_, std::move(dst));
                }
            }
        }
        MultivectorMatrix m(sig_, n, n);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                m.entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                    aug[static_cast<std::size_t>(r)][static_cast<std::size_t>(n + c)];
            }
        }
        return m;
    }

    [[nodiscard]] double distance_to(const MultivectorMatrix& o) const {
        check_shape(o);
        double max = 0.0;
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                max = std::max(
                    max,
                    entries_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]
                        .distance_to(o.entries_[static_cast<std::size_t>(r)]
                                            [static_cast<std::size_t>(c)]));
            }
        }
        return max;
    }

private:
    Signature sig_;
    int rows_ = 0;
    int cols_ = 0;
    std::vector<std::vector<Multivector>> entries_;

    void check_shape(const MultivectorMatrix& o) const {
        if (o.sig_ != sig_ || o.rows_ != rows_ || o.cols_ != cols_) {
            throw std::invalid_argument("matrix shape mismatch");
        }
    }

    void check_signature(const Multivector& value) const {
        if (value.signature() != sig_) {
            throw std::invalid_argument("different algebras");
        }
    }
};

} // namespace vsr::runtime
