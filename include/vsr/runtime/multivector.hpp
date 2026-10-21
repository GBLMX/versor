#pragma once

#include <vsr/runtime/signature.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vsr::runtime {

class Multivector;
struct GeneralInverse;

/**
 * Immutable dense multivector: one coefficient per basis blade of a Signature.
 *
 * The API intentionally follows versor-java's Multivector.  Operations return
 * new values; gp_accumulate is exposed for batch and matrix kernels that need
 * to avoid intermediate allocations.
 */
class Multivector {
public:
    Multivector() : Multivector(Signature{}) {}

    explicit Multivector(Signature sig)
        : sig_(std::move(sig)), c_(static_cast<std::size_t>(sig_.size()), 0.0) {}

    Multivector(Signature sig, std::vector<double> coefficients)
        : sig_(std::move(sig)), c_(std::move(coefficients)) {
        if (c_.size() != static_cast<std::size_t>(sig_.size())) {
            throw std::invalid_argument("coefficient length mismatch");
        }
    }

    [[nodiscard]] static Multivector zero(const Signature& sig) {
        return Multivector(sig);
    }

    [[nodiscard]] static Multivector scalar(const Signature& sig, double value) {
        Multivector out(sig);
        out.c_[0] = value;
        return out;
    }

    [[nodiscard]] static Multivector blade(const Signature& sig, int mask, double value) {
        if (mask < 0 || mask >= sig.size()) {
            throw std::out_of_range("blade mask outside signature");
        }
        Multivector out(sig);
        out.c_[static_cast<std::size_t>(mask)] = value;
        return out;
    }

    [[nodiscard]] static Multivector vector(const Signature& sig,
                                            std::initializer_list<double> coords) {
        return vector(sig, std::span<const double>(coords.begin(), coords.size()));
    }

    [[nodiscard]] static Multivector vector(const Signature& sig,
                                            std::span<const double> coords) {
        if (coords.size() != static_cast<std::size_t>(sig.n())) {
            throw std::invalid_argument("expected " + std::to_string(sig.n()) + " coordinates");
        }
        Multivector out(sig);
        for (int i = 0; i < sig.n(); ++i) {
            out.c_[static_cast<std::size_t>(1 << i)] = coords[static_cast<std::size_t>(i)];
        }
        return out;
    }

    [[nodiscard]] const Signature& signature() const noexcept { return sig_; }
    [[nodiscard]] int size() const noexcept { return sig_.size(); }

    [[nodiscard]] double get(int mask) const {
        return c_.at(static_cast<std::size_t>(mask));
    }

    [[nodiscard]] const std::vector<double>& coefficients() const noexcept { return c_; }

    [[nodiscard]] std::vector<double> to_array() const { return c_; }

    [[nodiscard]] double scalar_part() const noexcept { return c_[0]; }

    [[nodiscard]] Multivector add(const Multivector& o) const {
        check(o);
        std::vector<double> r = c_;
        for (std::size_t i = 0; i < r.size(); ++i) {
            r[i] += o.c_[i];
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector add(double s) const {
        std::vector<double> r = c_;
        r[0] += s;
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector sub(const Multivector& o) const {
        check(o);
        std::vector<double> r = c_;
        for (std::size_t i = 0; i < r.size(); ++i) {
            r[i] -= o.c_[i];
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector sub(double s) const { return add(-s); }

    [[nodiscard]] Multivector neg() const {
        std::vector<double> r(c_.size());
        for (std::size_t i = 0; i < r.size(); ++i) {
            r[i] = -c_[i];
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector scale(double s) const {
        std::vector<double> r(c_.size());
        for (std::size_t i = 0; i < r.size(); ++i) {
            r[i] = c_[i] * s;
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector divide(double s) const { return scale(1.0 / s); }

    [[nodiscard]] Multivector divide(const Multivector& divisor) const {
        return gp(divisor.inverse());
    }

    [[nodiscard]] double scalar_product(const Multivector& o) const {
        return gp(o).scalar_part();
    }

    [[nodiscard]] Multivector mul(const Multivector& o) const { return gp(o); }
    [[nodiscard]] Multivector mul(double s) const { return scale(s); }

    [[nodiscard]] Multivector gp(const Multivector& o) const {
        std::vector<double> r(c_.size(), 0.0);
        gp_accumulate(o, r);
        return Multivector(sig_, std::move(r));
    }

    /** Accumulates this * o into out.  out must have size() zeroed entries. */
    void gp_accumulate(const Multivector& o, std::span<double> out,
                       double scale = 1.0) const {
        if (scale == 0.0) {
            return;
        }
        check(o);
        if (out.size() != c_.size()) {
            throw std::invalid_argument("output length mismatch");
        }
        const int size = sig_.size();
        const std::int8_t* table = sig_.sign_table();
        const int a_count = non_zero_count(c_);
        const int b_count = non_zero_count(o.c_);
        if (static_cast<long long>(a_count) * b_count * 4 <
            static_cast<long long>(size) * size) {
            const std::vector<int> ai = non_zero_indices(c_, a_count);
            const std::vector<int> bi = non_zero_indices(o.c_, b_count);
            for (int ia : ai) {
                const double ca = scale * c_[static_cast<std::size_t>(ia)];
                for (int ib : bi) {
                    const int sign = table ? table[static_cast<std::size_t>(ia) * size + ib]
                                           : sig_.compute_blade_product_sign(ia, ib);
                    if (sign != 0) {
                        out[static_cast<std::size_t>(ia ^ ib)] +=
                            static_cast<double>(sign) * ca * o.c_[static_cast<std::size_t>(ib)];
                    }
                }
            }
            return;
        }
        for (int a = 0; a < size; ++a) {
            const double ca = scale * c_[static_cast<std::size_t>(a)];
            if (ca == 0.0) {
                continue;
            }
            for (int b = 0; b < size; ++b) {
                const double cb = o.c_[static_cast<std::size_t>(b)];
                if (cb == 0.0) {
                    continue;
                }
                const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                       : sig_.compute_blade_product_sign(a, b);
                if (sign != 0) {
                    out[static_cast<std::size_t>(a ^ b)] +=
                        static_cast<double>(sign) * ca * cb;
                }
            }
        }
    }

    void gp_accumulate(const Multivector& o, std::vector<double>& out,
                       double scale = 1.0) const {
        gp_accumulate(o, std::span<double>(out), scale);
    }

    [[nodiscard]] Multivector op(const Multivector& o) const {
        check(o);
        const int size = sig_.size();
        const int full = sig_.full_mask();
        const std::int8_t* table = sig_.sign_table();
        std::vector<double> r(c_.size(), 0.0);
        for (int a = 0; a < size; ++a) {
            const double ca = c_[static_cast<std::size_t>(a)];
            if (ca == 0.0) {
                continue;
            }
            const int comp = full ^ a;
            for (int b = comp;; b = (b - 1) & comp) {
                const double cb = o.c_[static_cast<std::size_t>(b)];
                if (cb != 0.0) {
                    const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                           : sig_.compute_blade_product_sign(a, b);
                    if (sign != 0) {
                        r[static_cast<std::size_t>(a ^ b)] +=
                            static_cast<double>(sign) * ca * cb;
                    }
                }
                if (b == 0) {
                    break;
                }
            }
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector wedge(const Multivector& o) const { return op(o); }

    [[nodiscard]] Multivector ip(const Multivector& o) const {
        check(o);
        const int size = sig_.size();
        const int full = sig_.full_mask();
        const std::int8_t* table = sig_.sign_table();
        std::vector<double> r(c_.size(), 0.0);
        for (int a = 0; a < size; ++a) {
            const double ca = c_[static_cast<std::size_t>(a)];
            if (ca == 0.0) {
                continue;
            }
            const int comp = full ^ a;
            for (int sub = comp;; sub = (sub - 1) & comp) {
                const int b = a | sub;
                const double cb = o.c_[static_cast<std::size_t>(b)];
                if (cb != 0.0) {
                    const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                           : sig_.compute_blade_product_sign(a, b);
                    if (sign != 0) {
                        r[static_cast<std::size_t>(sub)] +=
                            static_cast<double>(sign) * ca * cb;
                    }
                }
                if (sub == 0) {
                    break;
                }
            }
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector contract(const Multivector& o) const { return ip(o); }

    [[nodiscard]] Multivector right_contraction(const Multivector& o) const {
        check(o);
        const int size = sig_.size();
        const std::int8_t* table = sig_.sign_table();
        std::vector<double> r(c_.size(), 0.0);
        for (int a = 0; a < size; ++a) {
            const double ca = c_[static_cast<std::size_t>(a)];
            if (ca == 0.0) {
                continue;
            }
            for (int b = a;; b = (b - 1) & a) {
                const double cb = o.c_[static_cast<std::size_t>(b)];
                if (cb != 0.0) {
                    const int sign = table ? table[static_cast<std::size_t>(a) * size + b]
                                           : sig_.compute_blade_product_sign(a, b);
                    if (sign != 0) {
                        r[static_cast<std::size_t>(a ^ b)] +=
                            static_cast<double>(sign) * ca * cb;
                    }
                }
                if (b == 0) {
                    break;
                }
            }
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector commutator(const Multivector& o) const {
        return gp(o).sub(o.gp(*this)).scale(0.5);
    }

    [[nodiscard]] Multivector anticommutator(const Multivector& o) const {
        return gp(o).add(o.gp(*this)).scale(0.5);
    }

    [[nodiscard]] Multivector vee(const Multivector& o) const {
        return dual().op(o.dual()).dual();
    }

    [[nodiscard]] Multivector regressive(const Multivector& o) const { return vee(o); }

    [[nodiscard]] Multivector reverse() const {
        std::vector<double> r(c_.size());
        for (int m = 0; m < sig_.size(); ++m) {
            r[static_cast<std::size_t>(m)] =
                c_[static_cast<std::size_t>(m)] * Signature::reverse_sign(sig_.grade(m));
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector conjugate() const {
        std::vector<double> r(c_.size());
        for (int m = 0; m < sig_.size(); ++m) {
            r[static_cast<std::size_t>(m)] =
                c_[static_cast<std::size_t>(m)] * Signature::conjugate_sign(sig_.grade(m));
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector involution() const {
        std::vector<double> r(c_.size());
        for (int m = 0; m < sig_.size(); ++m) {
            r[static_cast<std::size_t>(m)] =
                c_[static_cast<std::size_t>(m)] * Signature::involution_sign(sig_.grade(m));
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector grade(int k) const {
        std::vector<double> r(c_.size(), 0.0);
        for (int m = 0; m < sig_.size(); ++m) {
            if (sig_.grade(m) == k) {
                r[static_cast<std::size_t>(m)] = c_[static_cast<std::size_t>(m)];
            }
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector even() const {
        std::vector<double> r(c_.size(), 0.0);
        for (int m = 0; m < sig_.size(); ++m) {
            if ((sig_.grade(m) & 1) == 0) {
                r[static_cast<std::size_t>(m)] = c_[static_cast<std::size_t>(m)];
            }
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector odd() const {
        std::vector<double> r(c_.size(), 0.0);
        for (int m = 0; m < sig_.size(); ++m) {
            if ((sig_.grade(m) & 1) != 0) {
                r[static_cast<std::size_t>(m)] = c_[static_cast<std::size_t>(m)];
            }
        }
        return Multivector(sig_, std::move(r));
    }

    /** Poincare dual: complement blade with the sign making blade ^ dual = I. */
    [[nodiscard]] Multivector dual() const {
        const int full = sig_.full_mask();
        std::vector<double> r(c_.size(), 0.0);
        for (int m = 0; m < sig_.size(); ++m) {
            if (c_[static_cast<std::size_t>(m)] == 0.0) {
                continue;
            }
            r[static_cast<std::size_t>(full ^ m)] +=
                sig_.dual_sign(m) * c_[static_cast<std::size_t>(m)];
        }
        return Multivector(sig_, std::move(r));
    }

    /** Inverse of dual() under the Poincare complement convention. */
    [[nodiscard]] Multivector undual() const {
        const int full = sig_.full_mask();
        std::vector<double> r(c_.size(), 0.0);
        for (int m = 0; m < sig_.size(); ++m) {
            const int comp = full ^ m;
            if (c_[static_cast<std::size_t>(comp)] == 0.0) {
                continue;
            }
            const int k = sig_.grade(m);
            const int sign = sig_.dual_sign(comp) *
                             (((k * (sig_.n() - k)) & 1) == 0 ? 1 : -1);
            r[static_cast<std::size_t>(m)] +=
                sign * c_[static_cast<std::size_t>(comp)];
        }
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] double norm2() const { return gp(reverse()).scalar_part(); }

    [[nodiscard]] double norm() const { return std::sqrt(std::abs(norm2())); }

    [[nodiscard]] Multivector unit() const {
        const double n = norm();
        return n == 0.0 ? *this : divide(n);
    }

    /** Fast inverse for versors and blades: ~A / (A ~A). */
    [[nodiscard]] Multivector inverse() const {
        const double n2 = norm2();
        if (n2 == 0.0) {
            throw std::domain_error("element is not invertible");
        }
        return reverse().divide(n2);
    }

    [[nodiscard]] GeneralInverse inverse_general() const;
    [[nodiscard]] GeneralInverse inverse_general(double tolerance) const;
    [[nodiscard]] GeneralInverse inverse_general_result() const;

    /** Sandwich product M x ~M. */
    [[nodiscard]] Multivector sandwich(const Multivector& x) const {
        check(x);
        if (is_e3() && only_even_grades() && x.only_blades(1, 2, 4)) {
            return sandwich_e3_vector(x);
        }
        if (is_pga3() && only_even_grades() && x.only_blades(7, 11, 13, 14)) {
            return sandwich_pga_point(x);
        }
        return gp(x).gp(reverse());
    }

    [[nodiscard]] Multivector apply(const Multivector& x) const { return sandwich(x); }

    /** Reflection by an odd unit versor: b x^ b~. */
    [[nodiscard]] Multivector reflect(const Multivector& reflector) const {
        check(reflector);
        return reflector.gp(involution()).gp(reflector.reverse());
    }

    /** Exponential, summed with scaling and squaring. */
    [[nodiscard]] Multivector exp() const {
        const double max = max_abs();
        if (max == 0.0) {
            return scalar(sig_, 1.0);
        }
        int squares = std::max(0, static_cast<int>(std::ceil(std::log2(max))));
        squares = std::min(squares, 1023);
        const Multivector a = scale(std::ldexp(1.0, -squares));
        Multivector term = scalar(sig_, 1.0);
        Multivector sum = scalar(sig_, 1.0);
        for (int k = 1; k <= 32; ++k) {
            term = term.gp(a).divide(static_cast<double>(k));
            sum = sum.add(term);
            if (term.max_abs() < 1e-18) {
                break;
            }
        }
        for (int i = 0; i < squares; ++i) {
            sum = sum.gp(sum);
        }
        return sum;
    }

    [[nodiscard]] double max_abs() const {
        double m = 0.0;
        for (double v : c_) {
            m = std::max(m, std::abs(v));
        }
        return m;
    }

    [[nodiscard]] bool is_zero(double eps = 1e-12) const { return max_abs() < eps; }

    [[nodiscard]] bool is_finite() const {
        for (double v : c_) {
            if (!std::isfinite(v)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool is_scalar(double eps = 1e-12) const {
        if (!is_finite()) {
            return false;
        }
        const double limit = eps * max_abs();
        for (std::size_t m = 1; m < c_.size(); ++m) {
            if (std::abs(c_[m]) > limit) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool is_homogeneous(double eps = 1e-12) const {
        if (!is_finite()) {
            return false;
        }
        const double limit = eps * max_abs();
        int grade = -1;
        for (int m = 0; m < sig_.size(); ++m) {
            if (std::abs(c_[static_cast<std::size_t>(m)]) <= limit) {
                continue;
            }
            const int k = sig_.grade(m);
            if (grade < 0) {
                grade = k;
            } else if (grade != k) {
                return false;
            }
        }
        return true;
    }

    /** True when this is a simple blade (rank test for grade > 1). */
    [[nodiscard]] bool is_blade(double eps = 1e-12) const {
        if (!is_finite() || !is_homogeneous(eps)) {
            return false;
        }
        const double limit = eps * max_abs();
        int grade = -1;
        for (int m = 0; m < sig_.size(); ++m) {
            if (std::abs(c_[static_cast<std::size_t>(m)]) > limit) {
                grade = sig_.grade(m);
                break;
            }
        }
        if (grade <= 1) {
            return true;
        }
        if (grade > sig_.n()) {
            return false;
        }
        std::vector<std::vector<double>> image(static_cast<std::size_t>(sig_.n()),
                                              std::vector<double>(c_.size(), 0.0));
        double scale = 0.0;
        for (int i = 0; i < sig_.n(); ++i) {
            const Multivector contracted = Multivector::blade(sig_, 1 << i, 1.0).ip(*this);
            image[static_cast<std::size_t>(i)] = contracted.c_;
            for (double v : contracted.c_) {
                scale = std::max(scale, std::abs(v));
            }
        }
        return rank(image, eps * scale) == grade;
    }

    [[nodiscard]] bool is_versor(double eps = 1e-12) const {
        if (!is_finite()) {
            return false;
        }
        const Multivector square = gp(reverse());
        if (!square.is_scalar(eps)) {
            return false;
        }
        return std::abs(square.scalar_part()) > eps * square.max_abs();
    }

    [[nodiscard]] double distance_to(const Multivector& o) const {
        check(o);
        double m = 0.0;
        for (std::size_t i = 0; i < c_.size(); ++i) {
            m = std::max(m, std::abs(c_[i] - o.c_[i]));
        }
        return m;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream out;
        out << std::setprecision(6);
        bool first = true;
        for (int m = 0; m < sig_.size(); ++m) {
            const double v = c_[static_cast<std::size_t>(m)];
            if (std::abs(v) < 1e-12) {
                continue;
            }
            if (!first) {
                out << (v < 0.0 ? " - " : " + ");
            } else if (v < 0.0) {
                out << '-';
            }
            out << std::abs(v);
            if (m != 0) {
                out << sig_.blade_name(m);
            }
            first = false;
        }
        return first ? "0" : out.str();
    }

    [[nodiscard]] bool operator==(const Multivector& o) const {
        return sig_ == o.sig_ && c_ == o.c_;
    }

    [[nodiscard]] bool operator!=(const Multivector& o) const { return !(*this == o); }

private:
    Signature sig_;
    std::vector<double> c_;

    void check(const Multivector& o) const {
        if (!(sig_ == o.sig_)) {
            throw std::invalid_argument("different algebras");
        }
    }

    static std::vector<double> invert_equilibrated(const std::vector<double>& a, int n);
    double inverse_residual(const std::vector<double>& x) const;
    std::vector<double> inverse_error_vector(const std::vector<double>& x) const;
    static std::vector<double> mat_vec(const std::vector<double>& matrix,
                                       const std::vector<double>& vector, int n);
    static double max_row_sum(const std::vector<double>& matrix, int n);
    static double max_abs(const std::vector<double>& values);

    static int non_zero_count(const std::vector<double>& values) {
        int count = 0;
        for (double v : values) {
            if (v != 0.0) {
                ++count;
            }
        }
        return count;
    }

    static std::vector<int> non_zero_indices(const std::vector<double>& values, int count) {
        std::vector<int> indices;
        indices.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < static_cast<int>(values.size()); ++i) {
            if (values[static_cast<std::size_t>(i)] != 0.0) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    [[nodiscard]] bool is_e3() const {
        return sig_.n() == 3 && sig_.metric(0) == 1 && sig_.metric(1) == 1 &&
               sig_.metric(2) == 1;
    }

    [[nodiscard]] bool is_pga3() const {
        return sig_.n() == 4 && sig_.metric(0) == 0 && sig_.metric(1) == 1 &&
               sig_.metric(2) == 1 && sig_.metric(3) == 1;
    }

    [[nodiscard]] bool only_even_grades() const {
        for (int m = 1; m < sig_.size(); ++m) {
            if (c_[static_cast<std::size_t>(m)] != 0.0 && (sig_.grade(m) & 1) != 0) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool only_blades(int m0, int m1, int m2) const {
        for (int m = 0; m < sig_.size(); ++m) {
            if (m != m0 && m != m1 && m != m2 && c_[static_cast<std::size_t>(m)] != 0.0) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool only_blades(int m0, int m1, int m2, int m3) const {
        for (int m = 0; m < sig_.size(); ++m) {
            if (m != m0 && m != m1 && m != m2 && m != m3 &&
                c_[static_cast<std::size_t>(m)] != 0.0) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Multivector sandwich_e3_vector(const Multivector& x) const {
        const double s = c_[0];
        const double b1 = c_[3];
        const double b2 = c_[5];
        const double b3 = c_[6];
        const double vx = x.c_[1];
        const double vy = x.c_[2];
        const double vz = x.c_[4];
        const double rx = (s * s - b1 * b1 - b2 * b2 + b3 * b3) * vx +
                          2.0 * (b1 * (s * vy + b3 * vz) + b2 * (s * vz - b3 * vy));
        const double ry = (s * s - b1 * b1 + b2 * b2 - b3 * b3) * vy +
                          2.0 * (-vx * (b1 * s + b2 * b3) + vz * (b3 * s - b1 * b2));
        const double rz = (s * s + b1 * b1 - b2 * b2 - b3 * b3) * vz +
                          2.0 * (vx * (b1 * b3 - b2 * s) - vy * (b1 * b2 + b3 * s));
        std::vector<double> r(8, 0.0);
        r[1] = rx;
        r[2] = ry;
        r[4] = rz;
        return Multivector(sig_, std::move(r));
    }

    [[nodiscard]] Multivector sandwich_pga_point(const Multivector& x) const {
        const double s = c_[0];
        const double b01 = c_[3];
        const double b02 = c_[5];
        const double b12 = c_[6];
        const double b03 = c_[9];
        const double b13 = c_[10];
        const double b23 = c_[12];
        const double i = c_[15];
        const double p7 = x.c_[7];
        const double p11 = x.c_[11];
        const double p13 = x.c_[13];
        const double p14 = x.c_[14];
        const double r7 =
            (s * s + b12 * b12 - b13 * b13 - b23 * b23) * p7 +
            2.0 * (b12 * (b13 * p11 + b23 * p13) + s * (b23 * p11 - b13 * p13) +
                   p14 * (i * b12 - b01 * b13 - b02 * b23 + b03 * s));
        const double r11 =
            (s * s - b12 * b12 + b13 * b13 - b23 * b23) * p11 +
            2.0 * (b12 * (b13 * p7 + s * p13) + b13 * b23 * p13 - s * b23 * p7 +
                   p14 * (i * b13 + b01 * b12 - b02 * s - b03 * b23));
        const double r13 =
            (s * s - b12 * b12 - b13 * b13 + b23 * b23) * p13 +
            2.0 * (b12 * (b23 * p7 - s * p11) + b13 * (b23 * p11 + s * p7) +
                   p14 * (i * b23 + b01 * s + b02 * b12 + b03 * b13));
        const double r14 = (s * s + b12 * b12 + b13 * b13 + b23 * b23) * p14;
        std::vector<double> r(16, 0.0);
        r[7] = r7;
        r[11] = r11;
        r[13] = r13;
        r[14] = r14;
        return Multivector(sig_, std::move(r));
    }

    static int rank(std::vector<std::vector<double>> matrix, double tolerance) {
        const int rows = static_cast<int>(matrix.size());
        const int cols = rows == 0 ? 0 : static_cast<int>(matrix[0].size());
        int rank = 0;
        int pivot_row = 0;
        for (int col = 0; col < cols && pivot_row < rows; ++col) {
            int best = pivot_row;
            double best_abs = std::abs(matrix[static_cast<std::size_t>(best)]
                                             [static_cast<std::size_t>(col)]);
            for (int r = pivot_row + 1; r < rows; ++r) {
                const double v = std::abs(matrix[static_cast<std::size_t>(r)]
                                                 [static_cast<std::size_t>(col)]);
                if (v > best_abs) {
                    best = r;
                    best_abs = v;
                }
            }
            if (best_abs <= tolerance) {
                continue;
            }
            if (best != pivot_row) {
                std::swap(matrix[static_cast<std::size_t>(pivot_row)],
                          matrix[static_cast<std::size_t>(best)]);
            }
            const double pivot = matrix[static_cast<std::size_t>(pivot_row)]
                                       [static_cast<std::size_t>(col)];
            for (int r = pivot_row + 1; r < rows; ++r) {
                const double factor = matrix[static_cast<std::size_t>(r)]
                                            [static_cast<std::size_t>(col)] /
                                      pivot;
                if (factor == 0.0) {
                    continue;
                }
                for (int k = col; k < cols; ++k) {
                    matrix[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)] -=
                        factor * matrix[static_cast<std::size_t>(pivot_row)]
                                       [static_cast<std::size_t>(k)];
                }
            }
            ++pivot_row;
            ++rank;
        }
        return rank;
    }
};

struct GeneralInverse {
    Multivector value;
    double residual = 0.0;
    double relative_residual = 0.0;
    double condition = 0.0;
};

inline constexpr double inverse_tolerance = 1e-12;
inline constexpr double inverse_condition_limit = 1e12;

inline GeneralInverse Multivector::inverse_general() const {
    GeneralInverse result = inverse_general_result();
    if (!(result.condition <= inverse_condition_limit)) {
        throw std::domain_error("ill-conditioned inverse: condition " +
                                std::to_string(result.condition) + " > " +
                                std::to_string(inverse_condition_limit));
    }
    return result;
}

inline GeneralInverse Multivector::inverse_general(double tolerance) const {
    GeneralInverse result = inverse_general_result();
    if (!(result.relative_residual <= tolerance)) {
        throw std::domain_error("ill-conditioned inverse: relative residual " +
                                std::to_string(result.relative_residual) + " > " +
                                std::to_string(tolerance) + ", condition ~ " +
                                std::to_string(result.condition));
    }
    return result;
}

inline GeneralInverse Multivector::inverse_general_result() const {
    const int n = sig_.size();
    std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
    for (int col = 0; col < n; ++col) {
        Multivector unit(sig_);
        unit.c_[static_cast<std::size_t>(col)] = 1.0;
        const Multivector product = gp(unit);
        for (int row = 0; row < n; ++row) {
            a[static_cast<std::size_t>(row) * n + col] =
                product.c_[static_cast<std::size_t>(row)];
        }
    }

    std::vector<double> row_scale(static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
        double max = 0.0;
        for (int col = 0; col < n; ++col) {
            max = std::max(max, std::abs(a[static_cast<std::size_t>(row) * n + col]));
        }
        if (max == 0.0) {
            throw std::domain_error("element is not invertible");
        }
        row_scale[static_cast<std::size_t>(row)] = max;
    }
    std::vector<double> col_scale(static_cast<std::size_t>(n), 0.0);
    for (int col = 0; col < n; ++col) {
        double max = 0.0;
        for (int row = 0; row < n; ++row) {
            max = std::max(max,
                           std::abs(a[static_cast<std::size_t>(row) * n + col]) /
                               row_scale[static_cast<std::size_t>(row)]);
        }
        if (max == 0.0) {
            throw std::domain_error("element is not invertible");
        }
        col_scale[static_cast<std::size_t>(col)] = max;
    }

    std::vector<double> s(static_cast<std::size_t>(n) * n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            s[static_cast<std::size_t>(row) * n + col] =
                a[static_cast<std::size_t>(row) * n + col] /
                (row_scale[static_cast<std::size_t>(row)] *
                 col_scale[static_cast<std::size_t>(col)]);
        }
    }
    const std::vector<double> s_inv = invert_equilibrated(s, n);
    std::vector<double> inv(static_cast<std::size_t>(n) * n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            inv[static_cast<std::size_t>(row) * n + col] =
                s_inv[static_cast<std::size_t>(row) * n + col] /
                (col_scale[static_cast<std::size_t>(row)] *
                 row_scale[static_cast<std::size_t>(col)]);
        }
    }

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
        x[static_cast<std::size_t>(row)] = inv[static_cast<std::size_t>(row) * n];
    }
    double residual = inverse_residual(x);
    for (int step = 0; step < 2 && residual > 1e-14; ++step) {
        const std::vector<double> correction =
            mat_vec(inv, inverse_error_vector(x), n);
        std::vector<double> candidate = x;
        for (int row = 0; row < n; ++row) {
            candidate[static_cast<std::size_t>(row)] -=
                correction[static_cast<std::size_t>(row)];
        }
        const double candidate_residual = inverse_residual(candidate);
        if (!(candidate_residual < residual)) {
            break;
        }
        x = std::move(candidate);
        residual = candidate_residual;
    }

    const double norm_a = max_row_sum(a, n);
    const double norm_inv = max_row_sum(inv, n);
    const double condition = norm_a * norm_inv;
    const double scale = norm_a * max_abs(x) + 1.0;
    const double relative_residual = residual / scale;
    if (!std::isfinite(relative_residual) || !std::isfinite(condition)) {
        throw std::domain_error("inverse is not numerically finite");
    }
    return GeneralInverse{Multivector(sig_, std::move(x)), residual, relative_residual,
                          condition};
}

inline std::vector<double> Multivector::invert_equilibrated(const std::vector<double>& a,
                                                            int n) {
    std::vector<double> m(static_cast<std::size_t>(n) * 2 * n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            m[static_cast<std::size_t>(row) * 2 * n + col] =
                a[static_cast<std::size_t>(row) * n + col];
        }
        m[static_cast<std::size_t>(row) * 2 * n + n + row] = 1.0;
    }
    const double pivot_tolerance = std::numeric_limits<double>::epsilon() * n;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(m[static_cast<std::size_t>(row) * 2 * n + col]) >
                std::abs(m[static_cast<std::size_t>(pivot) * 2 * n + col])) {
                pivot = row;
            }
        }
        if (!(std::abs(m[static_cast<std::size_t>(pivot) * 2 * n + col]) >
              pivot_tolerance)) {
            throw std::domain_error("element is not invertible");
        }
        if (pivot != col) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(m[static_cast<std::size_t>(col) * 2 * n + j],
                          m[static_cast<std::size_t>(pivot) * 2 * n + j]);
            }
        }
        const double p = m[static_cast<std::size_t>(col) * 2 * n + col];
        for (int j = col; j < 2 * n; ++j) {
            m[static_cast<std::size_t>(col) * 2 * n + j] /= p;
        }
        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double f = m[static_cast<std::size_t>(row) * 2 * n + col];
            if (f == 0.0) {
                continue;
            }
            for (int j = col; j < 2 * n; ++j) {
                m[static_cast<std::size_t>(row) * 2 * n + j] -=
                    f * m[static_cast<std::size_t>(col) * 2 * n + j];
            }
        }
    }
    std::vector<double> inv(static_cast<std::size_t>(n) * n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            inv[static_cast<std::size_t>(row) * n + col] =
                m[static_cast<std::size_t>(row) * 2 * n + n + col];
        }
    }
    return inv;
}

inline double Multivector::inverse_residual(const std::vector<double>& x) const {
    return max_abs(inverse_error_vector(x));
}

inline std::vector<double> Multivector::inverse_error_vector(
    const std::vector<double>& x) const {
    const Multivector ax = gp(Multivector(sig_, x));
    std::vector<double> error = ax.c_;
    error[0] -= 1.0;
    return error;
}

inline std::vector<double> Multivector::mat_vec(const std::vector<double>& matrix,
                                                const std::vector<double>& vector, int n) {
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += matrix[static_cast<std::size_t>(row) * n + col] *
                   vector[static_cast<std::size_t>(col)];
        }
        out[static_cast<std::size_t>(row)] = sum;
    }
    return out;
}

inline double Multivector::max_row_sum(const std::vector<double>& matrix, int n) {
    double max = 0.0;
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += std::abs(matrix[static_cast<std::size_t>(row) * n + col]);
        }
        max = std::max(max, sum);
    }
    return max;
}

inline double Multivector::max_abs(const std::vector<double>& values) {
    double max = 0.0;
    for (double value : values) {
        max = std::max(max, std::abs(value));
    }
    return max;
}

inline Multivector operator+(Multivector a, const Multivector& b) { return a.add(b); }
inline Multivector operator-(Multivector a, const Multivector& b) { return a.sub(b); }
inline Multivector operator-(Multivector a) { return a.neg(); }
inline Multivector operator*(const Multivector& a, const Multivector& b) { return a.gp(b); }
inline Multivector operator*(Multivector a, double s) { return a.scale(s); }
inline Multivector operator*(double s, Multivector a) { return a.scale(s); }
inline Multivector operator/(Multivector a, double s) { return a.divide(s); }

inline std::ostream& operator<<(std::ostream& os, const Multivector& value) {
    return os << value.to_string();
}

} // namespace vsr::runtime
