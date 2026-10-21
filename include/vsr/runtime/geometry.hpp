#pragma once

#include <vsr/runtime/multivector.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vsr::runtime {

/** 3D Euclidean GA (quaternion / rotor model). */
struct E3 {
    inline static const Signature SIG = Signature::e3();

    [[nodiscard]] static Multivector vec(double x, double y, double z) {
        return Multivector::vector(SIG, {x, y, z});
    }

    /** Bivector for an axis: dual(a) = a1 e23 + a2 e31 + a3 e12. */
    [[nodiscard]] static Multivector bivector(double ax, double ay, double az) {
        return Multivector::blade(SIG, 0b110, ax)
            .add(Multivector::blade(SIG, 0b101, -ay))
            .add(Multivector::blade(SIG, 0b011, az));
    }

    /** R = exp(-theta/2 * B) for the unit axis (ax, ay, az). */
    [[nodiscard]] static Multivector rotor(double ax, double ay, double az, double angle) {
        const double len = std::sqrt(ax * ax + ay * ay + az * az);
        if (len == 0.0) {
            throw std::invalid_argument("zero axis");
        }
        const Multivector b = bivector(ax / len, ay / len, az / len);
        return Multivector::scalar(SIG, std::cos(angle / 2.0))
            .sub(b.scale(std::sin(angle / 2.0)));
    }

    [[nodiscard]] static Multivector rotate(const Multivector& x, const Multivector& rotor) {
        return rotor.sandwich(x);
    }

    [[nodiscard]] static std::array<double, 3> coords(const Multivector& v) {
        return {v.get(1), v.get(2), v.get(4)};
    }
};

/**
 * Rotor and motor constructors following versor's Gen helpers.
 * Rotors are exponentials of bivectors, so no angle bookkeeping is hidden in
 * the type.
 */
struct Gen {
    /** exp(B) for a bivector B (simple or null), closed form. */
    [[nodiscard]] static Multivector exp_bivector(const Multivector& b) {
        const double b2 = b.gp(b).scalar_part();
        if (b2 < -1e-14) {
            const double r = std::sqrt(-b2);
            return Multivector::scalar(b.signature(), std::cos(r))
                .add(b.scale(std::sin(r) / r));
        }
        if (b2 > 1e-14) {
            const double r = std::sqrt(b2);
            return Multivector::scalar(b.signature(), std::cosh(r))
                .add(b.scale(std::sinh(r) / r));
        }
        return Multivector::scalar(b.signature(), 1.0).add(b);
    }

    [[nodiscard]] static Multivector rotor(const Multivector& bivector) {
        return exp_bivector(bivector);
    }

    [[nodiscard]] static Multivector log(const Multivector& rotor) {
        const Multivector b = rotor.grade(2);
        const double s = rotor.scalar_part();
        const double n = b.norm();
        if (n < 1e-14) {
            return s < 0.0 ? pi_bivector(rotor.signature()) : b;
        }
        const double angle = b.gp(b).scalar_part() > 1e-14 ? std::log(s + n)
                                                           : std::atan2(n, s);
        return b.scale(angle / n);
    }

    [[nodiscard]] static Multivector slerp(const Multivector& a, const Multivector& b,
                                           double t) {
        const Multivector relative = a.reverse().gp(b);
        return a.gp(exp_bivector(log(relative).scale(t)));
    }

    [[nodiscard]] static Multivector apply(const Multivector& versor,
                                           const Multivector& x) {
        return versor.sandwich(x);
    }

private:
    [[nodiscard]] static Multivector pi_bivector(const Signature& sig) {
        for (int m = 0; m < sig.size(); ++m) {
            if (sig.grade(m) != 2) {
                continue;
            }
            const int lo = std::countr_zero(static_cast<unsigned>(m));
            const int hi = std::bit_width(static_cast<unsigned>(m)) - 1;
            if (sig.metric(lo) == 1 && sig.metric(hi) == 1) {
                return Multivector::blade(sig, m, std::numbers::pi);
            }
        }
        throw std::domain_error("no Euclidean bivector in this algebra");
    }
};

/**
 * 3D projective GA for graphics: points are trivectors, planes are vectors,
 * lines are bivectors, rigid motions are motors.
 */
struct P3 {
    inline static const Signature SIG = Signature::pga3();
    inline static const Multivector I = Multivector::blade(SIG, 0b1111, 1.0);

    static constexpr int E01 = 0b0011;
    static constexpr int E02 = 0b0101;
    static constexpr int E03 = 0b1001;
    static constexpr int E12 = 0b0110;
    static constexpr int E13 = 0b1010;
    static constexpr int E23 = 0b1100;
    static constexpr int E123 = 0b1110;
    static constexpr int E012 = 0b0111;
    static constexpr int E013 = 0b1011;
    static constexpr int E023 = 0b1101;

    [[nodiscard]] static Multivector point(double x, double y, double z) {
        return Multivector::blade(SIG, E123, 1.0)
            .add(Multivector::blade(SIG, E012, -z))
            .add(Multivector::blade(SIG, E013, y))
            .add(Multivector::blade(SIG, E023, -x));
    }

    [[nodiscard]] static std::array<double, 3> point_coords(const Multivector& p) {
        check_pga(p);
        const double w = p.get(E123);
        if (std::abs(w) < 1e-12) {
            throw std::invalid_argument("point at infinity");
        }
        return {-p.get(E023) / w, p.get(E013) / w, -p.get(E012) / w};
    }

    [[nodiscard]] static Multivector plane(double nx, double ny, double nz, double d) {
        return Multivector::vector(SIG, {d, nx, ny, nz});
    }

    [[nodiscard]] static std::array<double, 4> plane_coords(const Multivector& plane) {
        check_pga(plane);
        return {plane.get(1 << 1), plane.get(1 << 2), plane.get(1 << 3), plane.get(1)};
    }

    [[nodiscard]] static Multivector translator(double tx, double ty, double tz) {
        return Multivector::scalar(SIG, 1.0)
            .sub(Multivector::blade(SIG, E01, 0.5 * tx))
            .sub(Multivector::blade(SIG, E02, 0.5 * ty))
            .sub(Multivector::blade(SIG, E03, 0.5 * tz));
    }

    [[nodiscard]] static Multivector rotor(double ax, double ay, double az, double angle) {
        const double len = std::sqrt(ax * ax + ay * ay + az * az);
        if (len == 0.0) {
            throw std::invalid_argument("zero axis");
        }
        const Multivector b = Multivector::blade(SIG, E23, ax / len)
            .add(Multivector::blade(SIG, E13, -ay / len))
            .add(Multivector::blade(SIG, E12, az / len));
        return Multivector::scalar(SIG, std::cos(angle / 2.0))
            .sub(b.scale(std::sin(angle / 2.0)));
    }

    [[nodiscard]] static Multivector motor(const Multivector& rotor,
                                           const Multivector& translator) {
        return translator.gp(rotor);
    }

    [[nodiscard]] static Multivector apply(const Multivector& motor,
                                           const Multivector& x) {
        return motor.sandwich(x);
    }

    [[nodiscard]] static Multivector line(const Multivector& p1, const Multivector& p2) {
        return p1.vee(p2);
    }

    [[nodiscard]] static std::array<double, 3> line_direction(const Multivector& line) {
        check_pga(line);
        return {line.get(E23), -line.get(E13), line.get(E12)};
    }

    [[nodiscard]] static std::array<double, 3> line_moment(const Multivector& line) {
        check_pga(line);
        return {line.get(E01), line.get(E02), line.get(E03)};
    }

    [[nodiscard]] static std::array<double, 3> line_anchor(const Multivector& line) {
        const auto d = line_direction(line);
        const double d2 = dot(d, d);
        if (d2 < 1e-24) {
            throw std::invalid_argument("degenerate line");
        }
        return scale(cross(d, line_moment(line)), 1.0 / d2);
    }

    [[nodiscard]] static Multivector closest_point_on_line(const Multivector& line,
                                                           const Multivector& query) {
        const auto p = point_coords(query);
        const auto d = line_direction(line);
        const double d2 = dot(d, d);
        if (d2 < 1e-24) {
            throw std::invalid_argument("degenerate line");
        }
        const auto a = line_anchor(line);
        const double t = dot(sub(p, a), d) / d2;
        return point(a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]);
    }

    [[nodiscard]] static double distance_point_line(const Multivector& line,
                                                    const Multivector& point_value) {
        return norm(sub(point_coords(point_value),
                        point_coords(closest_point_on_line(line, point_value))));
    }

    [[nodiscard]] static double signed_distance_point_plane(const Multivector& plane,
                                                            const Multivector& query) {
        const auto pi = plane_coords(plane);
        const double n2 = pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2];
        if (n2 < 1e-24) {
            throw std::invalid_argument("degenerate plane");
        }
        const auto p = point_coords(query);
        return (pi[0] * p[0] + pi[1] * p[1] + pi[2] * p[2] + pi[3]) / std::sqrt(n2);
    }

    [[nodiscard]] static double distance_point_plane(const Multivector& plane,
                                                     const Multivector& query) {
        return std::abs(signed_distance_point_plane(plane, query));
    }

    [[nodiscard]] static Multivector closest_point_on_plane(const Multivector& plane,
                                                            const Multivector& query) {
        const auto pi = plane_coords(plane);
        const double n2 = pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2];
        if (n2 < 1e-24) {
            throw std::invalid_argument("degenerate plane");
        }
        const auto p = point_coords(query);
        const double t =
            (pi[0] * p[0] + pi[1] * p[1] + pi[2] * p[2] + pi[3]) / n2;
        return point(p[0] - t * pi[0], p[1] - t * pi[1], p[2] - t * pi[2]);
    }

    [[nodiscard]] static Multivector meet_planes(const Multivector& pi1,
                                                 const Multivector& pi2) {
        return pi1.op(pi2);
    }

    [[nodiscard]] static Multivector meet_plane_line(const Multivector& plane,
                                                     const Multivector& line) {
        return plane.op(line);
    }

    [[nodiscard]] static Multivector point_at_infinity(double x, double y, double z) {
        return Multivector::blade(SIG, E012, -z)
            .add(Multivector::blade(SIG, E013, y))
            .add(Multivector::blade(SIG, E023, -x));
    }

    [[nodiscard]] static std::array<double, 3> point_at_infinity_coords(
        const Multivector& p) {
        return {-p.get(E023), p.get(E013), -p.get(E012)};
    }

    [[nodiscard]] static std::array<double, 3> translation_of(const Multivector& motor_value) {
        return point_coords(apply(motor_value, point(0, 0, 0)));
    }

    [[nodiscard]] static std::array<std::array<double, 3>, 3> rotation_matrix(
        const Multivector& motor_value) {
        std::array<std::array<double, 3>, 3> m{};
        const std::array<Multivector, 3> dirs = {
            point_at_infinity(1, 0, 0), point_at_infinity(0, 1, 0),
            point_at_infinity(0, 0, 1)};
        for (int col = 0; col < 3; ++col) {
            const auto c = point_at_infinity_coords(apply(motor_value, dirs[col]));
            for (int row = 0; row < 3; ++row) {
                m[row][col] = c[row];
            }
        }
        return m;
    }

    [[nodiscard]] static Multivector rotation_of(const Multivector& motor_value) {
        return rotor_from_matrix(rotation_matrix(motor_value));
    }

    /** Shepperd quaternion extraction; returns w - (x e23 - y e13 + z e12). */
    [[nodiscard]] static Multivector rotor_from_matrix(
        const std::array<std::array<double, 3>, 3>& m) {
        const double tr = m[0][0] + m[1][1] + m[2][2];
        double w = 0.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        if (tr > 0.0) {
            const double s = std::sqrt(tr + 1.0) * 2.0;
            w = 0.25 * s;
            x = (m[2][1] - m[1][2]) / s;
            y = (m[0][2] - m[2][0]) / s;
            z = (m[1][0] - m[0][1]) / s;
        } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
            const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
            w = (m[2][1] - m[1][2]) / s;
            x = 0.25 * s;
            y = (m[0][1] + m[1][0]) / s;
            z = (m[0][2] + m[2][0]) / s;
        } else if (m[1][1] > m[2][2]) {
            const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
            w = (m[0][2] - m[2][0]) / s;
            x = (m[0][1] + m[1][0]) / s;
            y = 0.25 * s;
            z = (m[1][2] + m[2][1]) / s;
        } else {
            const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
            w = (m[1][0] - m[0][1]) / s;
            x = (m[0][2] + m[2][0]) / s;
            y = (m[1][2] + m[2][1]) / s;
            z = 0.25 * s;
        }
        return Multivector::scalar(SIG, w)
            .sub(Multivector::blade(SIG, E23, x))
            .add(Multivector::blade(SIG, E13, y))
            .sub(Multivector::blade(SIG, E12, z));
    }

    [[nodiscard]] static Multivector slerp(const Multivector& a, const Multivector& b,
                                           double t) {
        const Multivector r = Gen::slerp(rotation_of(a), rotation_of(b), t);
        const auto ta = translation_of(a);
        const auto tb = translation_of(b);
        return motor(r, translator(ta[0] + (tb[0] - ta[0]) * t,
                                   ta[1] + (tb[1] - ta[1]) * t,
                                   ta[2] + (tb[2] - ta[2]) * t));
    }

private:
    static void check_pga(const Multivector& value) {
        if (value.signature() != SIG) {
            throw std::invalid_argument("not a PGA3 multivector");
        }
    }

    static double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    static std::array<double, 3> cross(const std::array<double, 3>& a,
                                       const std::array<double, 3>& b) {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]};
    }

    static std::array<double, 3> sub(const std::array<double, 3>& a,
                                     const std::array<double, 3>& b) {
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }

    static std::array<double, 3> scale(const std::array<double, 3>& a, double s) {
        return {a[0] * s, a[1] * s, a[2] * s};
    }

    static double norm(const std::array<double, 3>& a) {
        return std::sqrt(dot(a, a));
    }
};

/**
 * Conformal model of 3D Euclidean space in Cl(4,1).
 *
 * n0 = (e5 - e4)/2, ninf = e4 + e5, n0.ninf = -1,
 * P(x) = x + |x|^2/2 ninf + n0.
 */
struct Conformal {
    inline static const Signature SIG = Signature::cga3();
    inline static const Multivector NINF =
        Multivector::blade(SIG, 8, 1.0).add(Multivector::blade(SIG, 16, 1.0));
    inline static const Multivector N0 =
        Multivector::blade(SIG, 16, 0.5).sub(Multivector::blade(SIG, 8, 0.5));

    static constexpr int E1 = 1;
    static constexpr int E2 = 2;
    static constexpr int E3 = 4;
    static constexpr int E4 = 8;
    static constexpr int E5 = 16;

    [[nodiscard]] static Multivector point(double x, double y, double z) {
        const double half = 0.5 * (x * x + y * y + z * z);
        return Multivector::vector(SIG, {x, y, z, 0.0, 0.0})
            .add(NINF.scale(half))
            .add(N0);
    }

    [[nodiscard]] static std::array<double, 3> point_coords(const Multivector& p) {
        const double w = p.get(E5) - p.get(E4);
        if (std::abs(w) < 1e-12) {
            throw std::invalid_argument("point at infinity");
        }
        return {p.get(E1) / w, p.get(E2) / w, p.get(E3) / w};
    }

    [[nodiscard]] static Multivector sphere(double cx, double cy, double cz,
                                            double radius) {
        return point(cx, cy, cz).sub(NINF.scale(0.5 * radius * radius));
    }

    [[nodiscard]] static Multivector plane(double nx, double ny, double nz, double d) {
        return Multivector::vector(SIG, {nx, ny, nz, 0.0, 0.0})
            .add(NINF.scale(-d));
    }

    [[nodiscard]] static std::array<double, 4> sphere_coords(const Multivector& sphere) {
        check(sphere);
        const double w = sphere.get(E5) - sphere.get(E4);
        if (std::abs(w) < 1e-12) {
            throw std::invalid_argument("sphere at infinity");
        }
        std::array<double, 3> c = {sphere.get(E1) / w, sphere.get(E2) / w,
                                   sphere.get(E3) / w};
        const double r2 =
            c[0] * c[0] + c[1] * c[1] + c[2] * c[2] -
            (sphere.get(E4) + sphere.get(E5)) / w;
        if (r2 < -1e-9) {
            throw std::invalid_argument("sphere has negative radius squared");
        }
        return {c[0], c[1], c[2], std::sqrt(std::max(0.0, r2))};
    }

    [[nodiscard]] static std::array<double, 4> plane_coords(const Multivector& plane) {
        check(plane);
        return {plane.get(E1), plane.get(E2), plane.get(E3),
                -0.5 * (plane.get(E4) + plane.get(E5))};
    }

    [[nodiscard]] static double signed_distance_point_plane(const Multivector& plane,
                                                            const Multivector& query) {
        const auto pi = plane_coords(plane);
        const auto p = point_coords(query);
        const double n2 = pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2];
        if (n2 < 1e-24) {
            throw std::invalid_argument("degenerate plane");
        }
        return (pi[0] * p[0] + pi[1] * p[1] + pi[2] * p[2] + pi[3]) / std::sqrt(n2);
    }

    [[nodiscard]] static double distance_point_plane(const Multivector& plane,
                                                     const Multivector& query) {
        return std::abs(signed_distance_point_plane(plane, query));
    }

    [[nodiscard]] static Multivector closest_point_on_plane(const Multivector& plane,
                                                            const Multivector& query) {
        const auto pi = plane_coords(plane);
        const auto p = point_coords(query);
        const double n2 = pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2];
        if (n2 < 1e-24) {
            throw std::invalid_argument("degenerate plane");
        }
        const double t =
            (pi[0] * p[0] + pi[1] * p[1] + pi[2] * p[2] + pi[3]) / n2;
        return point(p[0] - t * pi[0], p[1] - t * pi[1], p[2] - t * pi[2]);
    }

    [[nodiscard]] static double distance_point_sphere(const Multivector& sphere,
                                                      const Multivector& query) {
        const auto s = sphere_coords(sphere);
        const auto p = point_coords(query);
        const double dx = p[0] - s[0];
        const double dy = p[1] - s[1];
        const double dz = p[2] - s[2];
        return std::abs(std::sqrt(dx * dx + dy * dy + dz * dz) - s[3]);
    }

    [[nodiscard]] static Multivector closest_point_on_sphere(const Multivector& sphere,
                                                             const Multivector& query) {
        const auto s = sphere_coords(sphere);
        const auto p = point_coords(query);
        const double dx = p[0] - s[0];
        const double dy = p[1] - s[1];
        const double dz = p[2] - s[2];
        const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (length < 1e-12) {
            return point(s[0] + s[3], s[1], s[2]);
        }
        const double scale = s[3] / length;
        return point(s[0] + dx * scale, s[1] + dy * scale, s[2] + dz * scale);
    }

    [[nodiscard]] static Multivector circle(double cx, double cy, double cz,
                                            double nx, double ny, double nz,
                                            double radius) {
        const double length = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (length < 1e-12) {
            throw std::invalid_argument("zero normal");
        }
        const double ux = nx / length;
        const double uy = ny / length;
        const double uz = nz / length;
        const double hx = std::abs(ux) < 0.9 ? 1.0 : 0.0;
        const double hy = std::abs(ux) < 0.9 ? 0.0 : 1.0;
        const double hz = 0.0;
        double ax = hy * uz - hz * uy;
        double ay = hz * ux - hx * uz;
        double az = hx * uy - hy * ux;
        const double a_length = std::sqrt(ax * ax + ay * ay + az * az);
        ax /= a_length;
        ay /= a_length;
        az /= a_length;
        const double bx = uy * az - uz * ay;
        const double by = uz * ax - ux * az;
        const double bz = ux * ay - uy * ax;
        const Multivector p1 = point(cx + radius * ax, cy + radius * ay, cz + radius * az);
        const Multivector p2 = point(cx + radius * bx, cy + radius * by, cz + radius * bz);
        const Multivector p3 = point(cx - radius * ax, cy - radius * ay, cz - radius * az);
        return p1.op(p2).op(p3);
    }

    [[nodiscard]] static double dot(const Multivector& a, const Multivector& b) {
        return a.gp(b).scalar_part();
    }

    [[nodiscard]] static double distance_sq(const Multivector& a, const Multivector& b) {
        return -2.0 * dot(a, b);
    }

private:
    static void check(const Multivector& m) {
        if (m.signature() != SIG) {
            throw std::invalid_argument("expected conformal Cl(4,1) multivector");
        }
    }
};

/**
 * Rigid-body kinematics in PGA3: M(t) = exp(t * twist).
 */
class Motion {
public:
    explicit Motion(Multivector twist)
        : twist_(std::move(twist)),
          twist2_(twist_.gp(twist_)),
          twist3_(twist2_.gp(twist_)) {
        if (twist_.grade(2).distance_to(twist_) > 1e-12) {
            throw std::invalid_argument("twist must be a bivector");
        }
    }

    [[nodiscard]] static Motion rotation(double ax, double ay, double az, double omega) {
        Multivector unit = Gen::log(P3::rotor(ax, ay, az, 1.0));
        return Motion(unit.scale(omega));
    }

    [[nodiscard]] static Motion translation(double vx, double vy, double vz) {
        return Motion(P3::translator(vx, vy, vz)
                          .sub(Multivector::scalar(P3::SIG, 1.0)));
    }

    [[nodiscard]] static Motion screw(double ax, double ay, double az, double omega,
                                      double vx, double vy, double vz) {
        const Multivector rot = Gen::log(P3::rotor(ax, ay, az, 1.0)).scale(omega);
        const Multivector trans =
            P3::translator(vx, vy, vz).sub(Multivector::scalar(P3::SIG, 1.0));
        return Motion(rot.add(trans));
    }

    [[nodiscard]] const Multivector& twist() const noexcept { return twist_; }

    [[nodiscard]] Multivector motor(double t) const {
        return twist_.scale(t).exp();
    }

    [[nodiscard]] Multivector point_at(double t, double x, double y, double z) const {
        return P3::apply(motor(t), P3::point(x, y, z));
    }

    [[nodiscard]] std::array<double, 3> position(double t, double x, double y,
                                                 double z) const {
        return P3::point_coords(point_at(t, x, y, z));
    }

    [[nodiscard]] std::array<double, 3> velocity(double t, double x, double y,
                                                 double z) const {
        const Multivector p = point_at(t, x, y, z);
        return P3::point_at_infinity_coords(commutator(twist_, p));
    }

    [[nodiscard]] std::array<double, 3> acceleration(double t, double x, double y,
                                                     double z) const {
        const Multivector p = point_at(t, x, y, z);
        const Multivector a = twist2_.gp(p).add(p.gp(twist2_))
            .sub(twist_.gp(p).gp(twist_).scale(2.0));
        return P3::point_at_infinity_coords(a);
    }

    [[nodiscard]] std::array<double, 3> jerk(double t, double x, double y,
                                             double z) const {
        const Multivector p = point_at(t, x, y, z);
        const Multivector j = twist3_.gp(p)
            .sub(twist2_.gp(p).gp(twist_).scale(3.0))
            .add(twist_.gp(p).gp(twist2_).scale(3.0))
            .sub(p.gp(twist3_));
        return P3::point_at_infinity_coords(j);
    }

private:
    Multivector twist_;
    Multivector twist2_;
    Multivector twist3_;

    [[nodiscard]] static Multivector commutator(const Multivector& a,
                                                const Multivector& b) {
        return a.gp(b).sub(b.gp(a));
    }
};

/** A multivector-valued field on R^3: F(x, y, z). */
class Field {
public:
    using Fn = std::function<Multivector(double, double, double)>;

    Field() = default;
    Field(Fn fn) : fn_(std::move(fn)) {}

    template <typename F>
        requires std::constructible_from<Fn, F>
    Field(F&& fn) : fn_(std::forward<F>(fn)) {}

    [[nodiscard]] Multivector at(double x, double y, double z) const {
        return fn_(x, y, z);
    }

    [[nodiscard]] Multivector at(const std::array<double, 3>& p) const {
        return at(p[0], p[1], p[2]);
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(fn_);
    }

private:
    Fn fn_;
};

/**
 * Numerical geometric calculus on R^3 (Cl(3,0)).
 *
 * nabla F = e1 dF/dx + e2 dF/dy + e3 dF/dz.  Its scalar part is divergence
 * and its bivector part encodes curl.
 */
struct Calculus {
    static constexpr double STEP = 1e-4;
    inline static const Multivector E1 = Multivector::blade(E3::SIG, 0b001, 1.0);
    inline static const Multivector E2 = Multivector::blade(E3::SIG, 0b010, 1.0);
    inline static const Multivector E3B = Multivector::blade(E3::SIG, 0b100, 1.0);
    inline static const Multivector I = Multivector::blade(E3::SIG, 0b111, 1.0);

    [[nodiscard]] static Multivector partial(const Field& f, double x, double y,
                                             double z, int axis) {
        return partial(f, x, y, z, axis, STEP);
    }

    [[nodiscard]] static Multivector partial(const Field& f, double x, double y,
                                             double z, int axis, double h) {
        if (axis < 0 || axis > 2) {
            throw std::invalid_argument("axis must be 0, 1 or 2");
        }
        std::array<double, 3> p = {x, y, z};
        const std::array<double, 3> a = p;
        p[static_cast<std::size_t>(axis)] += h;
        const Multivector plus = f.at(p[0], p[1], p[2]);
        p[static_cast<std::size_t>(axis)] = a[static_cast<std::size_t>(axis)] - h;
        const Multivector minus = f.at(p[0], p[1], p[2]);
        return plus.sub(minus).scale(0.5 / h);
    }

    [[nodiscard]] static Multivector directional(const Field& f, double x, double y,
                                                  double z, double vx, double vy,
                                                  double vz) {
        const double len = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (len == 0.0) {
            throw std::invalid_argument("zero direction");
        }
        const double ux = vx / len;
        const double uy = vy / len;
        const double uz = vz / len;
        const Multivector plus = f.at(x + STEP * ux, y + STEP * uy, z + STEP * uz);
        const Multivector minus = f.at(x - STEP * ux, y - STEP * uy, z - STEP * uz);
        return plus.sub(minus).scale(0.5 / STEP);
    }

    [[nodiscard]] static Multivector vector_derivative(const Field& f, double x,
                                                        double y, double z) {
        const Multivector dx = partial(f, x, y, z, 0);
        const Multivector dy = partial(f, x, y, z, 1);
        const Multivector dz = partial(f, x, y, z, 2);
        return E1.gp(dx).add(E2.gp(dy)).add(E3B.gp(dz));
    }

    [[nodiscard]] static Multivector gradient(const Field& f, double x, double y,
                                               double z) {
        return vector_derivative(f, x, y, z).grade(1);
    }

    [[nodiscard]] static double divergence(const Field& f, double x, double y,
                                           double z) {
        return vector_derivative(f, x, y, z).scalar_part();
    }

    [[nodiscard]] static double flux(const Field& f, const class Manifold& surface,
                                     double u0, double u1, double v0, double v1,
                                     int nu, int nv);

    [[nodiscard]] static Multivector curl(const Field& f, double x, double y,
                                           double z) {
        const Multivector bivector = vector_derivative(f, x, y, z).grade(2);
        return I.gp(bivector).neg();
    }
};

/**
 * An embedded submanifold of R^3 given by a parameterization.
 */
class Manifold {
public:
    using Param = std::function<std::vector<double>(const std::vector<double>&)>;

    static Manifold curve(Param param) { return Manifold(1, std::move(param)); }
    static Manifold surface(Param param) { return Manifold(2, std::move(param)); }

    [[nodiscard]] int dimension() const noexcept { return dimension_; }

    [[nodiscard]] std::vector<double> point(
        std::initializer_list<double> u) const {
        return point(std::vector<double>(u));
    }

    [[nodiscard]] std::vector<double> point(const std::vector<double>& u) const {
        return param_(u);
    }

    /** Partial derivative of the parameterization, as a vector. */
    [[nodiscard]] Multivector partial(const std::vector<double>& u, int axis) const {
        if (axis < 0 || axis >= dimension_) {
            throw std::invalid_argument("bad axis");
        }
        std::vector<double> plus = u;
        std::vector<double> minus = u;
        plus[static_cast<std::size_t>(axis)] += H;
        minus[static_cast<std::size_t>(axis)] -= H;
        const auto a = param_(plus);
        const auto b = param_(minus);
        return E3::vec((a[0] - b[0]) / (2.0 * H), (a[1] - b[1]) / (2.0 * H),
                       (a[2] - b[2]) / (2.0 * H));
    }

    [[nodiscard]] Multivector tangent_blade(const std::vector<double>& u) const {
        const Multivector t = partial(u, 0);
        if (dimension_ == 1) {
            return t;
        }
        return t.op(partial(u, 1));
    }

    [[nodiscard]] Multivector tangent_blade(std::initializer_list<double> u) const {
        return tangent_blade(std::vector<double>(u));
    }

    [[nodiscard]] double jacobian(const std::vector<double>& u) const {
        return dimension_ == 1 ? partial(u, 0).norm() : tangent_blade(u).norm();
    }

    [[nodiscard]] double jacobian(std::initializer_list<double> u) const {
        return jacobian(std::vector<double>(u));
    }

    [[nodiscard]] Multivector normal(const std::vector<double>& u) const {
        if (dimension_ != 2) {
            throw std::logic_error("normal is defined for surfaces");
        }
        const Multivector blade = I.gp(tangent_blade(u)).neg().grade(1);
        return blade.unit();
    }

    [[nodiscard]] Multivector normal(std::initializer_list<double> u) const {
        return normal(std::vector<double>(u));
    }

    [[nodiscard]] Multivector tangent(const std::vector<double>& u) const {
        if (dimension_ != 1) {
            throw std::logic_error("tangent is defined for curves");
        }
        return partial(u, 0).unit();
    }

    [[nodiscard]] Multivector tangent(std::initializer_list<double> u) const {
        return tangent(std::vector<double>(u));
    }

private:
    static constexpr double H = 1e-5;
    inline static const Multivector I = Multivector::blade(E3::SIG, 0b111, 1.0);

    Manifold(int dimension, Param param)
        : dimension_(dimension), param_(std::move(param)) {}

    int dimension_;
    Param param_;
};

inline double Calculus::flux(const Field& f, const Manifold& surface, double u0,
                             double u1, double v0, double v1, int nu, int nv) {
    if (surface.dimension() != 2) {
        throw std::invalid_argument("flux needs a surface");
    }
    if (nu < 1 || nv < 1) {
        throw std::invalid_argument("need at least one cell");
    }
    const double du = (u1 - u0) / nu;
    const double dv = (v1 - v0) / nv;
    double sum = 0.0;
    for (int i = 0; i < nu; ++i) {
        const double u = u0 + (i + 0.5) * du;
        for (int j = 0; j < nv; ++j) {
            const double v = v0 + (j + 0.5) * dv;
            const auto p = surface.point({u, v});
            const Multivector field = f.at(p[0], p[1], p[2]);
            const Multivector normal = surface.normal({u, v});
            sum += field.ip(normal).scalar_part() * surface.jacobian({u, v}) * du * dv;
        }
    }
    return sum;
}

} // namespace vsr::runtime
