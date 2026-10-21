#include <vsr/runtime.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace vsr::runtime;

namespace {

void expect_near(double actual, double expected, double tolerance = 1e-12) {
    EXPECT_NEAR(actual, expected, tolerance);
}

void expect_vector_near(const std::array<double, 3>& actual,
                        const std::array<double, 3>& expected,
                        double tolerance = 1e-12) {
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(actual[static_cast<std::size_t>(i)],
                    expected[static_cast<std::size_t>(i)], tolerance);
    }
}

} // namespace

TEST(RuntimeSignature, FactoriesAndNames) {
    const Signature sta = Signature::sta();
    EXPECT_EQ(sta.blade_product_sign(1, 1), 1);
    EXPECT_EQ(sta.blade_product_sign(2, 2), -1);
    EXPECT_EQ(sta.blade_name(1), "e0");

    const Signature cga = Signature::cga3();
    EXPECT_EQ(cga.blade_product_sign(1 << 4, 1 << 4), -1);
    EXPECT_EQ(cga.blade_name(1 << 4), "e5");

    const Signature mink = Signature::cl(3, 1);
    EXPECT_EQ(mink.n(), 4);
    EXPECT_EQ(mink.metric_copy(), (std::vector<int>{1, 1, 1, -1}));
    EXPECT_EQ(mink.blade_product_sign(1, 1), 1);
    EXPECT_EQ(mink.blade_product_sign(8, 8), -1);

    EXPECT_EQ(Signature::euclidean(2).blade_name(2), "e2");
    EXPECT_THROW((void)Signature::cl(-1, 2), std::invalid_argument);
    EXPECT_THROW((void)Signature::cl(8, 8), std::invalid_argument);
    EXPECT_THROW((void)Signature::euclidean(15), std::invalid_argument);
}

TEST(RuntimeMultivector, BasisAndProducts) {
    const Multivector e1 = Multivector::blade(E3::SIG, 1, 1.0);
    const Multivector e2 = Multivector::blade(E3::SIG, 2, 1.0);
    EXPECT_TRUE(e1.gp(e1).distance_to(Multivector::scalar(E3::SIG, 1.0)) < 1e-12);
    EXPECT_TRUE(e1.gp(e2).distance_to(e2.gp(e1).neg()) < 1e-12);
    EXPECT_TRUE(e1.gp(e2).gp(e2).distance_to(e1) < 1e-12);

    const Multivector a = E3::vec(1, 2, 3);
    const Multivector b = E3::vec(4, 5, 6);
    expect_near(a.scalar_product(b), 32.0);
    EXPECT_TRUE(a.divide(b).gp(b).distance_to(a) < 1e-12);

    const Multivector e12 = e1.gp(e2);
    EXPECT_TRUE(e12.scalar_product(e12) < 0.0);
    EXPECT_TRUE(e1.reflect(e1).distance_to(e1.neg()) < 1e-12);
    EXPECT_TRUE(e2.reflect(e1).distance_to(e2) < 1e-12);
}

TEST(RuntimeMultivector, ReverseDualAndValidation) {
    for (const Signature& sig : {E3::SIG, P3::SIG, Conformal::SIG, Signature::sta(),
                                 Signature::cl(2, 1)}) {
        for (int mask = 0; mask < sig.size(); ++mask) {
            const Multivector blade = Multivector::blade(sig, mask, 1.0);
            EXPECT_TRUE(blade.dual().undual().distance_to(blade) < 1e-12);
            EXPECT_TRUE(blade.undual().dual().distance_to(blade) < 1e-12);
        }
    }

    const Multivector v = E3::vec(1, 2, 3);
    EXPECT_TRUE(v.is_finite());
    EXPECT_FALSE(v.is_scalar());
    EXPECT_TRUE(v.is_homogeneous());
    EXPECT_TRUE(v.is_blade());
    EXPECT_TRUE(v.is_versor());
    EXPECT_TRUE(E3::rotor(0, 0, 1, 0.7).is_versor());
    EXPECT_FALSE(E3::rotor(0, 0, 1, 0.7).is_blade());
    EXPECT_FALSE(Multivector::scalar(E3::SIG, 1.0)
                     .add(E3::vec(1, 0, 0))
                     .is_versor());

    const Signature e4 = Signature::euclidean(4);
    const Multivector non_simple =
        Multivector::blade(e4, 0b0011, 1.0).add(Multivector::blade(e4, 0b1100, 1.0));
    EXPECT_FALSE(non_simple.is_blade());
}

TEST(RuntimeMultivector, GeneralInverse) {
    const Multivector arbitrary = Multivector::scalar(E3::SIG, 1.3)
        .add(E3::vec(0.4, -0.2, 0.7))
        .add(E3::bivector(0.1, 0.3, -0.2));
    const Multivector one = Multivector::scalar(E3::SIG, 1.0);
    EXPECT_TRUE(arbitrary.gp(arbitrary.inverse_general().value)
                    .distance_to(one) < 1e-12);
    EXPECT_TRUE(arbitrary.inverse_general().value.gp(arbitrary)
                    .distance_to(one) < 1e-12);

    const GeneralInverse info = arbitrary.inverse_general_result();
    EXPECT_LT(info.residual, 1e-12);
    EXPECT_LT(info.condition, 1e4);

    const Multivector tiny = arbitrary.scale(1e-20);
    const Multivector scaled_inv = arbitrary.inverse_general().value.scale(1e20);
    EXPECT_TRUE(tiny.inverse_general().value.distance_to(scaled_inv) /
                    scaled_inv.max_abs() < 1e-12);

    const Multivector singular = Multivector::scalar(E3::SIG, 0.5)
        .add(Multivector::blade(E3::SIG, 1, 0.5));
    EXPECT_THROW((void)singular.inverse_general(), std::domain_error);
}

TEST(RuntimeE3, RotorAndSlerp) {
    const Multivector e1 = E3::vec(1, 0, 0);
    const Multivector e2 = E3::vec(0, 1, 0);
    const Multivector r = E3::rotor(0, 0, 1, std::numbers::pi / 2);
    EXPECT_TRUE(E3::rotate(e1, r).distance_to(e2) < 1e-12);
    EXPECT_TRUE(E3::rotate(e2, r).distance_to(e1.neg()) < 1e-12);
    EXPECT_TRUE(r.gp(r.reverse())
                    .distance_to(Multivector::scalar(E3::SIG, 1.0)) < 1e-12);

    const Multivector a = E3::rotor(0, 0, 1, 0.0);
    const Multivector b = E3::rotor(0, 0, 1, std::numbers::pi / 2);
    const Multivector mid = Gen::slerp(a, b, 0.5);
    const Multivector expected = E3::rotor(0, 0, 1, std::numbers::pi / 4);
    EXPECT_TRUE(mid.distance_to(expected) < 1e-12);
    EXPECT_TRUE(Gen::exp_bivector(Gen::log(b)).distance_to(b) < 1e-12);

    const Multivector half_turn =
        Gen::slerp(Multivector::scalar(E3::SIG, 1.0),
                   Multivector::scalar(E3::SIG, -1.0), 0.5);
    EXPECT_TRUE(half_turn.distance_to(Multivector::blade(E3::SIG, 3, 1.0)) < 1e-12);
}

TEST(RuntimeP3, PointsTranslatorMotorAndQueries) {
    const Multivector p = P3::point(1, 2, 3);
    expect_vector_near(P3::point_coords(p), {1, 2, 3});

    const Multivector t = P3::translator(1, 2, 3);
    expect_vector_near(P3::point_coords(P3::apply(t, P3::point(0, 0, 0))), {1, 2, 3});
    expect_vector_near(P3::point_coords(P3::apply(t, P3::point(4, 5, 6))), {5, 7, 9});

    const Multivector r = P3::rotor(0, 0, 1, std::numbers::pi / 2);
    expect_vector_near(P3::point_coords(P3::apply(r, P3::point(1, 0, 0))), {0, 1, 0});
    const Multivector m = P3::motor(r, P3::translator(0, 2, 0));
    expect_vector_near(P3::point_coords(P3::apply(m, P3::point(1, 0, 0))), {0, 3, 0});
    expect_vector_near(P3::point_coords(P3::apply(m.inverse(), P3::apply(m, p))),
                       {1, 2, 3});

    const Multivector line =
        P3::line(P3::point(1, 2, 0), P3::point(1, 2, 1));
    expect_vector_near(P3::line_direction(line), {0, 0, 1});
    expect_vector_near(P3::line_moment(line), {2, -1, 0});
    expect_vector_near(P3::line_anchor(line), {1, 2, 0});
    expect_vector_near(P3::point_coords(P3::closest_point_on_line(
                           line, P3::point(0, 0, 0))), {1, 2, 0});
    expect_near(P3::distance_point_line(line, P3::point(0, 0, 0)), std::sqrt(5.0));

    const Multivector plane = P3::plane(0, 0, 1, -1);
    expect_near(P3::signed_distance_point_plane(plane, P3::point(0, 0, 0)), -1.0);
    expect_near(P3::distance_point_plane(plane, P3::point(0, 0, 3)), 2.0);
    expect_vector_near(P3::point_coords(P3::closest_point_on_plane(
                           plane, P3::point(0, 0, 0))), {0, 0, 1});

    const Multivector x = P3::plane(1, 0, 0, 0);
    const Multivector y = P3::plane(0, 1, 0, 0);
    const Multivector z_axis = P3::meet_planes(x, y);
    expect_vector_near(P3::point_coords(P3::meet_plane_line(
                           P3::plane(0, 0, 1, -2), z_axis)), {0, 0, 2});
    EXPECT_THROW((void)P3::line_anchor(Multivector::zero(P3::SIG)),
                 std::invalid_argument);
}

TEST(RuntimeConformal, PointsSpheresPlanesAndCircles) {
    const Multivector p = Conformal::point(1, 2, 3);
    expect_near(Conformal::dot(p, p), 0.0, 1e-9);
    expect_vector_near(Conformal::point_coords(p), {1, 2, 3}, 1e-9);
    expect_near(Conformal::distance_sq(Conformal::point(1, 2, 3),
                                       Conformal::point(4, 6, 3)), 25.0, 1e-9);

    const Multivector sphere = Conformal::sphere(1, 1, 1, 2);
    expect_near(Conformal::dot(sphere, sphere), 4.0, 1e-9);
    expect_near(Conformal::dot(Conformal::point(3, 1, 1), sphere), 0.0, 1e-9);
    expect_near(Conformal::dot(Conformal::point(1, 1, 1), sphere), 2.0, 1e-9);
    expect_near(Conformal::distance_point_sphere(sphere, Conformal::point(1, 1, 1)),
                2.0);

    const std::array<double, 4> sc = Conformal::sphere_coords(sphere);
    expect_near(sc[0], 1.0);
    expect_near(sc[1], 1.0);
    expect_near(sc[2], 1.0);
    expect_near(sc[3], 2.0);
    expect_vector_near(Conformal::point_coords(
                           Conformal::closest_point_on_sphere(
                               sphere, Conformal::point(1, 1, 1))), {3, 1, 1});

    const Multivector plane = Conformal::plane(0, 0, 1, -2);
    expect_near(Conformal::dot(Conformal::point(1, 1, 2), plane), 0.0, 1e-9);
    expect_near(Conformal::dot(Conformal::point(1, 1, 3), plane), 1.0, 1e-9);
    expect_near(Conformal::signed_distance_point_plane(
                    plane, Conformal::point(0, 0, 0)), -2.0);
    expect_vector_near(Conformal::point_coords(
                           Conformal::closest_point_on_plane(
                               plane, Conformal::point(0, 0, 0))), {0, 0, 2});

    const Multivector circle = Conformal::circle(0, 0, 2, 0, 0, 1, 1);
    EXPECT_TRUE(circle.grade(3).distance_to(circle) < 1e-12);
    EXPECT_LT(Conformal::point(1, 0, 2).op(circle).max_abs(), 1e-9);
    EXPECT_GT(Conformal::point(2, 0, 2).op(circle).max_abs(), 0.5);
}

TEST(RuntimeMotion, Kinematics) {
    const Motion spin = Motion::rotation(0, 0, 1, 1.0);
    for (double t : {0.0, 0.7, 1.2}) {
        const double c = std::cos(t);
        const double s = std::sin(t);
        expect_vector_near(spin.position(t, 1, 0, 0), {c, s, 0}, 1e-9);
        expect_vector_near(spin.velocity(t, 1, 0, 0), {-s, c, 0}, 1e-9);
        expect_vector_near(spin.acceleration(t, 1, 0, 0), {-c, -s, 0}, 1e-9);
        expect_vector_near(spin.jerk(t, 1, 0, 0), {s, -c, 0}, 1e-9);
    }

    const Motion drift = Motion::translation(2, 1, 0);
    expect_vector_near(drift.velocity(3.0, 1, 1, 1), {2, 1, 0});
    expect_vector_near(drift.acceleration(3.0, 1, 1, 1), {0, 0, 0});
    expect_vector_near(drift.position(3.0, 1, 1, 1), {7, 4, 1});

    const Motion screw = Motion::screw(0, 0, 1, 1.0, 0, 0, 1);
    const double t = 0.8;
    expect_vector_near(screw.position(t, 1, 0, 0),
                       {std::cos(t), std::sin(t), t}, 1e-9);

    const Motion general = Motion::screw(1, 2, 3, 0.8, 0.3, -0.2, 0.5);
    const double h = 1e-4;
    const auto plus = general.position(0.6 + h, 1, -1, 0.5);
    const auto minus = general.position(0.6 - h, 1, -1, 0.5);
    expect_vector_near(general.velocity(0.6, 1, -1, 0.5),
                       {(plus[0] - minus[0]) / (2 * h),
                        (plus[1] - minus[1]) / (2 * h),
                        (plus[2] - minus[2]) / (2 * h)}, 1e-5);

    EXPECT_THROW((void)Motion::rotation(0, 0, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)Motion(Multivector::scalar(P3::SIG, 1.0)),
                 std::invalid_argument);
}

TEST(RuntimeCalculus, FieldsAndManifolds) {
    const Field correct_swirl = [](double x, double y, double) {
        return E3::vec(-y, x, 0);
    };
    expect_near(Calculus::divergence(correct_swirl, 1, 2, 3), 0.0, 1e-6);
    expect_vector_near(E3::coords(Calculus::curl(correct_swirl, 1, 2, 3)),
                       {0, 0, 2}, 1e-6);
    const Multivector nabla_f = Calculus::vector_derivative(correct_swirl, 1, 2, 3);
    expect_near(nabla_f.get(0b011), 2.0, 1e-6);
    expect_near(nabla_f.scalar_part(), 0.0, 1e-6);

    const Field scalar = [](double x, double y, double z) {
        return Multivector::scalar(E3::SIG, x * x + 2 * y * y + 3 * z * z);
    };
    expect_vector_near(E3::coords(Calculus::gradient(scalar, 1, 2, 3)),
                       {2, 8, 18}, 1e-5);
    expect_near(Calculus::directional(scalar, 1, 2, 3, 1, 0, 0).scalar_part(),
                2.0, 1e-5);

    const Manifold plane = Manifold::surface(
        [](const std::vector<double>& u) { return std::vector<double>{u[0], u[1], 0}; });
    expect_vector_near(E3::coords(plane.normal({1, 2})), {0, 0, 1}, 1e-6);
    expect_near(plane.tangent_blade({1, 2}).get(0b011), 1.0, 1e-9);
    expect_near(plane.jacobian({1, 2}), 1.0, 1e-9);

    const Manifold sphere = Manifold::surface(
        [](const std::vector<double>& u) {
            return std::vector<double>{
                2 * std::cos(u[0]) * std::cos(u[1]),
                2 * std::sin(u[0]) * std::cos(u[1]),
                2 * std::sin(u[1])};
        });
    expect_vector_near(E3::coords(sphere.normal({0, 0})), {1, 0, 0}, 1e-5);
    expect_near(sphere.jacobian({0, 0}), 4.0, 1e-4);

    const Field radial = [](double x, double y, double z) {
        return E3::vec(x, y, z);
    };
    expect_near(Calculus::flux(radial, sphere, 0, 2 * std::numbers::pi,
                               -std::numbers::pi / 2, std::numbers::pi / 2, 64, 32),
                32 * std::numbers::pi, 0.1);

    const Manifold helix = Manifold::curve(
        [](const std::vector<double>& u) {
            return std::vector<double>{std::cos(u[0]), std::sin(u[0]), u[0]};
        });
    const double s = 1 / std::sqrt(2.0);
    expect_vector_near(E3::coords(helix.tangent({0})), {0, s, s}, 1e-5);
}

TEST(RuntimeContainers, SparseAndBatch) {
    const Multivector a = Multivector::vector(E3::SIG, {1, 2, 3});
    const Multivector b = Multivector::vector(E3::SIG, {4, 5, 6});
    const SparseMultivector sa = SparseMultivector::from_dense(a);
    const SparseMultivector sb = SparseMultivector::from_dense(b);
    EXPECT_TRUE(sa.gp(sb).to_dense().distance_to(a.gp(b)) < 1e-12);
    EXPECT_TRUE(sa.inverse().to_dense().distance_to(a.inverse()) < 1e-12);
    EXPECT_TRUE(sa.sandwich(sb).to_dense().distance_to(a.sandwich(b)) < 1e-12);

    const MultivectorBatch batch = MultivectorBatch::of({a, b});
    EXPECT_TRUE(batch.gp(a).get(1).distance_to(a.gp(b)) < 1e-12);
    const Multivector rotor = E3::rotor(0, 0, 1, 1.0);
    EXPECT_TRUE(batch.sandwich(rotor).get(0).distance_to(rotor.sandwich(a)) < 1e-12);
    MultivectorBatch out(E3::SIG, 2);
    (void)batch.sandwich(rotor, out);
    EXPECT_TRUE(out.get(1).distance_to(rotor.sandwich(b)) < 1e-12);

    const Multivector motor = P3::motor(
        P3::rotor(0.3, -0.7, 0.2, 0.9), P3::translator(1.5, -2.5, 0.75));
    const MultivectorBatch points = MultivectorBatch::of(
        {P3::point(0.2, -0.3, 0.9), P3::point(1.0, 2.0, -1.0),
         P3::point(-0.4, 0.8, 0.1)});
    const MultivectorBatch moved = points.sandwich(motor);
    for (int i = 0; i < points.size(); ++i) {
        EXPECT_TRUE(moved.get(i).distance_to(motor.sandwich(points.get(i))) < 1e-12);
    }

    const Signature e8 = Signature::euclidean(8);
    const SparseMultivector e1 = SparseMultivector::of(e8, {1}, {1.0});
    const SparseMultivector e2 = SparseMultivector::of(e8, {2}, {1.0});
    EXPECT_TRUE(e1.gp(e2).to_dense().distance_to(
                    Multivector::blade(e8, 3, 1.0)) < 1e-15);

    const Signature e12 = Signature::euclidean(12);
    const Multivector de1 = Multivector::blade(e12, 1, 1.0);
    const Multivector de2 = Multivector::blade(e12, 2, 1.0);
    EXPECT_TRUE(de1.gp(de2).distance_to(Multivector::blade(e12, 3, 1.0)) < 1e-15);
}

TEST(RuntimeContainers, Matrix) {
    const MultivectorMatrix identity = MultivectorMatrix::identity(P3::SIG, 3);
    MultivectorMatrix m(P3::SIG, 3, 3);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> direction(-0.5, 0.5);
    std::uniform_real_distribution<double> amount(0.0, 1.0);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m.set(r, c, P3::motor(
                           P3::rotor(direction(rng), direction(rng), direction(rng),
                                     amount(rng) * 2),
                           P3::translator(amount(rng), amount(rng), amount(rng))));
        }
    }
    const MultivectorMatrix inverse = m.inverse();
    EXPECT_TRUE(m.multiply(inverse).distance_to(identity) < 1e-9);
    EXPECT_TRUE(inverse.multiply(m).distance_to(identity) < 1e-9);

    MultivectorMatrix known(E3::SIG, 2, 2);
    known.set(0, 0, Multivector::scalar(E3::SIG, 2.0));
    known.set(1, 1, Multivector::scalar(E3::SIG, 4.0));
    MultivectorMatrix expected(E3::SIG, 2, 2);
    expected.set(0, 0, Multivector::scalar(E3::SIG, 0.5));
    expected.set(1, 1, Multivector::scalar(E3::SIG, 0.25));
    EXPECT_TRUE(known.inverse().distance_to(expected) < 1e-12);

    MultivectorMatrix singular(E3::SIG, 2, 2);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            singular.set(r, c, Multivector::scalar(E3::SIG, 1.0));
        }
    }
    EXPECT_THROW((void)singular.inverse(), std::domain_error);
}

TEST(RuntimeContainers, SandwichFastPathAgreesWithGeneric) {
    const Multivector rotor = Multivector::scalar(E3::SIG, 0.7)
        .add(E3::bivector(0.2, -0.4, 0.9));
    const Multivector vector = E3::vec(0.3, -0.8, 0.5);
    EXPECT_TRUE(rotor.sandwich(vector).distance_to(
                    rotor.gp(vector).gp(rotor.reverse())) < 1e-12);

    const Multivector motor = Multivector::scalar(P3::SIG, 0.8)
        .add(Multivector::blade(P3::SIG, 3, 0.2))
        .add(Multivector::blade(P3::SIG, 6, -0.1))
        .add(Multivector::blade(P3::SIG, 12, 0.4));
    const Multivector point = P3::point(0.2, -0.3, 0.9);
    EXPECT_TRUE(motor.sandwich(point).distance_to(
                    motor.gp(point).gp(motor.reverse())) < 1e-12);
}
