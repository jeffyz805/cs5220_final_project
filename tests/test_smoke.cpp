#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#if MAC_HAS_EIGEN
#include <Eigen/Dense>
#endif

TEST_CASE("smoke: build wires up", "[smoke]") {
    REQUIRE(1 + 1 == 2);
}

#if MAC_HAS_EIGEN
TEST_CASE("smoke: eigen oracle reachable", "[smoke][eigen]") {
    Eigen::Matrix2d A;
    A << 2, 0, 0, 3;
    Eigen::Vector2d b{1, 1};
    Eigen::Vector2d x = A.llt().solve(b);
    REQUIRE(x[0] == Catch::Approx(0.5));
    REQUIRE(x[1] == Catch::Approx(1.0 / 3.0));
}
#endif
