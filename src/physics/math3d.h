#pragma once

// Minimal 3D linear algebra: Vec3, Mat3, Quat. Header-only, value-semantics,
// constexpr where possible. Hand-rolled to keep the dependency surface tiny
// (no Eigen at runtime).

#include <array>
#include <cmath>

#include "la/sparse_matrix.h"  // mac::la::Real

namespace mac::physics {

using la::Real;

struct Vec3 {
    Real x, y, z;

    constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator-() const             { return {-x, -y, -z}; }
    constexpr Vec3 operator*(Real s)       const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(Real s)       const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(Real s)        { x *= s;  y *= s;  z *= s;  return *this; }

    constexpr Real dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    constexpr Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    Real norm()  const { return std::sqrt(dot(*this)); }
    Real norm2() const { return dot(*this); }
};

inline Vec3 operator*(Real s, const Vec3& v) { return v * s; }

// Row-major 3x3.
struct Mat3 {
    std::array<Real, 9> m{};

    static constexpr Mat3 identity() {
        Mat3 r{}; r.m[0] = r.m[4] = r.m[8] = 1.0; return r;
    }
    static constexpr Mat3 diag(Real a, Real b, Real c) {
        Mat3 r{}; r.m[0] = a; r.m[4] = b; r.m[8] = c; return r;
    }

    // Skew-symmetric matrix [v]_x s.t. [v]_x w = v × w.
    static constexpr Mat3 skew(const Vec3& v) {
        return Mat3{{0, -v.z, v.y,
                     v.z, 0, -v.x,
                    -v.y, v.x, 0}};
    }

    Real  operator()(int i, int j) const { return m[3 * i + j]; }
    Real& operator()(int i, int j)       { return m[3 * i + j]; }

    Vec3 operator*(const Vec3& v) const {
        return {
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z};
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = (*this)(i, 0) * o(0, j)
                        + (*this)(i, 1) * o(1, j)
                        + (*this)(i, 2) * o(2, j);
        return r;
    }
    Mat3 transpose() const {
        return Mat3{{m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]}};
    }
};

// Unit quaternion (w + xi + yj + zk). Convention: w is the scalar part.
struct Quat {
    Real w, x, y, z;

    static constexpr Quat identity() { return {1, 0, 0, 0}; }

    Quat normalized() const {
        Real n = std::sqrt(w * w + x * x + y * y + z * z);
        return {w / n, x / n, y / n, z / n};
    }

    // Hamilton product: (this) * (o), applied right-to-left to a vector.
    constexpr Quat operator*(const Quat& o) const {
        return {
            w * o.w - x * o.x - y * o.y - z * o.z,
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w};
    }

    // Rotate v by this quaternion. v' = q v q*.
    Vec3 rotate(const Vec3& v) const {
        // Optimized: v' = v + 2 r × (r × v + w v), r = (x,y,z).
        Vec3 r{x, y, z};
        Vec3 t = r.cross(v) * 2.0;
        return v + Vec3{w * t.x, w * t.y, w * t.z} + r.cross(t);
    }

    // 3x3 rotation matrix.
    Mat3 to_matrix() const {
        Real xx = x * x, yy = y * y, zz = z * z;
        Real xy = x * y, xz = x * z, yz = y * z;
        Real wx = w * x, wy = w * y, wz = w * z;
        return Mat3{{
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)
        }};
    }

    // Apply small angular increment omega*dt: q_new = exp(0.5 * omega*dt) * q.
    // Renormalizes to fight drift.
    Quat integrate(const Vec3& omega, Real dt) const {
        Vec3 half = omega * (0.5 * dt);
        Real angle = half.norm();
        Quat dq;
        if (angle < 1e-12) {
            dq = {1, half.x, half.y, half.z};
        } else {
            Real s = std::sin(angle) / angle;
            dq = {std::cos(angle), s * half.x, s * half.y, s * half.z};
        }
        return (dq * (*this)).normalized();
    }
};

} // namespace mac::physics
