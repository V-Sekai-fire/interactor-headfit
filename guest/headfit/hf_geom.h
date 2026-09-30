// hf_geom -- the head fit's geometry queries: a triangle BVH with closest
// point and ray queries, vertex normals, boundary edges. The correspondence
// search, not a residual: its results (closest points, normals, hits and
// barycentrics) are the fixed inputs of the Lean kernels. Double precision;
// every order is total (ties broken by index), so the guest's libstdc++ and
// a native libc++ build answer the same (AGENTS.md: tied sort keys).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <vector>

namespace hfg {

struct V3 {
	double x = 0, y = 0, z = 0;
	V3() = default;
	V3(double a, double b, double c) : x(a), y(b), z(c) {}
	V3 operator+(const V3 &o) const { return { x + o.x, y + o.y, z + o.z }; }
	V3 operator-(const V3 &o) const { return { x - o.x, y - o.y, z - o.z }; }
	V3 operator*(double s) const { return { x * s, y * s, z * s }; }
	double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
inline double dot(const V3 &a, const V3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3 &a, const V3 &b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
double norm(const V3 &a);
V3 normalized(const V3 &a);

// Closest point on triangle abc to p: the point, its barycentrics, and the
// feature it lies on (0 face interior, 1..3 edge ab/bc/ca, 4..6 vertex a/b/c).
V3 closest_on_triangle(const V3 &p, const V3 &a, const V3 &b, const V3 &c, double bary[3], int &feature);

struct Mesh {
	std::vector<V3> v;
	std::vector<uint32_t> t; // 3 per triangle
	size_t nt() const { return t.size() / 3; }
	V3 tri_normal(uint32_t f) const; // unit, from the winding
	std::vector<V3> vertex_normals() const; // area-weighted, unit
	// boundary[f*3+e]: edge e of triangle f (ab, bc, ca) has one triangle.
	std::vector<uint8_t> boundary_edges() const;
};

class Bvh {
public:
	void build(const Mesh &m);
	struct Closest {
		int32_t tri = -1;
		double d2 = 0;
		V3 q;
		double bary[3] = { 0, 0, 0 };
		int feature = 0;
	};
	// Nearest triangle to p (ties: the lower triangle index).
	Closest closest(const V3 &p) const;
	struct Hit {
		int32_t tri = -1;
		double t = 0;
		double bary[3] = { 0, 0, 0 };
	};
	// The hit with the smallest |t| along o + t d, |t| <= tmax, whose
	// triangle normal n satisfies dot(n, d) >= min_cos (d unit). Both
	// directions; ties: the lower triangle index.
	Hit ray(const V3 &o, const V3 &d, double tmax, double min_cos) const;

private:
	struct Node {
		V3 lo, hi;
		int32_t left = -1, right = -1; // children, or -1 at a leaf
		uint32_t first = 0, count = 0; // leaf range in order_
	};
	int32_t build_rec(uint32_t first, uint32_t count, int depth);
	const Mesh *m_ = nullptr;
	std::vector<Node> nodes_;
	std::vector<uint32_t> order_;
	std::vector<V3> cent_, tlo_, thi_;
};

} // namespace hfg
