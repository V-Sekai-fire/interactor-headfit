// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "hf_geom.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace hfg {

double norm(const V3 &a) {
	return std::sqrt(dot(a, a));
}

V3 normalized(const V3 &a) {
	const double n = norm(a);
	return n > 1e-300 ? a * (1.0 / n) : V3(0, 0, 0);
}

// Ericson, Real-Time Collision Detection, 5.1.5, with the feature recorded.
V3 closest_on_triangle(const V3 &p, const V3 &a, const V3 &b, const V3 &c, double bary[3], int &feature) {
	const V3 ab = b - a, ac = c - a, ap = p - a;
	const double d1 = dot(ab, ap), d2 = dot(ac, ap);
	if (d1 <= 0 && d2 <= 0) {
		bary[0] = 1, bary[1] = 0, bary[2] = 0, feature = 4;
		return a;
	}
	const V3 bp = p - b;
	const double d3 = dot(ab, bp), d4 = dot(ac, bp);
	if (d3 >= 0 && d4 <= d3) {
		bary[0] = 0, bary[1] = 1, bary[2] = 0, feature = 5;
		return b;
	}
	const double vc = d1 * d4 - d3 * d2;
	if (vc <= 0 && d1 >= 0 && d3 <= 0) {
		const double v = d1 / (d1 - d3);
		bary[0] = 1 - v, bary[1] = v, bary[2] = 0, feature = 1;
		return a + ab * v;
	}
	const V3 cp = p - c;
	const double d5 = dot(ab, cp), d6 = dot(ac, cp);
	if (d6 >= 0 && d5 <= d6) {
		bary[0] = 0, bary[1] = 0, bary[2] = 1, feature = 6;
		return c;
	}
	const double vb = d5 * d2 - d1 * d6;
	if (vb <= 0 && d2 >= 0 && d6 <= 0) {
		const double w = d2 / (d2 - d6);
		bary[0] = 1 - w, bary[1] = 0, bary[2] = w, feature = 3;
		return a + ac * w;
	}
	const double va = d3 * d6 - d5 * d4;
	if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
		const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		bary[0] = 0, bary[1] = 1 - w, bary[2] = w, feature = 2;
		return b + (c - b) * w;
	}
	const double denom = 1.0 / (va + vb + vc);
	const double v = vb * denom, w = vc * denom;
	bary[0] = 1 - v - w, bary[1] = v, bary[2] = w, feature = 0;
	return a + ab * v + ac * w;
}

V3 Mesh::tri_normal(uint32_t f) const {
	const V3 &a = v[t[f * 3]], &b = v[t[f * 3 + 1]], &c = v[t[f * 3 + 2]];
	return normalized(cross(b - a, c - a));
}

std::vector<V3> Mesh::vertex_normals() const {
	std::vector<V3> n(v.size());
	for (size_t f = 0; f < nt(); ++f) {
		const uint32_t i = t[f * 3], j = t[f * 3 + 1], k = t[f * 3 + 2];
		const V3 fn = cross(v[j] - v[i], v[k] - v[i]); // area-weighted
		n[i] = n[i] + fn, n[j] = n[j] + fn, n[k] = n[k] + fn;
	}
	for (V3 &x : n) {
		x = normalized(x);
	}
	return n;
}

std::vector<uint8_t> Mesh::boundary_edges() const {
	std::map<std::pair<uint32_t, uint32_t>, int> count;
	for (size_t f = 0; f < nt(); ++f) {
		for (int e = 0; e < 3; ++e) {
			uint32_t a = t[f * 3 + e], b = t[f * 3 + (e + 1) % 3];
			if (a > b) {
				std::swap(a, b);
			}
			++count[{ a, b }];
		}
	}
	std::vector<uint8_t> out(t.size(), 0);
	for (size_t f = 0; f < nt(); ++f) {
		for (int e = 0; e < 3; ++e) {
			uint32_t a = t[f * 3 + e], b = t[f * 3 + (e + 1) % 3];
			if (a > b) {
				std::swap(a, b);
			}
			out[f * 3 + e] = count[{ a, b }] == 1 ? 1 : 0;
		}
	}
	return out;
}

void Bvh::build(const Mesh &m) {
	m_ = &m;
	const uint32_t n = uint32_t(m.nt());
	order_.resize(n);
	cent_.resize(n);
	tlo_.resize(n);
	thi_.resize(n);
	for (uint32_t f = 0; f < n; ++f) {
		order_[f] = f;
		const V3 &a = m.v[m.t[f * 3]], &b = m.v[m.t[f * 3 + 1]], &c = m.v[m.t[f * 3 + 2]];
		tlo_[f] = V3(std::min({ a.x, b.x, c.x }), std::min({ a.y, b.y, c.y }), std::min({ a.z, b.z, c.z }));
		thi_[f] = V3(std::max({ a.x, b.x, c.x }), std::max({ a.y, b.y, c.y }), std::max({ a.z, b.z, c.z }));
		cent_[f] = (a + b + c) * (1.0 / 3.0);
	}
	nodes_.clear();
	nodes_.reserve(n * 2 + 1);
	if (n > 0) {
		build_rec(0, n, 0);
	}
}

int32_t Bvh::build_rec(uint32_t first, uint32_t count, int depth) {
	Node nd;
	nd.lo = V3(1e300, 1e300, 1e300);
	nd.hi = V3(-1e300, -1e300, -1e300);
	V3 clo(1e300, 1e300, 1e300), chi(-1e300, -1e300, -1e300);
	for (uint32_t i = first; i < first + count; ++i) {
		const uint32_t f = order_[i];
		nd.lo = V3(std::min(nd.lo.x, tlo_[f].x), std::min(nd.lo.y, tlo_[f].y), std::min(nd.lo.z, tlo_[f].z));
		nd.hi = V3(std::max(nd.hi.x, thi_[f].x), std::max(nd.hi.y, thi_[f].y), std::max(nd.hi.z, thi_[f].z));
		clo = V3(std::min(clo.x, cent_[f].x), std::min(clo.y, cent_[f].y), std::min(clo.z, cent_[f].z));
		chi = V3(std::max(chi.x, cent_[f].x), std::max(chi.y, cent_[f].y), std::max(chi.z, cent_[f].z));
	}
	const int32_t id = int32_t(nodes_.size());
	nodes_.push_back(nd);
	if (count <= 4 || depth > 48) {
		nodes_[id].first = first;
		nodes_[id].count = count;
		return id;
	}
	const V3 ext = chi - clo;
	const int ax = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2);
	// A total order: the centroid coordinate, then the triangle index.
	std::sort(order_.begin() + first, order_.begin() + first + count, [&](uint32_t a, uint32_t b) {
		const double ca = cent_[a][ax], cb = cent_[b][ax];
		return ca < cb || (ca == cb && a < b);
	});
	const uint32_t half = count / 2;
	const int32_t l = build_rec(first, half, depth + 1);
	const int32_t r = build_rec(first + half, count - half, depth + 1);
	nodes_[id].left = l;
	nodes_[id].right = r;
	return id;
}

namespace {

double box_d2(const V3 &p, const V3 &lo, const V3 &hi) {
	double d = 0;
	for (int i = 0; i < 3; ++i) {
		const double v = p[i] < lo[i] ? lo[i] - p[i] : (p[i] > hi[i] ? p[i] - hi[i] : 0.0);
		d += v * v;
	}
	return d;
}

// Slab test for the segment o + t d, t in [-tmax, tmax].
bool box_ray(const V3 &o, const V3 &d, double tmax, const V3 &lo, const V3 &hi) {
	double t0 = -tmax, t1 = tmax;
	for (int i = 0; i < 3; ++i) {
		if (std::fabs(d[i]) < 1e-300) {
			if (o[i] < lo[i] || o[i] > hi[i]) {
				return false;
			}
			continue;
		}
		double a = (lo[i] - o[i]) / d[i], b = (hi[i] - o[i]) / d[i];
		if (a > b) {
			std::swap(a, b);
		}
		t0 = std::max(t0, a);
		t1 = std::min(t1, b);
		if (t0 > t1) {
			return false;
		}
	}
	return true;
}

} // namespace

Bvh::Closest Bvh::closest(const V3 &p) const {
	Closest best;
	best.d2 = 1e300;
	if (nodes_.empty()) {
		return best;
	}
	std::vector<int32_t> stack;
	stack.reserve(128);
	stack.push_back(0);
	while (!stack.empty()) {
		const Node &nd = nodes_[stack.back()];
		stack.pop_back();
		if (box_d2(p, nd.lo, nd.hi) > best.d2) {
			continue;
		}
		if (nd.left < 0) {
			for (uint32_t i = nd.first; i < nd.first + nd.count; ++i) {
				const uint32_t f = order_[i];
				double bary[3];
				int feat = 0;
				const V3 q = closest_on_triangle(p, m_->v[m_->t[f * 3]], m_->v[m_->t[f * 3 + 1]],
						m_->v[m_->t[f * 3 + 2]], bary, feat);
				const V3 dq = q - p;
				const double d2 = dot(dq, dq);
				if (d2 < best.d2 || (d2 == best.d2 && int32_t(f) < best.tri)) {
					best.d2 = d2;
					best.tri = int32_t(f);
					best.q = q;
					best.bary[0] = bary[0], best.bary[1] = bary[1], best.bary[2] = bary[2];
					best.feature = feat;
				}
			}
			continue;
		}
		const double dl = box_d2(p, nodes_[nd.left].lo, nodes_[nd.left].hi);
		const double dr = box_d2(p, nodes_[nd.right].lo, nodes_[nd.right].hi);
		// Visit the nearer child first (pushed last).
		if (dl <= dr) {
			stack.push_back(nd.right);
			stack.push_back(nd.left);
		} else {
			stack.push_back(nd.left);
			stack.push_back(nd.right);
		}
	}
	return best;
}

Bvh::Hit Bvh::ray(const V3 &o, const V3 &d, double tmax, double min_cos) const {
	Hit best;
	double bestAbs = 1e300;
	if (nodes_.empty()) {
		return best;
	}
	std::vector<int32_t> stack;
	stack.reserve(128);
	stack.push_back(0);
	while (!stack.empty()) {
		const Node &nd = nodes_[stack.back()];
		stack.pop_back();
		if (!box_ray(o, d, std::min(tmax, bestAbs), nd.lo, nd.hi)) {
			continue;
		}
		if (nd.left >= 0) {
			stack.push_back(nd.right);
			stack.push_back(nd.left);
			continue;
		}
		for (uint32_t i = nd.first; i < nd.first + nd.count; ++i) {
			const uint32_t f = order_[i];
			const V3 &a = m_->v[m_->t[f * 3]], &b = m_->v[m_->t[f * 3 + 1]], &c = m_->v[m_->t[f * 3 + 2]];
			if (dot(m_->tri_normal(f), d) < min_cos) {
				continue; // the normal test: a surface facing away
			}
			// Moller-Trumbore, both directions.
			const V3 e1 = b - a, e2 = c - a;
			const V3 pv = cross(d, e2);
			const double det = dot(e1, pv);
			if (std::fabs(det) < 1e-300) {
				continue;
			}
			const double inv = 1.0 / det;
			const V3 tv = o - a;
			const double u = dot(tv, pv) * inv;
			const double eps = 1e-9;
			if (u < -eps || u > 1 + eps) {
				continue;
			}
			const V3 qv = cross(tv, e1);
			const double w = dot(d, qv) * inv;
			if (w < -eps || u + w > 1 + eps) {
				continue;
			}
			const double t = dot(e2, qv) * inv;
			const double at = std::fabs(t);
			if (at > tmax) {
				continue;
			}
			if (at < bestAbs || (at == bestAbs && int32_t(f) < best.tri)) {
				bestAbs = at;
				best.tri = int32_t(f);
				best.t = t;
				const double uu = std::min(std::max(u, 0.0), 1.0), ww = std::min(std::max(w, 0.0), 1.0 - uu);
				best.bary[0] = 1 - uu - ww, best.bary[1] = uu, best.bary[2] = ww;
			}
		}
	}
	return best;
}

} // namespace hfg
