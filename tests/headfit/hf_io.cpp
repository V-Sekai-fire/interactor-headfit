// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "hf_io.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sstream>
#include <sys/stat.h>

#include "hf_core.h"

namespace hfio {

bool read_file(const std::string &path, std::string &out) {
	FILE *f = std::fopen(path.c_str(), "rb");
	if (!f) {
		return false;
	}
	out.clear();
	char buf[1 << 16];
	size_t n;
	while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
		out.append(buf, n);
	}
	std::fclose(f);
	return true;
}

bool read_gz(const std::string &path, std::string &out) {
	struct stat sb;
	if (stat(path.c_str(), &sb) != 0) {
		return false;
	}
	const std::string cmd = "gzip -dc '" + path + "'";
	FILE *p = popen(cmd.c_str(), "r");
	if (!p) {
		return false;
	}
	out.clear();
	char buf[1 << 16];
	size_t n;
	while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) {
		out.append(buf, n);
	}
	return pclose(p) == 0;
}

static void walk(const std::string &root, const std::string &rel, std::vector<std::string> &out) {
	DIR *d = opendir((root + "/" + rel).c_str());
	if (!d) {
		return;
	}
	while (dirent *e = readdir(d)) {
		const std::string n = e->d_name;
		if (n == "." || n == "..") {
			continue;
		}
		const std::string r = rel.empty() ? n : rel + "/" + n;
		struct stat sb;
		if (stat((root + "/" + r).c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
			walk(root, r, out);
		} else {
			out.push_back(r);
		}
	}
	closedir(d);
}

std::vector<std::string> list_files(const std::string &dir) {
	std::vector<std::string> out;
	walk(dir, "", out);
	std::sort(out.begin(), out.end());
	return out;
}

static bool ends_with(const std::string &s, const char *suf) {
	const size_t n = std::strlen(suf);
	return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// One target: `<key>.target` from `<path>` (.gz inflated).
std::string g_corrupt; // the planted control: this key's first delta moved by 1 dm

void set_corrupt(const std::string &key) {
	g_corrupt = key;
}

static bool put_target(const std::string &path, const std::string &key, std::string &err) {
	std::string text;
	const bool ok = ends_with(path, ".gz") ? read_gz(path, text) : read_file(path, text);
	if (!ok) {
		err = "cannot read " + path;
		return false;
	}
	if (key == g_corrupt) {
		const size_t e = text.find('\n');
		long idx = 0;
		double d[3];
		if (std::sscanf(text.c_str(), "%ld %lf %lf %lf", &idx, &d[0], &d[1], &d[2]) == 4) {
			char b[128];
			std::snprintf(b, sizeof b, "%ld %.6f %.6f %.6f", idx, d[0] + 1.0, d[1], d[2]);
			text = std::string(b) + text.substr(e == std::string::npos ? text.size() : e);
		}
	}
	hf::data_put(key, text);
	if (!hf::last_error().empty()) {
		err = hf::last_error();
		return false;
	}
	return true;
}

int upload_anny(const std::string &anny, const std::string &dials_path, const std::string &map_path, UploadStats &st,
		std::string &err) {
	hf::data_clear();
	std::string text;
	// base.obj first: the targets are indexed on it.
	if (!read_file(anny + "/mpfb2/3dobjs/base.obj", text)) {
		err = "cannot read " + anny + "/mpfb2/3dobjs/base.obj";
		return -1;
	}
	hf::data_put("mpfb2/3dobjs/base.obj", text);
	if (!hf::last_error().empty()) {
		err = hf::last_error();
		return -1;
	}
	st.base = 1;
	const std::string mdir = anny + "/mpfb2/targets/macrodetails";
	for (const std::string &r : list_files(mdir)) {
		if (!ends_with(r, ".target.gz")) {
			continue;
		}
		if (!put_target(mdir + "/" + r, "mpfb2/targets/macrodetails/" + r.substr(0, r.size() - 3), err)) {
			return -1;
		}
		++st.macro;
	}
	if (!read_file(dials_path, text)) {
		err = "cannot read " + dials_path;
		return -1;
	}
	hf::data_put("local_dials.txt", text);
	++st.text;
	{
		std::istringstream is(text);
		std::string line;
		while (std::getline(is, line)) {
			if (line.empty() || line[0] == '#') {
				continue;
			}
			std::istringstream ls(line);
			std::string g, n, p, q;
			ls >> g >> n >> p >> q;
			for (std::string lst : { p, q }) {
				std::stringstream ss(lst);
				std::string f;
				while (std::getline(ss, f, ',')) {
					const std::string key = "mpfb2/targets/" + g + "/" + f + ".target";
					if (hf::data_has(key)) {
						continue;
					}
					if (!put_target(anny + "/" + key + ".gz", key, err)) {
						return -1;
					}
					++st.dial;
				}
			}
		}
	}
	const std::string fdir = anny + "/faceunits01/targets/faceunits";
	for (const std::string &r : list_files(fdir)) {
		if (!ends_with(r, ".target")) {
			continue;
		}
		if (!put_target(fdir + "/" + r, "faceunits01/targets/faceunits/" + r, err)) {
			return -1;
		}
		++st.facial;
	}
	const std::string udir = anny + "/mpfb2/targets/expression/units";
	for (const std::string &r : list_files(udir)) {
		if (!ends_with(r, ".target.gz")) {
			continue;
		}
		if (!put_target(udir + "/" + r, "mpfb2/targets/expression/units/" + r.substr(0, r.size() - 3), err)) {
			return -1;
		}
		++st.unit;
	}
	if (!read_file(map_path, text)) {
		err = "cannot read " + map_path;
		return -1;
	}
	hf::data_put("unified_expressions.map", text);
	++st.text;
	return st.base + st.macro + st.dial + st.facial + st.unit + st.text;
}

// The text between `<key> = [` and its closing `]`.
static bool usda_array(const std::string &s, const char *key, std::string &body) {
	const size_t k = s.find(key);
	if (k == std::string::npos) {
		return false;
	}
	const size_t eq = s.find('=', k);
	const size_t a = eq == std::string::npos ? eq : s.find('[', eq);
	const size_t b = s.find(']', a);
	if (a == std::string::npos || b == std::string::npos) {
		return false;
	}
	body = s.substr(a + 1, b - a - 1);
	return true;
}

static std::vector<double> numbers(const std::string &s) {
	std::vector<double> out;
	const char *p = s.c_str();
	while (*p) {
		if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.') {
			char *q = nullptr;
			const double v = std::strtod(p, &q);
			if (q == p) {
				++p;
				continue;
			}
			out.push_back(v);
			p = q;
		} else {
			++p;
		}
	}
	return out;
}

bool read_avatar(const std::string &usda, const std::string &shapes, Avatar &a, std::string &err) {
	std::string s;
	if (!read_file(usda, s)) {
		err = "cannot read " + usda;
		return false;
	}
	std::string body;
	if (!usda_array(s, "point3f[] points", body)) {
		err = "no points in " + usda;
		return false;
	}
	for (double v : numbers(body)) {
		a.xyz.push_back(float(v));
	}
	if (a.xyz.size() % 3) {
		err = "points not a multiple of 3";
		return false;
	}
	std::vector<double> counts, idx, sub;
	if (!usda_array(s, "int[] faceVertexCounts", body)) {
		err = "no faceVertexCounts";
		return false;
	}
	counts = numbers(body);
	if (!usda_array(s, "int[] faceVertexIndices", body)) {
		err = "no faceVertexIndices";
		return false;
	}
	idx = numbers(body);
	if (usda_array(s, "int[] primvars:submesh", body)) {
		sub = numbers(body);
	}
	const size_t nv = a.xyz.size() / 3;
	size_t o = 0;
	for (size_t f = 0; f < counts.size(); ++f) {
		if (counts[f] != 3) {
			err = "a face with " + std::to_string(int(counts[f])) + " corners (triangles only)";
			return false;
		}
		for (int k = 0; k < 3; ++k) {
			const double v = idx[o + k];
			if (v < 0 || v >= double(nv)) {
				err = "face index out of range";
				return false;
			}
			a.tri.push_back(int32_t(v));
		}
		a.submesh.push_back(f < sub.size() ? int32_t(sub[f]) : 0);
		o += 3;
	}
	if (o != idx.size()) {
		err = "faceVertexIndices and faceVertexCounts disagree";
		return false;
	}
	std::string t;
	if (!read_file(shapes, t)) {
		err = "cannot read " + shapes;
		return false;
	}
	std::istringstream is(t);
	std::string line;
	int left = 0;
	while (std::getline(is, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		if (left == 0) {
			std::istringstream ls(line);
			std::string kw, name;
			int n = 0;
			if (!(ls >> kw >> name >> n) || kw != "shape" || n < 0) {
				err = "shapes: expected `shape <name> <n>`";
				return false;
			}
			a.shape_names.push_back(name);
			a.shapes.emplace_back(nv * 3, 0.0f);
			left = n;
			continue;
		}
		long i;
		double dx, dy, dz;
		if (std::sscanf(line.c_str(), "%ld %lf %lf %lf", &i, &dx, &dy, &dz) != 4 || i < 0 || size_t(i) >= nv) {
			err = "shapes: bad delta line `" + line + "` (" + std::to_string(nv) + " vertices)";
			return false;
		}
		float *d = &a.shapes.back()[size_t(i) * 3];
		d[0] = float(dx), d[1] = float(dy), d[2] = float(dz);
		--left;
	}
	if (left != 0) {
		err = "shapes: truncated";
		return false;
	}
	return true;
}

bool read_marks(const std::string &path, std::vector<Mark> &m, std::string &err) {
	std::string t;
	if (!read_file(path, t)) {
		err = "cannot read " + path;
		return false;
	}
	std::istringstream is(t);
	std::string line;
	while (std::getline(is, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		Mark k;
		std::istringstream ls(line);
		if (!(ls >> k.name >> k.model >> k.avatar)) {
			err = "marks: expected `name model_id avatar_id`";
			return false;
		}
		m.push_back(k);
	}
	return true;
}

} // namespace hfio
