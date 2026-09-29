// hf_io -- the native head-fit runner's file side (tests/headfit): ANNY's
// sources read from disk (the .gz inflated with `gzip -dc`), the avatar
// fixture (USDA points and triangles, the artist shape deltas) and the marks
// file. Host-only: the guest has no filesystem, and headfit.elf gets the same
// bytes from GDScript (project/stages/headfit_stage.gd).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hfio {

bool read_file(const std::string &path, std::string &out);
// `gzip -dc path` into out.
bool read_gz(const std::string &path, std::string &out);
// Files under dir (recursive), relative to dir, sorted.
std::vector<std::string> list_files(const std::string &dir);

// Upload ANNY's sources into hf_core, base.obj first: every macrodetails
// target, the local dials' targets (dials_path), the 52 face units, the
// expression units of the three races, local_dials.txt and the map.
// Returns the number of files uploaded, or -1 (err says why).
struct UploadStats {
	int base = 0, macro = 0, dial = 0, facial = 0, unit = 0, text = 0;
};
int upload_anny(const std::string &anny_data, const std::string &dials_path, const std::string &map_path,
		UploadStats &st, std::string &err);

struct Avatar {
	std::vector<float> xyz; // metres
	std::vector<int32_t> tri;
	std::vector<int32_t> submesh; // per triangle
	std::vector<std::string> shape_names;
	std::vector<std::vector<float>> shapes; // dense, nv*3 metres
};
bool read_avatar(const std::string &usda, const std::string &shapes, Avatar &a, std::string &err);

// marks: `name model_id avatar_id` lines.
struct Mark {
	std::string name;
	int32_t model = -1, avatar = -1;
};
bool read_marks(const std::string &path, std::vector<Mark> &m, std::string &err);

// The planted control of the load gate: this key's first delta gets 1 dm
// added to dx before it is uploaded.
void set_corrupt(const std::string &key);

} // namespace hfio
