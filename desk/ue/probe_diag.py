import numpy as np, sys
exec(open("probe_close.py").read().split("ref = dense")[0])
ref = dense("face_wink_L"); pr = np.load("anime_close_L.npy")
vs = {s: np.unique(idx[sub == s].ravel()) for s in (0, 1, 2)}
for s, v in vs.items():
    m = np.linalg.norm(ref[v], axis=1); p = np.linalg.norm(pr[v], axis=1)
    print("submesh %d: artist moves %d (rms %.2f mm, mean dz %.2f dy %.2f), squash moves %d (rms %.2f)" % (s, (m > 1e-6).sum(), np.sqrt((m[m>1e-6]**2).mean())*1e3 if (m>1e-6).any() else 0, ref[v][m>1e-6,2].mean()*1e3 if (m>1e-6).any() else 0, ref[v][m>1e-6,1].mean()*1e3 if (m>1e-6).any() else 0, (p > 1e-6).sum(), np.sqrt((p[p>1e-6]**2).mean())*1e3 if (p>1e-6).any() else 0))
m = np.linalg.norm(ref, axis=1) > 1e-6
P = pts[m]; print("artist support box x %.3f..%.3f z %.3f..%.3f y %.3f..%.3f" % (P[:,0].min(), P[:,0].max(), P[:,2].min(), P[:,2].max(), P[:,1].min(), P[:,1].max()))
big = np.linalg.norm(ref, axis=1) > 0.005
P = pts[big]; print("artist >5mm: %d verts, box x %.3f..%.3f z %.3f..%.3f y %.3f..%.3f; mean d %s mm" % (big.sum(), P[:,0].min(), P[:,0].max(), P[:,2].min(), P[:,2].max(), P[:,1].min(), P[:,1].max(), np.round(ref[big].mean(0)*1e3,2)))
print("by submesh >5mm:", {s: int(big[v].sum()) for s, v in vs.items()})
