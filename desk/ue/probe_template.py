"""Desk probe (not committed): eye closes as a template in eye-normalised coordinates, learned from
the artists' own closes and scored leaving one avatar out.

For each eye: u = across the eye (0 inner/outer corner .. 1 the other, by the eye-role outline's x
range), v = height in eye heights above the outline's lower edge at that x, H = the outline's
largest height. The template is the mean up-motion dz/H in a (u, v) grid, bilinear, over every
candidate vertex (the eye side, not iris/brow, in front of the eye's back) of the training avatars,
moving or not. Prediction = the template at each candidate vertex, times H.
"""
import sys

import numpy as np

sys.argv = sys.argv[:1]
import case2  # noqa: E402
import probe_close2 as pc  # noqa: E402
import probe_close3 as p3  # noqa: E402
import probe_lid  # noqa: E402

U = np.linspace(-0.3, 1.3, 9)
V = np.linspace(-0.8, 1.8, 14)


def frame(c):
    P, up, hz = c["P"], c["up"], c["hz"]
    roles = p3.eye_roles(c)
    side = c["left_sign"]
    da = int(np.argmax(np.abs(c["fwd"])))
    E = P[(roles == "eye") & (np.sign(P[:, hz]) == side)]
    x0, x1 = E[:, hz].min(), E[:, hz].max()
    bins = np.linspace(x0, x1, 11)
    xs, lo, hi = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            xs.append(s[:, hz].mean()); lo.append(s[:, up].min()); hi.append(s[:, up].max())
    xs, lo, hi = map(np.array, (xs, lo, hi))
    H = float((hi - lo).max())
    front = (E[:, da] * c["fwd"][da]).min() - 0.25 * H
    vs = np.where((np.sign(P[:, hz]) == side) & ~np.isin(roles, ["iris", "brow"]) & (P[:, da] * c["fwd"][da] >= front))[0]
    # u runs from the inner corner (toward the nose, x = 0) outward, so both eyes and all avatars agree
    x = P[vs, hz]
    u = (x - x0) / (x1 - x0)
    if side < 0:
        u = 1 - u
    v = (P[vs, up] - np.interp(x, xs, lo)) / H
    keep = (u >= U[0]) & (u <= U[-1]) & (v >= V[0]) & (v <= V[-1])
    return vs[keep], u[keep], v[keep], H


def weights(u, v):
    iu = np.clip(np.searchsorted(U, u) - 1, 0, len(U) - 2)
    iv = np.clip(np.searchsorted(V, v) - 1, 0, len(V) - 2)
    fu = (u - U[iu]) / (U[iu + 1] - U[iu])
    fv = (v - V[iv]) / (V[iv + 1] - V[iv])
    return iu, iv, fu, fv


def fit(cases):
    num = np.zeros((len(U), len(V)))
    den = np.zeros((len(U), len(V)))
    for c in cases:
        vs, u, v, H = frame(c)
        dz = c["ref"][vs, c["up"]] / H
        iu, iv, fu, fv = weights(u, v)
        for du, dv, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
            np.add.at(num, (iu + du, iv + dv), w * dz)
            np.add.at(den, (iu + du, iv + dv), w)
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def predict(T, c):
    vs, u, v, H = frame(c)
    iu, iv, fu, fv = weights(u, v)
    t = (T[iu, iv] * (1 - fu) * (1 - fv) + T[iu + 1, iv] * fu * (1 - fv) + T[iu, iv + 1] * (1 - fu) * fv
         + T[iu + 1, iv + 1] * fu * fv)
    out = np.zeros_like(c["P"])
    out[vs, c["up"]] = t * H
    return out


def main():
    cases = {}
    for av, path, t in p3.VRM_REFS:
        cases[(av, t)] = case2.case(av, path, t)
    for t in p3.MIRE_REFS:
        cases[("Mire", t)] = pc.case("Mire", None, t)
    for key, c in cases.items():
        c["kind"] = pc.kind_of(c)[0]
    for kind in ("plain", "smile"):
        refs = [k for k, c in cases.items() if c["kind"] == kind]
        print("\n== %s template, leave one avatar out" % kind)
        for hold in sorted({a for a, _ in refs}):
            # one reference per avatar in training, so an avatar with four closes does not outvote the rest
            train = {}
            for r in refs:
                if r[0] != hold:
                    train.setdefault(r[0], r)
            T = fit([cases[r] for r in train.values()])
            for r in [r for r in refs if r[0] == hold]:
                v, e, m = probe_lid.score(predict(T, cases[r]), cases[r]["ref"], cases[r]["sel"])
                print("  HELD OUT %-15s %-16s rel %.3f (%.2f of %.2f mm)  trained on %s" % (r[0], r[1], v, e, m, sorted(train)))
        train = {}
        for r in refs:
            train.setdefault(r[0], r)
        T = fit([cases[r] for r in train.values()])
        np.save("template_%s.npy" % kind, T)
        other = "smile" if kind == "plain" else "plain"
        for r in [r for r in cases if cases[r]["kind"] == other]:
            print("  control: %s template on %s ref %s %s rel %.3f" % (kind, other, r[0], r[1],
                  probe_lid.score(predict(T, cases[r]), cases[r]["ref"], cases[r]["sel"])[0]))


if __name__ == "__main__":
    main()
