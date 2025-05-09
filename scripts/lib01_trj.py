import numpy as np
import copy
import struct
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl


def angle_harmonic_prob(t, k, t0, R=8.314/1000, T=298.15):
    """This function calculates the probability density function based on the
    harmonic angle force field function."""
    E = 0.5*k*(np.radians(t)-np.radians(t0))**2
    P = np.exp(-E/(R*T))*np.sin(np.radians(t))
    if np.sum(P) != 0:
        P = P/np.sum(P)
    return P


def angle_harmonic_prob_int(t, k, t0, R=8.314/1000, T=298.15):
    """This function calculates the cumulative density function based on the
    harmonic angle force field function."""
    P = angle_harmonic_prob(t, k, t0, R=R, T=T)
    return np.cumsum(P)


def harmonic_prob2(r, k, r0, R=8.314/1000, T=298.15):
    """This function calculates the probability density function based on the
    harmonic bond force field function."""
    E = 0.5*k*(r/10-r0/10)**2
    P = np.exp(-E/(R*T))*r**2
    if np.sum(P) != 0:
        P = P/np.sum(P)
    return P


def harmonic_prob2_int(r, k, r0, R=8.314/1000, T=298.15):
    """This function calculates the cumulative density function based on the
    harmonic bond force field function."""
    P = harmonic_prob2(r, k, r0, R=R, T=T)
    return np.cumsum(P)


def calc_angles(As, Bs, Cs):
    """Calculate angles for A-B-C atoms, based on their xyz coordinates."""
    v1 = As - Bs
    v2 = Cs - Bs
    cosine_angles = np.sum(np.multiply(As-Bs, Cs-Bs), axis=1) / \
        (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    angle = np.degrees(np.arccos(cosine_angles))
    return angle


def G96_prob2(t, k, t0, R=8.314/1000, T=298.15):
    """This function calculates the probability density function based on the
    G96 angle force field function."""
    E = 0.5*k*(np.cos(np.radians(t))-np.cos(np.radians(t0)))**2
    P = np.exp(-E/(R*T))*np.sin(np.radians(t))
    if np.sum(P) != 0:
        P = P/np.sum(P)
    return P


def G96_prob2_int(t, k, t0, R=8.314/1000, T=298.15):
    """This function calculates the cumulative density function based on the
    G96 angle force field function."""
    P = G96_prob2(t, k, t0, R=R, T=T)
    return np.cumsum(P)


def ReB_prob2(t, k, t0, R=8.314/1000, T=298.15):
    """This function calculates probability density function based on the
    GROMACS Restricted Bending potential"""
    E = 0.5*k*(np.cos(np.radians(t))-np.cos(np.radians(t0)))**2 / \
        np.sin(np.radians(t))**2
    P = np.exp(-E/(R*T))*np.sin(np.radians(t))
    if np.sum(P) != 0:
        P = P/np.sum(P)
    return P


def ReB_prob2_int(t, k, t0, R=8.314/1000, T=298.15):
    """This function calculates cumulative density function based on the
    GROMACS Restricted Bending potential"""
    P = ReB_prob2(t, k, t0, R=R, T=T)
    return np.cumsum(P)


def calc_dihedral_multiple(u1, u2, u3, u4):
    """Calculates dihedral angles for A-B-C-D atoms. Each input is an Nx3 matrix for N atoms.
    The result is an array of N anlge values in degrees."""
    a1 = u2 - u1
    a2 = u3 - u2
    a3 = u4 - u3
    v1 = np.cross(a1, a2, axis=1)
    v1 = v1/np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2 = np.cross(a2, a3, axis=1)
    v2 = v2/np.linalg.norm(v2, axis=1)[:, np.newaxis]
    porm = np.sign(np.sum(np.multiply(v1, a3), axis=1))
    rad = np.sum(v1*v2, axis=1)
    rad = rad/np.sqrt((np.sum(v1**2, axis=1)*np.sum(v2**2, axis=1)))
    rad = np.arccos(rad)
    rad = porm*rad
    return np.degrees(rad)


def calc_dihedral(u1, u2, u3, u4):
    """ Calculate dihedral angle method. From bioPython.PDB
    (adapted to np.array)
    Calculate the dihedral angle between 4 vectors
    representing 4 connected points. The angle is in
    [-pi, pi].
    """
    a1 = u2 - u1
    a2 = u3 - u2
    a3 = u4 - u3
    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1)**0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1)**0.5
    porm = np.sign((v1 * a3).sum(-1))
    rad = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
    if not porm == 0:
        rad = rad * porm
    return rad

# a = np.array([1219.7, 4106.1, -7366.7])
# b = np.array([1277.1, 4016.6, -7447.1])
# c = np.array([1398.6, 3944.8, -7407.8])
# d = np.array([1501.2, 3943.2, -7521.3])
# alpha = calc_dihedral(a,b,c,d)
# print("Dihedral angle in radians:", alpha, "and degrees:", alpha*180/np.pi)
# a = np.array([[1219.7, 4106.1, -7366.7],
#               [1119.7, 4106.1, -7366.7]])
# b = np.array([[1277.1, 4016.6, -7447.1],
#               [1177.1, 4016.6, -7447.1]])
# c = np.array([[1398.6, 3944.8, -7407.8],
#               [1198.6, 3944.8, -7407.8]])
# d = np.array([[1501.2, 3943.2, -7521.3],
#               [1101.2, 3943.2, -7521.3]])
# alpha = calc_dihedral_multiple(a,b,c,d)
# print("Dihedral angle in radians:", alpha, "and degrees:", np.degrees(alpha))


def get_bead_positions(trj, bead_definitions, bead_name):
    """Generates CG bead positions for a trajectory based on CG mapping."""
    tempbeads = copy.deepcopy(bead_definitions)
    beadlabels = [bead[0] for bead in tempbeads]
    bead_index = beadlabels.index(bead_name)
    # check if the bead is virtual bead
    if isinstance(tempbeads[bead_index][1], str):
        temptrj = []
        for beadlabel in tempbeads[bead_index][2]:
            bead_index_real = beadlabels.index(beadlabel)
            atoms = np.array(tempbeads[bead_index_real][1])
            atoms = atoms[np.argsort(atoms)]-1
            temptrj.append(np.average(trj[:, atoms], axis=1))
        temptrj = np.array(temptrj)
        bead_positions = np.average(temptrj, axis=0)
    else:                                           # if the bead is regular bead
        atoms = np.array(tempbeads[bead_index][1])
        atoms = atoms[np.argsort(atoms)]-1
        bead_positions = np.average(trj[:, atoms], axis=1)
    return bead_positions

# def dihed_prob(t,k,t0,A,n=1,R=8.314/1000,T=298.15):
#     E=np.abs(k)*(1+np.cos(n*np.radians(t)-np.radians(t0)))
#     P=A*np.exp(-E/(R*T))
#     return P


def dihed_prob2(t, k, t0, n=1, R=8.314/1000, T=298.15):
    """This function calculates the probability density function based on the
    dihedral force field function."""
    E = np.abs(k)*(1+np.cos(n*np.radians(t)-np.radians(t0)))
    P = np.exp(-E/(R*T))
    P = P/np.sum(P)
    return P


def dihed_prob2_int(t, k, t0, n=1, R=8.314/1000, T=298.15):
    """This function calculates the cumulative density function based on the
    dihedral force field function."""
    P = dihed_prob2(t, k, t0, n=1, R=R, T=T)
    return np.cumsum(P)

# def import_dcd(dcdfile):
#     """Import trajectory from a .dcd file"""
#     B = open(dcdfile, "rb").read()
#     nframes=list(struct.unpack("<i", B[8  :12 ]))[0]
#     natoms =list(struct.unpack("<i", B[268:272]))[0]
#     crds = B[280:len(B)]
#     del B
#     byte = 0
#     celldims=np.zeros(shape=(nframes,3))
#     trj=np.zeros(shape=(nframes,natoms,3))
#     for frame in range(nframes):
#         celldims[frame,0]=list(struct.unpack("<d", crds[byte   :byte+8 ]))[0]
#         celldims[frame,1]=list(struct.unpack("<d", crds[byte+16:byte+24]))[0]
#         celldims[frame,2]=list(struct.unpack("<d", crds[byte+40:byte+48]))[0]
#         byte +=56
#         for atom in range(natoms):
#             trj[frame,atom,0]=list(struct.unpack("<f", crds[byte:byte+4]))[0]
#             byte += 4
#         byte += 8
#         for atom in range(natoms):
#             trj[frame,atom,1]=list(struct.unpack("<f", crds[byte:byte+4]))[0]
#             byte += 4
#         byte += 8
#         for atom in range(natoms):
#             trj[frame,atom,2]=list(struct.unpack("<f", crds[byte:byte+4]))[0]
#             byte += 4
#         byte += 8
#     celldims=np.array(celldims)
#     return trj,celldims


def import_dcd(dcdfile):
    """Import trajectory from a .dcd file"""
    with open(dcdfile, "rb") as f:
        B = f.read()

    nframes = struct.unpack("<i", B[8:12])[0]
    natoms = struct.unpack("<i", B[268:272])[0]
    crds = memoryview(B[280:])  # Use memoryview to avoid unnecessary copies

    celldims = np.zeros((nframes, 3))
    trj = np.zeros((nframes, natoms, 3), dtype=np.float32)
    byte = 0

    for frame in range(nframes):
        # celldims[frame] = struct.unpack("<d d d", crds[byte:byte + 24])
        celldims[frame, 0] = list(struct.unpack("<d", crds[byte:byte+8]))[0]
        celldims[frame, 1] = list(
            struct.unpack("<d", crds[byte+16:byte+24]))[0]
        celldims[frame, 2] = list(
            struct.unpack("<d", crds[byte+40:byte+48]))[0]
        byte += 56  # Skip to trajectory data

        for coord in range(3):
            trj[frame, :, coord] = np.frombuffer(
                crds[byte:byte + 4 * natoms], dtype=np.float32)
            byte += 4 * natoms + 8  # Skip padding

    return trj, celldims


def get_R_rotation(vector1, vector2):
    """Rotation matrix based on Rodriges' formula"""
    a, b = [0, 0, 0], [0, 0, 0]
    for i in range(0, 3, 1):
        a[i] = vector1[i] / np.linalg.norm(vector1)
        b[i] = vector2[i] / np.linalg.norm(vector2)
    v = np.cross(a, b)
    c = 1 / (1 + np.dot(a, b))
    vx = np.array([[0.0, -v[2],  v[1]],
                   [v[2],  0.0, -v[0]],
                   [-v[1], v[0],  0.0]])
    vx2 = np.array(np.dot(vx, vx))
    for i in range(0, 3, 1):
        for j in range(0, 3, 1):
            vx2[i, j] *= c
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = np.array(np.add(np.add(I, vx), vx2))
    return R


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)


def average_bond_results(r0_list, k_list, N=500, T=298.15):
    """Average bond probability density functions based on Earthmovers' distance"""
    r0s = np.array(r0_list)
    ks = np.array(k_list)
    if ks[0] == 0:
        return [0, np.average(r0s)]
    if len(r0s) == 1:
        return [ks[0], r0s[0]]
    xs = np.linspace(0, 10, N)
    ys_target = np.zeros(shape=N)
    for r0, k in zip(r0s, ks):
        ys_target = ys_target+harmonic_prob2(xs, r0=r0, k=k)
    ys_target = ys_target/sum(ys_target)

    lsq_min = -1
    def f_pdf(p): return EMD(ys_target, harmonic_prob2(xs, k=p[0], r0=p[1], T=T)) if (
        0 <= p[0] <= 999999.9 and p[1] > 0) else 999999.9

    def f_cdf(centers, k, r0): return harmonic_prob2_int(xs, k=k, r0=r0, T=T)
    # bounds=optimize.Bounds([0, 0], [np.inf, 100])
    for shift in np.linspace(np.min(r0s), np.max(r0s), 8):
        popt_temp, _ = optimize.curve_fit(f_cdf, xs, np.cumsum(ys_target), p0=(np.min(ks), shift), maxfev=50000,
                                          bounds=((0, 0), (999999.9, 100)))
        # res=optimize.minimize(f, popt_temp, bounds=bounds, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
        # res=optimize.minimize(f_pdf, popt_temp, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
        try:
            res = optimize.minimize(
                f_pdf, popt_temp, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
            res = res.x
        except:
            res = popt_temp
            print("Error in optimization")
        ys = harmonic_prob2(xs, k=res[0], r0=res[1], T=T)
        lsq = EMD(ys_target, ys)
        if lsq_min < 0 or lsq < lsq_min:
            lsq_min = lsq
            popt = res
    return popt


def average_G96_results(t0_list, k_list, N=500, T=298.15):
    t0s = np.array(t0_list)
    ks = np.array(k_list)
    if len(t0s) == 1:
        return [ks[0], t0s[0]]
    xs = np.linspace(0, 180, N)
    ys_target = np.zeros(shape=N)
    for t0, k in zip(t0s, ks):
        ys_target = ys_target+G96_prob2(xs, t0=t0, k=k)
    ys_target = ys_target/sum(ys_target)

    lsq_min = -1
    def f_pdf(p): return EMD(ys_target, G96_prob2(xs, k=p[0], t0=p[1], T=T)) if (
        0 <= p[0] <= 9999.9 and 0 <= p[1] < 180) else 999999.9

    def f_cdf(centers, k, t0): return G96_prob2_int(xs, k=k, t0=t0, T=T)
    # bounds=optimize.Bounds([0, 0], [np.inf, np.inf])
    for shift in np.linspace(max(np.min(t0s), 1), min(np.max(t0s), 179), 8):
        popt_temp, _ = optimize.curve_fit(f_cdf, xs, np.cumsum(ys_target), p0=(np.min(ks), shift), maxfev=50000,
                                          bounds=((0, 0), (9999.9, 180.0)))
        # res=optimize.minimize(f, popt_temp, bounds=bounds, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
        # res=optimize.minimize(f_pdf, popt_temp, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
        try:
            res = optimize.minimize(
                f_pdf, popt_temp, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
            res = res.x
        except:
            res = popt_temp
            print("Error in optimization")
        ys = G96_prob2(xs, k=res[0], t0=res[1], T=T)
        lsq = EMD(ys_target, ys)
        if lsq_min < 0 or lsq < lsq_min:
            lsq_min = lsq
            popt = res
    return popt


def average_angle_harmonic_results(t0_list, k_list, N=500, T=298.15):
    t0s = np.array(t0_list)
    ks = np.array(k_list)
    if len(t0s) == 1:
        return [ks[0], t0s[0]]
    xs = np.linspace(0, 180, N)
    ys_target = np.zeros(shape=N)
    for t0, k in zip(t0s, ks):
        ys_target = ys_target+angle_harmonic_prob(xs, t0=t0, k=k)
    ys_target = ys_target/sum(ys_target)

    lsq_min = -1

    def f_pdf(p): return EMD(ys_target, angle_harmonic_prob(xs, k=p[0], t0=p[1], T=T)) if (
        0 <= p[0] <= 9999.9 and 0 <= p[1] < 180) else 999999.9
    def f_cdf(centers, k, t0): return angle_harmonic_prob_int(
        xs, k=k, t0=t0, T=T)
    # bounds=optimize.Bounds([0, 0], [np.inf, np.inf])
    for shift in np.linspace(max(np.min(t0s), 1), min(np.max(t0s), 179), 8):
        popt_temp, _ = optimize.curve_fit(f_cdf, xs, np.cumsum(ys_target), p0=(np.min(ks), shift), maxfev=50000,
                                          bounds=((0, 0), (9999.9, 180.0)))
        # res=optimize.minimize(f, popt_temp, bounds=bounds, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
        # res=optimize.minimize(f_pdf, popt_temp, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
        try:
            res = optimize.minimize(
                f_pdf, popt_temp, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
            res = res.x
        except:
            res = popt_temp
            print("Error in optimization")
        ys = angle_harmonic_prob(xs, k=res[0], t0=res[1], T=T)
        lsq = EMD(ys_target, ys)
        if lsq_min < 0 or lsq < lsq_min:
            lsq_min = lsq
            popt = res
    return popt


def average_dihedral_results(t0_list, k_list, N=500, T=298.15):
    t0s = np.array(t0_list)
    ks = np.array(k_list)
    if len(t0s) == 1:
        return [ks[0], t0s[0]]
    xs = np.linspace(-360, 360, N)
    ys_target = np.zeros(shape=N)
    for t0, k in zip(t0s, ks):
        ys_target = ys_target+dihed_prob2(xs, t0=t0, k=k)
    ys_target = ys_target/sum(ys_target)

    lsq_min = -1
    def f_pdf(p): return EMD(ys_target, dihed_prob2(xs, k=p[0], t0=p[1], T=T)) if (
        0 <= p[0] <= 9999.9 and -360 <= p[1] < 360) else 999999.9

    def f_cdf(centers, k, t0): return dihed_prob2_int(xs, k=k, t0=t0, T=T)
    # bounds=optimize.Bounds([0, -360], [np.inf, 360])
    for shift in np.linspace(-180, 180, 8):
        try:
            popt_temp, _ = optimize.curve_fit(f_cdf, xs, np.cumsum(ys_target), p0=(np.min(ks), shift), maxfev=50000,
                                              bounds=((0, -360), (9999.9, 360)))
            # res=optimize.minimize(f, popt_temp, bounds=bounds, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
            # res=optimize.minimize(f_pdf, popt_temp, method='Nelder-Mead',tol=1e-8,options={'maxiter': 50000})
            try:
                res = optimize.minimize(
                    f_pdf, popt_temp, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
                res = res.x
            except:
                res = popt_temp
            print("Error in optimization")
            ys = dihed_prob2(xs, k=res[0], t0=res[1], T=T)
            lsq = EMD(ys_target, ys)
            if lsq_min < 0 or lsq < lsq_min:
                lsq_min = lsq
                popt = res
        except:
            pass
    if popt[1] < -180:
        popt[1] = popt[1]+360
    if popt[1] > 180:
        popt[1] = popt[1]-360
    return popt


def EMD(ys1, ys2):
    """Earthmovers' distance calculation"""
    return np.sum(np.abs((np.cumsum(ys1)/np.sum(ys1)-np.cumsum(ys2)/np.sum(ys2))))


def get_distances(trj, bead_definitions, bond_definitions, T=298.15,
                  N=128, hw=0.3, pw=None, dpi=100, tosave=False, isplot=True, printvalues=False):
    labels = ["{:s}-{:s}".format(*np.sort(np.array([bond[0], bond[1]])))
              for bond in bond_definitions]
    types = [bond[2] for bond in bond_definitions]
    header = "i -j            r0(A)   k(kJ/mol/A2)"
    A_all = []
    centers_all = []
    xs_all = []
    ys_all = []
    results = []
    aver_all = []
    sdev_all = []

    for i, bond in enumerate(bond_definitions):
        bead_index1 = [bead[0] for bead in bead_definitions].index(bond[0])
        bead_index2 = [bead[0] for bead in bead_definitions].index(bond[1])
        pos1 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=bond[0])
        pos2 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=bond[1])
        distances = np.linalg.norm(pos1-pos2, axis=1)
        distances = distances[np.isfinite(distances)]
        aver = np.average(distances)
        sdev = np.std(distances)
        aver_all.append(aver)
        sdev_all.append(sdev)

        if bond[3] != 'F':
            width = np.max(distances)-np.min(distances)
            xs = np.linspace(np.min(distances)-width,
                             np.max(distances)+width, N)
            xs_all.append(xs)
            A, B = np.histogram(distances, bins=xs, density=True)
            A = A/np.sum(A)
            A_all.append(A*hw/(xs[1]-xs[0]))
            centers = (B[1:]+B[:-1])/2
            centers_all.append(centers)

            def f_pdf(p): return EMD(A, harmonic_prob2(centers, k=p[0], r0=p[1], T=T)) if (
                0 <= p[0] <= 999999.9 and 0 <= p[1]) else 999999.9

            def f_cdf(centers, k, r0): return harmonic_prob2_int(
                centers, k=k, r0=r0, T=T)
            # bounds=optimize.Bounds([0, 0], [np.inf, 100.0])
            lsq_min = -1.0
            # for shift in np.linspace(np.min(distances),np.max(distances),16):
            for shift in np.linspace(aver-sdev*3, aver+sdev*3, 8):
                popt_temp, _ = optimize.curve_fit(f_cdf, centers, np.cumsum(A), p0=(1000, shift), maxfev=50000,
                                                  bounds=((0, 0), (999999.9, 100.0)))
                p0 = popt_temp
                # res=optimize.minimize(f, p0, bounds=bounds, method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
                # res=optimize.minimize(f_pdf, p0, method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
                try:
                    res = optimize.minimize(
                        f_pdf, p0, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
                    res = res.x
                except:
                    res = p0
                ys = harmonic_prob2(centers, k=res[0], r0=res[1], T=T)
                lsq = EMD(A, ys)
                if lsq_min < 0 or lsq < lsq_min:
                    lsq_min = lsq
                    popt = res

            ys = harmonic_prob2(centers, k=popt[0], r0=popt[1], T=T)
            ys_all.append(ys/np.sum(ys)*hw/(xs[1]-xs[0]))
            results.append([labels[i], types[i], popt[1],
                           popt[0], bond[3], bond[4]])
            if printvalues:
                print("{:<13s}{:<13s}{:>10.4f}{:>10.1f}  {:>3d} {:>3d}".format(labels[i], types[i], popt[1], popt[0],
                                                                               bead_index1+1, bead_index2+1))
        else:
            A_all.append([np.nan])
            centers_all.append([np.nan])
            xs_all.append([np.nan])
            ys_all.append([np.nan])
            results.append([labels[i], types[i], aver, 0.0, bond[3], bond[4]])
            if printvalues:
                print("{:<13s}{:<13s}{:>10.4f}{:10s}  {:>3d} {:>3d}".format(labels[i], types[i], aver, " ",
                                                                            bead_index1+1, bead_index2+1))

    if isplot == True:
        if pw is None:
            pw = len(bond_definitions)*3/5+1
        fig, ax = plt.subplots(figsize=(pw, 3), dpi=dpi)
        mpl.rcParams['hatch.linewidth'] = 2.0
        i = 0
        for A, centers, xs, ys, aver in zip(A_all, centers_all, xs_all, ys_all, aver_all):
            color = bond_definitions[i][5]
            if len(xs) == 1:
                plt.scatter(i+1.0, aver, s=100, lw=2.0,
                            alpha=0.65, color=color, hatch='////')
                plt.plot([i+1, i+1], [aver+sdev, aver-sdev], color=color,
                         lw=2.0, solid_capstyle='round', marker='_', mew=2.0, ms=8)
                plt.plot([i+1-0.2, i+1+0.2], [aver, aver],
                         color='k', lw=1.0, solid_capstyle='round')
            else:
                plt.plot(i+1.0-A/2, centers, color=color,
                         lw=2.0, solid_capstyle='round')
                plt.plot(i+1.0+A/2, centers, color=color,
                         lw=2.0, solid_capstyle='round')
                plt.fill_betweenx(centers, i+1.0-A/2, i+1.0+A/2,
                                  color=color, lw=0.0, alpha=0.65, hatch='////')
                plt.plot(+ys/2+i+1, centers, color='k',
                         lw=1.0, solid_capstyle='round')
            i += 1
        for i in range(1, len(labels)):
            plt.axvline(i+0.5, color='k', ls='--', lw=0.5)
        plt.xticks([*range(1, len(labels)+1)], labels=labels)
        plt.xlim(0.5, len(labels)+0.5)
        plt.ylabel(r"Bead-bead distance, $\mathrm{\AA}$")
        plt.tight_layout()
    return results, header


def get_angles(trj, bead_definitions, angle_definitions, T=298.15,
               N=128, hw=13, pw=None, dpi=100, tosave=False, isplot=True, printvalues=False):
    labels = ["{:s}-{:s}-{:s}".format(angle[0], angle[1], angle[2])
              for angle in angle_definitions]
    types = [angle[3] for angle in angle_definitions]

    header = "                  G96 angles            ReB angles "
    header = header+"i  j  k          Angle      k          Angle      k"

    A_all = []
    centers_all = []
    xs_all = []
    ys1_all = []
    ys2_all = []
    results = []

    for i, angle in enumerate(angle_definitions):

        bead_index1 = [bead[0] for bead in bead_definitions].index(angle[0])
        bead_index2 = [bead[0] for bead in bead_definitions].index(angle[1])
        bead_index3 = [bead[0] for bead in bead_definitions].index(angle[2])

        pos1 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=angle[0])
        pos2 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=angle[1])
        pos3 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=angle[2])

        angles = calc_angles(pos1, pos2, pos3)
        angles = angles[np.isfinite(angles)]
        angles[angles < 0] = 0
        angles[angles > 180] = 180
        width = np.max(angles)-np.min(angles)
        xs = np.linspace(max(np.min(angles)-width, 0),
                         min(np.max(angles)+width, 180), N)
        xs_all.append(xs)
        A, B = np.histogram(angles, bins=xs, density=True)
        A = A/np.sum(A)
        A_all.append(A*hw/(xs[1]-xs[0]))
        centers = (B[1:]+B[:-1])/2
        centers_all.append(centers)
        aver = np.average(angles)
        sdev = np.std(angles)

        def f_pdf(p): return EMD(A, angle_harmonic_prob(centers, k=p[0], t0=p[1], T=T)) if (
            0 <= p[0] <= 9999.9 and 0 <= p[1] < 180) else 999999.9

        def f_cdf(centers, k, t0): return angle_harmonic_prob_int(
            centers, k=k, t0=t0, T=T)
        # bounds=optimize.Bounds([0, 0], [np.inf, 180])
        lsq_min = -1.0
        # for shift in np.linspace(max(np.min(angles),0.5),min(np.max(angles),179.5),16):
        for shift in np.linspace(max(aver-sdev*3, 0.5), min(np.max(angles), 179.5), 8):
            popt_temp, _ = optimize.curve_fit(f_cdf, centers, np.cumsum(A), p0=(50.0, shift), maxfev=50000,
                                              bounds=((0, 0), (9999.9, 180)))
            p0 = popt_temp
            # res=optimize.minimize(f, p0, bounds=bounds, method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
            # res=optimize.minimize(f_pdf, p0, method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
            try:
                res = optimize.minimize(
                    f_pdf, p0, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
                res = res.x
            except:
                res = p0
            ys = angle_harmonic_prob(centers, k=res[0], t0=res[1], T=T)
            lsq = EMD(A, ys)
            if lsq_min < 0 or lsq < lsq_min:
                lsq_min = lsq
                popt = res

        popt1 = popt
        ys = angle_harmonic_prob(centers, k=popt1[0], t0=popt1[1], T=T)
        ys1_all.append(ys/np.sum(ys)*hw/(xs[1]-xs[0]))

        results.append([labels[i], types[i], popt1[1], popt1[0]])

        if printvalues:
            print("{:<15s}{:<13s}{:>8.1f}  {:>8.1f}  {:>3d} {:>3d} {:>3d}".format(
                labels[i], types[i], popt1[1], popt1[0],
                bead_index1+1, bead_index2+1, bead_index3+1))

    if isplot == True:
        if pw is None:
            pw = len(angle_definitions)*4.5/5+1
        fig, ax = plt.subplots(figsize=(pw, 3), dpi=dpi)
        mpl.rcParams['hatch.linewidth'] = 2.0
        i = 0
        for A, centers, xs, ys1 in zip(A_all, centers_all, xs_all, ys1_all):
            # A=A*hw
            # ys1=ys1*hw
            # ys2=ys2*hw
            color = angle_definitions[i][4]
            plt.plot(i+1.0-A/2, centers, color=color,
                     lw=2.0, solid_capstyle='round')
            plt.plot(i+1.0+A/2, centers, color=color,
                     lw=2.0, solid_capstyle='round')
            plt.fill_betweenx(centers, i+1.0-A/2, i+1.0+A/2,
                              color=color, lw=0.0, alpha=0.6, hatch='////')
            plt.plot(+ys1/2+i+1, centers, color='k',
                     lw=1.0, solid_capstyle='round')
            # plt.plot(-ys2/2+i+1,xs,color='k',lw=1.0,dash_capstyle='round',ls=(0,(2,2)))
            i += 1

        for i in range(1, len(angle_definitions)):
            plt.axvline(i+0.5, color='k', ls='--', lw=0.5)
        plt.xlim(0.5, len(angle_definitions)+0.5)
        plt.xticks([*range(1, len(labels)+1)], labels=labels)
        plt.yticks(np.arange(0, 181, 30))
        plt.ylim(0, 180)
        plt.ylabel("B-B-B angle, degrees")
        plt.tight_layout()
    return results, header


def get_dihedrals(trj, bead_definitions, dihedral_definitions, T=298.15,
                  N=128, hw=31.5, pw=None, dpi=100, tosave=False, isplot=True, printvalues=False):
    labels = ["{:s}-{:s}-{:s}-{:s}".format(dihed[0], dihed[1], dihed[2], dihed[3])
              for dihed in dihedral_definitions]
    types = [dihedral[4] for dihedral in dihedral_definitions]

    header = "i  j  k  l        Angle      k"

    A_all = []
    centers_all = []
    xs_all = []
    ys_all = []
    results = []

    for i, dihed in enumerate(dihedral_definitions):

        bead_index1 = [bead[0] for bead in bead_definitions].index(dihed[0])
        bead_index2 = [bead[0] for bead in bead_definitions].index(dihed[1])
        bead_index3 = [bead[0] for bead in bead_definitions].index(dihed[2])
        bead_index4 = [bead[0] for bead in bead_definitions].index(dihed[3])

        pos1 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=dihed[0])
        pos2 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=dihed[1])
        pos3 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=dihed[2])
        pos4 = get_bead_positions(
            trj, bead_definitions=bead_definitions, bead_name=dihed[3])

        dihedrals = calc_dihedral_multiple(pos1, pos2, pos3, pos4)
        dihedrals = dihedrals[np.isfinite(dihedrals)]
        average = np.degrees(np.arctan2(
            np.sum(np.sin(np.radians(dihedrals))), np.sum(np.cos(np.radians(dihedrals)))))
        dihedrals[dihedrals > average +
                  180] = dihedrals[dihedrals > average+180]-360
        dihedrals[dihedrals < average -
                  180] = dihedrals[dihedrals < average-180]+360

        xs = np.linspace(average-179.9, average+179.9, N)
        xs_all.append(xs)
        A, B = np.histogram(dihedrals, bins=xs, density=True)
        A = A/np.sum(A)
        A_all.append(A*hw/(xs[1]-xs[0]))
        centers = (B[1:]+B[:-1])/2
        centers_all.append(centers)

        def f_pdf(p): return EMD(A, dihed_prob2(centers, k=p[0], t0=p[1], T=T)) if (
            0 <= p[0] <= 9999.9 and -360 <= p[1] < 360) else 999999.9
        def f_cdf(centers, k, t0): return dihed_prob2_int(
            centers, k=k, t0=t0, T=T)
        # bounds=optimize.Bounds([0, -360], [np.inf, 360])
        lsq_min = -1.0
        for t0 in np.linspace(-170.0, 170, 16):
            popt_temp, _ = optimize.curve_fit(f_cdf, centers, np.cumsum(A), p0=(1, t0), maxfev=50000,
                                              bounds=((0, -360), (9999.9, 360)))
            p0 = popt_temp
            # res=optimize.minimize(f, p0, bounds=bounds, method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
            # res=optimize.minimize(f_pdf, p0, method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
            try:
                res = optimize.minimize(
                    f_pdf, p0, method='Nelder-Mead', tol=1e-3, options={'maxiter': 50000})
                res = res.x
            except:
                res = p0
            ys = dihed_prob2(centers, k=res[0], t0=res[1], T=T)
            lsq = EMD(A, ys)
            if lsq_min < 0 or lsq < lsq_min:
                lsq_min = lsq
                popt = res

        # popt[0]=np.degrees(np.arccos(np.cos(np.radians(popt[0]))))
        # popt[0]=np.abs(popt[0])
        if popt[1] > 180:
            while popt[1] > 180:
                popt[1] = popt[1]-360
        if popt[1] < -180:
            while popt[1] < -180:
                popt[1] = popt[1]+360

        ys = dihed_prob2(centers, k=popt[0], t0=popt[1], T=T)
        ys_all.append(ys/np.sum(ys)*hw/(xs[1]-xs[0]))

        results.append([labels[i], types[i], popt[1], popt[0], dihed[5]])
        if printvalues:
            print("{:<20s}{:<15s}{:>10.1f} {:>8.1f} {:>3d} {:>3d} {:>3d} {:>3d}".format(
                  labels[i], types[i], popt[1], popt[0],
                  bead_index1+1, bead_index2+1, bead_index3+1, bead_index4+1), end="")
            if dihed[5] == "F":
                print("   *")
            else:
                print("   ")

    if isplot == True:
        if pw is None:
            pw = len(dihedral_definitions)*5/4+1
        ncols = max(len(dihedral_definitions), 2)
        fig, axs = plt.subplots(ncols=ncols, figsize=(pw, 3), dpi=dpi)
        mpl.rcParams['hatch.linewidth'] = 2.0

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.spines.bottom.set_visible(False)

        i = 0
        for A, centers, xs, ys in zip(A_all, centers_all, xs_all, ys_all):
            color = dihedral_definitions[i][6]
            axs[i].spines.left.set_visible(True)
            axs[i].plot(1.0-A/2, centers, color=color, lw=2.0,
                        solid_capstyle='round', zorder=5)
            axs[i].plot(1.0+A/2, centers, color=color, lw=2.0,
                        solid_capstyle='round', zorder=5)
            axs[i].fill_betweenx(
                centers, 1.0-A/2, 1.0+A/2, color=color, lw=0.0, alpha=0.6, hatch='////', zorder=4)
            axs[i].plot(+ys/2+1, centers, color='k', lw=1.0,
                        solid_capstyle='round', zorder=5)
            axs[i].set_xticks([1.0], labels=[labels[i]])
            axs[i].set_yticks([*range(-360, 361, 45)])
            axs[i].set_ylim(xs[0], xs[-1])
            axs[i].set_xlim(0.6, 1.6)

            i += 1
        axs[0].set_ylabel("B-B-B-B dihedral angle, degrees")
    return results, header


def get_vs3(ri, rj, rk, rl):
    """Get virtual site vs3 vector parameters based on the GROMACS definition."""
    def vs(p): return (ri+p[0]*(rj-ri)+p[1] *
                       (rk-ri)+p[2]*np.cross(rj-ri, rk-ri))

    def dist(p): return np.sum(np.linalg.norm(vs(p)-rl, axis=1))
    res = optimize.minimize(dist, x0=[
                            1.0, 1.0, 1.0], method='Nelder-Mead', tol=1e-8, options={'maxiter': 50000})
    return res.x


def summarize_bond_data(result_list, printvalues=False):
    datalines = []
    for results in result_list:
        for res in results:
            bead1 = res[0].split("-")[0]
            bead2 = res[0].split("-")[1]
            if bead1 < bead2:
                datalines.append(["{:s}-{:s}".format(bead1, bead2), *res[1:]])
            else:
                datalines.append(["{:s}-{:s}".format(bead2, bead1), *res[1:]])
    datalines = sorted(datalines, key=lambda x: x[0])
    temp = [list(x) for x in zip(*datalines)]
    names = temp[0]
    types = temp[1]
    for i, type in enumerate(types):
        if type == 'name':
            types[i] = names[i]
    names = np.array(names)
    types = np.array(types)
    dists = np.array(temp[2])
    fcs = np.array(temp[3])
    isfrozen = np.array(temp[4])
    scales = np.array(temp[5])

    results = []
    for type in np.unique(types):
        sorted_values = np.ndarray.tolist(dists[types == type])
        (k, r0) = average_bond_results(r0_list=dists[types == type],
                                       k_list=fcs[types == type])
        results.append([type,
                        r0,
                        k,
                        isfrozen[types == type][0],
                        scales[types == type][0],
                        names[types == type][0],
                        len(types[types == type]),
                        sorted_values])
    constants, parameters = [], []
    for res in results:
        type = res[0]
        distance = res[1]
        fc = res[2]
        isfrozen = res[3]
        scale = res[4]
        example = res[5]
        pops = res[6]
        sorted_values = res[7]
        resline = [type, distance*scale, fc, pops, example]
        if scale == 1:
            resline.append('not_scaled')
        else:
            resline.append('scaled')
        resline.append(sorted_values)
        if isfrozen == 'F':
            constants.append(resline)
        if isfrozen == 'B':
            parameters.append(resline)
    if printvalues:
        print("Constants")
        for line in constants:
            print("{:<8s} {:8.4f} {:8.0f}    {:<3d}   {:<8s}  {:<10s}".format(
                *line[:-1]), end=" ")
            for value in line[-1]:
                print("{:>7.4f}".format(value), end=" ")
            print(" ")
        print("Parameters")
        for line in parameters:
            print("{:<8s} {:8.4f} {:8.0f}    {:<3d}   {:<8s}  {:<10s}".format(
                *line[:-1]), end=" ")
            for value in line[-1]:
                print("{:>7.4f}".format(value), end=" ")
            print(" ")
    return parameters, constants


def summarize_angle_data(result_list, printvalues=False):
    datalines = []
    for results in result_list:
        for res in results:
            bead1 = res[0].split("-")[0]
            bead2 = res[0].split("-")[1]
            bead3 = res[0].split("-")[2]
            if bead1 < bead3:
                datalines.append(
                    ["{:s}-{:s}-{:s}".format(bead1, bead2, bead3), *res[1:]])
            else:
                datalines.append(
                    ["{:s}-{:s}-{:s}".format(bead3, bead2, bead1), *res[1:]])
    datalines = sorted(datalines, key=lambda x: x[0])
    # print(datalines)
    temp = [list(x) for x in zip(*datalines)]
    names = temp[0]
    types = temp[1]
    for i, type in enumerate(types):
        if type == 'name':
            types[i] = names[i]
    names = np.array(names)
    types = np.array(types)
    angles = np.array(temp[2])
    fcs = np.array(temp[3])

    results = []
    for type in np.unique(types):
        sorted_values = np.ndarray.tolist(angles[types == type])
        (k, t0) = average_angle_harmonic_results(t0_list=angles[types == type],
                                                 k_list=fcs[types == type])
        results.append([type,
                        t0,
                        k,
                        names[types == type][0],
                        len(types[types == type]),
                        sorted_values])
    parameters, constants = [], []
    for res in results:
        type = res[0]
        angle = res[1]
        fc = res[2]
        example = res[3]
        pops = res[4]
        sorted_values = res[5]
        resline = [type, angle, fc, pops, example, sorted_values]
        parameters.append(resline)
    if printvalues:
        print("Parameters")
        for line in parameters:
            print("{:<12s} {:9.1f} {:9.1f}    {:<3d}  {:<12s}".format(
                *line[:-1]), end=" ")
            for value in line[-1]:
                print("{:>7.2f}".format(value), end=" ")
            print(" ")
    return parameters, constants


def summarize_dihedral_data(result_list, printvalues=False):
    datalines = []
    for results in result_list:
        for res in results:
            bead1 = res[0].split("-")[0]
            bead2 = res[0].split("-")[1]
            bead3 = res[0].split("-")[2]
            bead4 = res[0].split("-")[3]
            if bead2 < bead3:
                datalines.append(
                    ["{:s}-{:s}-{:s}-{:s}".format(bead1, bead2, bead3, bead4), *res[1:]])
            else:
                datalines.append(
                    ["{:s}-{:s}-{:s}-{:s}".format(bead4, bead3, bead2, bead1), *res[1:]])
    datalines = sorted(datalines, key=lambda x: x[0])
    temp = [list(x) for x in zip(*datalines)]
    names = temp[0]
    types = temp[1]
    for i, type in enumerate(types):
        if type == 'name':
            types[i] = names[i]
    names = np.array(names)
    types = np.array(types)
    diheds = np.array(temp[2])
    fcs = np.array(temp[3])
    isfrozen = np.array(temp[4])

    results = []
    for type in np.unique(types):
        sorted_values = np.ndarray.tolist(diheds[types == type])
        (k, t0) = average_dihedral_results(t0_list=diheds[types == type],
                                           k_list=fcs[types == type])
        results.append([type,
                        t0,
                        k,
                        names[types == type][0],
                        len(types[types == type]),
                        isfrozen[types == type][0],
                        sorted_values])
    constants, parameters = [], []
    for res in results:
        type = res[0]
        dihed = res[1]
        fc = res[2]
        example = res[3]
        pops = res[4]
        sorted_values = res[6]
        resline = [type, dihed, fc, pops, example, sorted_values]
        if res[5] == "F":
            constants.append(resline)
        else:
            parameters.append(resline)
    if printvalues:
        print("Constants")
        for line in constants:
            print("{:<16s} {:9.1f} {:9.1f}    {:<3d}  {:<16s}".format(
                *line[:-1]), end=" ")
            for value in line[-1]:
                print("{:>8.2f}".format(value), end=" ")
            print(" ")
        print("Parameters")
        for line in parameters:
            print("{:<16s} {:9.1f} {:9.1f}    {:<3d}  {:<16s}".format(
                *line[:-1]), end=" ")
            for value in line[-1]:
                print("{:>8.2f}".format(value), end=" ")
            print(" ")
    return parameters, constants
