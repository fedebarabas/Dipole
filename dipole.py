# -*- coding: utf-8 -*-
"""
This file gives the dipole radiation (E and B field) in the far field, the full
radiation (near field + far field) and the near field radiation only

@author: manu / Federico Barabas
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import jv
import scipy.interpolate
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

c = 299792458.
pi = np.pi
mu0 = 4*pi*1e-7
eps0 = 1./(mu0*c**2)


def hertz_dipole(r, p, R, phi, f, t=0, epsr=1.):
    """
    Calculate E and B field strength radiated by hertzian dipole(s).

    args:

    p: array of dipole moments
    [[px0, py0, pz0], [px1, py1, pz1], ..., [pxn, pyn, pzn]]

    R: array of dipole positions
    [[X0, Y0, Z0], [X1, Y1, Z1], ..., [Xn, Yn, Zn]]

    r: observation point
    [x, y, z]

    f: array of frequencies
    [f0, f1, ...]

    t: time

    phi: array with dipole phase angles (0-2pi)
    [phi0, phi1, ..., phin]

    return: fields values at observation point r at time t for every frequency
    in f. E and B are (3 components, number of frequencies) arrays.
    """
    nf = len(f)
    rprime = r - R    # r' = r-R
    if np.ndim(p) < 2:
        magrprime = np.sqrt(np.sum((rprime)**2))
        magrprimep = np.tile(magrprime,  (nf, 1)).T
        phip = np.tile(phi, (nf, 1))
        w = 2*pi*f            # \omega
        k = w/c               # wave number
        krp = k*magrprimep    # k|r'|
        rprime_cross_p = np.cross(rprime,  p)           # r'x p
        rp_c_p_c_rp = np.cross(rprime_cross_p, rprime)  # (r' x p) x r'
        rprime_dot_p = np.sum(rprime*p)
        expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
        e_ff = (np.tile(rp_c_p_c_rp, (nf, 1))).T
        e_nf = np.tile(3*rprime*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p.T, (nf, 1)).T
        b_ff = w**2*np.tile(rprime_cross_p, (nf, 1)).T
        E = expfac*(w**2/(c**2*magrprimep**3) * e_ff + (1/magrprimep**3-w*1j/(c*magrprimep**2))*e_nf)
        B = expfac/(magrprimep**2*c**3)*b_ff*(1-c/(1j*w*magrprimep))
    else:
        magrprime = np.sqrt(np.sum((rprime)**2, axis=1))
        magrprimep = np.tile(magrprime, (nf, 1)).T
        phip = np.tile(phi, (nf, 1))
        fp = np.tile(f, (len(magrprime), 1))
        w = 2*pi*fp           # \omega
        k = w/c               # wave number
        krp = k*magrprimep    # k|r'|
        rprime_cross_p = np.cross(rprime, p)            # r' x p
        rp_c_p_c_rp = np.cross(rprime_cross_p, rprime)  # (r' x p) x r'
        rprime_dot_p = np.sum(rprime*p, axis=1)
        expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
        e0 = w**2/(c**2*magrprimep**3)
        Ex = expfac*(e0 * (np.tile(rp_c_p_c_rp[:, 0], (nf, 1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:, 0]*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p[:, 0].T, (nf, 1)).T))
        Ey = expfac*(e0 * (np.tile(rp_c_p_c_rp[:, 1], (nf, 1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:, 1]*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p[:, 1].T, (nf, 1)).T))
        Ez = expfac*(e0 * (np.tile(rp_c_p_c_rp[:, 2], (nf, 1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:, 2]*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p[:, 2].T, (nf, 1)).T))
        Bx = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:, 0], (nf, 1)).T)*(1-c/(1j*w*magrprimep))
        By = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:, 1], (nf, 1)).T)*(1-c/(1j*w*magrprimep))
        Bz = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:, 2], (nf, 1)).T)*(1-c/(1j*w*magrprimep))
        E = np.vstack((np.sum(Ex, axis=0), np.sum(Ey, axis=0),
                       np.sum(Ez, axis=0)))
        B = np.vstack((np.sum(Bx, axis=0), np.sum(By, axis=0),
                       np.sum(Bz, axis=0)))
    return E, B


def hertz_dipole_ff(r, p, R, phi, f, t=0, epsr=1.):
    """
    Calculate E and B field strength radiated by hertzian dipole(s)
    in the far field.

    args:

    p: array of dipole moments
    [[px0, py0, pz0], [px1, py1, pz1], ..., [pxn, pyn, pzn]]

    R: array of dipole positions
    [[X0, Y0, Z0], [X1, Y1, Z1], ..., [Xn, Yn, Zn]]

    r: observation point
    [x, y, z]

    f: array of frequencies
    [f0, f1, ...]

    t: time

    phi: array with dipole phase angles (0-2pi)
    [phi0, phi1, ..., phin]

    return: fields values at observation point r at time t for every frequency
    in f. E and B are (3 components, number of frequencies) arrays.
    """
    nf = len(f)
    rprime = r-R    # r' = r-R
    if np.ndim(p) < 2:
        magrprime = np.sqrt(np.sum((rprime)**2))
        w = 2*pi*f             # \omega
        krp = w*magrprime/c    # k|r'|
        rprime_cross_p = np.cross(rprime,  p)            # r'x p
        rp_c_p_c_rp = np.cross(rprime_cross_p,  rprime)  # (r' x p) x r'
        expfac = np.exp(1j*(w*t-krp+phi))/(4*pi*eps0*epsr)
        e0 = w**2/(c**2*magrprime**3)
        b0 = expfac/(magrprime**2*c**3)
        E = e0 * expfac * rp_c_p_c_rp[:, np.newaxis]
        B = b0 * w**2 * rprime_cross_p[:, np.newaxis]
    else:
        magrprime = np.sqrt(np.sum((rprime)**2, axis=1))    # |r'|
        magrprimep = np.tile(magrprime,  (nf, 1)).T
        phip = np.tile(phi,  (nf, 1))
        fp = np.tile(f, (len(magrprime), 1))
        w = 2*pi*fp           # \omega
        k = w/c               # wave number
        krp = k*magrprimep    # k|r'|
        rprime_cross_p = np.cross(rprime,  p)               # r'x p
        rp_c_p_c_rp = np.cross(rprime_cross_p,  rprime)     # (r' x p) x r'
        expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
        e0 = w**2/(c**2*magrprimep**3)
        b0 = expfac/(magrprimep**2*c**3)
        Ex = (e0 * expfac) * (np.tile(rp_c_p_c_rp[:, 0], (nf, 1))).T
        Ey = (e0 * expfac) * (np.tile(rp_c_p_c_rp[:, 1], (nf, 1))).T
        Ez = (e0 * expfac) * (np.tile(rp_c_p_c_rp[:, 2], (nf, 1))).T
        Bx = b0*(w**2*np.tile(rprime_cross_p[:, 0], (nf, 1)).T)
        By = b0*(w**2*np.tile(rprime_cross_p[:, 1], (nf, 1)).T)
        Bz = b0*(w**2*np.tile(rprime_cross_p[:, 2], (nf, 1)).T)
        E = np.vstack((np.sum(Ex, axis=0), np.sum(Ey, axis=0),
                       np.sum(Ez, axis=0)))
        B = np.vstack((np.sum(Bx, axis=0), np.sum(By, axis=0),
                       np.sum(Bz, axis=0)))
    return E, B


def hertz_dipole_nf(r, p, R, phi, f, t=0, epsr=1.):
    """
    Calculate E and B field strength radiated by hertzian dipole(s) in the near
    field.
    args:

    p: array of dipole moments
    [[px0, py0, pz0], [px1, py1, pz1], ..., [pxn, pyn, pzn]]

    R: array of dipole positions
    [[X0, Y0, Z0], [X1, Y1, Z1], ..., [Xn, Yn, Zn]]

    r: observation point
    [x, y, z]

    f: array of frequencies
    [f0, f1, ...]

    t: time

    phi: array with dipole phase angles (0-2pi)
    [phi0, phi1, ..., phin]

    return: fields values at observation point r at time t for every frequency
    in f. E and B are (3 components, number of frequencies) arrays.
    """
    nf = len(f)
    rprime = r-R    # r'=r-R
    if np.ndim(p) < 2:
        magrprime = np.sqrt(np.sum((rprime)**2))
        magrprimep = np.tile(magrprime,  (nf, 1)).T
        phip = np.tile(phi, (nf, 1))
        w = 2*pi*f      # \omega
        k = w/c         # wave number
        krp = k*magrprimep    # k|r'|
        rprime_cross_p = np.cross(rprime,  p)       # r'x p
        rprime_dot_p = np.sum(rprime*p)
        expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
        E = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p.T, (nf, 1)).T))
        B = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p, (nf, 1)).T)*1j
    else:
        magrprime = np.sqrt(np.sum((rprime)**2, axis=1))    # |r'|
        magrprimep = np.tile(magrprime,  (nf, 1)).T
        phip = np.tile(phi,  (nf, 1))
        fp = np.tile(f, (len(magrprime), 1))
        w = 2*pi*fp     # \omega
        k = w/c         # wave number
        krp = k*magrprimep    # k|r'|
        rprime_cross_p = np.cross(rprime,  p)    # r' x p
        rprime_dot_p = np.sum(rprime*p, axis=1)  # r'.p
        expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
        Ex = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:, 0]*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p[:, 0].T, (nf, 1)).T))
        Ey = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:, 1]*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p[:, 1].T, (nf, 1)).T))
        Ez = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:, 2]*rprime_dot_p, (nf, 1)).T/magrprimep**2-np.tile(p[:, 2].T, (nf, 1)).T))
        Bx = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[:, 0], (nf, 1)).T)*1j
        By = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[:, 1], (nf, 1)).T)*1j
        Bz = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[:, 2], (nf, 1)).T)*1j
        E = np.vstack((np.sum(Ex, axis=0), np.sum(Ey, axis=0),
                       np.sum(Ez, axis=0)))
        B = np.vstack((np.sum(Bx, axis=0), np.sum(By, axis=0),
                       np.sum(Bz, axis=0)))
    return E, B


def plot_hertz():

    # observation points
    nx = 401
    xmax = 2
    nz = 201
    zmax = 1
    x = np.linspace(-xmax, xmax, nx)
    y = 0
    z = np.linspace(-zmax, zmax, nz)

    # dipole
    f = np.array([1000e6])
    # dipole moment
    # total time averaged radiated power P= 1 W dipole moment
    # => |p| = sqrt(12pi*c*P/muO/w**4)
    Pow = 1
    norm_p = np.sqrt(12*pi*c*Pow/(mu0*(2*pi*f)**4))
    # dipole moment
    p = np.array([0, 0, norm_p[0]])
    R = np.array([0, 0, 0])
    # dipole phases
    phi = 0

    t0 = 1/f/10
    t1 = 5/f
    nt = int(t1/t0)
    t = np.linspace(t0, t1, nt)

    print("Computing the radiation...")
#    fig = plt.figure(num=1, figsize=(10, 6), dpi=300)
#    for k in np.arange(nt):
    for k in [0]:
        P = np.zeros((nx, nz))
        for i in np.arange(nx):
            for j in np.arange(nz):
                r = np.array([x[i], y, z[j]])
                E, B = hertz_dipole_ff(r, p, R, phi, f, t[k], epsr=1.)
                S = np.real(E)**2    # 0.5*np.cross(E.T, conjugate(B.T))
                P[i, j] = sum(S)
#        print(('%2.1f/100' % ((k+1)/nt*100)))
        # Radiation diagram
    plt.figure()
    plt.pcolormesh(x, z, P[:, :].T, cmap='hot')
#    fname = 'img_%s' % (k)
    plt.clim(0, 1000)
    plt.axis('scaled')
    plt.xlim(-xmax, xmax)
    plt.ylim(-zmax, zmax)
    plt.xlabel(r'$x/$m')
    plt.ylabel(r'$z/$m')
    plt.title(r'$t=%2.2f$ ns' % (t[k]/1e-9))
#        print('Saving frame', fname)
#        fig.savefig(fname+'.png', bbox='tight')
    plt.show()
#        plt.clf()


def psf(r, z, lamb=670e-09):
    """ Widefield PSF. The integral is real, so in this case I don't need to
    separate in real and imaginary integrals."""

    k = 2*pi/lamb

    def func(th):
        j0 = jv(0, k*r*np.sin(th))
        return np.sqrt(np.cos(th))*j0*np.exp(-1j*k*z*np.cos(th))*np.sin(th)

#    def real_func(x):
#        return scipy.real(func(x))
#    def imag_func(x):
#        return scipy.imag(func(x))
#    real_int = quad(real_func, 0, 1.2)
#    imag_int = quad(imag_func, 0, 1.2)
#    return np.sqrt(real_int[0]**2 + imag_int[0]**2)**2

    result = quad(func, 0, 1.2)
    return result[0]**2


def plot_psf(r, z, h):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(r*1e6, z*1e6, h, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_zslice(h, z):

    thetas = np.radians(np.linspace(0, 360, h[z].shape[1]))
    rhos = np.arange(0, 1, 0.01)    # [um]
    xx, yy = np.meshgrid(thetas, rhos)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.pcolormesh(xx, yy, h[z])
    plt.show()


def plot_xslice(h, x):

    data = h[:, :, x]
    h2 = np.zeros((data.shape[0], 2*data.shape[1]))
    h2[:, 0.5*h2.shape[1]:] = data
    h2[:, :0.5*h2.shape[1]:] = np.fliplr(data)

    plt.imshow(h2, interpolation='None')


def cyl_to_cart(rho_coord, theta_coord, z_coord, data):
    rhoflat = np.array(rho_coord.flat)
    theflat = np.array(theta_coord.flat)
    zflat = np.array(z_coord.flat)
    coordpoints = np.concatenate([rhoflat[:, np.newaxis],
                                  theflat[:, np.newaxis],
                                  zflat[:, np.newaxis]], axis=1)
    rtz_interpolator = scipy.interpolate.LinearNDInterpolator(coordpoints,
                                                              data.flat)
    return rtz_interpolator


def xyz2rtz(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/z)
    return (rho, theta, z)


def sph_to_cart(r_coord, theta_coord, phi_coord, data):
    rflat = np.array(r_coord.flat)
    tflat = np.array(theta_coord.flat)
    pflat = np.array(phi_coord.flat)
    coordpoints = np.concatenate([rflat[:, np.newaxis], tflat[:, np.newaxis],
                                  pflat[:, np.newaxis]], axis=1)
    rtp_interpolator = scipy.interpolate.LinearNDInterpolator(coordpoints,
                                                              data.flat)
    return rtp_interpolator


def xyz2rtp(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    t = np.acos(z/r)
    p = np.arctan2(y, x)
    return (r, t, p)


datac = np.zeros((100, 100, 100))
for (xi, yi, zi) in zip(xx.ravel(), yy.ravel(), zz.ravel()):
    datac[int(xi*1e08), int(yi*1e08), int(zi*1e08)] = rtp_interpolator(xyz2rtz(xi, yi, zi))


# now you can get the interpolated value for any (x,y,z) coordinate you want.
# val = rtpinterpolator(xyz2rtp(x, y, z))

if __name__ == "__main__":

    rhomax = 1e-06
    zmax = 2e-06
    n = 100
    drho, dz, dtheta = rhomax/n, zmax/n, 2*pi/n
    rho, z = np.mgrid[0:rhomax:drho, -zmax:zmax:dz]
    rho = rho.T
    z = z.T

    h = np.array([psf(ri, zi) for (ri, zi) in zip(rho.ravel(), z.ravel())])
    h = h.reshape(rho.shape)

    # This problem has theta symmetry
    h2 = np.zeros((h.shape[0], 100, h.shape[1]))
    for (x, y), value in np.ndenumerate(h):
        h2[x, :, y] = h[x, y]

#    plot_psf(r, z, h)
