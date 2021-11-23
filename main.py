import os
import numpy as np
import laspy as lp
import matplotlib.pyplot as plt
import math
from scipy import special, interpolate, signal
from scipy.optimize import curve_fit, least_squares
from scipy.stats import linregress
import uncertainties
from sklearn import linear_model
import geopandas as gpd
import json
import statsmodels.api as sm

# def scarp_ss(x, H, D, b ):
#     Q = H / D
#     u = (H * special.erf(x / (2 * np.sqrt(D)))) + ((Q * x**2)/2)*(special.erf(x / (2*np.sqrt(D))) - np.sign(x)) + \
#         ((Q * x) * np.sqrt(D / math.pi) * np.exp((-1 * x**2)/(4 * D))) + (b * x)
#     return u


def scarp_ss(x, h, d):
    """
    Generates a scarp profile without far-field slope (b*x). This is using the steady-state uplift case of Hanks (2000).
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    h : float
        Scarp half-height
    d : float
        Value of kappa * t for scarp, where kappa is the diffusion constant and t is time.

    Returns
    -------
    u : numpy.ndarray
        Elevation measurements along the scarp. These are detrended from the external slope (b).

    """
    q = h / d
    u = (h * special.erf(x / (2 * np.sqrt(d)))) + ((q * x**2)/2)*(special.erf(x / (2*np.sqrt(d))) - np.sign(x)) + \
        ((q * x) * np.sqrt(d / math.pi) * np.exp((-1 * x**2)/(4 * d)))
    return u


# def scarp_1e(x, H, D, b):
#     u = ((H) * special.erf(x / (2 * np.sqrt(D)))) + (b * x)
#     return u


def scarp_1e(x, h, d):
    """
    Generates a scarp profile without far-field slope (b*x). This is using the single-event uplift case of Hanks (2000).
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    h : float
        Scarp half-height
    d : float
        Value of kappa * t for scarp, where kappa is the diffusion constant and t is time.

    Returns
    -------
    u : numpy.ndarray
        Elevation measurements along the scarp. These are detrended from the external slope (b).

    """
    u = (h * special.erf(x / (2 * np.sqrt(d))))
    return u


def init_geom(x, h, b):
    """
    Generates a scarp profile without any degradation.
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    h : float
        Scarp half-height
    b : numpy.ndarray
        values of far-field slope (1-d array like x)

    Returns
    -------
    z : numpy.ndarray
        Elevation measurements along the scarp.

    """
    z = ((h * np.sign(x)) + (b * x))
    return z


def gen_b_from_two(x, b1, b2):
    """
    Generates b vector from uphill and downhill slope.
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    b1 : float
        Lower slope value
    b2 : float
        Upper slope value

    Returns
    -------
    b : numpy.ndarray
        values of far-field slope (1-d array like x)

    """
    xs = np.sign(x)
    b = np.empty_like(xs)
    b[xs < 0] = b1
    b[xs > 0] = b2
    b[xs == 0] = 0
    return b


def geom_two_slopes(x, h, b1, b2):
    """
    Generates a scarp profile without any degradation, with both slope values directly input.
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    h : float
        Scarp half-height
    b1 : float
        Lower slope value
    b2 : float
        Upper slope value

    Returns
    -------
    z : numpy.ndarray
        Elevation measurements along the scarp.

    """
    b = gen_b_from_two(x, b1, b2)
    z = ((h * np.sign(x)) + (b * x))
    return z


def grid_search_d(fun, d_min, d_max, d_step, x, z, h):
    """
    Performs a grid-search for the D parameter for a degradation model.
    Parameters
    ----------
    fun : function
        Function to perform grid search on. Either "scarp_ss" or "scarp_1e".
    d_min : float
        Minimum D for grid space
    d_max : float
        Maximum D for grid space
    d_step : float
        Step for D iteration
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    z : numpy.ndarray
        Elevation measurements along the scarp. Detrended from b (slope).
    h : float
        Scarp half-height

    Returns
    -------
    opt_d : float
            Optimum D that minimizes RMSE.

    """
    x_step = 2.5
    d_range = np.arange(d_min, d_max, d_step)
    d_out = np.empty((d_range.shape[0], 3))
    x_max = np.floor(x.max())
    x_min = np.ceil(x.min())
    x_new = np.arange(x_min, x_max, x_step)
    gfg = interpolate.interp1d(x, z)
    z_new = gfg(x_new)
    for ind, d in enumerate(d_range):
        u = fun(x_new, h, d)
        rmse = np.sqrt(np.mean((z_new - u)**2))
        mae = np.mean(np.abs(u - z_new))
        d_out[ind, 0] = d
        d_out[ind, 1] = rmse
        d_out[ind, 2] = mae
    minrmse = np.amin(d_out[:, 1])
    minmae = np.amin(d_out[:, 2])
    opt_ind = np.where(d_out[:, 1] == minrmse)
    opt_d = d_out[opt_ind, 0][0]
    return opt_d


def fit_prof_mid(x, z):
    """
    Fits profile midpoint. Somewhat depreciated in my eyes (sgp), please use "dsp_scarp_identify" instead.
    Parameters
    ----------
    x : numpy.ndarray
        x-values (1-d array) along scarp, either starting at beginning or end.
    z : numpy.ndarray
        z-values (1-d array) along scarp.

    Returns
    -------
    opt_midx : float
                Solved midpoint (x)
    opt_midz : float
                Solved midpoint (z)
    H_guess : float
            Best guess for scarp half-height (H)
    b_guess : float
            Best Guess for scarp far-field slope (b)

    """
    if z[0] > z[-1]:
        I = np.argsort(x)
        x = x[I]
        z = z[I]
        z = np.flip(z)
        x = x[-1] - np.flip(x)
    if x.shape[0] > 5000:
        samp_step = 100
        n_out_rows = (x.shape[0] // samp_step) + 1
    else:
        samp_step = 1
        n_out_rows = x.shape[0]
    output = np.empty((n_out_rows, 6))
    inc = 0
    for i in range(0, x.shape[0], samp_step):
        midz = z[i]
        midx = x[i]
        z1 = z - midz
        x1 = x - midx
        popt, pcov = curve_fit(init_geom, x1, z1)
        rmse = np.sqrt(np.mean((z1 - init_geom(x1, *popt))**2))
        mean_resid = np.mean(z1 - init_geom(x1, *popt))
        outrow = [midx, midz, popt[0], popt[1], rmse, mean_resid]
        output[inc, :] = outrow
        inc += 1

    minrmse = np.amin(output[:, 4])
    opt_ind = np.where(output[:, 4] == minrmse)
    opt_results = output[opt_ind, :][0]
    opt_midx = opt_results[0, 0]
    opt_midz = opt_results[0, 1]
    H_guess = opt_results[0, 2]
    b_guess = opt_results[0, 3]
    return opt_midx, opt_midz, H_guess, b_guess


def refine_b(x, z, midx, midz):
    """
    Better fits results from fit_prof_mid. Again, use dsp_scarp_identify.
    Parameters
    ----------
    x : numpy.ndarray
        x-values (1-d array) along scarp, either starting at beginning or end.
    z : numpy.ndarray
        z-values (1-d array) along scarp.
    midx : float
        Solved midpoint (x)
    midz : float
        Solved midpoint (z)

    Returns
    -------
    H : float
        Scarp half-height.
    b1 : float
        Lower scarp far-field slope.
    b2 : float
        Upper scarp far-field slope.

    """
    if z[0] > z[-1]:
        I = np.argsort(x)
        x = x[I]
        z = z[I]
        z = np.flip(z)
        x = x[-1] - np.flip(x)
    x1 = x - midx
    z1 = z - midz
    # bmin = b_guess - (b_guess * 1.5)
    # bmax = b_guess + (b_guess * 1.5)
    # bound1 = ([bmin, bmin], [bmax, bmax])
    # guess = [b_guess, b_guess]
    popt, pcov = curve_fit(geom_two_slopes, x1, z1)
    (H, b1, b2) = uncertainties.correlated_values(popt, pcov)
    return H, b1, b2


def dsp_scarp_identify(x, z):
    """
    Fits profile midpoint. Uses signal processing tools to find midpoint.
    Parameters
    ----------
    x : numpy.ndarray
        x-values (1-d array) along scarp, either starting at beginning or end.
    z : numpy.ndarray
        z-values (1-d array) along scarp.

    Returns
    -------
    opt_midx : float
                Scarp midpoint (x)
    opt_midz : float
                Scarp midpoint (z)
    b1_out : uncertainties.ufloat
            Lower far-field slope (b)
    b2_out : uncertainties.ufloat
            Upper far-field slope (b)
    opt_h : uncertainties.ufloat
            Scarp half-height (H). Uncertainty does not include covariance with b values.
    """
    sample_dist = 1
    Hmin = 0.1
    Hmax = z.max() - z.min()
    I = np.argsort(x)
    x = x[I]
    z = z[I]
    if z[0] > z[-1]:
        z = np.flip(z)
        x = x[-1] - np.flip(x)
    x_max = np.floor(x.max())
    if x_max < 100:
        sample_dist = 0.5
    x_new = np.arange(0, x_max, sample_dist)
    gfg = interpolate.interp1d(x, z, fill_value='extrapolate')
    z_new = gfg(x_new)
    z_filt = signal.savgol_filter(signal.detrend(z_new), 17, 1)
    z_slope = signal.savgol_filter(signal.detrend(z_new), 17, 1, deriv=1)
    peaks, _ = signal.find_peaks(z_slope)
    peak_width_res = signal.peak_widths(z_slope, peaks, rel_height=1)
    max_peak = peaks[np.argmax(peak_width_res[0])]
    max_peak_width = np.max(peak_width_res[0])
    opt_midx = x_new[max_peak]
    opt_midz = z_new[max_peak]

    x_upper = x_new[(z_slope < 0) & (z_new > opt_midz)] - opt_midx
    x_upper = x_upper.reshape((-1, 1))
    x_lower = x_new[(z_slope < 0) & (z_new < opt_midz)] - opt_midx
    x_lower = x_lower.reshape((-1, 1))
    z_upper = z_new[(z_slope < 0) & (z_new > opt_midz)] - opt_midz
    z_lower = z_new[(z_slope < 0) & (z_new < opt_midz)] - opt_midz
    # upper_model = linear_model.LinearRegression().fit(x_upper, z_upper)
    # lower_model = linear_model.LinearRegression().fit(x_lower, z_lower)
    # b1 = lower_model.coef_
    # b2 = upper_model.coef_
    lower_ols_x = sm.add_constant(x_lower, prepend=False)
    upper_ols_x = sm.add_constant(x_upper, prepend=False)
    lower_model = sm.OLS(z_lower, lower_ols_x)
    upper_model = sm.OLS(z_upper, upper_ols_x)

    lower_results = lower_model.fit()
    upper_results = upper_model.fit()
    b1 = lower_results.params[0]
    b2 = upper_results.params[0]
    int1 = lower_results.params[1]
    int2 = upper_results.params[1]
    lower_SE = lower_results.HC0_se
    upper_SE = upper_results.HC0_se
    b1_se = lower_SE[0]
    b2_se = upper_SE[0]
    int1_se = lower_SE[1]
    int2_se = upper_SE[1]

    b1_out = uncertainties.ufloat(b1, b1_se)
    b2_out = uncertainties.ufloat(b2, b2_se)
    int_1_out = uncertainties.ufloat(int1, int1_se)
    int_2_out = uncertainties.ufloat(int2, int2_se)

    x_lower_ind = (np.abs(x - opt_midx - x_lower.max())).argmin()
    x_upper_ind = (np.abs(x - opt_midx - x_upper.min())).argmin()

    x_real_lower = x[0:x_lower_ind] - opt_midx
    x_real_lower = x_real_lower.reshape((-1, 1))
    x_real_lower = sm.add_constant(x_real_lower, prepend=False)
    x_real_upper = x[x_upper_ind:-1] - opt_midx
    x_real_upper = x_real_upper.reshape((-1, 1))
    x_real_upper = sm.add_constant(x_real_upper, prepend=False)
    z_real_lower = z[0:x_lower_ind] - opt_midz
    z_real_upper = z[x_upper_ind:-1] - opt_midz
    z_lower_reg = lower_results.get_prediction(x_real_lower).predicted_mean
    z_upper_reg = upper_results.get_prediction(x_real_upper).predicted_mean

    lower_resid = np.mean(z_real_lower - z_lower_reg)
    upper_resid = np.mean(z_real_upper - z_upper_reg)

    avg_resid = np.mean([lower_resid, upper_resid])

    if np.abs(avg_resid) > 0.05:
        opt_midz = opt_midz - avg_resid
        idx = (np.abs(z_new - opt_midz)).argmin()
        opt_midx = x_new[idx]

    # H is half-height. Difference between y-intercepts in a system centered on the scarp
    opt_h = (int_2_out - int_1_out) / 2

    # IF mid-distance of scarp is not equal to lower intercept, adjust so that midpoint is at actual midpoint
    z_adj = opt_h.n - np.abs(int_1_out.n)
    if np.abs(z_adj) > 0.05:
        opt_midz = opt_midz + z_adj
        idx = (np.abs(z_new - opt_midz)).argmin()
        opt_midx = x_new[idx]

    return opt_midx, opt_midz, b1_out, b2_out, opt_h


def fit_1event(x, z, xmid, zmid, b, H_guess):
    """
    Fits single-event model.
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    z : numpy.ndarray
        Elevation measurements along the scarp.
    xmid : float
                Scarp midpoint (x)
    zmid : float
                Scarp midpoint (z)
    b : numpy.ndarray
        Slope values along scarp (b) (1-d array)
    H_guess : uncertainties.ufloat
            Solved H value

    Returns
    -------
    H : uncertainties.ufloat
        Scarp half-height.
    D : uncertainties.ufloat
        Solved value for kappa*t. Uncertainty is just derived from H unc.

    """
    # H_unc = 0.15
    # H_unc_abs = np.abs(H_guess * H_unc)
    H_unc = H_guess.s / H_guess.n
    x1 = x - xmid
    z1 = z - zmid
    D_min = 0.1
    D_max = 750.
    D_step = 0.1
    # b = gen_b_from_two(x1, b1, b2)
    z1 = z1 - (x1 * b)
    opt_d = grid_search_d(scarp_1e, D_min, D_max, D_step, x1, z1, H_guess.n)
    D = uncertainties.ufloat(opt_d, (H_unc * opt_d))
    H = H_guess
    # H = uncertainties.ufloat(H_guess, H_unc_abs)
    return H, D


def fit_ss_uplift(x, z, xmid, zmid, b, H_guess):
    """
    Fits steady-state uplift model.
    Parameters
    ----------
    x : numpy.ndarray
        x values (1-d array) centered on scarp midpoint
    z : numpy.ndarray
        Elevation measurements along the scarp.
    xmid : float
                Scarp midpoint (x)
    zmid : float
                Scarp midpoint (z)
    b : numpy.ndarray
        Slope values along scarp (b) (1-d array)
    H_guess : uncertainties.ufloat
            Solved H value

    Returns
    -------
    H : uncertainties.ufloat
        Scarp half-height.
    D : uncertainties.ufloat
        Solved value for kappa*t. Uncertainty is just derived from H unc.

    """
    # H_unc = 0.15
    # H_unc_abs = np.abs(H_guess *
    H_unc = H_guess.s / H_guess.n
    x1 = x - xmid
    z1 = z - zmid
    D_min = 0.1
    D_max = 5000
    D_step = 0.1
    # b = gen_b_from_two(x1, b1, b2)
    z1 = z1 - (x1 * b)
    # popt, pcov = curve_fit(scarp_ss, x1, z1, p0=guess, bounds=bound1)
    # (H, D) = uncertainties.correlated_values(popt, pcov)
    opt_d = grid_search_d(scarp_ss, D_min, D_max, D_step, x1, z1, H_guess.n)
    D = uncertainties.ufloat(opt_d, (H_unc * opt_d))
    H = H_guess
    return H, D


def condition_las_profile(file):
    """
    Conditions .las (or equivalent) file for input.
    Parameters
    ----------
    file : str
        Path to .las (or other pointcloud readable from laspy) file.

    Returns
    -------
    x : numpy.ndarray
        1-d array of x-values
    z : numpy.ndarray
        1-d array of z-values

    """
    lasfile = lp.read(file)
    lin_results = linregress(lasfile.x, lasfile.y)
    slope = lin_results.slope
    azimuth = math.atan(slope)
    c, s = np.cos(azimuth), np.sin(azimuth)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    pts = np.array([lasfile.x, lasfile.y, lasfile.z])
    pts = pts.T
    new_pts = pts @ R.T
    I = np.argsort(new_pts[:, 0])
    new_pts = new_pts[I, :]
    x = new_pts[:, 0] - new_pts[0, 0]
    z = new_pts[:, 2]

    return x, z


def extract_lidar_profile(profile, tindex, profname, prof_buff=2.5, out_las_dir="./"):
    """
    Creates json files for use in pdal pipeline. Run pipeline as pdal pipeline "file"
    Parameters
    ----------
    profile : shapely.LineString
        line geometry indicating profile. Use helper function to extract
    tindex : path
        Path to tile index
    profname : str
            Name of profile
    prof_buff : float
                Width of profile
    out_las_dir : str
                path to where output files are saved

    Returns
    -------

    """
    profile_wide = profile.buffer(prof_buff, cap_style=2)
    tiles = gpd.read_file(tindex)
    tile_sel = tiles[tiles.intersects(profile_wide)]
    list_of_tiles = tile_sel.location.values
    out_las = "".join([profname, ".las"])
    out_json_name = "".join([profname, ".json"])
    # outdir = os.path.dirname(tindex)
    out_path = os.path.join(out_las_dir, out_las)
    out_json = os.path.join(out_las_dir, out_json_name)
    crop_filter = {"type": "filters.crop", "polygon": profile_wide.wkt}
    writer_text = out_path
    if len(list_of_tiles) > 1:
        inputs = list_of_tiles.tolist()
        inputs.append({"type": "filters.merge"})
        inputs.append(crop_filter)
        inputs.append(writer_text)
        with open(out_json, 'w') as outfile:
            json.dump(inputs, outfile, indent=4)
        # json_input = json.dumps(inputs)
        # reader = pdal.Pipeline(json_input)
        # count = reader.execute()

    elif len(list_of_tiles) == 1:
        inputs = list_of_tiles.tolist()
        inputs.append(crop_filter)
        inputs.append(writer_text)
        with open(out_json, 'w') as outfile:
            json.dump(inputs, outfile, indent=4)
        # json_input = json.dumps(inputs)
        # reader = pdal.Pipeline(json_input)
        # count = reader.execute()

    else:
        raise ValueError("No overlapping tiles")

    return


def iterate_prof_shp(prof_shp, tindex, out_dir="./"):
    """
    Iterate over a shapefile to generate files to extract profiles from pointclouds
    Parameters
    ----------
    prof_shp : path
                Path to shapefile (or other geopandas compatible file) containing profiles
    tindex : path
            Path to pdal tile index (tindex)
    out_dir : path
            Path to output files

    Returns
    -------

    """
    profs = gpd.read_file(prof_shp)
    for prof_ind in range(len(profs)):
        print(prof_ind)
        prof = profs.iloc[prof_ind]
        geom = prof.geometry
        prof_name = prof['ProfName']

        extract_lidar_profile(geom, tindex, prof_name, out_las_dir=out_dir)
    return


class Scarp:
    """
    General class to manage scarps. Initiate with x, z, name= (optional)
    """
    def __init__(self, x, z, b_fit='dsp', name=''):
        self.default_unc = 0.001
        self.aspect = 1
        self.name = name
        self.num_sim = 1000
        x_step = 2.5

        I = np.argsort(x)
        self.x = x[I]
        self.z = z[I]
        if z[0] > z[-1]:
            self.z = np.flip(self.z)
            self.x = x[-1] - np.flip(x)
        x_max = np.floor(self.x.max())
        if x_max < 100:
            x_step = 0.25
        x_min = np.ceil(self.x.min())
        self.x_lowres = np.arange(x_min, x_max, x_step)
        interp_scarp = interpolate.interp1d(self.x, self.z)
        self.z_lowres = interp_scarp(self.x_lowres)
        if b_fit == 'dsp':
            self.midx, self.midz, self.b1, self.b2, self.Hinit = dsp_scarp_identify(self.x, self.z)
            # self.Hinit = 1.0
        else:

            self.midx, self.midz, Hinit, binit = fit_prof_mid(self.x_lowres, self.z_lowres)

            # Hinit, self.b1, self.b2 = refine_b(x_lowres, z_lowres, self.midx, self.midz)
            binit = uncertainties.ufloat(binit, self.default_unc)
            self.Hinit = uncertainties.ufloat(Hinit, self.default_unc * Hinit)
            self.b1 = binit
            self.b2 = binit
        self.b = gen_b_from_two(self.x - self.midx, self.b1.n, self.b2.n)
        self.b_sim = np.empty((self.x.shape[0], self.num_sim), dtype=float)
        b1_sims = (self.b1.s * np.random.randn(self.num_sim)) + self.b1.n
        b2_sims = (self.b2.s * np.random.randn(self.num_sim)) + self.b2.n
        for i in range(self.num_sim):
            b_n = gen_b_from_two(self.x - self.midx, b1_sims[i], b2_sims[i])
            self.b_sim[:, i] = b_n
        self.H1 = None
        self.D1 = None
        self.Hs = None
        self.Ds = None
        self.H1_sim = None
        self.D1_sim = None
        self.Hs_sim = None
        self.Ds_sim = None
        self.active_fig = None

    def gen_1e(self):
        """
        Generates results for 1 event model (no uncertainty handling)
        Returns
        -------

        """
        self.H1, self.D1 = fit_1event(self.x, self.z, self.midx, self.midz, self.b, self.Hinit)

    def sim_1e(self):
        """
        Generates results for 1 event model with uncertainty handling using monte carlo simulation.
        Returns
        -------

        """
        H1_sim = np.empty(self.num_sim)
        D1_sim = np.empty(self.num_sim)
        x1 = self.x - self.midx
        z1 = self.z - self.midz
        D_min = 1
        D_max = 750.
        D_step = 1
        H_init = (self.Hinit.s * np.random.randn(self.num_sim)) + self.Hinit.n
        for i in range(self.num_sim):
            b = self.b_sim[:, i]
            z2 = z1 - (x1 * b)
            opt_d = grid_search_d(scarp_1e, D_min, D_max, D_step, x1, z2, H_init[i])
            H1_sim[i] = H_init[i]
            D1_sim[i] = opt_d
        self.H1_sim = uncertainties.ufloat(np.mean(H1_sim), np.std(H1_sim))
        self.D1_sim = uncertainties.ufloat(np.mean(D1_sim), np.std(D1_sim))

    def gen_ss(self):
        """
        Generates steady-state results (no uncertainty handling)
        Returns
        -------

        """
        self.Hs, self.Ds = fit_ss_uplift(self.x, self.z, self.midx, self.midz, self.b, self.Hinit)

    def sim_ss(self):
        """
        Generates steady-state results with uncertainty handling using monte carlo simulation.
        Returns
        -------

        """
        Hs_sim = np.empty(self.num_sim)
        Ds_sim = np.empty(self.num_sim)
        x1 = self.x - self.midx
        z1 = self.z - self.midz
        D_min = 1
        D_max = 5000
        D_step = 1
        H_init = (self.Hinit.s * np.random.randn(self.num_sim)) + self.Hinit.n
        for i in range(self.num_sim):
            b = self.b_sim[:, i]
            z2 = z1 - (x1 * b)
            opt_d = grid_search_d(scarp_1e, D_min, D_max, D_step, x1, z2, H_init[i])
            Hs_sim[i] = H_init[i]
            Ds_sim[i] = opt_d
        self.Hs_sim = uncertainties.ufloat(np.mean(Hs_sim), np.std(Hs_sim))
        self.Ds_sim = uncertainties.ufloat(np.mean(Ds_sim), np.std(Ds_sim))

    def plot_scarp(self, type, unc=False):
        self.active_fig = plt.figure(figsize=[8.5, 5.5], dpi=300)
        ax = self.active_fig.add_subplot(111)
        x1 = self.x - self.midx
        z1 = self.z - self.midz
        ax.plot(x1, z1, 'o', color='k')
        # H = self.Hinit
        if type == "se":
            if unc:
                H = self.H1_sim
                D = self.D1_sim
            else:
                H = self.H1
                D = self.D1
            scarp_model = scarp_1e(x1, H.n, D.n) + (x1 * self.b)
        elif type == "ss":
            if unc:
                H = self.Hs_sim
                D = self.Ds_sim
            else:
                H = self.Hs
                D = self.Ds
            scarp_model = scarp_ss(x1, H.n, D.n) + (x1 * self.b)
        else:
            raise ValueError("Undefined scarp type")

        ax.plot(x1, init_geom(x1, H.n, self.b), '--', c='darkgray')
        ax.plot(x1, scarp_model, ':', c='red')
        ax.set_aspect(self.aspect)

        name_text = self.name + "\n"
        if unc:
            D_text = "$\kappa t$ = {:.1f} ".format(D.n) + "+/- {:.1f}  [$m^2$]\n".format(D.s)
            H_text = "H = {:.2f} ".format(H.n) + "+/- {:.2f} [m] \n".format(H.s)
            b_text = "Far-Field Slope = {:.2f} \n".format(np.mean([self.b1.n, self.b2.n]))
        else:
            D_text = "$\kappa t$ = {:.1f} [$m^2$] \n".format(D.n)
            H_text = "H = {:.2f} [m] \n".format(H.n)
            b_text = "Far-Field Slope = {:.2f} \n".format(np.mean([self.b1.n, self.b2.n]))
        midx_text = "Mid X = {:.0f} [m] \n".format(self.midx)
        midz_text = "Mid Z =  {:.0f} [m]".format(self.midz)
        if type == "ss":
            model_text = "Steady State Uplift \n"
        elif type == "se":
            model_text = "Single Event Uplift \n "
        else:
            model_text = ""
        if unc:
            unc_text = "+/- 1-$\sigma$ \n"
        else:
            unc_text = ""
        # annot_text = model_text + H_text + D_text + b_text + midx_text + midz_text
        annot_text = name_text + model_text + H_text + D_text + unc_text + b_text
        ax.text(-0.1, -0.1, annot_text, transform=ax.transAxes, horizontalalignment='right', bbox=dict(facecolor='white',
                                                                                                      alpha=0.7))
        plt.show()

    def save_scarp_fig(self, path):
        self.active_fig.savefig(path, dpi=300, bbox_inches="tight")
