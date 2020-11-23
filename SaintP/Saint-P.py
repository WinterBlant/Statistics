import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit


def least_sq_norm(x, loc, scale):
    return st.norm.pdf(x, loc, scale)


def least_sq_gamma(x, a, loc, scale):
    return st.gamma.pdf(x, a, loc, scale)


def least_sq_gumbel_l(x, loc, scale):
    return st.gumbel_l.pdf(x, loc, scale)


def hist_and_kernel(value):
    plt.clf()
    plt.hist(value, bins=15, density=True)
    density = st.kde.gaussian_kde(value)
    xgrid = np.linspace(value.min(), value.max(), 100)
    plt.plot(xgrid, density(xgrid), 'r-')


def stats(data):
    mean = data.mean()
    var = data.var()
    std = data.std()
    median = np.median(data)
    mod = st.mode(data)[0].item()
    print("Выборочные параметры: Медиана - {0}, Мода - {1}, СКО - {2}, Среднее - {3}, Дисперсия - {4}".format(median, mod, std, mean, var))
    norm_q95 = st.norm.ppf(0.95)
    mean_conf = norm_q95 * std / np.sqrt(len(data))
    chi2_q95_left = st.chi2.ppf((1 - 0.05 / 2.0), df=len(data) - 1)
    chi2_q95_right = st.chi2.ppf(0.05 / 2.0, df=len(data) - 1)
    var_conf_left = var * (len(data) - 1) / chi2_q95_left
    var_conf_right = var * (len(data) - 1) / chi2_q95_right
    std_conf_left = np.sqrt(var_conf_left)
    std_conf_right = np.sqrt(var_conf_right)
    print("Выборочное среднее: %0.3f +/- %0.3f" % (mean, mean_conf))
    print("95%% Доверительный интервал выборочной дисперсии : (%0.3f; %0.3f)"
          % (var_conf_left, var_conf_right))
    print("95%% Доверительный интервал выборочного СКО: (%0.3f; %0.3f)"
          % (std_conf_left, std_conf_right))


data = pd.read_csv("Saint-P.csv", ';')
winter_temp = data.loc[(pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month <= 2) | (pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month >= 12)]
spring_temp = data.loc[(pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month <= 5) & (pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month >= 3)]
summer_temp = data.loc[(pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month <= 8) & (pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month >= 6)]
autumn_temp = data.loc[(pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month <= 11) & (pd.to_datetime(data["DATE"], format="%d.%m.%Y").dt.month >= 9)]
winter = np.array([float(x.replace(",", ".")) for x in winter_temp["TEMP"] if x != "9999,9"])
winter_score = st.zscore(winter)
winter = np.array([num for pos, num in enumerate(winter) if abs(winter_score[pos]) <= 3])
spring = np.array([float(x.replace(",", ".")) for x in spring_temp["TEMP"] if x != "9999,9"])
spring_score = st.zscore(spring)
spring = np.array([num for pos, num in enumerate(spring) if abs(spring_score[pos]) <= 3])
summer = np.array([float(x.replace(",", ".")) for x in summer_temp["TEMP"] if x != "9999,9"])
summer_score = st.zscore(summer)
summer = np.array([num for pos, num in enumerate(summer) if abs(summer_score[pos]) <= 3])
autumn = np.array([float(x.replace(",", ".")) for x in autumn_temp["TEMP"] if x != "9999,9"])
autumn_score = st.zscore(autumn)
autumn = np.array([num for pos, num in enumerate(autumn) if abs(autumn_score[pos]) <= 3])
z = np.array([float(z.replace(",", ".")) for z in data["WDSP"] if z != "999,9"])
z_score = st.zscore(z)
z = np.array([num for pos, num in enumerate(z) if abs(z_score[pos]) <= 3])
hist_and_kernel(winter)
plt.savefig("winter-temp.png", dpi=1000)
hist_and_kernel(spring)
plt.savefig("spring-temp.png", dpi=1000)
hist_and_kernel(summer)
plt.savefig("summer-temp.png", dpi=1000)
hist_and_kernel(autumn)
plt.savefig("autumn-temp.png", dpi=1000)
hist_and_kernel(z)
plt.savefig("wdsp.png", dpi=1000)
plt.clf()
stats(winter)
stats(spring)
stats(summer)
stats(autumn)
stats(z)
plt.boxplot(winter, vert=False)
plt.savefig("winter-temp_box.png", dpi=1000)
plt.show()
plt.boxplot(spring, vert=False)
plt.savefig("spring-temp_box.png", dpi=1000)
plt.show()
plt.boxplot(summer, vert=False)
plt.savefig("summer-temp_box.png", dpi=1000)
plt.show()
plt.boxplot(autumn, vert=False)
plt.savefig("autumn-temp_box.png", dpi=1000)
plt.show()
plt.boxplot(z, vert=False)
plt.savefig("wdsp_box.png", dpi=1000)
plt.show()
params_winter = st.gumbel_l.fit(winter)
params_spring = st.norm.fit(spring)
params_summer = st.norm.fit(summer)
params_autumn = st.norm.fit(autumn)
params_z = st.gamma.fit(z)
print(params_winter)
print(params_spring)
print(params_summer)
print(params_autumn)
print(params_z)
params_winter_lsq, pcovWinter = curve_fit(least_sq_gumbel_l, winter, st.gumbel_l.pdf(winter, *params_winter))
params_spring_lsq, pcovSpring = curve_fit(least_sq_norm, spring, st.norm.pdf(spring, *params_spring))
params_summer_lsq, pcovSummer = curve_fit(least_sq_norm, summer, st.norm.pdf(summer, *params_summer))
params_autumn_lsq, pcovAutumn = curve_fit(least_sq_norm, autumn, st.norm.pdf(autumn, *params_autumn))
params_z_lsq, pcovZ = curve_fit(least_sq_gamma, z, st.gamma.pdf(z, *params_z))
print(params_winter_lsq)
print(params_spring_lsq)
print(params_summer_lsq)
print(params_autumn_lsq)
print(params_z_lsq)
percs = np.linspace(0, 100, 21)
qn_first = np.percentile(winter, percs)
qn_norm = st.gumbel_l.ppf(percs / 100.0, *params_winter)
plt.figure(figsize=(10, 10))
plt.plot(qn_first, qn_norm, ls="", marker="o", markersize=6)
winter_lin = np.linspace(winter.min(), winter.max())
plt.plot(winter_lin, winter_lin, color="k", ls="--")
plt.xlabel(f'Эмпирическое распределение')
plt.ylabel('Теоретическое (Гумбеля) распределение')
plt.savefig("winter-temp_qn.png", dpi=1000)
plt.show()

qn_first = np.percentile(spring, percs)
qn_norm = st.norm.ppf(percs / 100.0, *params_spring)
plt.figure(figsize=(10, 10))
plt.plot(qn_first, qn_norm, ls="", marker="o", markersize=6)
spring_lin = np.linspace(spring.min(), spring.max())
plt.plot(spring_lin, spring_lin, color="k", ls="--")
plt.xlabel(f'Эмпирическое распределение')
plt.ylabel('Теоретическое (нормальное) распределение')
plt.savefig("spring-temp_qn.png", dpi=1000)
plt.show()

qn_first = np.percentile(summer, percs)
qn_norm = st.norm.ppf(percs / 100.0, *params_summer)
plt.figure(figsize=(10, 10))
plt.plot(qn_first, qn_norm, ls="", marker="o", markersize=6)
summer_lin = np.linspace(summer.min(), summer.max())
plt.plot(summer_lin, summer_lin, color="k", ls="--")
plt.xlabel(f'Эмпирическое распределение')
plt.ylabel('Теоретическое (нормальное) распределение')
plt.savefig("summer-temp_qn.png", dpi=1000)
plt.show()

qn_first = np.percentile(autumn, percs)
qn_norm = st.norm.ppf(percs / 100.0, *params_autumn)
plt.figure(figsize=(10, 10))
plt.plot(qn_first, qn_norm, ls="", marker="o", markersize=6)
autumn_lin = np.linspace(autumn.min(), autumn.max())
plt.plot(autumn_lin, autumn_lin, color="k", ls="--")
plt.xlabel(f'Эмпирическое распределение')
plt.ylabel('Теоретическое (нормальное) распределение')
plt.savefig("autumn-temp_qn.png", dpi=1000)
plt.show()

qn_first = np.percentile(z, percs)
qn_gamma = st.gamma.ppf(percs / 100.0, *params_z)
plt.figure(figsize=(10, 10))
plt.plot(qn_first, qn_gamma, ls="", marker="o", markersize=6)
z_lin = np.linspace(z.min(), z.max())
plt.plot(z_lin, z_lin, color="k", ls="--")
plt.xlabel(f'Эмпирическое распределение')
plt.ylabel('Теоретическое (гамма) распределение')
plt.savefig("wdsp_qn.png", dpi=1000)
plt.show()

ks = st.kstest(winter, 'gumbel_l', params_winter)
print(ks)

ks = st.kstest(spring, 'norm', params_spring)
print(ks)

ks = st.kstest(summer, 'norm', params_summer)
print(ks)

ks = st.kstest(autumn, 'norm', params_autumn)
print(ks)

ks = st.kstest(z, 'gamma', params_z)
print(ks)

ks = st.kstest(winter, 'gumbel_l', params_winter_lsq)
print(ks)

ks = st.kstest(spring, 'norm', params_spring_lsq)
print(ks)

ks = st.kstest(summer, 'norm', params_summer_lsq)
print(ks)

ks = st.kstest(autumn, 'norm', params_autumn_lsq)
print(ks)

ks = st.kstest(z, 'gamma', params_z_lsq)
print(ks)