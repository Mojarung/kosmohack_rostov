"""Устойчивое условное среднее при опубликованных линейных ограничениях на NDVI."""

import numpy as np
from threadpoolctl import threadpool_limits

from gapfill.research.constraints import linear_projection


def log_density(error, sigma):
    """Плотность Student-t с тремя степенями свободы, без постоянных множителей."""
    return -2. * np.log1p((error / sigma) ** 2 / 3.)


def reflection_moves(x, center, null, prior, sigma, rng, repeats):
    """Симметричные отражения переносят большие остатки, не меняя известных сумм."""
    for _ in range(repeats):
        i, j = rng.choice(len(prior), 2, replace=False)
        direction = null[:, i] - null[:, j]
        norm = direction @ direction
        if norm < 1e-10:
            continue
        candidate = x - 2 * ((x - center) @ direction)[:, None] * direction / norm
        old_lp = log_density(x - prior, sigma).sum(1)
        new_lp = log_density(candidate - prior, sigma).sum(1)
        accept = np.log(rng.random(len(x))) < new_lp - old_lp
        x[accept] = candidate[accept]
    return x


def _posterior(prior, a, b, scale, steps, seed, chains, reflections):
    """Гиббс: скрытая гамма-точность и нормальная выборка на плоскости A x = b."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((s > 1e-9).sum())
    if rank == 0:
        return prior.copy(), 0.
    basis, sums = vt[:rank], u[:, :rank].T @ b / s[:rank]
    if rank == len(prior):
        return basis.T @ sums, 0.
    rng = np.random.default_rng(seed)
    sigma = np.full(len(prior), .04) if scale is None else scale
    x = np.tile(linear_projection(prior, a, b, scale=sigma), (chains, 1))
    center = x[0].copy()
    null = np.eye(len(prior)) - basis.T @ basis
    total = np.zeros_like(x)
    rhs_mean = sums - basis @ prior
    for iteration in range(steps + 300):
        precision = rng.gamma(2., 2. / (3. + ((x - prior) / sigma) ** 2))
        variance = sigma ** 2 / np.maximum(precision, 1e-8)
        cross = variance[:, :, None] * basis.T[None]
        cov = basis[None] @ cross
        noise = rng.normal(size=x.shape) * np.sqrt(variance)
        rhs = np.stack([np.broadcast_to(rhs_mean, (len(x), rank)), noise @ basis.T], axis=-1)
        correction = cross @ np.linalg.solve(cov, rhs)
        mean = prior + correction[:, :, 0]
        x = mean + noise - correction[:, :, 1]
        x = reflection_moves(x, center, null, prior, sigma, rng, reflections)
        if iteration >= 300:
            total += mean
    means = total / steps
    return means.mean(0), float(np.max(means.std(0) / np.sqrt(len(means))))


def posterior(prior, a, b, scale=None, steps=2500, seed=42, chains=64, reflections=8):
    """Возвращает среднее и диагностический разброс цепей, не интервал ошибки самого прогноза.

    Равенства соблюдаются каждой выборкой. Усредняются условные нормальные средние, что снижает
    численный шум. Фиксированный seed обеспечивает воспроизводимость; малый разброс цепей сам
    по себе не доказывает полную сходимость в многомодальном случае.
    """
    with threadpool_limits(limits=1):
        return _posterior(prior, a, b, scale, steps, seed, chains, reflections)
