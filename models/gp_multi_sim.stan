data {
  int<lower=1> N;
  int<lower=1> P; # number of replicates
  int<lower=1> K; # number of latent functions
  matrix[P,K] design;

  real<lower=0> length_scale;
  real<lower=0> alpha;
  real<lower=0> sigma;
}
transformed data {
  vector[N] zeros;
  zeros = rep_vector(0, N);
}
model {}
generated quantities {
  real x[N];
  row_vector[N] y[P];
  matrix[K, N] f;
  for (n in 1:N)
    x[n] = uniform_rng(-2,2);

  for (i in 1:K)
  {
    matrix[N, N] cov;
    matrix[N, N] L_cov;
    cov = cov_exp_quad(x, alpha, length_scale);
    for (n in 1:N)
      cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);
    f[i] = multi_normal_cholesky_rng(zeros, L_cov)';
  }

  for (i in 1:P)
    for (j in 1:N)
      y[i][j] = normal_rng((design[i]*f)[j], sigma);
  // for (n in 1:N)
  //   {
  //     for (p in 1:P)
  //       y[n,p] = normal_rng((f'*design)[n,p], sigma);
  //   }
}
