data {
  int<lower=1> N;
  int<lower=1> P; # number of replicates
  int<lower=1> K; # number of latent functions

  matrix[P,K] design;
  row_vector[N] y[P];
  real x[N];
}
parameters {
  real<lower=0> length_scale;
  real<lower=0> alpha;
  real<lower=0> sigma;
  vector[N] f_eta[K];
}
transformed parameters {
  matrix[K,N] f;

  for (k in 1:K)
  {
    matrix[N, N] L_cov;
    matrix[N, N] cov;
    cov = cov_exp_quad(x, alpha, length_scale);
    for (n in 1:N)
      cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);
    f[k] = (L_cov * f_eta[k])';
    // f[k] = (L_cov * f_eta[k]);
  }
}
model {
  // length_scale ~ gamma(2, 2);
  // alpha ~ normal(0, 1);
  // sigma ~ normal(0, 1);

  length_scale ~ cauchy(0,5);
  alpha ~ cauchy(0,5);
  sigma ~ cauchy(0,5);

  for (i in 1:K)
    f_eta[i] ~ normal(0, 1);

  for (i in 1:P)
    // y[i] ~ normal((f'*design[i]'), sigma);
    y[i] ~ normal(design[i]*f, sigma);
}
