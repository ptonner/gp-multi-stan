data {
  int<lower=1> N;
  int<lower=0> N_mis; # number of missing datapoints
  int<lower=1> P; # number of replicates
  int<lower=1> K; # number of latent functions

  matrix[P,K] design;
  row_vector[N] y[P];
  real x[N];
  real x_mis[N_mis]; # missing data point locations
}
transformed data{
  int N_tot = N + N_mis;

  // missing obs positions
  real x_tot[N_tot];
  for (i in 1:N) x_tot[i] = x[i];
  for (i in 1:N_mis) x_tot[i+N] = x_mis[i];
}
parameters {
  real<lower=0> length_scale;
  real<lower=0> alpha;
  real<lower=0> sigma;
  vector[N_tot] f_eta[K];

  row_vector[N_mis] y_mis[P]; // missing data
}
transformed parameters {
  matrix[K,N_tot] f;
  row_vector[N_tot] y_tot[P];

  for (k in 1:K)
  {
    matrix[N_tot, N_tot] L_cov;
    matrix[N_tot, N_tot] cov;
    cov = cov_exp_quad(x_tot, alpha, length_scale);
    for (n in 1:N_tot)
      cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);
    f[k] = (L_cov * f_eta[k])';
  }

  for (p in 1:P)
  {
    for (i in 1:N) y_tot[p,i] = y[p,i];
    for (i in 1:N_mis) y_tot[p,i+N] = y_mis[p,i];
  }

}
model {

  length_scale ~ cauchy(0,5);
  alpha ~ cauchy(0,5);
  sigma ~ cauchy(0,5);

  for (i in 1:K)
    f_eta[i] ~ normal(0, 1);

  for (i in 1:P)
    // y[i] ~ normal((f'*design[i]'), sigma);
    y_tot[i] ~ normal(design[i]*f, sigma);
}
