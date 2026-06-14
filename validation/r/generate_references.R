#!/usr/bin/env Rscript
# =============================================================================
# generate_references.R
#
# Generates CSV reference files using phenofit (R) for comparison with
# phenoatlas (Python).
#
# Usage:
#   Rscript validation/r/generate_references.R
#
# Outputs (in validation/reference_data/):
#   curves.csv          - curve function values for all methods
#   whittaker.csv       - smooth_wWHIT output
#   curvefit_params.csv - fitted parameters from curvefit()
#   curvefit_fitted.csv - fitted values at t from curvefit()
#   curvefit_gof.csv    - GOF metrics from get_GOF()
#   pheno_trs.csv       - TRS phenology dates
#   pheno_der.csv       - DER phenology dates
#   pheno_gu.csv        - GU phenology dates
#   pheno_zhang.csv     - ZHANG phenology dates
# =============================================================================

library(phenofit)

OUT_DIR <- "validation/reference_data"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

# --------------------------------------------------------------------------
# Shared synthetic data
# --------------------------------------------------------------------------
N    <- 46
t    <- seq(1, 365, length.out = N)
tout <- t

# True Beck parameters (used to generate synthetic y)
p_beck_true <- c(mn = 0.10, mx = 0.85, sos = 90.0, rsp = 0.12, eos = 270.0, rau = 0.10)
y_clean     <- doubleLog.Beck(p_beck_true, t)
noise       <- rnorm(N, 0, 0.02)
y_noisy     <- y_clean + noise
w           <- rep(1.0, N)

# --------------------------------------------------------------------------
# 1. Curve functions — exact values at t for known params
# --------------------------------------------------------------------------
p_AG     <- c(t0 = 180, mn = 0.10, mx = 0.75, rsp = 0.04, a3 = 2.0, rau = 0.04, a5 = 2.0)
p_Beck   <- c(mn = 0.10, mx = 0.85, sos = 90.0, rsp = 0.12, eos = 270.0, rau = 0.10)
p_Elmore <- c(mn = 0.10, mx = 0.85, sos = 90.0, rsp = 0.12, eos = 270.0, rau = 0.10, m7 = 0.0)
p_Gu     <- c(y0 = 0.10, a1 = 0.75, a2 = 0.75, sos = 90.0, rsp = 0.12, eos = 270.0, rau = 0.10, c1 = 1.0, c2 = 1.0)
p_Zhang  <- c(t0 = 180, mn = 0.10, mx = 0.85, sos = 90.0, rsp = 0.12, eos = 270.0, rau = 0.10)

curves_df <- data.frame(
  t       = t,
  AG      = doubleLog.AG(p_AG, t),
  Beck    = doubleLog.Beck(p_Beck, t),
  Elmore  = doubleLog.Elmore(p_Elmore, t),
  Gu      = doubleLog.Gu(p_Gu, t),
  Zhang   = doubleLog.Zhang(p_Zhang, t)
)
write.csv(curves_df, file.path(OUT_DIR, "curves.csv"), row.names = FALSE)
cat("curves.csv written\n")

# --------------------------------------------------------------------------
# 2. Whittaker smoother
# --------------------------------------------------------------------------
ylu <- range(y_noisy) + c(-0.05, 0.05)

for (lam in c(10, 100, 1000)) {
  z_list <- smooth_wWHIT(y_noisy, w = w, ylu = ylu, lambda = lam, nptperyear = 23, iters = 1)
  z      <- z_list$zs[[1]]
  fname  <- file.path(OUT_DIR, sprintf("whittaker_lam%d.csv", lam))
  write.csv(data.frame(t = t, y = y_noisy, z = z, w = w), fname, row.names = FALSE)
  cat(sprintf("whittaker_lam%d.csv written\n", lam))
}

# --------------------------------------------------------------------------
# 3. curvefit — params, fitted values, GOF
# --------------------------------------------------------------------------
methods <- c("Beck", "Elmore", "Zhang", "AG", "Gu", "Klos")
fit     <- curvefit(y_noisy, t = t, tout = tout, methods = methods, w = w)

# Params
params_list <- lapply(methods, function(m) {
  p <- as.numeric(fit$model[[m]]$par)  # flatten 1×k matrix → vector
  data.frame(method = m, param_idx = seq_along(p), value = p)
})
params_df <- do.call(rbind, params_list)
write.csv(params_df, file.path(OUT_DIR, "curvefit_params.csv"), row.names = FALSE)
cat("curvefit_params.csv written\n")

# Fitted values at t
fitted_list <- lapply(methods, function(m) {
  zs <- fit$model[[m]]$zs
  z  <- if (is.list(zs)) zs[[length(zs)]] else zs
  data.frame(t = t, method = m, fitted = z)
})
fitted_df <- do.call(rbind, fitted_list)
write.csv(fitted_df, file.path(OUT_DIR, "curvefit_fitted.csv"), row.names = FALSE)
cat("curvefit_fitted.csv written\n")

# GOF
gof_df <- as.data.frame(get_GOF(fit))
write.csv(gof_df, file.path(OUT_DIR, "curvefit_gof.csv"), row.names = FALSE)
cat("curvefit_gof.csv written\n")

# --------------------------------------------------------------------------
# 4. Phenology extraction
# --------------------------------------------------------------------------
# Use only Beck for phenology (most stable)
# Note: get_pheno(fit, method) — 'method' selects which *curve* to use (Beck/Elmore/etc.)
# It always returns all phenology types: TRS2, TRS5, DER, GU, ZHANG
fit_beck <- curvefit(y_noisy, t = t, tout = tout, methods = "Beck", w = w)
pheno    <- get_pheno(fit_beck)   # uses Beck curve by default

# TRS (only TRS2 and TRS5 available in phenofit)
trs2 <- pheno$TRS2; trs5 <- pheno$TRS5
trs_df <- data.frame(
  method = c("TRS2", "TRS5"),
  SOS    = c(as.numeric(trs2["sos"]), as.numeric(trs5["sos"])),
  EOS    = c(as.numeric(trs2["eos"]), as.numeric(trs5["eos"]))
)
write.csv(trs_df, file.path(OUT_DIR, "pheno_trs.csv"), row.names = FALSE)
cat("pheno_trs.csv written\n")

# DER
der <- pheno$DER
der_df <- data.frame(
  DER.SOS = as.numeric(der["sos"]),
  DER.POS = as.numeric(der["pos"]),
  DER.EOS = as.numeric(der["eos"])
)
write.csv(der_df, file.path(OUT_DIR, "pheno_der.csv"), row.names = FALSE)
cat("pheno_der.csv written\n")

# GU
gu <- pheno$GU
gu_df <- data.frame(
  UD = as.numeric(gu["UD"]),
  SD = as.numeric(gu["SD"]),
  DD = as.numeric(gu["DD"]),
  RD = as.numeric(gu["RD"])
)
write.csv(gu_df, file.path(OUT_DIR, "pheno_gu.csv"), row.names = FALSE)
cat("pheno_gu.csv written\n")

# ZHANG
zh <- pheno$ZHANG
zh_df <- data.frame(
  GreenUp    = as.numeric(zh["Greenup"]),
  Maturity   = as.numeric(zh["Maturity"]),
  Senescence = as.numeric(zh["Senescence"]),
  Dormancy   = as.numeric(zh["Dormancy"])
)
write.csv(zh_df, file.path(OUT_DIR, "pheno_zhang.csv"), row.names = FALSE)
cat("pheno_zhang.csv written\n")

# --------------------------------------------------------------------------
# 5. Save input data used (for Python to reproduce exactly)
# --------------------------------------------------------------------------
input_df <- data.frame(t = t, y_clean = y_clean, y_noisy = y_noisy, w = w,
                       p_beck_mn  = p_beck_true["mn"],
                       p_beck_mx  = p_beck_true["mx"],
                       p_beck_sos = p_beck_true["sos"],
                       p_beck_rsp = p_beck_true["rsp"],
                       p_beck_eos = p_beck_true["eos"],
                       p_beck_rau = p_beck_true["rau"])
write.csv(input_df, file.path(OUT_DIR, "input_data.csv"), row.names = FALSE)
cat("input_data.csv written\n")

# Params used for curve function tests
curve_params_df <- data.frame(
  method = c("AG","Beck","Elmore","Gu","Zhang"),
  p1  = c(p_AG[1],  p_Beck[1],  p_Elmore[1],  p_Gu[1],  p_Zhang[1]),
  p2  = c(p_AG[2],  p_Beck[2],  p_Elmore[2],  p_Gu[2],  p_Zhang[2]),
  p3  = c(p_AG[3],  p_Beck[3],  p_Elmore[3],  p_Gu[3],  p_Zhang[3]),
  p4  = c(p_AG[4],  p_Beck[4],  p_Elmore[4],  p_Gu[4],  p_Zhang[4]),
  p5  = c(p_AG[5],  p_Beck[5],  p_Elmore[5],  p_Gu[5],  p_Zhang[5]),
  p6  = c(p_AG[6],  p_Beck[6],  p_Elmore[6],  p_Gu[6],  p_Zhang[6]),
  p7  = c(p_AG[7],  NA,         p_Elmore[7],  p_Gu[7],  p_Zhang[7]),
  p8  = c(NA,       NA,         NA,            p_Gu[8],  NA),
  p9  = c(NA,       NA,         NA,            p_Gu[9],  NA)
)
write.csv(curve_params_df, file.path(OUT_DIR, "curve_params.csv"), row.names = FALSE)
cat("curve_params.csv written\n")

cat("\nAll reference files generated in:", OUT_DIR, "\n")
