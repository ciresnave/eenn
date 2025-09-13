Functions to Add So They Can Be Used in Neurons:

Template matching ⇢ convolution / cross-correlation (already what CNNs do).
Normalized cross-correlation (NCC) as a layer (helps illumination invariance).
Edge/texture banks (Sobel/Scharr/Gabor): fixed or learnable filter banks.
Morphology (dilate/erode/open/close): use soft-max/soft-min or max-pool relaxations.
Hough/Radon transforms (lines/circles): differentiable voting/accumulators exist.
K-means / soft K-means (cluster memberships as features).
GMM / mixture density (likelihoods or responsibilities as features/losses).
LDA/QDA / Mahalanobis distance (class-conditional Gaussian scores).
Naive Bayes (log-likelihood ratio as a linear layer with priors).
Parzen windows / KDE (random Fourier features or soft kernels).
RBF networks (Gaussian basis functions; classic “traditional→neural” bridge).
KNN / LVQ (soft-KNN; note: attention is essentially soft KNN over a memory).
SVM (hinge margin as loss atop a learned representation; “deep SVM”).
CRF / MRF potentials (CRF-RNN style layers for structured outputs).
HMM / forward–backward (neural HMMs; differentiable DP with log-sum-exp).
Kalman / linear state-space (deep Kalman filters, S4/SSM-style cells).
DTW (soft-DTW layer for time-series alignment).
PCA/whitening (differentiable SVD or orthogonality-regularized linear layer).
ICA / sparse coding (Infomax losses; LISTA: learned ISTA unrolled as a net).
Wavelets / scattering networks (fixed or learnable multiscale banks).
Histogram/HOG-like features (soft binning; differentiable histograms).
Decision trees/forests (soft gating → neural decision forests).
Graphical/grammar structure (GNN message-passing; differentiable parsing via Gumbel-softmax/straight-through).
Non-max suppression / median (soft-NMS, weighted-median relaxations).
