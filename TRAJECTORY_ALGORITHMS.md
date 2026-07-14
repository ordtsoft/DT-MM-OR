# Trajectory algorithms

Every method is **causal**: it predicts a frame before seeing that frame's
position. All methods start empty and learn from the same observations.

## Persistence

Predicts that the entity remains at its last known position. It is the simplest
baseline and often works well when detections are noisy or motion is slow.

## Smoothed constant velocity

Measures velocity between observations and exponentially smooths it. It follows
steady movement quickly, but sudden direction changes or noisy centroids can
cause overshoot.

## Rolling regression

Fits straight lines `x(time)` and `y(time)` over a recent observation window.
It reduces individual-frame noise, but can lag behind curved or rapidly changing
motion.

## Alpha-beta filter

Maintains position and velocity, predicts the next position, then corrects both
using the new observation. It is a lightweight tracking filter between constant
velocity and a full Kalman filter.

## Online neural network

A small one-hidden-layer multilayer perceptron (MLP) predicts the next X/Y
displacement from recent velocity, acceleration, speed, and training progress.
It starts with deliberately broad random output weights, so its early predictions
are poor. After every true position arrives, it learns by back-propagation and
stochastic gradient descent. It never trains on a position before predicting it.

This is an online-learning demonstration rather than a pre-trained model. The
`trained` count in the viewer shows how many movement examples it has received.

## Memory k-nearest neighbors

This deliberately data-hungry instance learner stores motion examples rather
than fitting fixed parameters. For the current velocity, acceleration, and
speed, it finds the nine most similar historical contexts and averages their
next velocities with distance-based weights.

It stays in warm-up for its first 40 examples, so it draws no forecast and
receives no score during that period. This avoids pretending that an empty
memory is a trained model. The metrics panel reports its smaller evaluation
count (`n`) and stored-memory size so comparisons remain transparent.

## Adaptive expert ensemble

This online ensemble mixes six motion experts, from exact persistence through
increasingly reactive velocity models. After each observation, exponential
weighting rewards the experts that just predicted well and suppresses those that
did not. Because persistence is included, the ensemble can become conservative
on noisy data; when movement is steady, it shifts weight toward velocity.

The method is intended to beat persistence when the sequence contains learnable
motion, not to promise a win on every dataset. On the noisy `circulator` track
it approximately ties persistence. On a separate synthetic steady-motion test,
it reduced MAE from about `0.0231 m` to `0.0078 m` without changing parameters.

## Comparison metrics

- **Current error:** distance between the latest prediction and observation at
  the viewer's selected prediction horizon.
- **MAE:** average prediction distance; used for the live ranking.
- **RMSE:** like MAE but penalizes occasional large errors more strongly.

Lower values are better. Missing detections are skipped equally for every
algorithm. The k-NN warm-up is also excluded from its score, so check `n` when
comparing it with methods that predict immediately. A more complex model is not
guaranteed to outperform persistence.
