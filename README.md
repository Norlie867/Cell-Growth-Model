# Cell Growth Prediction Model
Understanding and anticipating cell culture growth is essential for running a bioprocess. This project explores how four environmental conditions :temperature, pH, glucose and lactate shape the specific growth rate. Its goal is to optimise bioprocesses by training and comparing polynomial and k-nearest neighbours models, while checking how reliable the predictions are. Cross-validation, learning curves and a reliability domain are used throughout to avoid unsafe extrapolations and keep the conclusions applicable to biological work.

## Highlights
* Developed an end-to-end regression workflow for bioprocess data
* Tuned polynomial and k-NN models with cross-validated grid search
* Compared models using held-out tests and learning curves
* Interpreted polynomial weights and one-factor response plots
* Mapped reliable operating regions using nearest-neighbor distances

## Environment
* Python 3.14.7
* matplotlib 3.11.0
* numpy 2.5.2
* pandas 3.0.5
* scikit-learn 1.9.0

## Code Design

Running [`main.py`](main.py) executes the whole study in sequence and saves every result in [`figures/`](figures/) and [`tables/`](tables/). The script only decides the order of the steps. Paths, column names, random seeds, and search ranges live in [`src/constants.py`](src/constants.py), and the code behind each step lives in [`src/functions.py`](src/functions.py), so I can change a search range by editing a single file.

### Data Loading and Validation

I start by reading the CSV file and checking that the five expected columns exist and that no value is missing. Catching a misnamed column or an empty cell here is much cheaper than finding out later why a model behaves strangely. The validated table is the only source of data for every other step.

### Exploratory Analysis

Next I summarize each variable (mean, spread, range, number of zeros), compute the correlation matrix, and plot histograms and growth rate against each factor with binned averages. This tells me whether the factors cover a useful range, whether they are correlated with each other (which would make their effects harder to separate), and whether the relationships are curved. The curvature I saw, especially for pH, is why I test polynomial and neighbor-based models instead of settling for a straight line.

### Train-Test Split

I convert the data to NumPy arrays and randomly keep 80 % of the samples (254) for development while setting 20 % (64) aside. The held-out samples are used only once, at the end, so the final error reflects conditions the model has never seen instead of memorization. A fixed random seed makes the split identical on every run.

### Candidate Models

I define two model families as scikit-learn pipelines. The polynomial model standardizes the four features, expands them into powers and interaction terms, and fits a linear regression on those terms. The k-nearest neighbors (k-NN) model standardizes the features and averages the growth rates of the closest training samples. Putting the scaler inside the pipeline means it is refit on the training part of each cross-validation fold, so validation samples never leak into preprocessing. Standardizing also puts temperature (30 to 42 °C) and lactate (0 to 40 mM) on comparable scales, which matters for distances and makes the polynomial weights comparable.

### Hyperparameter Tuning

Some settings cannot be learned from the data: the polynomial degree (1 to 10) and, for k-NN, the number of neighbors (1 to 40) and the weighting (uniform or distance-based). I use a grid search with 4-fold cross-validation on the training set, scored by root mean square error (RMSE). I also record the training error, because a large gap between training and validation error signals overfitting. Separate searches are run for a linear model, the polynomial family, and each k-NN weighting, so every alternative is compared at its own best settings. This step gives each family its best hyperparameters and a validation error to compare.

### Model Comparison and Selection

I compare the tuned models with a baseline that always predicts the mean growth rate, which shows how much error the models actually remove. The selection rule is fixed in advance: the model with the lowest validation RMSE wins, unless the polynomial is within one standard deviation of it, in which case I keep the polynomial because it is simpler and its weights can be read directly. Setting the rule before looking at the test set keeps the choice reproducible and stops the test data from influencing it.

### Learning Curves

To see whether more experiments would help, I repeat the tuning on random subsets of growing size, from 60 samples to the full dataset, each with its own split and grid search. This step does not influence model selection. It tells me whether the amount of data is the bottleneck.

### Evaluation on the Test Set

The selected model and its closest alternative are scored on the held-out samples with RMSE, mean absolute error, and R², and shown in a measured-versus-predicted plot and a residual plot. Negative predictions are set to zero because a growth rate cannot be negative. This is the only step that uses the test set, so it gives the performance I report, and the plots show whether the errors are random or systematic.

### Interpretation

I then refit the selected model on all 318 samples so that every observation contributes. Because the inputs are standardized, its weights can be compared directly: squared terms point to an optimum and cross terms to an interaction. To compare with a model of a different type, I vary one factor at a time over its observed range while holding the others at their mean, for both the polynomial and the k-NN model, and I draw two-dimensional maps of temperature versus pH and glucose versus lactate. This turns a fitted model into something I can explain biologically.

### Operating Region and Reliability

Finally, I search a four-dimensional grid of conditions for the highest and lowest predicted growth rates. A condition is kept only if it is close to the training data, measured by the distance to its fifth nearest sample compared with typical distances inside the dataset. The same boundary appears on the maps, and I report the error for each quartile of each factor. This separates interpolation from extrapolation and shows where the model should not be trusted.
## Analysis

### Dataset Overview

The dataset has 318 simulated experiments and no missing values ([`tables/data_summary.csv`](tables/data_summary.csv)). Growth rates go from 0 to 0.75 d⁻¹ with a mean of 0.283 d⁻¹. A total of 24 samples (7.5 %) have exactly zero growth and a few sit exactly at 0.75 d⁻¹, which looks like a measurement floor and ceiling. The four factors are almost uncorrelated with each other (largest |r| = 0.07, [`tables/correlation_matrix.csv`](tables/correlation_matrix.csv)), so their effects can be separated. Their linear correlations with growth are weak: lactate −0.29, glucose +0.18, temperature −0.12, and pH +0.01. The near-zero value for pH is misleading, because a linear correlation cannot detect an optimum. Figure 1 shows growth peaking around pH 7 and dropping on both sides. The temperature signal is noisier, with the lowest growth in the hottest samples.

![figure_growth_vs_features.png](figures/figure_growth_vs_features.png)

**Figure 1**-  Growth rate versus each factor. Blue points are individual samples; the red line shows the mean growth rate per quantile bin.

### Choice of Model Architecture
The selected model is a second-degree polynomial regression with standardized inputs, fitted by ordinary least squares and refit on all data for interpretation. Degree 2 was chosen for three reasons. First, the data call for curvature: temperature and pH both show an interior optimum, which only a squared term can represent. Second, cross terms allow interactions such as nutrient limitation by glucose being weakened by lactate inhibition. Third, validation error was lowest at degree 2. With four features this model has only 14 weights for 254 training samples (about 18 samples per weight), so no extra regularization was needed.

**Table 1**- Training and validation error for polynomial degrees 1 to 6

|Degree|Train RMSE|Valid RMSE|Valid RMSE SD|Train R2|Valid R2    |
|------|----------|----------|-------------|--------|------------|
|1     |0.1717    |0.1739    |0.008        |0.123   |0.0841      |
|2     |0.143     |0.1542    |0.0104       |0.3922  |0.283       |
|3     |0.1299    |0.1645    |0.018        |0.4987  |0.1871      |
|4     |0.1111    |0.2289    |0.058        |0.6329  |-0.6757     |
|5     |0.0833    |0.6043    |0.088        |0.7927  |-10.245     |
|6     |0.0       |10.5678   |1.9979       |1.0     |-3414.5163  |
|7     |0.0       |8.9491    |1.6196       |1.0     |-2521.5482  |
|8     |0.0       |15.53     |4.8509       |1.0     |-7996.9822  |
|9     |0.0008    |45.4061   |8.5678       |0.9999  |-62176.8604 |
|10    |0.0145    |59.7274   |13.7556      |0.9922  |-113868.3134|

Training error keeps falling as the degree rises, but validation error is lowest at degree 2 and then gets worse, with degree 3 already less accurate and less stable. At degree 6 the model has 209 terms, more than the roughly 190 training samples in each fold, so it reproduces the training data perfectly (training RMSE of 0) and fails on new data. This is a clear case of overfitting, and Figure 2 shows the gap opening.

![figure_validation_curve_poly.png](figures/figure_validation_curve_poly.png)

**Figure 2** - Training and validation RMSE versus polynomial degree (log scale). Validation error is minimised at degree 2 and explodes for degree ≥ 4.

### Comparison with Linear and k-NN Models

Every candidate was tuned on the training set only and then scored once on the held-out test set (Table 2).

**Table 2.** Comparison of all candidate models

|Model          |Hyperparameters          |Valid RMSE|Valid RMSE SD|Train RMSE|Test RMSE|Test MAE|Test R2|
|---------------|-------------------------|----------|-------------|----------|---------|--------|-------|
|Baseline (mean)|none                     |0.1838    |0.0118       |0.1835    |0.2068   |0.1707  |-0.0686|
|Linear         |degree = 1               |0.1739    |0.008        |0.172     |0.1907   |0.1643  |0.0917 |
|Polynomial     |degree = 2               |0.1542    |0.0104       |0.1414    |0.157    |0.1309  |0.3841 |
|kNN (uniform)  |k = 9, weights = uniform |0.1533    |0.0113       |0.1418    |0.1621   |0.1363  |0.3434 |
|kNN (distance) |k = 9, weights = distance|0.1515    |0.0112       |0.0       |0.1587   |0.1341  |0.3707 |

The linear model underfits: its training and validation errors are nearly equal (0.172 and 0.174) but both are high, and its test R² is only 0.09. The curvature captured by degree 2 lowers validation RMSE by about 11 % relative to the linear model. The k-NN models reach the same accuracy range. The best k is 9 for both weightings, with validation RMSE of 0.1533 (uniform) and 0.1515 (distance). Distance weighting gives a training RMSE of exactly 0, because each training sample is its own nearest neighbor at distance zero and is reproduced perfectly, so it memorizes the noise in the training data even though its validation error is similar.

![figure_validation_curve_knn.png](figures/figure_validation_curve_knn.png)

**Figure 3.** Training and validation RMSE versus k for both k-NN weightings. The training curve for distance weighting lies on the horizontal axis (RMSE of 0).

The gap between the best k-NN model and the polynomial in validation RMSE (0.0027 d⁻¹) is far smaller than the fold-to-fold standard deviation (about 0.010 d⁻¹), and on the test set the polynomial is slightly better (0.1570 versus 0.1587 d⁻¹). The two families are therefore tied on accuracy, and I chose between them on other grounds. The polynomial is described by 15 numbers, gives smooth predictions, and can be read for optima and interactions. The k-NN model has to store all the samples, gives staircase-like surfaces, and can never predict outside the range of growth rates it has seen. I selected the polynomial. Both families cut the test RMSE by only 22 to 24 % compared with the mean baseline (0.2068 d⁻¹), so most of the variability is not explained by these four factors.


### Learning Curve
The learning curve repeats the whole tuning procedure on subsets of 60 to 318 samples.

![figure_learning_curve.png](figures/figure_learning_curve.png)

**Figure 4.** Training and test RMSE versus the number of training samples. Source: [`tables/learning_curve.csv`](tables/learning_curve.csv).

With 48 training samples the polynomial test RMSE is 0.256 d⁻¹, since there is too little data to estimate the curvature. From 144 samples onward the polynomial always selects degree 2 and its test RMSE stays between 0.14 and 0.17 d⁻¹. The k-NN test RMSE stays between 0.13 and 0.18 d⁻¹ (the lowest value comes from a test set of only 12 samples), and its training RMSE is 0 in seven of eight cases. Both curves level off, so more data of the same kind is unlikely to help much; new factors or less noisy measurements probably would. The curves are noisy because every point uses a different subset and a different small test split, as the bump at 208 samples shows, so only the overall trend should be trusted.


### Model Performance and Residuals
On the 64 held-out samples the selected model reaches an RMSE of 0.157 d⁻¹, a mean absolute error of 0.131 d⁻¹, and an R² of 0.384. About 42 % of test predictions fall within ±0.1 d⁻¹ of the measurement and 77 % within ±0.2 d⁻¹. The cross-validated R² (0.283) is lower than the test R², and with only 64 test samples both values are uncertain, so a realistic expectation is an R² between roughly 0.3 and 0.4.

![figure_parity.png](figures/figure_parity.png)

**Figure 5** - Measured versus predicted growth rate on the test set, for the polynomial and k-NN models.

![figure_residuals.png](figures/figure_residuals.png)

**Figure 6.** Residuals of the polynomial model on the test set.

The errors are not random. The predictions are squeezed toward the middle: they stay between about 0.1 and 0.5 d⁻¹ while measured values reach 0.7 d⁻¹, so high growth rates are underestimated and low ones overestimated (samples measured at zero are predicted at roughly 0.2 to 0.25 d⁻¹). The residuals also fan out as the predicted rate increases, reaching about ±0.3 d⁻¹. The model is therefore better suited to ranking conditions and showing trends than to predicting one individual culture.

### Influence of Temperature, pH, Glucose, and Lactate

**Table 3**- Largest weights of the selected polynomial model with standardized inputs

|Term                     |Weight |
|-------------------------|-------|
|Intercept (baseline)     |0.3631 |
|[pH]^2                   |-0.0645|
|[Lactate]                |-0.0615|
|[Glucose]                |0.0318 |
|[Temperature]^2          |-0.0244|
|[Temperature]            |-0.0199|
|[Glucose] * [Lactate]    |-0.0194|
|[Temperature] * [pH]     |-0.0187|
|[Lactate]^2              |0.0112 |
|[Temperature] * [Glucose]|-0.0111|
|[pH]                     |-0.0098|
|[Temperature] * [Lactate]|-0.007 |
|[pH] * [Glucose]         |-0.0056|
|[pH] * [Lactate]         |0.0051 |
|[Glucose]^2              |-0.0002|

The intercept (0.363 d⁻¹) is the predicted growth rate at average conditions, and every other term modifies it. The largest weight is the squared pH term, which indicates a strong optimum in pH. With the other factors at their means, predicted growth peaks at pH 6.97 (0.364 d⁻¹) and falls to zero below about pH 6.2 and above about pH 7.8, so pH has the largest effect of the four factors.

![figure_one_factor_effects.png](figures/figure_one_factor_effects.png)

**Figure 7.** Predicted growth rate when one factor varies and the others stay at their mean. The tick marks at the bottom show where the training samples are.

* **pH** has the largest effect. The squared pH term is the biggest weight, and growth peaks near pH 7 at about 0.37 d⁻¹, falling to zero below about pH 6.2 and above about pH 7.8.
* **Lactate** inhibits growth. Its weight is the second largest and negative, and predicted growth falls from about 0.49 d⁻¹ at 0 mM to 0.28 d⁻¹ at 40 mM, mostly within the first 20 mM.
* **Temperature** has an optimum near 36 °C (about 0.37 d⁻¹), and growth declines above about 38 °C in both models. The low-temperature side is less certain: the polynomial drops to 0.15 d⁻¹ at 30 °C while the k-NN stays near 0.3 d⁻¹, and only 8 samples lie below 33 °C.
* **Glucose** helps, with predicted growth rising from about 0.31 to 0.45 d⁻¹ across its range. It is the smallest of the four effects.
* **Interactions** matter. The negative glucose–lactate weight means glucose helps most when lactate is low, and the right map in Figure 8 shows the best growth in the high-glucose, low-lactate corner. The negative temperature–pH weight means the best pH shifts slightly lower as temperature rises.

The k-NN model agrees with the polynomial on the direction of every effect and on the optima of temperature and pH, so these conclusions do not come from the polynomial form alone. The two models differ mainly at the edges of the ranges, where data are sparse.

![figure_interaction_maps.png](figures/figure_interaction_maps.png)

**Figure 8** - Predicted growth rate in the temperature–pH and glucose–lactate planes. The dashed white line encloses the reliable region

### Operating Conditions and Reliability

A search over the region covered by the data (5th to 95th percentile of each factor, restricted to conditions near real samples) gives the summary in Table 4. The first row is the single best condition; the other rows give the median and, in parentheses, the 5th to 95th percentile range of the top and bottom 2 % of conditions.

**Table 4**- Predicted growth rate and conditions of the best, high-growth, and low-growth groups

|Group                 |N  |Predicted growth rate|Temperature [C]    |pH              |Glucose [mM]       |Lactate [mM]       |
|----------------------|---|---------------------|-------------------|----------------|-------------------|-------------------|
|Best single condition |1  |0.599                |35.83              |6.92            |28.83              |0.00               |
|High growth (top 2%)  |991|0.531                |36.27 (34.09-38.89)|6.92 (6.70-7.23)|26.77 (20.59-28.83)|0.00 (0.00-4.40)   |
|Low growth (bottom 2%)|991|0.059                |39.76 (34.52-40.20)|7.53 (6.47-7.53)|14.41 (0.00-26.77) |24.21 (13.21-30.81)|

**High growth** (about 0.5 to 0.6 d⁻¹) needs a temperature of 34 to 39 °C (median 36), a pH of 6.7 to 7.2 (median 6.9), almost no lactate (below about 4.4 mM), and fairly high glucose (21 to 29 mM). **Low growth** (below about 0.1 d⁻¹) appears with high lactate (13 to 31 mM) together with a warm culture near 40 °C and a pH near 7.5, which is the upper edge of the searched range. Even the best predicted rate (0.60 d⁻¹) is below the highest measured value (0.75 d⁻¹), another sign that predictions are pulled toward the mean.

**Predictions are not reliable** in these situations:

* **Extremes of one factor.** Few samples lie below 33 °C (8), above 40.5 °C (10), below pH 6.45 (13), above pH 7.55 (15), above 30 mM glucose (13), or above 30 mM lactate (21).
* **Combinations of extremes.** With four factors the space is sparse, so many combinations, such as high glucose with high lactate, are far from any sample. The dashed line in Figure 8 shows where this starts.
* **Outside the studied ranges.** A quadratic eventually turns downward, so values beyond 30 to 42 °C, pH 6.0 to 7.8, and 0 to 40 mM for glucose and lactate should not be trusted.
* **Zero growth.** The 24 zero-growth samples appear even under ordinary conditions, so they cannot be predicted from these four factors.

The error by quartile ([`tables/reliability_by_range.csv`](tables/reliability_by_range.csv)) stays between 0.127 and 0.162 d⁻¹, so no range is dramatically worse. It is highest where growth is highest, for example lactate below 7.35 mM (0.162), glucose above 20.4 mM (0.161), and temperature between 36.9 and 38.3 °C (0.160).

### Advantages and Limitations

The polynomial is simple, fast, and interpretable: 15 numbers describe the whole model, optima and interactions can be read directly, training takes a fraction of a second, and its accuracy matches the more flexible k-NN model. Its smooth predictions also suit optimization.

Its limitations are just as clear. It explains only about 30 to 40 % of the variance and compresses extreme growth rates toward the mean. It can only clip negative values and cannot represent the floor at 0 and the ceiling at 0.75 d⁻¹. A quadratic forces symmetric optima and extrapolates poorly. Only four factors are included, so oxygen, other nutrients, or cell age are absorbed into the noise. The test set is small (64 samples), so the performance estimate is itself uncertain, and the data are simulated, so a real culture may behave differently.

### Potential Applications

The model can screen operating windows before running experiments, for example to decide which pH and temperature setpoints are worth testing and to narrow the design space of a new process. It can support feeding strategies by showing the trade-off between glucose supply and lactate buildup: keeping lactate low, for instance with controlled feeding or medium exchange, lets a high glucose level pay off. In a monitoring system it could give the expected growth rate for the current readings, so a large deviation flags a possible process problem. It could also supply growth-rate estimates to larger bioprocess simulations. In every case, predictions should stay inside the reliable region, and a new process would need the model to be retrained on its own data.