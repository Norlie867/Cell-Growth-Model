# Cell Growth Model
This project predict and analyze the specific growth rate of cell cultures based on four key environmental factors: temperature, pH, glucose, and lactate. It solves the challenge of bioprocess optimization by rigorously training and comparing polynomial and k-Nearest Neighbors models. Furthermore, it integrates cross-validation, learning curves, and a strict reliability domain mechanism to prevent dangerous extrapolations, ensuring safe and actionable insights for biological research  

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
The project runs end to end in order, and each step writes its results (numbers, tables) to the figures/ and tables/ folders. The main.py script only contains the sequence of steps: file paths, column names, random seeds and hyperparameter ranges are centralised in src/constants.py, while the logic of each step lives in src/functions.py. Keeping parameters separate from logic is practical: to try a different search range, you only need to edit one file, without touching the function code.

### Data loading and validation
The workflow starts by reading the CSV file and running a check: the five expected columns must exist and no value must be missing. The validated table becomes the single source of data for all following steps, which avoids re-reading the file in several places and risking inconsistencies.

### Exploratory analysis
The program then summarises each variable (mean, spread, range, number of zeros) along with the correlation matrix. It also plots histograms and the growth rate against each factor, with binned means. This step serves three purposes: checking that the factors cover a relevant range, spotting possible correlations between factors (which would blur the interpretation later), and seeing whether the relationships are curved. This observed curvature is precisely what justifies using polynomial and nearest-neighbour models rather than a simple straight line.

### Train / test split
The data is converted into NumPy arrays and then randomly split: 80% of the samples (254) are used for development and 20% (64) are set aside. The test set is used only once, at the very end, to report the final performance. This split guarantees that the reported error is an honest estimate on conditions never seen before, rather than a measure of memorisation. A fixed random seed ensures the split is identical on every run.

### Candidate models
Two model families are defined as pipelines. Polynomial regression standardises the four variables, expands them into powers and interaction terms, then fits a linear regression on this extended set. The k-nearest neighbours (k-NN) model standardises the variables and averages the growth rates of the closest training samples. Standardisation is placed inside the pipeline so that the means and standard deviations are learned only on the training part of each cross-validation fold: this way, validation samples do not "leak" into training. It also puts temperature (around 30–42 °C) and lactate (0–40 mM) on comparable scales, which matters for distance calculations and makes the polynomial weights comparable with each other.

### Hyperparameter tuning
Each family has settings that cannot be learned directly from the data: the polynomial degree (1 to 10) and, for k-NN, the number of neighbours (1 to 40) as well as the weighting scheme (uniform or distance-based). A grid search with 4-fold cross-validation is run on the training set: each combination is trained on three folds and evaluated on the fourth using RMSE. The training error is also recorded, because the gap between training and validation error reveals underfitting and overfitting. Separate searches are run for a purely linear model, for the polynomial family, and for each k-NN weighting, so that each alternative is compared with its own best settings.

### Model comparison and selection
The tuned candidates are compared against a baseline that always predicts the mean growth rate; this shows the real share of the error that the models manage to explain. The final choice follows an explicit rule: the model with the lowest RMSE wins, unless the polynomial is within one standard deviation of it, in which case the polynomial is kept because it is simpler and its weights can be read directly. Stating the rule makes the selection reproducible and prevents the test set from influencing it.

### Learning curves
To find out whether more experiments would help, the program repeats the fit on random subsets of increasing size, from 60 samples up to the full dataset. Each subset gets its own train/test split and its own grid search, and training and test errors are recorded for both families. This step does not affect model selection: it only serves to diagnose whether the amount of data is sufficient.

### Evaluation on the test set
The selected model and its closest alternative are evaluated on the held-out data with RMSE, mean absolute error and R². The results are shown in a measured-vs-predicted plot and a residual plot. Negative predictions are set to zero, since a growth rate cannot be negative. These plots make it possible to see whether the errors are random or systematic — for example, whether high growth rates are consistently underestimated.

### Interpretation
The selected model is then refitted on all 318 samples, so that every observation contributes to the final model. Its weights are exported: since the inputs are standardised, they are directly comparable, with squared terms signalling an optimum and cross terms signalling an interaction. For a view that is independent of the model type, each factor is varied over its observed range while the others are held at their mean, and this is done for both the polynomial and the k-NN model. The two-dimensional maps show the temperature–pH and glucose–lactate relationships. The fact that two very different model families reach similar conclusions strengthens confidence in the interpretation.

### Operating region and reliability
Finally, the program scans a four-dimensional grid of conditions to identify the highest and lowest predicted growth rates. A condition is only accepted if it is close to the training data, which is measured by the distance to its fifth-nearest sample compared with the typical distances observed in the dataset. The same boundary is drawn on the maps, and the residual error is reported for each quartile of each factor. This step turns the model from a simple curve fit into a practical guide, while also specifying the zones where it should not be trusted.


## Analysis
### Dataset Characteristics
The dataset contains 318 simulated experiments with no missing values. Growth rates range from 0 to 0.75 d⁻¹ (mean 0.283 d⁻¹); 24 samples (7.5 %) have exactly zero growth and 4 sit at exactly 0.75 d⁻¹, which suggests a floor and a ceiling in the measurements. The four factors are essentially uncorrelated with one another (largest |r| = 0.073), so their effects can be separated cleanly. Pearson correlations with growth are modest: lactate −0.29, glucose +0.18, temperature −0.12, and pH +0.01. The near-zero value for pH does not mean pH is unimportant. A linear correlation cannot detect an optimum, and the binned means below show growth peaking around pH 7 and dropping toward both pH extremes, while for temperature the clearest signal is a lower growth rate in the hottest samples (above about 39 °C).
![figure_growth_vs_features.png](figures/figure_growth_vs_features.png)

### Choice of Model Architecture
The selected model is a second-degree polynomial regression with standardized inputs, fitted by ordinary least squares and refit on all data for interpretation. Degree 2 was chosen for three reasons. First, the data call for curvature: temperature and pH both show an interior optimum, which only a squared term can represent. Second, cross terms allow interactions such as nutrient limitation by glucose being weakened by lactate inhibition. Third, validation error was lowest at degree 2. With four features this model has only 14 weights for 254 training samples (about 18 samples per weight), so no extra regularization was needed.

Table 1. Training and validation error for polynomial degrees 1 to 6

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

As Table 1 shows, training error keeps falling as the degree rises, but validation error is lowest at degree 2 (0.1542 d⁻¹) and then worsens: degree 3 is already worse and less stable (0.1645 ± 0.0180), and degrees 5 and 6 are disastrous. At degree 6 the model has 209 terms, more than the roughly 190 training samples in each fold, so it interpolates the training data perfectly (training RMSE of 0) and fails completely on new data. This is the typical signature of overfitting, and the widening gap between training and validation error is visible in the validation curve.
![figure_validation_curve_poly.png](figures/figure_validation_curve_poly.png)

### Comparison with Alternatives

All candidates were tuned on the training set only and then scored once on the held-out test set (Table 2).

Table 2. Comparison of all candidate models

|Model          |Hyperparameters          |Valid RMSE|Valid RMSE SD|Train RMSE|Test RMSE|Test MAE|Test R2|
|---------------|-------------------------|----------|-------------|----------|---------|--------|-------|
|Baseline (mean)|none                     |0.1838    |0.0118       |0.1835    |0.2068   |0.1707  |-0.0686|
|Linear         |degree = 1               |0.1739    |0.008        |0.172     |0.1907   |0.1643  |0.0917 |
|Polynomial     |degree = 2               |0.1542    |0.0104       |0.1414    |0.157    |0.1309  |0.3841 |
|kNN (uniform)  |k = 9, weights = uniform |0.1533    |0.0113       |0.1418    |0.1621   |0.1363  |0.3434 |
|kNN (distance) |k = 9, weights = distance|0.1515    |0.0112       |0.0       |0.1587   |0.1341  |0.3707 |

The linear model underfits: its training and validation errors are nearly equal (0.172 and 0.174) but both are high, and its test R² is only 0.09. The curvature captured by degree 2 lowers validation RMSE by about 11 % relative to the linear model. The k-NN models reach the same accuracy range. The best k is 9 for both weightings, with validation RMSE of 0.1533 (uniform) and 0.1515 (distance). Distance weighting gives a training RMSE of exactly 0, because each training sample is its own nearest neighbor at distance zero and is reproduced perfectly, so it memorizes the noise in the training data even though its validation error is similar.
![figure_validation_curve_knn.png](figures/figure_validation_curve_knn.png)

The difference between the best k-NN model and the polynomial (0.0027 d⁻¹) is far smaller than the fold-to-fold standard deviation (about 0.010 d⁻¹), and on the test set the polynomial is actually slightly better (0.1570 versus 0.1587 d⁻¹). The two families are therefore statistically indistinguishable in accuracy, and the choice rests on other properties. The polynomial is described by 15 interpretable numbers, produces smooth predictions, and can be inspected for optima and interactions. The k-NN model must store all samples, produces staircase-like surfaces, and cannot predict outside the range of growth rates it has seen. Because the accuracy is equivalent, the simpler and more interpretable polynomial was selected. The polynomial and k-NN models cut the test RMSE by only about 22 to 24 % relative to the mean baseline (0.2068 d⁻¹), which indicates a large amount of variability that none of the four measured factors explains.

### Learning Curve
The learning curve repeats the whole tuning procedure on subsets of 60 to 318 samples.
![figure_learning_curve.png](figures/figure_learning_curve.png)

With only 48 training samples the polynomial test RMSE is 0.256 d⁻¹, because there is too little data to estimate the curvature reliably. From about 140 training samples onward it settles at roughly 0.14 to 0.17 d⁻¹, and the polynomial then consistently uses degree 2. The k-NN test RMSE stays between 0.13 and 0.18 d⁻¹ (its lowest value, at 48 samples, comes from a test set of only 12 samples), while its training RMSE is 0 in almost every case. Both curves roughly level off, so collecting more of the same kind of data is unlikely to improve accuracy much; adding new factors or reducing measurement noise would probably help more. The curves are noisy, since each point uses a different random subset and a different small test split (the bump at 208 samples is an example), so only their overall trend should be trusted.

### Model Performance and Residuals
On the 64 held-out samples the selected model reaches an RMSE of 0.157 d⁻¹, a mean absolute error of 0.131 d⁻¹, and an R² of 0.384. About 42 % of test predictions fall within ±0.1 d⁻¹ of the measurement and 77 % within ±0.2 d⁻¹. The cross-validated R² (0.283) is lower than the test R², and with only 64 test samples both values are uncertain, so a realistic expectation is an R² between roughly 0.3 and 0.4.
![figure_parity.png](figures/figure_parity.png)
![figure_residuals.png](figures/figure_residuals.png)

The errors are not purely random. Predictions are pulled toward the mean: in out-of-fold predictions, samples with measured growth above 0.5 d⁻¹ are underestimated by 0.20 d⁻¹ on average, while samples below 0.1 d⁻¹ are overestimated by 0.16 d⁻¹. This is expected when the inputs explain only part of the variance. The residual spread is also larger where growth is high (about 0.16 d⁻¹ for the upper half of predictions versus about 0.14 d⁻¹ for the lower half). The model should therefore be used to rank conditions and estimate trends, not to predict an individual culture precisely.

### Influence of Temperature, pH, Glucose, and Lactate

Table 3. Largest weights of the selected polynomial model with standardized inputs

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
Temperature has a moderate optimum at about 36.1 °C (0.367 d⁻¹), falling to about 0.15 d⁻¹ at both 30 °C and 42 °C. Lactate is inhibitory: growth falls from 0.49 d⁻¹ at 0 mM to 0.28 d⁻¹ at 40 mM, with the steepest drop below about 20 mM. Glucose is beneficial, rising from 0.31 d⁻¹ at 0 mM to 0.45 d⁻¹ at 40 mM, but this is the smallest of the four effects. The negative glucose–lactate weight shows that the factors interact: raising glucose from 0 to 40 mM increases growth by 0.29 d⁻¹ when lactate is absent, by 0.10 d⁻¹ when lactate is 20 mM, and slightly decreases it at 40 mM. The last result is an artifact of the quadratic surface in a sparsely sampled region and should not be interpreted biologically. The temperature–pH weight is also negative, so the best pH decreases somewhat as the temperature rises. The k-NN curves follow the same shapes, which confirms that the polynomial captured real structure and not an artifact of its form.
![figure_interaction_maps.png](figures/figure_interaction_maps.png)

### Operating Conditions and Reliability

A search over the region covered by the data (5th to 95th percentile of each factor, restricted to conditions near real samples) gives the summary in Table 4. The first row is the single best condition; the other rows give the median and, in parentheses, the 5th to 95th percentile range of the top and bottom 2 % of conditions.

Table 4. Predicted growth rate and conditions of the best, high-growth, and low-growth groups

|Group                 |N  |Predicted growth rate|Temperature [C]    |pH              |Glucose [mM]       |Lactate [mM]       |
|----------------------|---|---------------------|-------------------|----------------|-------------------|-------------------|
|Best single condition |1  |0.599                |35.83              |6.92            |28.83              |0.00               |
|High growth (top 2%)  |991|0.531                |36.27 (34.09-38.89)|6.92 (6.70-7.23)|26.77 (20.59-28.83)|0.00 (0.00-4.40)   |
|Low growth (bottom 2%)|991|0.059                |39.76 (34.52-40.20)|7.53 (6.47-7.53)|14.41 (0.00-26.77) |24.21 (13.21-30.81)|

High growth (about 0.5 to 0.6 d⁻¹) requires a temperature of about 34 to 39 °C (median 36 °C), a pH of about 6.7 to 7.2 (median 6.9), almost no lactate (below about 4.5 mM), and relatively high glucose (about 20 to 29 mM). Low growth (below about 0.1 d⁻¹) occurs when lactate is high (13 to 31 mM) and the temperature or pH is away from its optimum, typically a warm culture around 40 °C with a pH near 7.5, the upper edge of the searched range. Even the best predicted rate (0.60 d⁻¹) is below the highest measured value (0.75 d⁻¹), another sign of the compression toward the mean.

Predictions are least reliable in the following situations:

* Extremes of a single factor. Few samples exist below 33 °C (8), above 40.5 °C (10), below pH 6.45 (13), above pH 7.55 (15), above 30 mM glucose (13), or above 30 mM lactate (21).
* Combinations of extremes. Four factors make the space very sparse, so many combinations are far from any sample, for instance high glucose with high lactate. The dashed line on the maps marks where this starts.
* Outside the studied ranges. The quadratic form eventually turns downward or becomes negative, so values beyond 30 to 42 °C, pH 6.0 to 7.8, 0 to 40 mM glucose, and 0 to 40 mM lactate should not be trusted.
* Zero and near-ceiling growth. Zero-growth samples sit, on average, close to ordinary conditions (37.8 °C, pH 7.1, 12 mM glucose, and 18 mM lactate), so these values are not predictable from the four factors.

Residual error by quartile (tables/reliability_by_range.csv) varies only between 0.127 and 0.162 d⁻¹, so no single range of one factor is clearly worse, but errors are highest where growth is highest, for example lactate below 7.35 mM (0.162 d⁻¹) or glucose above 20.4 mM (0.161 d⁻¹).

### Advantages and Limitations

The main advantages of the selected model are interpretability, simplicity, and speed: 15 numbers describe the whole model, the optima and interactions can be read directly, training takes a fraction of a second, and its accuracy matches a more flexible k-NN model. Its smooth predictions are also suitable for optimization.

Its limitations are equally clear. It explains only about 30 to 40 % of the variance, compresses extreme growth rates toward the mean, and cannot represent the measurement floor at 0 and ceiling at 0.75 d⁻¹ other than by clipping negative values. The quadratic form forces symmetric optima and extrapolates poorly. Only four factors are considered, so effects such as dissolved oxygen, other nutrients, or cell age are absorbed into noise. The test set is small (64 samples), so the performance estimate itself is uncertain, and the data are simulated, so real cultures may behave differently.

### Potential Applications

The model can screen operating windows before running experiments, for example to identify which pH and temperature setpoints are worth testing and to narrow the design space of a new process. It can support feeding strategies by quantifying the trade-off between glucose supply and lactate accumulation; the interaction suggests that keeping lactate low (for instance through controlled feeding or medium exchange) allows high glucose to pay off. Inside a monitoring system, it can provide an expected growth rate for the current pH, temperature, glucose, and lactate readings, so that a large deviation flags a possible process problem. It can also supply growth-rate inputs to larger bioprocess simulations. In every case, predictions should be limited to the reliable region, and a new process would need to be retrained with its own experimental data.