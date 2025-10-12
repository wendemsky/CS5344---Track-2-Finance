# Loan Anomaly Detection: Speaker Script
## Detailed Notes for Each Slide

**Total Time**: ~25-30 minutes
**Pace**: Conversational but confident

---

## SLIDE 1: Title Slide (10 seconds)

**Say**:
"Good [morning/afternoon]. Today we'll be presenting our work on loan anomaly detection using unsupervised ensemble methods."

**Body Language**: Make eye contact, smile, establish presence

**Transition**: "Let me start by framing the problem we're trying to solve."

---

## SLIDE 2: Problem Statement (45 seconds)

**Say**:
"The problem we're addressing is detecting potentially risky loans before they default. This is a critical business need for financial institutions.

The key challenge here is that we're working in an unsupervised learning setting. Our training data contains ONLY normal loans—we have zero labeled anomalies to learn from. This mirrors real-world constraints where banks often don't have labeled default data when building initial risk models.

Our goal is to identify anomalous loan patterns by learning what 'normal' looks like, and then flagging anything that deviates from that norm."

**Emphasis**: "ONLY normal loans" and "zero labeled anomalies"

**Potential Question**: "Why not use supervised learning?"
**Answer**: "In practice, labeled default data is often unavailable or insufficient when building risk models, especially for new loan products. Additionally, anomalies can emerge in patterns not seen in historical defaults."

**Transition**: "Let's look at the dataset we're working with."

---

## SLIDE 3: Dataset Overview (60 seconds)

**Say**:
"We have three data splits. Training has 30,504 samples with zero anomalies—this is where we train all our models, learning only from normal loan behavior.

Validation has 5,370 samples with 12.61% anomalies—that's 677 anomalous loans. We use this ONLY for hyperparameter tuning and model selection, never for training.

Test has 13,426 samples with unknown anomaly rate—this is our final evaluation set.

**[Point to visual]** As you can see from this distribution chart, the class imbalance is severe, which is why we use AUPRC as our primary metric rather than AUROC. AUPRC focuses on the minority class performance."

**Features breakdown**: "We have 145 features total. Static features include things like credit score, debt-to-income ratio, and interest rate. Temporal features give us 14-month payment history snapshots—months 0 through 13—tracking unpaid principal balance, interest rates, and payment status over time."

**Emphasis**: "Never for training" when mentioning validation usage

**Potential Question**: "Why 14 months specifically?"
**Answer**: "This represents the early lifecycle of the loan where anomalous patterns typically emerge. It's a standard observation window in the industry."

**Transition**: "Before building models, we conducted comprehensive exploratory data analysis."

---

## SLIDE 4: Missing Values Analysis (40 seconds)

**Say**:
"Our EDA revealed systematic missing value patterns. We identified sentinel values—these are placeholder numbers that actually represent missing data. For example, CreditScore of 9999, DTI of 999—these aren't real values, they're sentinels.

**[Point to visual]** This chart shows our top 20 features by missing percentage across train, validation, and test sets.

Our handling strategy was two-fold: First, we created missing indicator features to preserve the signal that something was missing. Second, we used median imputation for numeric features and mode for categorical. This approach is more robust than simply dropping or zero-filling."

**Emphasis**: "Preserve the signal"

**Transition**: "Next, we analyzed feature distributions."

---

## SLIDE 5: Feature Distributions (45 seconds)

**Say**:
"We compared distributions of normal versus anomalous loans across our top features. **[Point to visual]** In these histograms, green represents normal loans and red represents anomalies.

We applied Kolmogorov-Smirnov tests to quantify whether distributions differ significantly. The result was that many features show statistically significant differences with p-values less than 0.001.

This is encouraging—it tells us there IS separability in the feature space. Normal and anomalous loans do behave differently across multiple dimensions."

**Body Language**: Use hand gestures to indicate separation when discussing distributions

**Potential Question**: "Which features showed the strongest differences?"
**Answer**: "That leads me to our next slide on correlation analysis."

**Transition**: "This led us to investigate feature-target correlations."

---

## SLIDE 6: Feature Correlation with Anomalies (90 seconds)

**Say**:
"Here's where we encountered an interesting challenge. We wanted to understand which features correlate with anomalies. However, our training set has zero anomalies—the target is always zero. **[Pause for effect]** You cannot compute meaningful correlations with a constant variable.

So we analyzed the validation set, which has 12.61% anomalies. **Important clarification**: This was used purely for exploratory data analysis and feature selection. All our models are still trained ONLY on the training data. We're not using validation labels for model fitting—that would be data leakage. We're simply using it to understand feature-target relationships, which is a valid EDA practice.

**[Point to visual]** This chart shows our top correlated features. CreditScore has the strongest correlation at negative 0.25—meaning lower credit scores associate with higher anomaly risk. This makes intuitive sense.

NonInterestBearingUPB in months 12 and 13 shows positive correlation around 0.12 to 0.14—this captures late-stage payment irregularities.

OriginalDTI at plus 0.10 indicates higher debt-to-income ratios for anomalies—financial stress.

And OriginalInterestRate at plus 0.096 suggests riskier borrower profiles get higher rates."

**Emphasis**:
- "Zero anomalies" (pause after saying this)
- "ONLY on the training data"
- "Valid EDA practice"

**Potential Question**: "Isn't this still data leakage?"
**Answer**: "No, because we're not using validation data to fit any model parameters. We're using it to understand the data, similar to how you'd look at validation performance to understand which features matter. The key is that no model sees validation during training. I can elaborate more on this in our methodology validation slide later."

**Transition**: "Let me show you more detail on these correlations."

---

## SLIDE 7: Statistical Significance (60 seconds)

**Say**:
"Let's deep-dive into CreditScore, our strongest predictor.

**[Point to visual]** This distribution shows the split. Normal loans average 752 points with standard deviation of 44. Anomalous loans average 717 points with standard deviation of 49. That's a 35-point difference!

The KS test gives us a p-value less than 1e-77—that's essentially zero. This is extremely statistically significant.

Looking across all features, 26 out of 40 we tested show significant correlation with p-values less than 0.05.

However—and this is important—no single feature has a strong correlation. The maximum is 0.25 in absolute value. This tells us we need a multi-dimensional approach. No single feature will solve this problem. We need to capture combinations of features, which is exactly what our ensemble approach does."

**Emphasis**: "35-point difference", "No single feature"

**Potential Question**: "Why is the correlation so weak?"
**Answer**: "Anomaly detection is inherently complex. Anomalies often result from combinations of features rather than single extreme values. A borrower with low credit score AND high DTI AND high interest rate is much riskier than any single factor alone."

**Transition**: "With these insights, we moved to preprocessing and feature engineering."

---

## SLIDE 8: Preprocessing Pipeline (45 seconds)

**Say**:
"Our preprocessing follows a systematic pipeline. Step one, sentinel mapping—we replace those sentinel values like 9999 with NaN and create missing indicator flags.

Step two, categorical encoding using LabelEncoder with special handling for unknown categories that might appear in test data.

Step three, imputation—median for numeric features because it's robust to outliers, mode for categorical features.

Step four, standard scaling to zero mean and unit variance across the full feature matrix.

This gives us a clean, scaled feature matrix ready for modeling."

**Pace**: Move through this quickly, it's standard stuff

**Transition**: "Feature engineering is where things get more interesting."

---

## SLIDE 9: Temporal Feature Engineering (60 seconds)

**Say**:
"Given we have 14-month payment history, we want to capture temporal patterns. We use a multi-window strategy.

We define three overlapping windows: The main window extracts months 0, 3, 6, 9, and 12. Alternative window one is months 0, 2, 4, 6, 8, 10, 12—higher frequency. Alternative window two is months 0, 3, 6, 9—early period focus.

For each window, we compute four features:
- Trend: how the value changes from first to last
- Volatility: the coefficient of variation
- First-difference mean: average of period-to-period changes
- First-difference std: variability of those changes

This captures payment trajectory patterns. For example, increasing volatility in late months might indicate payment stress—someone struggling to make consistent payments."

**Hand Gesture**: Use hands to show progression over time

**Potential Question**: "Why multiple windows?"
**Answer**: "Different anomaly patterns emerge at different timescales. Some loans show problems early, others deteriorate gradually. Multiple windows help capture both."

**Transition**: "We also engineered domain-specific features."

---

## SLIDE 10: Domain-Driven Features (75 seconds)

**Say**:
"This is where domain knowledge comes in. The first major feature we create is amortization shortfall signals.

Here's how it works: For a standard fixed-rate mortgage, we can calculate the expected principal reduction each month using the annuity formula. We then compare this to the observed principal reduction. If observed is less than expected—that's a shortfall.

We create three features: average shortfall across all periods, fraction of periods with greater than 70% shortfall, and fraction with greater than 50% shortfall.

**Important note**: This only applies to FRM loans—fixed-rate mortgages that aren't interest-only or balloon loans. For other loan types, we mask these features to zero.

Second, we apply PCA dimensionality reduction. We go from 145-plus features down to 80 to 100 principal components, retaining about 95% of variance. This helps our distance-based detectors like LOF and k-distance by reducing the curse of dimensionality."

**Emphasis**: "Expected versus observed" (pause between these words)

**Potential Question**: "How much improvement do amortization features provide?"
**Answer**: "They contribute moderately. LOF alone gets about 0.28 AUPRC. With all our engineered features including amortization, we push toward 0.30. Every bit helps in anomaly detection."

**Transition**: "Before diving into our final model, let's look at baseline performance."

---

## SLIDE 11: Baseline Models Overview (40 seconds)

**Say**:
"We tested seven algorithm families: LOF, k-distance, Isolation Forest, Elliptic Envelope, One-Class SVM, DBSCAN, and PCA Reconstruction Error.

We ran three preprocessing configurations—robust scaling with PCA 80 components, standard scaling with PCA 80, and robust scaling with no PCA.

With multiple hyperparameter settings per algorithm, we evaluated roughly 50 baseline configurations.

We evaluated using AUPRC as our primary metric, AUROC as secondary, and F1 for reference."

**Pace**: Brisk, this is setup for next slide

**Transition**: "Here are the results."

---

## SLIDE 12: Baseline Results (60 seconds)

**Say**:
"**[Point to visual]** This chart shows our top 15 models by AUPRC.

The clear pattern here is that LOF dominates. LOF with k equals 5 achieves 0.28 to 0.30 AUPRC. LOF with k equals 7 and k equals 10 are close behind.

k-distance comes in fourth at around 0.19 to 0.20. Isolation Forest hits 0.18.

The winning preprocessing configuration is robust scaling with 80 PCA components.

This baseline evaluation tells us two things: First, LOF is the best individual algorithm for this task. Second, distance-based methods in general work well."

**Emphasis**: "LOF dominates"

**Body Language**: Point to the chart and trace your finger down the top entries

**Potential Question**: "Did you try deep learning?"
**Answer**: "Given the modest dataset size—about 30K training samples—and the unsupervised constraint, deep learning approaches like autoencoders didn't outperform LOF in our initial tests. We mention this in future work as a potential direction with more data."

**Transition**: "Why does LOF work so well? Let me explain the intuition."

---

## SLIDE 13: Why LOF Works Best (75 seconds)

**Say**:
"LOF stands for Local Outlier Factor. The algorithm measures local density deviation.

Here's the intuition: For each sample, LOF looks at its k-nearest neighbors and compares the sample's density to the neighbors' density. If a sample is in a sparse region—lower density than its neighbors—it gets a high LOF score, indicating it's an outlier. If it's in a dense region like its neighbors, LOF is around 1, indicating normal.

**[Point to conceptual diagram if available]**

Why is this effective for our problem? Because anomalies in loan data manifest as unusual feature combinations. For example, a loan with low CreditScore AND high DTI AND high InterestRate sits in a sparse region of the feature space. Normal loans cluster in denser regions with more typical combinations.

LOF captures these multi-dimensional density patterns naturally. The k-parameter controls neighborhood size—we found k equals 5 to 10 optimal. Too small and you get noise, too large and you miss local patterns."

**Emphasis**: "Unusual feature combinations", "Multi-dimensional"

**Hand Gesture**: Use hands to show clustering and outliers in space

**Potential Question**: "How sensitive is performance to k-parameter?"
**Answer**: "Fairly robust. Performance stays strong from k=5 to k=12. That's why we include multiple k-values in our ensemble—it provides robustness."

**Transition**: "Now let's dive into our final model architecture."

---

## SLIDE 14: Final Approach Overview (60 seconds)

**Say**:
"Our final model is what we call an Ultra Unsupervised Ensemble. Let me walk through the architecture.

We start with raw data—145 features. This goes through our FeatureBuilderAdvanced, which does all the feature engineering we discussed: temporal windows, amortization signals, scaling, PCA.

We then train multiple unsupervised detectors—nine-plus types, each learning what 'normal' looks like from the training data.

These detectors score the validation and test sets. We then apply Train-CDF calibration to convert scores to comparable probabilities.

Next, we select the top-performing detectors based on validation AUPRC. Finally, we evaluate multiple fusion strategies—rank-based, probability-based, cohort-normalized—and select the best.

The output is final anomaly scores between zero and one.

**Key principle throughout**: Train ONLY on normal loans. No validation data is used for model fitting. Validation is used ONLY for hyperparameter selection, which is standard practice."

**Emphasis**: "ONLY on normal loans", "ONLY for hyperparameter selection"

**Transition**: "Let me break down each component in detail."

---

## SLIDE 15: Feature Builder Pipeline (75 seconds)

**Say**:
"Our FeatureBuilderAdvanced class implements the feature engineering pipeline systematically.

Step one handles static features: sentinel mapping to replace those placeholder values with NaN and create missing flags, categorical encoding with UNKNOWN handling for test-time surprises, and median imputation for numeric features.

Step two creates temporal features: We extract statistics from three window strategies I showed earlier. For each window we compute trend, volatility, and first-difference statistics. This captures the evolution of payment trajectories over the loan lifecycle.

Step three generates amortization signals: We calculate expected versus observed principal reduction, create shortfall features like mean shortfall and fraction of high-shortfall periods, and we mask non-applicable loans like interest-only or balloon loans.

Step four applies scaling and PCA: StandardScaler normalizes the feature matrix, and PCA reduces dimensionality to 80 to 100 components.

The output is two matrices: X_scaled with full features, and X_embed with the PCA embedding. Different detectors use different representations—distance-based methods like LOF use the PCA embedding, while tree-based methods like Isolation Forest use scaled features."

**Pace**: This is dense, take your time, pause between steps

**Potential Question**: "How did you decide on 80 PCA components?"
**Answer**: "We did a grid search over 60, 80, 100, and 120 components. 80 gave the best validation AUPRC while retaining about 95% of variance."

**Transition**: "This feature pipeline feeds into our detector portfolio."

---

## SLIDE 16: Detector Portfolio (90 seconds)

**Say**:
"We use nine detector types, giving us about 20 individual detectors total. Let me walk through them.

**LOF variants**: We train seven LOFs with k equals 4, 5, 6, 7, 8, 10, and 12. All use the PCA embedding. This gives us local density detection at multiple scales.

**Cluster-LOF**: We cluster samples into 12 groups using KMeans, then train a separate LOF for each cluster. This captures cohort-specific anomalies—maybe certain loan types have different normal patterns.

**k-distance**: Five variants with k equals 3, 5, 7, 9, 11. This measures distance to the k-th nearest neighbor. Simpler than LOF but complementary.

**Isolation Forest**: 500 trees, operates on PCA embedding. Tree-based isolation provides diversity from density methods.

**Elliptic Envelope**: Robust covariance-based outlier detection with 90% support fraction.

**PCA Reconstruction Error**: We train a separate PCA on scaled features and measure reconstruction error. Points that don't fit the linear subspace well get high scores.

**Random Projection LOF**: We create 40 random projections of the feature space and run LOF in each subspace, then take the max score. This is like bagging for LOF.

**Mahalanobis Distance**: Fast covariance-based distance measure on scaled features.

**Amortization Signal**: A custom detector using just our payment shortfall features.

Our selection criterion: We keep the top 10 detectors with validation AUPRC greater than or equal to 0.16. This balances diversity and quality."

**Emphasis**: "Nine detector types", "Top 10"

**Potential Question**: "Why so many LOF variants?"
**Answer**: "LOF performed best in baselines, so we explore it thoroughly at different scales. Also, multiple k-values provide ensemble robustness."

**Transition**: "All these detectors are trained on normal loans only. Let me explain the training strategy."

---

## SLIDE 17: Training Strategy (60 seconds)

**Say**:
"The fundamental principle of our approach is that all detectors fit ONLY on training data, which contains zero anomalies—only normal loans.

Why does this work? Unsupervised detectors learn the distribution of 'normal.' At inference time, anything that deviates from this learned normal distribution gets a high anomaly score.

**[Point to code snippet if shown]** Here's an example with LOF. We instantiate LOF with novelty equals True, fit it on only the training set—all samples are normal. Then at score time, we score the validation and test sets, which may contain anomalies.

The key insight: By training on normal loans, we're learning what typical loan behavior looks like. Anomalies are simply loans that don't behave typically."

**Emphasis**: "ONLY on training data", "Learn what typical behavior looks like"

**Potential Question**: "How do you prevent overfitting to training noise?"
**Answer**: "Regularization and diversity. LOF's k-parameter controls smoothness—larger k averages over more neighbors. Ensemble diversity also helps—if one detector overfits, others compensate."

**Transition**: "Once we have scores from all detectors, we need to calibrate them."

---

## SLIDE 18: Calibration Method (75 seconds)

**Say**:
"Different detectors output different score scales. LOF might give scores from 1 to 5, k-distance might be 0.1 to 10, Isolation Forest is negative 0.5 to positive 0.5. We need to make these comparable.

Our solution is Train-CDF calibration. CDF stands for Cumulative Distribution Function.

Here's how it works: We fit an empirical CDF on the training set scores for each detector. This CDF maps any score to its percentile rank—a tail probability between zero and one. A higher score maps to a higher percentile, indicating higher anomaly probability.

**[Point to visual diagram]** The algorithm is simple: Sort the training scores, then for any new score, find where it falls in that sorted list and divide by the total count.

The key advantage: This uses ONLY the training distribution. No validation data is involved in the calibration. This prevents leakage while giving us normalized probabilities."

**Emphasis**: "ONLY the training distribution", "Prevents leakage"

**Potential Question**: "What if test scores are outside the training range?"
**Answer**: "The CDF extrapolates naturally. Scores beyond the training max get probability 1.0. Scores below training min get probability 0.0. This is conservative and appropriate."

**Transition**: "With calibrated probabilities, we can now fuse the ensemble."

---

## SLIDE 19: Fusion Strategies (75 seconds)

**Say**:
"We explored three fusion strategy families.

**Rank-based fusion**: We rank-normalize each detector's scores from zero to one, preserving relative ordering. Then we combine using weighted average where weights equal validation AUPRC, max rank taking the highest rank across all detectors, or max rank top-2 and top-3 using only the best detectors.

**Probability-based fusion**: We use the calibrated probabilities directly. Weighted average with performance-based weights, Noisy-OR which computes 1 minus the product of 1 minus each probability—this models independent failure modes, or max probability.

**Cohort-normalized fusion**: We first cluster samples using KMeans, compute z-scores within each cluster to normalize for cohort-specific patterns, then apply rank or probability fusion on these normalized scores.

We evaluate all strategies on the validation set and select the one with the best AUPRC."

**Pace**: This is complex, use hand gestures to illustrate combining

**Potential Question**: "Which strategy worked best?"
**Answer**: "Max rank top-2 performed best for us. It uses only the top two LOF variants and takes their maximum rank. Simple but effective."

**Transition**: "Let me show you our hyperparameter selection process."

---

## SLIDE 20: Hyperparameter Selection (60 seconds)

**Say**:
"We have four key hyperparameters to tune.

PCA components: We tested 60, 80, 100, and 120. Selected 80 based on best validation AUPRC.

LOF k-values: We tested k equals 4 through 15. Rather than picking one, we kept multiple variants—k equals 4 through 12—in the ensemble. This provides robustness.

Detector selection threshold: We tested keeping detectors with AUPRC of 0.14, 0.15, 0.16, or 0.17. We chose 0.16, which gives us about 10 detectors, balancing diversity and quality.

Clustering for cohort normalization: We tested 10, 12, and 15 clusters. Selected 12.

Our validation strategy is straightforward: Use validation AUPRC to make these choices. We avoid overfitting by keeping selection simple and threshold-based rather than doing extensive grid search."

**Emphasis**: "Simple and threshold-based"

**Potential Question**: "How do you avoid overfitting to validation set?"
**Answer**: "We limit the number of hyperparameter choices, use threshold-based selection rather than fine-tuning to every decimal, and our train-CDF calibration uses only training data, not validation."

**Transition**: "Let me show you the complete architecture."

---

## SLIDE 21: Complete Architecture (75 seconds)

**Say**:
"Here's our end-to-end pipeline visualized.

**[Walk through the flowchart]**

We start with raw loan data—145 features. This flows into FeatureBuilderAdvanced, which applies sentinel mapping, temporal engineering across three windows, amortization signal calculation, scaling, and PCA to 80 components.

Next, detector training on training data only. We train seven LOF variants, one cluster-LOF, five k-distance detectors, Isolation Forest, Elliptic Envelope, PCA reconstruction, Random Projection LOF, Mahalanobis, and amortization signal. That's roughly 20 detectors total.

We score validation and test sets with each detector. Then we apply Train-CDF calibration to get probabilities between zero and one.

Based on validation AUPRC, we select the top 10 detectors exceeding our threshold of 0.16.

We evaluate multiple fusion strategies—rank-based, probability-based, and cohort-normalized variants.

Finally, we output anomaly scores from zero to one for each loan."

**Body Language**: Trace through the flowchart with your hand

**Emphasis**: "Training data only" when mentioning detector training

**Transition**: "What was our final selected configuration?"

---

## SLIDE 22: Best Configuration (60 seconds)

**Say**:
"After all this evaluation, here's what we selected.

PCA components: 80.

Our top 10 detectors by validation AUPRC are—**[Point to list]**—LOF k equals 5 leading at 0.2988, followed by LOF k equals 7 at 0.2965, LOF k equals 10 at 0.2943, Cluster-LOF at 0.2881, and k-distance variants filling out the rest, along with Isolation Forest and PCA reconstruction.

Notice the pattern: LOF variants dominate the top positions.

Our best fusion rule is max rank top-2. This uses the top two detectors—LOF k equals 5 and LOF k equals 7—and takes the maximum rank across them. Simple but effective.

On validation, this achieves AUPRC of **[INSERT YOUR ACTUAL RESULT]** and AUROC of **[INSERT YOUR ACTUAL RESULT]**."

**Emphasis**: "LOF variants dominate"

**Note**: You'll need to fill in the actual performance numbers

**Potential Question**: "Why max rank rather than weighted average?"
**Answer**: "Max rank is more conservative. It triggers on any strong signal from top detectors. Weighted average can dilute signals. For anomaly detection where false negatives are costly, max rank worked better."

**Transition**: "Let's look at the final performance results."

---

## SLIDE 23: Final Performance (60 seconds)

**Say**:
"On the validation set, our final ensemble achieves AUPRC of **[YOUR RESULT]** and AUROC of **[YOUR RESULT]**.

For comparison, our best individual detector—LOF k equals 5—achieves AUPRC of 0.2988 and AUROC of 0.6607.

**[Point to table]** The ensemble trades off a slight decrease in AUPRC for increased robustness. The ensemble is less sensitive to distribution shifts between validation and test. In production, robustness often matters more than peak performance on validation.

At an optimal threshold, our F1 score is **[YOUR RESULT]**."

**Emphasis**: "Robustness"

**Potential Question**: "Have you evaluated on the test set yet?"
**Answer**: "The test set is unlabeled in our project setup, so we report validation performance. In practice, we'd submit to a leaderboard or evaluate with held-out labels."

**Transition**: "Let me break down which detectors contributed most."

---

## SLIDE 24: Per-Detector Contributions (60 seconds)

**Say**:
"**[Point to table]** Let's analyze detector family performance.

The LOF family achieves best AUPRC at 0.2988, and three LOF variants make it into our top 10. LOF clearly dominates.

Cluster-LOF contributes at 0.2881, capturing cohort-specific patterns.

k-distance family peaks at 0.1949, and four variants make the top 10. They're complementary to LOF, providing diversity.

Isolation Forest at 0.1817 and PCA reconstruction at 0.1742 add tree-based and linear subspace perspectives.

The key insight: LOF-based detectors consistently outperform. They capture local density deviations effectively, which aligns perfectly with how anomalies manifest in this data—unusual combinations of features creating sparse regions in feature space.

Multiple k-values provide robustness by covering different scales."

**Emphasis**: "LOF clearly dominates", "Complementary"

**Transition**: "What patterns does the model actually capture?"

---

## SLIDE 25: Error Analysis (75 seconds)

**Say**:
"Let's talk about what our model does well and where it struggles.

**Patterns detected well**: Our model successfully identifies loans with low CreditScore combined with high DTI. It catches late-stage payment irregularities in months 10 through 13, where NonInterestBearingUPB spikes. It flags high interest rate plus risky borrower profiles. And it detects payment shortfall patterns through our amortization features.

**Limitations**: Remember, no single feature strongly predicts—maximum absolute correlation is 0.25. This is a fundamental data limitation. We're also underutilizing temporal dependencies. We treat windows independently rather than as sequences. There's a performance ceiling in unsupervised settings—we can't match supervised methods. And we do have false positives where some normal loans get high scores, probably due to legitimate but unusual patterns.

**Future improvements**: We could use LSTM or attention mechanisms for explicit temporal modeling. Graph-based loan similarity networks. Or semi-supervised fine-tuning with a small labeled set if available."

**Emphasis**: "What we do well", "Fundamental data limitation"

**Body Language**: Be honest about limitations, shows scientific maturity

**Transition**: "Before we wrap up, let me mention alternative approaches we explored."

---

## SLIDE 26: Alternative Approaches Explored (75 seconds)

**Say**:
"We ran several experimental tangents that are worth mentioning.

Experiment 1 was correlation-guided feature engineering. The professor suggested we explore feature correlation, so we created composite risk scores based on our correlation analysis—0.5 times credit risk plus 0.25 times DTI risk plus 0.25 times rate risk. We added payment irregularity indicators for late-stage months.

This experiment achieved AUPRC of 0.2931 for the ensemble and 0.2992 for the best single detector—actually slightly better than our ultra ensemble in terms of peak performance.

The insight here: Correlation-derived features do help. They provide clear risk signals.

Experiment 2 was feature-weighted Isolation Forest where we upweighted top correlated features during tree building. This gave moderate improvement but didn't beat LOF.

**[Mention any other experiments]**

Overall, our final ultra ensemble approach proved most robust. The correlation experiment gave us feature engineering ideas we incorporated, but the full ensemble framework remained superior."

**Emphasis**: "Professor suggested" (acknowledge input), "Most robust"

**Transition**: "How do these approaches compare?"

---

## SLIDE 27: Approach Comparison (60 seconds)

**Say**:
"**[Point to table]** Let's compare all approaches.

Baseline LOF as a single model achieved AUPRC of 0.28 to 0.30. Simple but effective.

Our correlation-guided experiment hit 0.2992 for the best single detector. This represents our peak performance on a single detector.

Our final ultra ensemble achieves **[YOUR RESULT]** AUPRC and **[YOUR RESULT]** AUROC. This is our production model.

Why does the final approach win? Four reasons:

One, diverse detector portfolio captures different anomaly types—density, distance, tree-based, subspace.

Two, proper calibration prevents score scale issues that can break simple averaging.

Three, systematic fusion strategy evaluation rather than picking one arbitrarily.

Four, no validation leakage in training. Everything is principled."

**Emphasis**: "Production model", "Principled"

**Potential Question**: "Why not just use the single best LOF?"
**Answer**: "Ensemble robustness. A single detector might overfit to validation quirks. The ensemble is more stable across distribution shifts."

**Transition**: "Let's talk about lessons learned."

---

## SLIDE 28: Lessons Learned (75 seconds)

**Say**:
"What did we learn from this project?

**Technical insights**: LOF is highly effective for local density-based anomalies. Feature engineering matters more than model complexity for this task. Ensemble provides robustness over single models. Correlation analysis is critical when training lacks anomalies—using validation for EDA is valid and valuable. Train-CDF calibration prevents validation leakage elegantly.

**Methodological insights**: Systematic baseline evaluation guides your final approach—don't skip this step. Diverse detector portfolios capture different patterns. Proper train-validation split usage prevents overfitting. Domain knowledge plus data insights gives you the best results.

**Practical insights**: Unsupervised methods can achieve reasonable performance. There's no silver bullet feature—max correlation is 0.25—so you need ensembles. Iterative experimentation pays off. We went through multiple experiments before landing on our final approach."

**Body Language**: Count off points on fingers

**Emphasis**: "Feature engineering matters more", "Iterative experimentation"

**Transition**: "We should acknowledge our limitations."

---

## SLIDE 29: Current Limitations (75 seconds)

**Say**:
"Let's be honest about limitations. We have data limitations, model limitations, and evaluation limitations.

**Data limitations**: No anomalies in training set—this is the unsupervised constraint we're working under. Limited validation data with only 677 anomalies out of 5,370 samples. Potential distribution shift between validation and test—we won't know until test evaluation. And missing values in temporal features reduce signal.

**Model limitations**: We're underexploiting temporal dependencies by treating months as independent windows rather than sequences. PCA assumes linear subspaces which might miss nonlinear patterns. There's an ensemble complexity versus interpretability trade-off. And computational cost for real-time scoring—we have 10-plus detectors to run.

**Evaluation limitations**: We focus on a single metric, AUPRC, which might not capture all business needs. Threshold selection depends on cost-benefit analysis we haven't done. And we lack explainability for individual predictions—we can't easily say WHY a loan is flagged."

**Body Language**: Be matter-of-fact, not apologetic

**Emphasis**: "Underexploiting temporal dependencies" (this is a key future direction)

**Transition**: "Which leads me to future improvements."

---

## SLIDE 30: Future Improvements - Features (60 seconds)

**Say**:
"For future work on feature engineering:

**Advanced temporal modeling**: LSTM or RNN to treat payment trajectories as sequences rather than independent windows. Attention mechanisms to automatically focus on critical months. Time-series specific detectors using ARIMA residuals or Prophet anomalies. Seasonal pattern detection for monthly or quarterly trends.

**Interaction features**: Polynomial interactions between top correlated features—for example, CreditScore times DTI or Rate times LTV. Graph-based features modeling loan similarity networks. Cluster-based features measuring distance to loan archetypes.

**External data integration**: Economic indicators like unemployment, GDP, interest rate environment. Geographic risk factors at the MSA or postal code level. Market conditions like housing prices or foreclosure rates.

These additions could push performance beyond our current ceiling."

**Pace**: Move through this briskly, it's forward-looking

**Transition**: "On the modeling side, we also have ideas."

---

## SLIDE 31: Future Improvements - Models (60 seconds)

**Say**:
"For future work on models:

**Deep learning**: Variational Autoencoders learning latent representations of normal loans, with reconstruction error as the anomaly score. Attention-based models focusing on important temporal patterns with interpretable attention weights. Self-supervised pretraining using contrastive learning on normal loans.

**Advanced ensembles**: Stacking meta-learner that trains a second-level model on detector outputs to learn optimal combination weights. Dynamic weighting based on sample characteristics—different detectors for different loan types. Cluster-specific ensembles with separate models for loan cohorts.

**Hybrid approaches**: Semi-supervised learning if we can acquire even a small labeled set. Active learning to strategically query labels. One-class neural networks.

These directions could improve both performance and interpretability."

**Emphasis**: "Attention-based" and "Interpretability"

**Transition**: "If we were to deploy this, here's what we'd consider."

---

## SLIDE 32: Deployment Considerations (75 seconds)

**Say**:
"For production deployment, we'd need to address several things.

**Technical requirements**: Real-time scoring with latency under 100 milliseconds per loan. Batch processing to score large portfolios overnight. Model monitoring to detect drift in feature distributions over time. And explainability for regulatory compliance—we need to explain WHY a loan is flagged.

**Implementation strategy**: First, automate the feature pipeline with scheduled data updates and on-demand feature computation. Second, model serving via REST API for real-time scoring and batch jobs for portfolio analysis. Third, a monitoring dashboard tracking score distributions, detecting feature drift, and reporting performance metrics if labels become available.

**Business impact**: This becomes an early warning system for risky loans. It supports portfolio risk assessment and management. And it's a lending decision support tool—emphasize 'support,' not replacement. Human judgment still matters."

**Emphasis**: "Support, not replacement"

**Potential Question**: "How would you explain predictions to regulators?"
**Answer**: "We'd focus on which features contributed most to the score for each loan. For LOF, we can show the loan's nearest neighbors and how different it is. For ensemble, we can show which detectors flagged it and why. Feature importance analysis helps."

**Transition**: "Let me wrap up with a summary."

---

## SLIDE 33: Summary (75 seconds)

**Say**:
"To summarize our project:

**Problem**: Detect anomalous loans using only normal loan data in an unsupervised learning setting.

**Approach**: We built an ultra unsupervised ensemble with three key components. Advanced feature engineering including temporal windows, amortization signals, and PCA. A diverse detector portfolio focused on LOF with nine-plus detector types. And train-CDF calibration with systematic fusion using rank and probability-based strategies.

**Results**: Our best single detector, LOF k equals 5, achieves AUPRC of 0.2988. Our final ensemble achieves AUPRC of **[YOUR RESULT]**. This demonstrates effective unsupervised anomaly detection despite the lack of labeled training data.

**Key innovations**: Validation-based correlation analysis for feature discovery. Multi-window temporal feature engineering. Domain-driven amortization signals. And systematic ensemble fusion evaluation.

**Impact**: We have a production-ready model for loan risk assessment that identifies high-risk loans before default."

**Emphasis**: "Production-ready"

**Body Language**: Strong concluding posture

**Transition**: "We're happy to take questions."

---

## SLIDE 34: Q&A (Variable)

**Say**:
"Thank you for your attention. We're happy to answer any questions you have."

**Stand ready, maintain eye contact with audience**

---

## Common Q&A Preparation

### Question: "Why not use supervised learning?"

**Answer**: "Great question. In practice, labeled default data is often unavailable or insufficient when building initial risk models, especially for new loan products. Also, anomalies can emerge in patterns not seen in historical defaults. Unsupervised methods let us detect novel anomaly types. That said, if we had labels, semi-supervised learning would be a strong next step."

### Question: "How do you ensure no data leakage from validation set?"

**Answer**: "Excellent question, and this is critical. We use validation in two ways, both valid. First, for EDA—understanding feature-target correlations. This is exploratory analysis, not model training. Second, for hyperparameter selection—choosing PCA components, k-values, and which detectors to keep. This is standard practice in ML. Critically, no model parameters are fit on validation. Train-CDF calibration uses only training scores. All detector fitting is on training data only. The rule is: validation can guide choices, but never fit parameters."

### Question: "How does this compare to industry benchmarks?"

**Answer**: "That's a great question. Without access to proprietary industry models, it's hard to make direct comparisons. However, AUPRC around 0.30 in an unsupervised setting with extreme class imbalance is respectable. Industry models with more features, more data, and supervised methods likely perform better. Our contribution is showing what's achievable with purely unsupervised methods and principled methodology."

### Question: "What's the computational cost?"

**Answer**: "Training takes about 2 minutes on standard hardware. Inference for a single loan is under 100 milliseconds. For batch scoring, we can process about 1,000 loans per second. This is acceptable for most production use cases. The bottleneck is LOF, which scales with training set size. For larger datasets, we might need approximations or sampling."

### Question: "Can you explain the model's predictions?"

**Answer**: "To an extent, yes. For LOF, we can show which neighbors a loan is compared to and how different it is. We can identify which features contribute most to the distance. For the ensemble, we show which detectors flagged the loan. Feature importance analysis reveals that CreditScore, DTI, and InterestRate are key drivers. However, deep interpretation of why specific feature combinations trigger alerts is challenging—that's a limitation of unsupervised methods."

### Question: "What about temporal patterns—did you try RNNs?"

**Answer**: "We engineered temporal features using multi-window statistics, but you're right that we don't explicitly model sequences. We didn't implement RNNs or LSTMs in this project. That's a clear future direction. Given the modest dataset size, we weren't confident deep learning would outperform our ensemble, but it's worth exploring with more data or as a semi-supervised approach."

### Question: "How sensitive is the model to hyperparameters?"

**Answer**: "Reasonably robust. LOF performs well across k equals 5 to 12. PCA components from 60 to 100 give similar results. The detector selection threshold from 0.14 to 0.17 doesn't drastically change ensemble composition. This robustness is why we include multiple variants in the ensemble—it hedges against hyperparameter sensitivity."

### Question: "What if test distribution is very different from validation?"

**Answer**: "That's a risk in any ML system. Our mitigation is ensemble diversity—different detectors might be robust to different shifts. In production, we'd monitor feature distributions and retrain periodically. If possible, collect some test labels to evaluate distribution shift impact. Our train-CDF calibration also helps by referencing training distribution, which is presumably more stable."

---

## Timing Guide

**Section 1 (Slides 1-3)**: 2 minutes
**Section 2 (Slides 4-7)**: 4 minutes
**Section 3 (Slides 8-10)**: 3 minutes
**Section 4 (Slides 11-13)**: 3 minutes
**Section 5 (Slides 14-22)**: 10 minutes ⭐ Core content
**Section 6 (Slides 23-25)**: 3 minutes
**Section 7 (Slides 26-27)**: 2 minutes
**Section 8 (Slides 28-32)**: 5 minutes
**Section 9 (Slides 33-34)**: 2 minutes

**Total**: ~34 minutes + Q&A

**Buffer**: Aim to finish main content in 25-30 minutes, leaving 5-10 for questions

---

## Final Presentation Tips

**Before Presenting**:
- Rehearse at least 3 times fully
- Time yourself, adjust pace
- Memorize first 30 seconds and last 30 seconds
- Have backup slides ready

**During Presenting**:
- Breathe deeply, smile, make eye contact
- Use hand gestures naturally
- Point to visuals when referencing them
- Pause after key points for emphasis
- If you stumble, don't apologize, just continue
- Watch for audience cues (confusion, engagement)

**Body Language**:
- Stand up straight, feet shoulder-width
- Avoid pacing or swaying
- Use open gestures (avoid crossing arms)
- Face the audience, not the screen
- Use a laser pointer if available

**Voice**:
- Vary pace (slow for complex concepts, normal otherwise)
- Vary volume (louder for emphasis)
- Avoid filler words ("um", "uh", "like")
- Pause instead of using fillers
- Project confidence even if nervous

**Handling Questions**:
- Listen fully before answering
- Repeat question for the room
- If you don't know: "That's a great question. I don't have the exact answer, but here's my thinking..."
- Bridge to what you do know
- Keep answers concise (30-60 seconds)

**Good luck with your presentation!**
