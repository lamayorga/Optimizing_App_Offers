# Optimizing App Offers with Starbucks
Analyzing Starbucks app data to reveal customer behavior trends and build a machine learning model to predict offer success rates.

## Project Overview

The first part of this project focuses on analyzing transaction, demographic, and offer data to reveal customer behavior trends and earnings. These insights are then used to build a machine learning model that predicts offer success rate.

The [Starbucks](https://www.starbucks.com/) datasets provided by [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_n&utm_campaign=2045115106_c&utm_term=77922608608&utm_keyword=udacity%20data%20science%20nanodegree_e&gclid=EAIaIQobChMI_cjDz7q06wIVw9SzCh20og-pEAAYAiAAEgKG3vD_BwE), mimic customer behavior on the [Starbucks rewards mobile app](https://www.starbucks.com/coffeehouse/mobile-apps). Similar datasets can be found [here](https://www.kaggle.com/blacktile/starbucks-app-customer-reward-program-data) on Kaggle. Starbucks sends out an offer to users every few days through the mobile app. These offers can be an advertisement, discount, or a buy one get one free (BOGO) promotion. Not all users recieve the same promotions and offer frequency varies per user. In this analysis, transaction, demographic, and offer data is analyzed to determine which demographic groups respond best to each offer type. The datasets used in this project are a simplified version of the real Starbucks app. The simulator only contains one product whereas Starbucks sells dozens of products.

## Datasets
The data is contained in these three files (can be found in repo):
* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

## Software
* Python and libraries:
  * JSON
  * math
  * Matplotlib
  * NumPy
  * pandas
  * Progressbar
  * re
  * scikit-learn
  * SciPy
  * Seaborn
* Jupyter Notebooks

## Summary of Findings

A summary of findings and visuals can be found on [this blog post](https://towardsdatascience.com/optimizing-starbucks-app-offers-e9f10689972).

### Results
  * **Starbucks user demographics:**
    * 57% of users are males, 41% are females, and 1.4% are other genders. While the majority of users are males, females spend more with average expenses totalling to 140.91 USD. Total spending average for other genders is $124.32 and $99.59 and for males.   
    * Majority of users in this dataset are in the 50-70 year age group. As age increases, users spend more.
    * The $55,000-78,000 salary range is most common amongst users in the dataset. Customers with greater income spend more.

  * **Starbucks offer performance:**
    * The BOGO promotions returned the greatest income of $138.43 in average earnings per customer. The discount promotions returned an average of $135.23 and informational offers averaged at $113.74 per customer.
    * The training data suggests that the top five features based on their importance are:
      1. Duration (how long offer runs before expiring)
      2. Reward (stars earned by customer when offer is redeemed)
      3. Difficulty (minimum spending required to redeem offer)
      4. Income
      5. Social (advertisements on social media)

  * **Customer behavior based on demographics:**
    * Females spent the most per transaction in all offer categories
    * Average transaction value increased with increasing age and declined past 75 years in all promotional categories
    * Customers with greater income spent more per transaction in all three offer types

  * **Machine learning model:**
    * Of the models built (logistic regression, random forest, and gradient boosting), random forest classification was selected as the best based on performance metrics, high quality output, and the ability to quickly train this model. Random forest can handle a large amount of training data efficiently and inherently so it is well suited for future use with multi-class problems from Starbucks' [13 million](https://www.forbes.com/sites/bernardmarr/2018/05/28/starbucks-using-big-data-analytics-and-artificial-intelligence-to-boost-performance/#266d9d8a65cd) active users. The random forest classifier model was then refined using hyperparameters from grid search. The resulting model has a training data accuracy of 0.824 and an f1-score of 0.881. The test data accuracy of 0.695 and f1-score of 0.797 suggests that the random forest model constructed did not overfit the training data.

### Recommendations
  1. Starbucks should continue running discounts and BOGO promotions since they are redeemed by customers of all demographics, have a large percent success rate, and return high earnings. Informational offers fall in last place when looking at earnings, percent success, and feature importance. There is also no way to track whether informational offers are redeemed (can only see if user received or opened informational offer). Based on these results, I would suggest that Starbucks only continue BOGO and discount offers. Informational offers should instead be viewed as an advertisement since there are no measurable properties to track earnings that are comparable to BOGO and discount offers.
  2. To maximize average transaction earnings, Starbucks should target females, age groups of 40 and above, and users with salaries of 60k and higher.
  3. Starbucks should use the random forest classifier model built to predict whether or not a customer will respond to an offer. This will minimize the use of resources and expenses involved in launching an offer and increase the likelihood of maximum return.

### Continued Exploration
  * **Customer data:**
      * Customer spending is greater with increasing age and income. It might be worth exploring whether Starbucks should increase the frequency of promotions based on age and salary. Since females spend more than all other genders, it would also be interesting to explore the reasoning behind these higher numbers.
          * i.e. Do females buy higher priced items? Do they purchasing more frequently? Are they buying more items? Is there any implication that they're purchasing for a group of individuals?
  * **Random forest model:**
      * Since the top three features are associated with customer offers, the model could be improved by creating features that describe an offer's success rate as a function of offer duration, difficulty, and reward. The addition of these features should construct a better decision boundary of separating successful and unsuccessful customer offers, returning a model with more reliable performance metrics.

## Limitations
* The underlying simulator only has one product whereas Starbucks sells dozens
* Transactions can not be directly linked to redeemed offers
* Results are not indicative of user behavior at the individual level


## Credits
* [Accuracy, precision, recall, f1-score interpretation of performance metrics](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/_)
* [Forbes Starbucks using big data analytics and articial intelligence to boost performance](https://www.forbes.com/sites/bernardmarr/2018/05/28/starbucks-using-big-data-analytics-and-artificial-intelligence-to-boost-performance/#266d9d8a65cd)
* [Kaggle](https://www.kaggle.com/blacktile/starbucks-app-customer-reward-program-data)
* [Machine learning cheatsheet logistic_regression](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)
* [Medium gradient boosting vs random forest](https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80)
* [Python graph gallery add text annotations on scatterplots](https://python-graph-gallery.com/46-add-text-annotation-on-scatterplot/)
* [Seaborn distribution tutorial](https://seaborn.pydata.org/tutorial/distributions.html)
* [Seaborn distribution plot options](https://seaborn.pydata.org/examples/distplot_options.html)
* [Seaborn color palettes](https://python-graph-gallery.com/100-calling-a-color-with-seaborn/)
* [Stackoverflow change default colors for multiple plots in matplotlib](https://stackoverflow.com/questions/46768859/how-to-change-the-default-colors-for-multiple-plots-in-matplotlib)
* [Stackoverflow pandas plot value counts barplot descending](https://stackoverflow.com/questions/49059956/pandas-plot-value-counts-barplot-in-descending-manner)
* [Stackoverflow pandas bar plot with two bars and y axis](https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis)
* [Stackoverflow modify tick label text](https://stackoverflow.com/questions/11244514/modify-tick-label-text)
* [Stackoverflow make single legend for subplots](https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib)
* [Stackoverflow position legend](https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot)
* [Stackoverflow combine line and histogram](https://stackoverflow.com/questions/48749972/combine-line-and-histogram-in-matplotlib?noredirect=1&lq=1)
* [Stackoverflow lambda using multiple variables within df](https://stackoverflow.com/questions/41433983/python-pandas-lambda-using-multiple-variables-lambda-within-dataframe)
* [Starbucks](https://www.starbucks.com/coffeehouse/mobile-apps)
* [Towardsdatascience combo charts with seaborn and python](https://towardsdatascience.com/combo-charts-with-seaborn-and-python-2bc911a08950)
* [Towardsdatascience popular machine learning metrics](https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce)
* [Udacity](https://www.udacity.com/course/data-scientist-nanodegree)
