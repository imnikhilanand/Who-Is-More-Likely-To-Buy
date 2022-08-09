# Who-Is-More-Likely-To-Buy
Uplift Modeling to identify the pursuable group of customers from all the users in order to send them encouragement (in terms of coupons or other offers) to buy the product more without spending resources to convert those users who are not willing or interested to buy the product even after encouragement.

## Uplift Modeling


## Dataset Description

This is a synthetic dataset created for research purpose. 
<br><br>
Synthetic Data Set for Uplift Modeling [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3653141

<table>

<tr><th>S No.</th><th>Feature Name</th><th>Feature Description</th></tr>
<tr><td>1</td><td>treatment_group_key</td><td>Experiment group label</td></tr>
<tr><td>2</td><td>conversion</td><td>Outcome variable</td></tr>
<tr><td>3</td><td>x1_informative</td><td>informative feature</td></tr>
<tr><td>4</td><td>x2_informative</td><td>informative feature</td></tr>
<tr><td>5</td><td>x3_informative</td><td>informative feature</td></tr>
<tr><td>6</td><td>x4_informative</td><td>informative feature</td></tr>
<tr><td>7</td><td>x5_informative</td><td>informative feature</td></tr>
<tr><td>8</td><td>x6_informative</td><td>informative feature</td></tr>
<tr><td>9</td><td>x7_informative</td><td>informative feature</td></tr>
<tr><td>10</td><td>x8_informative</td><td>informative feature</td></tr>
<tr><td>11</td><td>x9_informative</td><td>informative feature</td></tr>
<tr><td>12</td><td>x10_informative</td><td>informative feature</td></tr>

<tr><td>13</td><td>x11_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>14</td><td>x12_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>15</td><td>x13_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>16</td><td>x14_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>17</td><td>x15_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>18</td><td>x16_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>19</td><td>x17_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>20</td><td>x18_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>21</td><td>x19_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>22</td><td>x20_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>23</td><td>x21_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>24</td><td>x22_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>25</td><td>x23_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>26</td><td>x24_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>27</td><td>x25_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>28</td><td>x26_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>29</td><td>x27_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>30</td><td>x28_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>31</td><td>x29_irrelevant</td><td>irrelevant feature</td></tr>
<tr><td>32</td><td>x30_irrelevant</td><td>irrelevant feature</td></tr>

</table>

## Exploratory Data Analysis

Data points in each of the two groups (Control and Treatment)

<table>
<tr><th>Groups</th><th>Data points</th></tr>
<tr><td>Control</td><td>5000</td></tr>
<tr><td>Treatment</td><td>5000</td></tr>
</table>

Percentage of conversions in the two groups (Control and Treatment)

<table>
<tr><th>Groups</th><th>Conversion Rate</th></tr>
<tr><td>Control</td><td>0.2670</td></tr>
<tr><td>Treatment</td><td>0.3712</td></tr>
</table>
<img src='src/conversion_graph_2.png' width='50%'>

Let's obeserve the proportion Z-test results among the two groups.
 
<table>
<tr><th>Statistic</th><th>Values</th></tr>
<tr><td>Z-stat</td><td>-11.17</td></tr>
<tr><td>P-value</td><td>5.27e-29</td></tr>
</table>
 
**Observations:**

- The data points are randomized and equally distributed among control and treatment groups.
- The ATE (Average Treatment Effect) is positive and is approximately 10%. 
- From the proportion Z-test between the two groups, we observed that the difference in the conversion between the two groups is significant as the p-value is less than 0.05.

These observations clearly states that we can move ahead with the uplift modeling. We will be creating a machine learning model to classify users based on how likely they will be purchase the product. Once we build the model, we will use the model to estimate the difference between the conversion of indivisual to see who are likely to get converted under treatment conditions. 
 
## Modeling

### S-Learner XGBoost

A single model (S-Learner) was developed to predict the binary outcome (conversion). For this model XGBooost was used which resulted in an AUC score of 0.7554.

**Hyperparameters**

<table>
<tr><th>Hyperparmeters</th><th>Values</th></tr>
<tr><td>eta</td><td>0.1</td></tr>
<tr><td>max_depth</td><td>5</td></tr>
<tr><td>alpha</td><td>1</td></tr>
<tr><td>gamma</td><td>1</td></tr>
</table>

<br>

**Model Performance**
<img src='src/S_learner_xgb_auc_plot.png' width='80%'>























