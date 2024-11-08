#!/usr/bin/env python
# coding: utf-8

# # STA130 Homework 06
# 
# Please see the course [wiki-textbook](https://github.com/pointOfive/STA130_ChatGPT/wiki) for the list of topics covered in this homework assignment, and a list of topics that might appear during ChatBot conversations which are "out of scope" for the purposes of this homework assignment (and hence can be safely ignored if encountered)

# <details class="details-example">
#     <summary style="color:blue"><u>Introduction</u></summary>
# 
# A reasonable characterization of STA130 Homework is that it simply defines a weekly reading comprehension assignment. 
# Indeed, STA130 Homework essentially boils down to completing various understanding confirmation exercises oriented around coding and writing tasks.
# However, rather than reading a textbook, STA130 Homework is based on ChatBots so students can interactively follow up to clarify questions or confusion that they may still have regarding learning objective assignments.
# 
# > Communication is a fundamental skill underlying statistics and data science, so STA130 Homework based on ChatBots helps practice effective two-way communication as part of a "realistic" dialogue activity supporting underlying conceptual understanding building. 
# 
# It will likely become increasingly tempting to rely on ChatBots to "do the work for you". But when you find yourself frustrated with a ChatBots inability to give you the results you're looking for, this is a "hint" that you've become overreliant on the ChatBots. Your objective should not be to have ChatBots "do the work for you", but to use ChatBots to help you build your understanding so you can efficiently leverage ChatBots (and other resources) to help you work more efficiently.<br><br>
# 
# </details>
# 
# <details class="details-example">
#     <summary style="color:blue"><u>Instructions</u></summary>
# 
# 1. Code and write all your answers (for both the "Prelecture" and "Postlecture" HW) in a python notebook (in code and markdown cells) 
#     
#     > It is *suggested but not mandatory* that you complete the "Prelecture" HW prior to the Monday LEC since (a) all HW is due at the same time; but, (b) completing some of the HW early will mean better readiness for LEC and less of a "procrastentation cruch" towards the end of the week...
#     
# 2. Paste summaries of your ChatBot sessions (including link(s) to chat log histories if you're using ChatGPT) within your notebook
#     
#     > Create summaries of your ChatBot sessions by using concluding prompts such as "Please provide a summary of our exchanges here so I can submit them as a record of our interactions as part of a homework assignment" or, "Please provide me with the final working verson of the code that we created together"
#     
# 3. Save your python jupyter notebook in your own account and "repo" on [github.com](github.com) and submit a link to that notebook though Quercus for assignment marking<br><br>
# 
# </details>
# 
# <details class="details-example">
#     <summary style="color:blue"><u>Prompt Engineering?</u></summary>
# 
# The questions (as copy-pasted prompts) are designed to initialize appropriate ChatBot conversations which can be explored in the manner of an interactive and dynamic textbook; but, it is nonetheless **strongly recommendated** that your rephrase the questions in a way that you find natural to ensure a clear understanding of the question. Given sensible prompts the represent a question well, the two primary challenges observed to arise from ChatBots are 
# 
# 1. conversations going beyond the intended scope of the material addressed by the question; and, 
# 2. unrecoverable confusion as a result of sequential layers logial inquiry that cannot be resolved. 
# 
# In the case of the former (1), adding constraints specifying the limits of considerations of interest tends to be helpful; whereas, the latter (2) is often the result of initial prompting that leads to poor developments in navigating the material, which are likely just best resolve by a "hard reset" with a new initial approach to prompting.  Indeed, this is exactly the behavior [hardcoded into copilot](https://answers.microsoft.com/en-us/bing/forum/all/is-this-even-normal/0b6dcab3-7d6c-4373-8efe-d74158af3c00)...
# 
# </details>
# 
# ### Marking Rubric (which may award partial credit) 
# 
# - [0.1 points]: All relevant ChatBot summaries [including link(s) to chat log histories if you're using ChatGPT] are reported within the notebook
# - [0.2 points]: Evaluation of correctness and clarity in written communication for Question "3"
# - [0.2 points]: Evaluation of correctness and clarity in written communication for Question "4"
# - [0.3 points]: Evaluation of submitted work and conclusions for Question "9"
# - [0.2 points]: Evaluation of written communication of the "big picture" differences and correct evidence assessement for Question "11"
# 

# ### "Prelecture" versus "Postlecture" HW? 
# 
# #### *Your HW submission is due prior to the Nov08 TUT on Friday after you return from Reading Week; however, this homework assignment is longer since it covers material from both the Oct21 and Nov04 LEC (rather than a single LEC); so, we'll brake the assignment into "Week of Oct21" and "Week of Nov04" HW and it will be due prior to the Nov08 TUT*
# 

# ## "Week of Oct21" HW [*due prior to the Nov08 TUT*]
# 
# 1. Explain the theoretical **Simple Linear Regression** model in your own words by describing its components (of predictor and outcome variables, slope and intercept coefficients, and an error term) and how they combine to form a sample from **normal distribution**; then, (working with a ChatBot if needed) create `python` code explicitly demonstrating your explanation using `numpy` and `scipy.stats`<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
#     > Your answer can be quite concise and will likely just address the "mathematical" and "statistical" aspects of the process of a **Simple Linear Model** specification, perhaps giving an intuitive interpretation summary of the result as a whole<br><br>
#     > 
#     > 1. Your code could be based on values for `n`, `x`, `beta0`, `beta1`, and `sigma`; and, then create the `errors` and `y`<br><br>
#     > 
#     > 2. The predictors $x_i$ can be fixed arbitrarily to start the process (perhaps sampled using `stats.uniform`), and they are conceptually different from the creation of **error** (or **noise**) terms $\epsilon_i$ which are sampled from a **normal distribution** (with some aribtrarily *a priori* chosen **standard deviation** `scale` parameter $\sigma$) which are then combined with $x_i$ through the **Simple Linear Model** equation (based on aribtrarily *a priori* chosen **slope** and **intercept coefficients**) to produce the $y_i$ outcomes<br><br>
#     > 
#     > 3. It should be fairly easy to visualize the "a + bx" line defined by the **Simple Linear Model** equation, and some **simulated** data points around the line in a `plotly` figure using the help of a ChatBot
#     > 
#     > If you use a ChatBot (as expected for this problem), **don't forget to ask for summaries of your ChatBot session(s) and paste these into your homework notebook (including link(s) to chat log histories if you're using ChatGPT)**
#     >
#     > 
#     > **Question Scope Warning:** *Be careful when using a ChatBot to help you with creating an example dataset and coding up a visualization though, because it might suggest creating (and visualizing) a fitted model for to your data (rather than the theoretical model); but, this is not what this question is asking you to demonstrate*. This question is not asking about how to produce a fitted **Simple Linear Regression** model or explain how model **slope** and **intercept coefficients** are calculated (e.g., using "ordinary least squares" or analytical equations to estimate the **coefficients**  for an observed dataset) 
#     > ```python
#     > # There are two distinct ways to use `plotly` here
#     >
#     > import plotly.express as px
#     > px.scatter(df, x='x',  y='y', color='Data', 
#     >            trendline='ols', title='y vs. x')
#     >        
#     > import plotly.graph_objects as go
#     > fig = go.Figure()
#     > fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
#     > 
#     > # The latter is preferable since `trendline='ols'` in the former 
#     > # creates a fitted model for the data and adds it to the figure
#     > ```
#     
#     </details><br>
# 
# 2. Continuing the previous question... (working with a ChatBot if needed) use a dataset **simulated** from your theoretical **Simple Linear Regression** model to demonstrate how to create and visualize a fitted **Simple Linear Regression** model using `pandas` and `import statsmodels.formula.api as smf`<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#         
#     > 1. Combine the **simulated** `x` and `y` into a `pandas` data frame object named `df` with the column names "x" and "y"<br><br>
#     > 
#     > 2. Replace the inline question comments below with their answers (working with a ChatBot if needed)
#     >
#     > ```python
#     > import statsmodels.formula.api as smf  # what is this library for?
#     > import plotly.express as px  # this is a ploting library
#     >
#     > # what are the following two steps doing?
#     > model_data_specification = smf.ols("y~x", data=df) 
#     > fitted_model = model_data_specification.fit() 
#     >
#     > # what do each of the following provide?
#     > fitted_model.summary()  # simple explanation? 
#     > fitted_model.summary().tables[1]  # simple explanation?
#     > fitted_model.params  # simple explanation?
#     > fitted_model.params.values  # simple explanation?
#     > fitted_model.rsquared  # simple explanation?
#     >
#     > # what two things does this add onto the figure?
#     > df['Data'] = 'Data' # hack to add data to legend 
#     > fig = px.scatter(df, x='x',  y='y', color='Data', 
#     >                  trendline='ols', title='y vs. x')
#     >
#     > # This is essentially what above `trendline='ols'` does
#     > fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
#     >                 line=dict(color='blue'), name="trendline='ols'")
#     > 
#     > fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
#     > ```
#     >
#     > The plotting here uses the `plotly.express` form `fig.add_scatter(x=x, y=y)` rather than the `plotly.graph_objects` form `fig.add_trace(go.Scatter(x=x, y=y))`. The difference between these two was noted in the "Further Guidance" comments in the previous question; but, the preference for the former in this case is because `px` allows us to access `trendline='ols'` through `px.scatter(df, x='x',  y='y', trendline='ols')`
# 
#     </details><br>
# 
# 3. Continuing the previous questions... (working with a ChatBot if needed) add the line from Question 1 on the figure of Question 2 and explain the difference between the nature of the two lines in your own words<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
#     > This question is effectively asking you to explain what the combined code you produced for Questions 1 and 2 is trying to demonstrate overall. If you're working with a ChatBot (as expected), giving these two sets of code as context, and asking what the purpose of comparing these lines could be would be a way to get some help in formulating your answer
#     > 
#     > The graphical visualization aspect of this question could be accomplished by appending the following code to the code provided in Question 2.
#     > 
#     > ```python
#     > # what does this add onto the figure in constrast to `trendline='ols'`?
#     > x_range = np.array([df['x'].min(), df['x'].max()])
#     > # beta0 and beta1 are assumed to be defined
#     > y_line = beta0 + beta1 * x_range
#     > fig.add_scatter(x=x_range, y=y_line, mode='lines',
#     >                 name=str(beta0)+' + '+str(beta1)+' * x', 
#     >                 line=dict(dash='dot', color='orange'))
#     >
#     > fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
#     > ```
#     > 
#     > *The right way to interactively "see" the answer to this question is to repeatedly create different dataset **simulations** using your theoretical model and the corresponding fitted models, and repeatedly visualize the data and the two lines over and over... this would be as easy as rerunning a single cell containing your simulation and visualization code...*
#     
#     </details><br>
#  
# 4. Continuing the previous questions... (working with a ChatBot if needed) explain how `fitted_model.fittedvalues` are derived on the basis of `fitted_model.summary().tables[1]` (or more specifically  `fitted_model.params` or `fitted_model.params.values`)<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
#     > The previous questions used code to explore the distinction between theoretical (true) $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i [\text{where } \epsilon_i \sim \mathcal{N}(0, \sigma)]$ and fitted (estimated) $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ **Simple Linear Regression** models
#     >
#     > This question asks you to explicitly illustrate what the latter fitted **Simple Linear Regression** model $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ is (in contrast to the linear equation of the theoretical model)
# 
#     </details><br>
# 
# 5. Building on the previous questions... (working with a ChatBot if needed) explain concisely in your own words what line is chosen for the fitted model based on observed data using the "ordinary least squares" method (as is done by `trendline='ols'` and `smf.ols(...).fit()`) and why it requires "squares"<br><br>
#     
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
#     > This question addresses the use of **residuals** $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i$ (in contrast to the **error** terms $\epsilon_i$ of the theoretical model), and particularly, asks for an explanation based on the following visualization
#     >
#     > ```python 
#     > import scipy.stats as stats
#     > import pandas as pd
#     > import statsmodels.formula.api as smf
#     > import plotly.express as px
#     > 
#     > n,x_min,x_range,beta0,beta1,sigma = 20,5,5,2,3,5
#     > x = stats.uniform(x_min, x_range).rvs(size=n)
#     > errors = stats.norm(loc=0, scale=sigma).rvs(size=n)
#     > y = beta0 + beta1 * x + errors
#     > 
#     > df = pd.DataFrame({'x': x, 'y': y})
#     > model_data_specification = smf.ols("y~x", data=df) 
#     > fitted_model = model_data_specification.fit() 
#     > 
#     > df['Data'] = 'Data' # hack to add data to legend 
#     > fig = px.scatter(df, x='x',  y='y', color='Data', 
#     >                  trendline='ols', title='y vs. x')
#     > 
#     > # This is what `trendline='ols'` is
#     > fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
#     >                 line=dict(color='blue'), name="trendline='ols'")
#     > 
#     > x_range = np.array([df['x'].min(), df['x'].max()])
#     > y_line = beta0 + beta1 * x_range
#     > fig.add_scatter(x=x_range, y=y_line, mode='lines',
#     >                 name=str(beta0)+' + '+str(beta1)+' * x', 
#     >                 line=dict(dash='dot', color='orange'))
#     > 
#     > # Add vertical lines for residuals
#     > for i in range(len(df)):
#     >     fig.add_scatter(x=[df['x'][i], df['x'][i]],
#     >                     y=[fitted_model.fittedvalues[i], df['y'][i]],
#     >                     mode='lines',
#     >                     line=dict(color='red', dash='dash'),
#     >                     showlegend=False)
#     >     
#     > # Add horizontal line at y-bar
#     > fig.add_scatter(x=x_range, y=[df['y'].mean()]*2, mode='lines',
#     >                 line=dict(color='black', dash='dot'), name='y-bar')
#     > 
#     > fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS
#     > ```
#     >
#     > **Question Scope Warning**: we are not looking for any explanation realted to the mathematical equations for the line chosen for the **Simple Linear Regression** model by the "ordinary least squares" method, which happen to be
#     > 
#     > $$\hat \beta_1 = r_{xy}\frac{s_y}{s_x} \quad \text{ and } \quad  \hat\beta_0 = \bar {y}-\hat \beta_1\bar {x}$$
#     >
#     > where $r_{xy}$ is the **correlation** between $x$ and $y$ and $s_y$ and $s_x$ are their **sample standard deviations**
#     
#     </details><br>
#     
# 5. Building on the previous questions... confirm that the following explain what the two `np.corrcoef...` expressions capture, why the final expression can be interpreted as "the proportion of variation in (outcome) y explained by the model (fitted_model.fittedvalues)", and therefore why `fitted_model.rsquared` can be interpreted as a measure of the accuracy of the model<br><br>
# 
#     1. `fitted_model.rsquared`
#     2. `np.corrcoef(y,x)[0,1]**2`
#     3. `np.corrcoef(y,fitted_model.fittedvalues)[0,1]**2`
#     4. `1-((y-fitted_model.fittedvalues)**2).sum()/((y-y.mean())**2).sum()`<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
#     > **R-squared** is the "the proportion of variation in (outcome) $y$ explained by the model ($\hat y_i$)" and is defined as
#     >
#     > $R^2 = 1 - \frac{\sum_{i=1}^n(y_i-\hat y)^2}{\sum_{i=1}^n(y_i-\bar y)^2}$
#     >
#     > The visuzation provided in the previous problem can be used to consider $(y_i-\bar y)^2$ as the squared distance of the $y_i$ to their sample average $\bar y$ as opposed to the squared **residuals** $(y_i-\hat y)^2$ which is the squared distance of the $y_i$ to their fitted (predicted) values $y_i$.
#     </details><br>
#     
# 7. Indicate a couple of the assumptions of the **Simple Linear Regression** model specification do not seem compatible with the example data below
#  

# In[3]:


1.
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(0)

# Step 1: Randomly generate parameters
n = np.random.randint(50, 200)  # Sample size between 50 and 200
beta0 = np.random.uniform(-10, 10)  # Intercept between -10 and 10
beta1 = np.random.uniform(-5, 5)    # Slope between -5 and 5
sigma = np.random.uniform(1, 5)     # Standard deviation of error between 1 and 5

# Step 2: Generate explanatory variable X from a uniform distribution
x = stats.uniform.rvs(0, 10, size=n)

# Step 3: Generate error term and response variable y
e = np.random.normal(0, sigma, size=n)
y = beta0 + beta1 * x + e

# Add intercept term for statsmodels
X_with_const = sm.add_constant(x)
model = sm.OLS(y, X_with_const).fit()

# Create DataFrame for Plotly
df = pd.DataFrame({'x': x, 'y': y})

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))


# the outcome 
# 𝑦
#  is influenced by a predictor 
# 𝑋
# , modified by coefficients 
# β 
# 0
# ​
#   and 
# 
# β 
# 1
# ​
#  , and subject to random noise 
# 𝑒
# , creating a realistic sample that approximates observations drawn from a normal distribution centered on the linear relationship.

# In[4]:


2.
import statsmodels.formula.api as smf  # what is this library for?
import plotly.express as px  # this is a ploting library

# what are the following two steps doing?
model_data_specification = smf.ols("y~x", data=df) 
fitted_model = model_data_specification.fit() 

# what do each of the following provide?
fitted_model.summary()  # simple explanation? 


# In[5]:


fitted_model.summary().tables[1]  # simple explanation?


# In[6]:


fitted_model.params  # simple explanation?


# In[7]:


fitted_model.params.values  # simple explanation?


# In[8]:


fitted_model.rsquared  # simple explanation?

# what two things does this add onto the figure?
df['Data'] = 'Data' # hack to add data to legend 
fig = px.scatter(df, x='x',  y='y', color='Data', 
                 trendline='ols', title='y vs. x')

# This is essentially what above `trendline='ols'` does
fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
                line=dict(color='blue'), name="trendline='ols'")

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[9]:


3.
fig = px.scatter(df, x='x',  y='y', color='Data', 
                 trendline='ols', title='y vs. x')
# what does this add onto the figure in constrast to `trendline='ols'`?
x_range = np.array([df['x'].min(), df['x'].max()])
# beta0 and beta1 are assumed to be defined
y_line = beta0 + beta1 * x_range
fig.add_scatter(x=x_range, y=y_line, mode='lines',
                name=str(beta0)+' + '+str(beta1)+' * x', 
                line=dict(dash='dot', color='orange'))

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[10]:


4.
fitted_model.summary().tables[1]


# In[11]:


5.
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf
import plotly.express as px

n,x_min,x_range,beta0,beta1,sigma = 20,5,5,2,3,5
x = stats.uniform(x_min, x_range).rvs(size=n)
errors = stats.norm(loc=0, scale=sigma).rvs(size=n)
y = beta0 + beta1 * x + errors

df = pd.DataFrame({'x': x, 'y': y})
model_data_specification = smf.ols("y~x", data=df) 
fitted_model = model_data_specification.fit() 

df['Data'] = 'Data' # hack to add data to legend 
fig = px.scatter(df, x='x',  y='y', color='Data', 
                 trendline='ols', title='y vs. x')

# This is what `trendline='ols'` is
fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
                line=dict(color='blue'), name="trendline='ols'")

x_range = np.array([df['x'].min(), df['x'].max()])
y_line = beta0 + beta1 * x_range
fig.add_scatter(x=x_range, y=y_line, mode='lines',
                name=str(beta0)+' + '+str(beta1)+' * x', 
                line=dict(dash='dot', color='orange'))

# Add vertical lines for residuals
for i in range(len(df)):
    fig.add_scatter(x=[df['x'][i], df['x'][i]],
                    y=[fitted_model.fittedvalues[i], df['y'][i]],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False)
    
# Add horizontal line at y-bar
fig.add_scatter(x=x_range, y=[df['y'].mean()]*2, mode='lines',
                line=dict(color='black', dash='dot'), name='y-bar')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# #1.
# Define the Linear Model: Assume a linear relationship 
# 𝑦
# =
# 𝛽
# 0
# +
# 𝛽
# 1
# 𝑥
# +
# 𝑒
# y=β 
# 0
# ​
#  +β 
# 1
# ​
#  x+e, where 
# 𝛽
# 0
# β 
# 0
# ​
#   is the intercept, 
# 𝛽
# 1
# β 
# 1
# ​
#   is the slope, and 
# 𝑒
# e is the error term.
# 

# #2.Calculate Residuals: For each data point, the residual is the difference between the observed value 
# 𝑦
# y and the predicted value 
# 𝑦
# ^
# =
# 𝛽
# 0
# +
# 𝛽
# 1
# 𝑥
# y
# ^
# ​
#  =β 
# 0
# ​
#  +β 
# 1
# ​
#  x.
# 

# #3.
# Minimize the Sum of Squared Residuals (SSR): The least squares method seeks values for 
# 
# β 
# 0
# ​ and 
# 
# β 
# 1
# ​
#   that minimize 
# 𝑆
# 𝑆
# 𝑅
# =
# ∑
# (
# 𝑦
# 𝑖
# −
# (
# 𝛽
# 0
# +
# 𝛽
# 1
# 𝑥
# 𝑖
# )
# )
# 2
# SSR=∑(y 
# i
# ​
#  −(β 
# 0
# ​
#  +β 
# 1
# ​
#  x 
# i
# ​
#  )) 
# 2
#  .
# 

# #4.Solve for 
# 𝛽
# 0
# β 
# 0
# ​
#   and 
# 𝛽
# 1
# β 
# 1
# ​
#  : Using formulas derived from minimizing the SSR, we calculate the slope 
# 𝛽
# 1
# β 
# 1
# ​
#   and intercept 
# 𝛽
# 0
# β 
# 0
# ​
#   based on the data’s 
# 𝑥
# x and 
# 𝑦
# y values.
# 
# 

# In[12]:


6.
fitted_model.rsquared


# In[13]:


np.corrcoef(y,x)


# In[14]:


np.corrcoef(y,x)[0,1]**2


# In[15]:


np.corrcoef(y,fitted_model.fittedvalues)[0,1]**2


# In[16]:


1-((y-fitted_model.fittedvalues)**2).sum()/((y-y.mean())**2).sum()


# In[17]:


7.
import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots

# This data shows the relationship between the amount of fertilizer used and crop yield
data = {'Amount of Fertilizer (kg) (x)': [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 
                                          2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 
                                          4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 
                                          6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8, 
                                          8.2, 8.4, 8.6, 8.8,9, 9.2, 9.4, 9.6],
        'Crop Yield (tons) (y)': [18.7, 16.9, 16.1, 13.4, 48.4, 51.9, 31.8, 51.3, 
                                  63.9, 50.6, 58.7, 82.4, 66.7, 81.2, 96.5, 112.2, 
                                  132.5, 119.8, 127.7, 136.3, 148.5, 169.4, 177.9, 
                                  186.7, 198.1, 215.7, 230.7, 250.4, 258. , 267.8, 
                                  320.4, 302. , 307.2, 331.5, 375.3, 403.4, 393.5,
                                  434.9, 431.9, 451.1, 491.2, 546.8, 546.4, 558.9]}
df = pd.DataFrame(data)
fig1 = px.scatter(df, x='Amount of Fertilizer (kg) (x)', y='Crop Yield (tons) (y)',
                  trendline='ols', title='Crop Yield vs. Amount of Fertilizer')

# Perform linear regression using scipy.stats
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(df['Amount of Fertilizer (kg) (x)'], df['Crop Yield (tons) (y)'])
# Predict the values and calculate residuals
y_hat = intercept + slope * df['Amount of Fertilizer (kg) (x)']
residuals = df['Crop Yield (tons) (y)'] - y_hat
df['Residuals'] = residuals
fig2 = px.histogram(df, x='Residuals', nbins=10, title='Histogram of Residuals',
                    labels={'Residuals': 'Residuals'})

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Crop Yield vs. Amount of Fertilizer', 
                                    'Histogram of Residuals'))
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
fig.update_layout(title='Scatter Plot and Histogram of Residuals',
    xaxis_title='Amount of Fertilizer (kg)', yaxis_title='Crop Yield (tons)',
    xaxis2_title='Residuals', yaxis2_title='Frequency', showlegend=False)

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# Linearity: Simple linear regression assumes a straight-line relationship between 
# 𝑋
# X (Fertilizer) and 
# 𝑌
# Y (Crop Yield). However, the data suggests a curved relationship, as yield seems to increase more quickly with higher fertilizer levels, not in a straight line.

# Constant Variance of Errors: The model assumes that errors (the differences between observed and predicted values) have a similar spread across all values of 
# 𝑋
# X. In this data, the errors appear to get larger with more fertilizer, meaning the spread isn’t consistent.

# https://chatgpt.com/share/67296003-81e0-8006-ba6a-537f23291bd9

# In summary, the assumptions of linearity and constant variance in simple linear regression may not be met in this example. The data shows a curved relationship between fertilizer and crop yield, suggesting that a straight line is not the best fit. Additionally, the spread of errors seems to increase with more fertilizer, violating the assumption of constant error variance. Therefore, a nonlinear model may better capture the relationship.

# ## "Week of Nov04" HW [due prior to the Nov08 TUT]
# 
# #### In place of the "Project" format we introduced for the previous weeks HW, the remaining questions will be a collection of exercises based around the following data
# 
# > The details of the LOWESS Trendline shown below are not a part of the intended scope of the activities here, but it is included since it is suggestive of the questions we will consider and address here
# 

# In[18]:


import plotly.express as px
import seaborn as sns
import statsmodels.api as sm

# The "Classic" Old Faithful Geyser dataset: ask a ChatBot for more details if desired
old_faithful = sns.load_dataset('geyser')

# Create a scatter plot with a Simple Linear Regression trendline
fig = px.scatter(old_faithful, x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions", 
                 trendline='ols')#'lowess'

# Add a smoothed LOWESS Trendline to the scatter plot
lowess = sm.nonparametric.lowess  # Adjust 'frac' to change "smoothness bandwidth"
smoothed = lowess(old_faithful['duration'], old_faithful['waiting'], frac=0.25)  
smoothed_df = pd.DataFrame(smoothed, columns=['waiting', 'smoothed_duration'])
fig.add_scatter(x=smoothed_df['waiting'], y=smoothed_df['smoothed_duration'], 
                mode='lines', name='LOWESS Trendline')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# 8. Specify a **null hypothesis** of "no linear association (on average)" in terms of the relevant **parameter** of the **Simple Linear Regression** model, and use the code below to characterize the evidence in the data relative to the **null hypothesis** and interpret your subsequent beliefs regarding the Old Faithful Geyser dataset
# 
#    > ```python
#    > import seaborn as sns
#    > import statsmodels.formula.api as smf
#    >
#    > # The "Classic" Old Faithful Geyser dataset
#    > old_faithful = sns.load_dataset('geyser')
#    > 
#    > linear_for_specification = 'duration ~ waiting'
#    > model = smf.ols(linear_for_specification, data=old_faithful)
#    > fitted_model = model.fit()
#    > fitted_model.summary()
#    > ```

# $H_0:\beta_1=0$

# $H_a:\beta_1\neq=0$

# In[19]:


import seaborn as sns
import statsmodels.formula.api as smf

# The "Classic" Old Faithful Geyser dataset
old_faithful = sns.load_dataset('geyser')

linear_for_specification = 'duration ~ waiting'
model = smf.ols(linear_for_specification, data=old_faithful)
fitted_model = model.fit()
fitted_model.summary()


# 9. As seen in the introductory figure above, if the delay of the geyser eruption since the previous geyser eruption exceeds approximately 63 minutes, there is a notable increase in the duration of the geyser eruption itself. In the figure below we therefore restrict the dataset to only short wait times. Within the context of only short wait times, is there evidence in the data for a relationship between duration and wait time in the same manner as in the full data set? Using the following code, characterize the evidence against the **null hypothesis** in the context of short wait times which are less than  `short_wait_limit` values of `62`, `64`, `66`.<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
#     Remember that **Hypothesis Ttesting** is not a "mathematical proof"
# 
#     1. We do not prove $H_0$ false, we instead give evidence against the $H_0$: "We reject the null hypothesis with a p-value of XYZ, meaning we have ABC evidence against the null hypothesis"
#     2. We do not prove $H_0$ is true, we instead do not have evidence to reject $H_0$: "We fail to reject the null hypothesis with a p-value of XYZ"<br><br>
# 
#     |p-value|Evidence|
#     |-|-|
#     |$$p > 0.1$$|No evidence against the null hypothesis|
#     |$$0.1 \ge p > 0.05$$|Weak evidence against the null hypothesis|
#     |$$0.05 \ge p > 0.01$$|Moderate evidence against the null hypothesis|
#     |$$0.01 \ge p > 0.001$$|Strong evidence against the null hypothesis|
#     |$$0.001 \ge p$$|Very strong evidence against the null hypothesis|
# 
#     </details>    

# In[20]:


import plotly.express as px
px.scatter(old_faithful, x='waiting',  y='duration',  
           trendline='ols', title='y vs. x')


# In[21]:


import plotly.express as px

short_wait_limit = 62 # 64 # 66 #
short_wait = old_faithful.waiting < short_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[short_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# A p-value of 0.238 suggests there is not enough evidence to reject the null hypothesis at typical significance levels (like 0.05 or 0.01).
# This implies that, when considering only the data with waiting times less than 62, the linear relationship between waiting and duration is not statistically significant. In other words, we cannot confidently say that there is a linear association in this subset.

# In[24]:


import plotly.express as px

short_wait_limit = 64 # 66 #
short_wait = old_faithful.waiting < short_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[short_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[23]:


import plotly.express as px

short_wait_limit = 66 #
short_wait = old_faithful.waiting < short_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[short_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# A p-value of 0 indicates strong evidence against the null hypothesis. This is often interpreted as there being a significant linear relationship between waiting and duration.
# By increasing the short_wait_limit to 64, the analysis includes additional data points, which may exhibit a stronger relationship with duration and thus increase the evidence of a linear association.

# 10. Let's now consider just the (`n=160`) long wait times (as specified in the code below), and write code to<br><br> 
# 
#     1. create fitted **Simple Linear Regression** models for **boostrap samples** and collect and visualize the **bootstrapped sampling distribution** of the **fitted slope coefficients** of the fitted models;<br><br> 
#     
#     2. **simulate** samples (of size `n=160`) from a **Simple Linear Regression** model that uses $\beta_0 = 1.65$, $\beta_1 = 0$, $\sigma = 0.37$ along with the values of `waiting` for $x$ to create **simuations** of $y$ and use these collect and visualize the **sampling distribution** of the **fitted slope coefficients** under a **null hypothesis** assumption of "no linear association (on average)"; then,<br><br>
#     
#     3. report if $0$ contained within a 95\% **bootstrapped confidence interval**; and if the **simulated p-value** matches `smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1]`?<br><br>
# 
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
# 
#     > You'll need to create `for` loops to repeatedly create fitted **Simple Linear Regression** models using different samples, collecting the **fitted slope coeffient** created in each `for` loop "step" in order to visualize the **simulated sampling distributions**<br><br>
#     > 
#     > 1. A **bootstrapped sample** of the "long wait times" dataset can be created with `old_faithful[long_wait].sample(n=long_wait.sum(), replace=True)`<br><br>
#     >
#     > 2. A **simulated** version of the "long wait times" dataset can be created by first creating `old_faithful_simulation = old_faithful[long_wait].copy()` and then assigning the **simulated** it values with `old_faithful_simulation['duration'] = 1.65 + 0*old_faithful_simulation.waiting + stats.norm(loc=0, scale=0.235).rvs(size=long_wait.sum())` 
#     >
#     >  The values $\beta_0 = 1.65$ and $\sigma = 0.37$ are chosen to match what is actually observed in the data, while $\beta_1 = 0$ is chosen to reflect a **null hypothesis** assumption of "no linear assocaition (on average)"; and, make sure that you understand why it is that<br><br>
#     >
#     >
#     > 1. if `bootstrapped_slope_coefficients` is the `np.array` of your **bootstrapped slope coefficients** then `np.quantile(bootstrapped_slope_coefficients, [0.025, 0.975])` is a 95\% **bootstrapped confidence interval**<br><br>
#     > 
#     > 2. if `simulated_slope_coefficients` is the `np.array` of your **fitted slope coefficients** **simulated** under a **null hypothesis** "no linear association (on average)" then `(np.abs(simulated_slope_coefficients) >= smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().params[1]).mean()` is the **p-value** for the **simulated** **simulated sampling distribution of the slope coeficients** under a **null hypothesis** "no linear association (on average)"
#     
#     </details>
#     

# In[39]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import statsmodels.formula.api as smf
from scipy import stats

# Load the Old Faithful dataset
old_faithful = sns.load_dataset('geyser')

# Filter the dataset for long wait times
long_wait_limit = 71
long_wait = old_faithful.waiting > long_wait_limit
long_wait_data = old_faithful[long_wait]

# Define parameters for the null hypothesis simulation
beta_0 = 1.65
beta_1 = 0
sigma = 0.37
n_bootstrap_samples = 1000



# In[40]:


# A. Bootstrapping to collect sampling distribution of slope coefficients
bootstrapped_slope_coefficients = []

for _ in range(n_bootstrap_samples):
    # Create a bootstrap sample with replacement
    bootstrap_sample = long_wait_data.sample(n=long_wait.sum(), replace=True)
    
    # Fit a simple linear regression model on the bootstrap sample
    model = smf.ols('duration ~ waiting', data=bootstrap_sample).fit()
    
    # Collect the slope coefficient
    bootstrapped_slope_coefficients.append(model.params[1])

# Convert to numpy array for easier manipulation
bootstrapped_slope_coefficients = np.array(bootstrapped_slope_coefficients)

# Visualize the bootstrapped sampling distribution of slope coefficients
fig = px.histogram(bootstrapped_slope_coefficients, nbins=30, title="Bootstrapped Sampling Distribution of Slope Coefficients")
fig.show()


# In[41]:


# B. Simulate samples under the null hypothesis (beta_1 = 0) and collect the slope coefficients
simulated_slope_coefficients = []

for _ in range(n_bootstrap_samples):
    # Create a simulated dataset under the null hypothesis
    old_faithful_simulation = old_faithful[long_wait].copy()
    old_faithful_simulation['duration'] = beta_0 + beta_1 * old_faithful_simulation.waiting + stats.norm(loc=0, scale=sigma).rvs(size=long_wait.sum())
    
    # Fit a simple linear regression model on the simulated dataset
    model = smf.ols('duration ~ waiting', data=old_faithful_simulation).fit()
    
    # Collect the slope coefficient
    simulated_slope_coefficients.append(model.params[1])

# Convert to numpy array for easier manipulation
simulated_slope_coefficients = np.array(simulated_slope_coefficients)

# Visualize the simulated sampling distribution of slope coefficients
fig = px.histogram(simulated_slope_coefficients, nbins=30, title="Simulated Sampling Distribution of Slope Coefficients (Under Null Hypothesis)")
fig.show()


# In[42]:


# C. Confidence interval and p-value calculations

# 95% Bootstrapped Confidence Interval for the slope
bootstrapped_confidence_interval = np.quantile(bootstrapped_slope_coefficients, [0.025, 0.975])
print("95% Bootstrapped Confidence Interval for Slope:", bootstrapped_confidence_interval)

# Check if 0 is within the bootstrapped confidence interval
contains_zero = 0 >= bootstrapped_confidence_interval[0] and 0 <= bootstrapped_confidence_interval[1]
print("Does the 95% Bootstrapped Confidence Interval contain 0?:", contains_zero)

# p-value for the simulated distribution under the null hypothesis
observed_slope = smf.ols('duration ~ waiting', data=long_wait_data).fit().params[1]
simulated_p_value = (np.abs(simulated_slope_coefficients) >= np.abs(observed_slope)).mean()
print("Simulated p-value under null hypothesis:", simulated_p_value)

# Display the summary of the actual model for comparison
print(smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1])


# In[35]:


import plotly.express as px

long_wait_limit = 71
long_wait = old_faithful.waiting > long_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[long_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# 11. Since we've considered wait times of `<64` "short" and wait times of `>71` "long", let's instead just divide the data and insead call wait times of `<68` "short" and otherwise just call them "long". Consider the **Simple Linear Regression** model specification using an **indicator variable** of the wait time length where we use $k_i$ (rather than $x_i$) (to refer to the "kind" or "katagory" or "kontrast") column (that you may have noticed was already a part) of the original dataset 
# 
#     $$\large Y_i = \beta_{\text{intercept}} + 1_{[\text{"long"}]}(\text{k_i})\beta_{\text{contrast}} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$
#     
#     and explain the "big picture" differences between this model specification and the previously considered model specifications and report the evidence against a **null hypothesis** of "no difference between groups "on average") for the new **indicator variable** based model
#     
#     - `smf.ols('duration ~ waiting', data=old_faithful)`
#     - `smf.ols('duration ~ waiting', data=old_faithful[short_wait])`
#     - `smf.ols('duration ~ waiting', data=old_faithful[long_wait])`
#     

# $$\large Y_i = \beta_{\text{intercept}} + 1_{[\text{"long"}]}(\text{k_i})\beta_{\text{contrast}} + \epsilon_i \quad \text{ where } \quad \epsilon_i \sim \mathcal N\left(0, \sigma\right)$$
#     

# $y_{short}=\beta_0$
# 
# $y_{long}=\beta_0+\beta_1$

# In[28]:


from IPython.display import display

display(smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().summary().tables[1])

fig = px.box(old_faithful, x='kind', y='duration', 
             title='duration ~ kind',
             category_orders={'kind': ['short', 'long']})
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[44]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import statsmodels.formula.api as smf
from IPython.display import display

# Load the Old Faithful dataset and define the "kind" column based on new criteria
old_faithful = sns.load_dataset('geyser')
old_faithful['kind'] = np.where(old_faithful['waiting'] < 68, 'short', 'long')

# Fit the linear regression model using the indicator variable for "kind"
model = smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit()

# Display the summary table for the fitted model
display(model.summary().tables[1])

# Create a box plot to visualize the differences in eruption duration between "short" and "long" wait times
fig = px.box(old_faithful, x='kind', y='duration', 
             title='Eruption Duration by Kind of Wait Time',
             category_orders={'kind': ['short', 'long']})
fig.show()


# ## Interpretation of Results

# P-Value for β contrast: The p-value for the coefficient associated with "kind" (if small, typically < 0.05) would indicate that there is a significant difference in average eruption durations between "short" and "long" wait times.

# Coefficient β contrast : The magnitude and sign of β contrast indicate the direction and size of the difference in duration. A positive value would suggest that "long" wait times are associated with longer eruption durations on average, while a negative value would suggest the opposite.

# By examining the coefficient for C(kind, Treatment(reference="short")), this model directly addresses whether the categorization of wait times significantly affects eruption duration, offering an alternative perspective to the continuous-variable models.

# https://chatgpt.com/share/672d7331-f40c-8006-96a6-1f371592cdec

# 12. As discussed in question 2 of the **Communication Activity #2** of the Oct25 TUT (addressing an **omitted** section of the TUT), the assumption in **Simple Linear Regression** that the **error** terms $\epsilon_i \sim \mathcal N\left(0, \sigma\right)$ is diagnostically assessed by evaluating distributional shape of the **residuals** $\text{e}_i = \hat \epsilon_i = y_i - \hat y_i$
# 
#   Which of the histograms suggests plausibility of the assumption that the distribution of **error** terms is normal for each of the models, and why don't the other three?

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats

model_residuals = {
    '<br>Model 1:<br>All Data using slope': smf.ols('duration ~ waiting', data=old_faithful).fit().resid,
    '<br>Model 2:<br>Short Wait Data': smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().resid,
    '<br>Model 3:<br>Long Wait Data': smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().resid,
    '<br>Model 4:<br>All Data using indicator': smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().resid
}

fig = make_subplots(rows=2, cols=2, subplot_titles=list(model_residuals.keys()))
for i, (title, resid) in enumerate(model_residuals.items()):

    if i == 1:  # Apply different bins only to the second histogram (index 1)
        bin_size = dict(start=-1.9, end=1.9, size=0.2)
    else:
        bin_size = dict(start=-1.95, end=1.95, size=0.3)

    fig.add_trace(go.Histogram(x=resid, name=title, xbins=bin_size, histnorm='probability density'), 
                  row=int(i/2)+1, col=(i%2)+1)
    fig.update_xaxes(title_text="n="+str(len(resid)), row=int(i/2)+1, col=(i%2)+1)    
    
    normal_range = np.arange(-3*resid.std(),3*resid.std(),0.01)
    fig.add_trace(go.Scatter(x=normal_range, mode='lines', opacity=0.5,
                             y=stats.norm(loc=0, scale=resid.std()).pdf(normal_range),
                             line=dict(color='black', dash='dot', width=2),
                             name='Normal Distribution<br>(99.7% of its area)'), 
                  row=int(i/2)+1, col=(i%2)+1)
    
fig.update_layout(title_text='Histograms of Residuals from Different Models')
fig.update_xaxes(range=[-2,2])
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# 13. Since the "short" and "long" wait times are not "before and after" measurements there are not natural pairs on which to base differences; so, we can't do a "one sample" (paired differences) **hypothesis test**; but, we can do a proper "two sample" hypothesis testing using a **permuation test**; and, we could create a 95% **bootstrap confidence interval** for the difference in means of the two populations; namely,<br><br> 
# 
#     1. test $H_0: \mu_{\text{short}}=\mu_{\text{long}} \quad \text{no difference in duration between short and long groups}$ by "shuffling" the labels<br><br>
#     
#     2. provide `np.quantile(bootstrapped_mean_differences, [0.025, 0.975])` by repeatedly bootstrapping each each group and collecting the difference between the sample means
#     
#     and once you've finished (a) explain how the sampling approaches work for the two simulations, then (b) compare and contrast these two methods with the **indicator variable** based model approach used in Question 10, explaining how they're similar and different<br><br>
#     
#     <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
#     
#     > You'll need to create `for` loops for repeated (shuffling simulation) **permutation** and (subgroup) **bootstrapping**, where<br><br> 
#     >
#     > 1. "shuffling" for **permutation testing** is done like this `old_faithful.assign(kind_shuffled=old_faithful['kind'].sample(n=len(old_faithful), replace=False).values)#.groupby('kind').size()`; then, the **mean difference statistic** is then calculated using `.groupby('kind_shuffled')['duration'].mean().iloc[::-1].diff().values[1]` (so the **observed statistic** is `old_faithful.groupby('kind')['duration'].mean().iloc[::-1].diff().values[1]`<br><br>
#     > 
#     > 2. "two sample" **bootstrapping** is done like this `old_faithful.groupby('kind').apply(lambda x: x.sample(n=len(x), replace=True)).reset_index(drop=True)#.groupby('kind').size()`; then, the **bootstrapped mean difference statistic** is then calculated using `.groupby('kind')['duration'].mean().iloc[::-1].diff().values[1]` (like the **observed statistic** except this is applied to the **bootstrapped** resampling of `old_faithful`)
#     
#     </details><br> 
#         
# 14. Have you reviewed the course [wiki-textbook](https://github.com/pointOfive/STA130_ChatGPT/wiki) and interacted with a ChatBot (or, if that wasn't sufficient, real people in the course piazza discussion board or TA office hours) to help you understand all the material in the tutorial and lecture that you didn't quite follow when you first saw it?<br><br>
#     
#     <details class="details-example">
#     <summary style="color:blue"><u>Further Guidance</u></summary>
#     <br><em>Just answering "Yes" or "No" or "Somewhat" or "Mostly" or whatever here is fine as this question isn't a part of the rubric; but, the midterm and final exams may ask questions that are based on the tutorial and lecture materials; and, your own skills will be limited by your familiarity with these materials (which will determine your ability to actually do actual things effectively with these skills... like the course project...)<br><br></em>
#     </details>    

# # Recommended Additional Useful Activities [Optional]
# 
# The "Ethical Profesionalism Considerations" and "Current Course Project Capability Level" sections below **are not a part of the required homework assignment**; rather, they are regular weekly guides covering (a) relevant considerations regarding professional and ethical conduct, and (b) the analysis steps for the STA130 course project that are feasible at the current stage of the course 
# 
# <br>
# <details class="details-example">
# <summary style="color:blue"><u>Ethical Professionalism Considerations</u></summary>
# 
# The TUT and HW both addressed some of the assumptions used in **Simple Linear Regression**. The **p-values** provided by `statsmodels` via `smf.ols(...).fit()` depend on these assumptions, so if they are not (at least approximately) correct, the **p-values** (and any subsequent claims regarding the "evidience against" the **null hypothesis**) are not reliable. In light of this consideration, describe how you could diagnostically check the first three assumptions (given below) when using analyses based on **Simple Linear regression** model. From an Ethical and Professional perspective, do you think doing diagnostic checks on the assumptions of a **Simple Linear regression** model is something you can and should do whenever you're doing this kind of analysis? 
#             
# > The first three assumptions associated with the **Simple Linear regression** model are that<br>
# > 
# > 1. the $\epsilon_i$ **errors** (sometimes referred to as the **noise**) are **normally distributed**
# > 2. the $\epsilon_i$ **errors** are **homoscedastic** (so their distributional variance $\sigma^2$ does not change as a function of $x_i$)
# > 3. the linear form is [at least reasonably approximately] "true" (in the sense that the above two remain [at least reasonably approximately] "true") so that then behavior of the $Y_i$ **outcomes** are represented/determined on average by the **linear equation**)<br>
# > 
# >    and there are additional assumptions; but, a deeper reflection on these is "beyond the scope" of STA130; nonetheless, they are that<br><br>
# > 4. the $x_i$ **predictor variable** is **measured without error**
# > 5. and the $\epsilon_i$ **errors** are **statistically independent** (so their values do not depend on each other)
# > 6. and the $\epsilon_i$ **errors** are **unbiased** relative to the **expected value** of **outcome** $E[Y_i|x_i]=\beta_0 + \beta_1x_i$ (which is equivalently stated by saying that the mean of the **error distribution** is $0$, or again equivalently, that the **expected value** of the **errors** $E[\epsilon_i] = 0$)
#     
# </details>
# 
# <details class="details-example">
#     <summary style="color:blue"><u>Current Course Project Capability Level</u></summary>
# 
# #### Remember to abide by the [data use agreement](https://static1.squarespace.com/static/60283c2e174c122f8ebe0f39/t/6239c284d610f76fed5a2e69/1647952517436/Data+Use+Agreement+for+the+Canadian+Social+Connection+Survey.pdf) at all times
# 
# At this point in the course you should be able to do a **Simple Linear Regression** analysis for data from the Canadian Social Connection Survey data
#     
# 1. Create and test a **null hypothesis** of no linear association "on average" for a couple of columns of interest in the Canadian Social Connection Survey data using **Simple Linear Regression**
# 
# 2. Use the **residuals** of a fitted **Simple Linear Regression** model to diagnostically assess some of the assumptions of the analysis
# 
# 3. Use an **indicator variable** based **Simple Linear Regression** model to compare two groups from the Canadian Social Connection Survey data
# 
# 4. Compare and contrast the results of an **indicator variable** based **Simple Linear Regression** model to analyses based on a **permutation test** and a **bootstrapped confidence interval**   
#     
# </details>    

# In[ ]:




