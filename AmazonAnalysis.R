library(vroom)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

library(tidymodels)
library(embed)
library(ggplot2)

# 1. Exploratory Plots
ggplot(trainData, aes(x = RESOURCE)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of RESOURCE", x = "RESOURCE", y = "Count")

ggplot(trainData, aes(x = factor(ACTION))) +
  geom_bar(fill = "orange") +
  labs(title = "Count of ACTION (0 vs 1)", x = "ACTION", y = "Count")

# 2. Create Recipe
target_var <- "ACTION"  
# Numeric features to convert to factors
vars_I_want_to_mutate <- c("RESOURCE", "MGR_ID", "ROLE_ROLLUP_1")
# Categorical variables to combine rare categories
vars_I_want_other_cat_in <- c("ROLE_TITLE", "ROLE_FAMILY_DESC")
# Categorical variables for dummy variable encoding
vars_I_want_to_dummy <- c("ROLE_DEPTNAME", "ROLE_FAMILY")
# Categorical variable for target encoding
vars_I_want_to_target_encode <- c("ROLE_ROLLUP_2")

# Recipe definition
my_recipe <- recipe(as.formula(paste(target_var, "~ .")), data = trainData) %>%
  step_mutate_at(all_of(vars_I_want_to_mutate), fn = factor) %>%  # Convert numeric features to factors
  step_mutate_at(all_of(vars_I_want_other_cat_in), fn = factor) %>%  # Convert other categorical features to factors
  step_mutate_at(all_of(vars_I_want_to_dummy), fn = factor) %>%  # Convert dummy variable candidates to factors
  step_mutate_at(all_of(vars_I_want_to_target_encode), fn = factor) %>%  # Convert target encoding variable to factor
  step_other(all_of(vars_I_want_other_cat_in), threshold = 0.001) %>%  # Combine rare categories
  step_dummy(all_of(vars_I_want_to_dummy)) %>%  # Dummy variable encoding
  step_lencode_mixed(all_of(vars_I_want_to_target_encode), outcome = vars(target_var))  # Target encoding

# 3. Preprocess the Data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)

glimpse(baked)  
