# Meilleur Data Scientist de France 2018

This is my code for the 2-hour competition "Meilleur Data Scientist de France 2018" (Best data scientist of France 2018).

https://www.meilleurdatascientistdefrance.com/

I've reached a multilogloss of 0.95629, rank 44 out of 233 ranked (and 350+ participants)

## Context

This is a competition to predict the time to sell on Label Emmaüs marketplace.

Label Emmaüs is a branch of Emmaüs one of the most famous French non-profit which is particularly known
for providing clothes to homeless people.

Label Emmaüs is their marketplace which is hiring among the most vulnerable people in France, profits are reinjected into Emmaüs.

## Difficulties

### 1. 2 hours is extremely short

I struggled one hour to get a baseline up.

Then I lost again 30 minutes because we had to output 3 probabilities
(selling between 0-10 days, 10-60 days and 60+ days) but XGBoost wasn't ordering the output columns
in a natural order.

### 2. The dataset

was quite dirty to limit the use of automl/datascience platforms
