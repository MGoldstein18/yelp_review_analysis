# import relevent modules
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd

# load data from JSON files
businesses = pd.read_json('yelp_business.json', lines=True)
reviews = pd.read_json('yelp_review.json', lines=True)
users = pd.read_json('yelp_user.json', lines=True)
checkins = pd.read_json('yelp_checkin.json', lines=True)
tips = pd.read_json('yelp_tip.json', lines=True)
photos = pd.read_json('yelp_photo.json', lines=True)

pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500

# Explore the data
print(businesses.head())

print(reviews.head())

print(users.head())

print(checkins.head())

print(tips.head())

print(photos.head())

# Number of businesses in data
print(businesses.business_id.nunique())

# merge the dataframes into one dataframe
df = pd.merge(businesses, reviews, how='left', on='business_id')
print(len(df))
df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')

# display columns
print(df.columns)

# remove non-numeric columns
features_to_remove = ['address', 'attributes', 'business_id', 'categories', 'city', 'hours',
                      'is_open', 'latitude', 'longitude', 'name', 'neighborhood', 'postal_code', 'state', 'time']
df.drop(features_to_remove, axis=1, inplace=True)


# Clean Data

# check for any NaN values
print(df.isna().any())

# replace the NaN values
df.fillna({
    'weekday_checkins': 0,
    'weekend_checkins': 0,
    'average_tip_length':  0,
    'number_tips': 0,
    'average_caption_length': 0,
    'number_pics': 0
}, inplace=True)

# Analysis

# check correlation in the dataframe
print(df.corr())

# create and display a number of scatter plots to show correlation between different variables and number of stars of a restaurant
plt.scatter(df.average_review_sentiment, df.stars, alpha=0.1)
plt.show()


plt.scatter(df.average_review_length, df.stars, alpha=0.1)
plt.show()

plt.scatter(df.average_review_age, df.stars, alpha=0.1)
plt.show()


plt.scatter(df.number_funny_votes, df.stars, alpha=0.1)
plt.show()

# Build a Linear Regression Model

features = df[['average_review_length', 'average_review_age']]
ratings = df['stars']


x_train, x_test, y_train, y_test = train_test_split(
    features, ratings, test_size=0.2, random_state=1)


model = LinearRegression()

model.fit(x_train, y_train)


model.score(x_train, y_train)
model.score(x_test, y_test)

y_predict = model.predict(x_test)
plt.scatter(y_test, y_predict, alpha=0.1)
plt.show()

# Create set of a number of different variables to to check which model is most effective

sentiment = ['average_review_sentiment']


binary_features = ['alcohol?', 'has_bike_parking', 'takes_credit_cards',
                   'good_for_kids', 'take_reservations', 'has_wifi']


numeric_features = ['review_count', 'price_range', 'average_caption_length', 'number_pics', 'average_review_age', 'average_review_length', 'average_review_sentiment', 'number_funny_votes', 'number_cool_votes',
                    'number_useful_votes', 'average_tip_length', 'number_tips', 'average_number_friends', 'average_days_on_yelp', 'average_number_fans', 'average_review_count', 'average_number_years_elite', 'weekday_checkins', 'weekend_checkins']


all_features = binary_features + numeric_features

feature_subset = ['alcohol?', 'has_wifi']

# create function to take a list of features and make, train and plot a linear regression model


def model_these_features(feature_list):

    ratings = df.loc[:, 'stars']
    features = df.loc[:, feature_list]

    X_train, X_test, y_train, y_test = train_test_split(
        features, ratings, test_size=0.2, random_state=1)

    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1, 1)
        X_test = np.array(X_test).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print('Train Score:', model.score(X_train, y_train))
    print('Test Score:', model.score(X_test, y_test))

    print(sorted(list(zip(feature_list, model.coef_)),
                 key=lambda x: abs(x[1]), reverse=True))

    y_predicted = model.predict(X_test)

    plt.scatter(y_test, y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1, 5)
    plt.show()


# Test which of the above lists of features creates the best model
model_these_features(sentiment)

model_these_features(binary_features)

model_these_features(numeric_features)

model_these_features(all_features)

model_these_features(feature_subset)


# create a linear regression model with the best model from the ones tested above
features = df.loc[:, all_features]
ratings = df.loc[:, 'stars']
X_train, X_test, y_train, y_test = train_test_split(
    features, ratings, test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)


danielles_delicious_delicacies = np.array(
    [1, 0, 1, 1, 1, 1, 32, 1, 3, 1.5, 1175, 600, 0.55, 16, 18, 43, 46, 6, 105, 2005, 12, 122, 1, 45, 50]).reshape(1, -1)

# use that model to predict the stars of a fictional store with simulated data
model.predict(danielles_delicious_delicacies)
