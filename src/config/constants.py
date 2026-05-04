"""Shared constants for all production scripts."""

# Target
TARGET     = 'price'
TARGET_CAP = 249

# Experiment setup
GLOBAL_SEED     = 42
TEST_SPLIT_SIZE = 0.2

# Architecture
ARCHITECTURE = {
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'activation': 'relu',
}

# Price segments
SEGMENTS = [
    {'name': 'Low',         'min': 0,   'max': 79},
    {'name': 'Low_Medium',  'min': 79,  'max': 110},
    {'name': 'Medium',      'min': 110, 'max': 150},
    {'name': 'Medium_High', 'min': 150, 'max': 193},
    {'name': 'High',        'min': 193, 'max': 249},
]

# Input features
NUMERICAL_STANDARD_FEATURES = [
    'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
    'amenities_count', 'host_days_active', 'days_since_first_review',
    'days_since_last_review', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
    'calculated_host_listings_count', 'availability_30', 'availability_60',
    'availability_90', 'availability_365', 'number_of_reviews',
    'host_is_superhost', 'host_identity_verified', 'instant_bookable',
]

NUMERICAL_REVIEW_FEATURES = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value', 'reviews_per_month',
]

CATEGORICAL_FEATURES = [
    'neighbourhood_group_cleansed', 'neighbourhood_cleansed',
    'room_type', 'property_type', 'host_response_time',
]

# Columns dropped at load time
COLS_TO_DROP = [
    'id', 'host_id', 'scrape_id', 'listing_url', 'picture_url', 'host_url',
    'host_thumbnail_url', 'host_picture_url', 'last_scraped', 'source',
    'calendar_last_scraped', 'name', 'description', 'host_name',
    'host_verifications', 'calendar_updated', 'license',
    'host_response_rate', 'host_acceptance_rate', 'neighborhood_overview',
    'host_neighbourhood', 'neighbourhood', 'host_location', 'host_about',
    'estimated_occupancy_l365d', 'estimated_revenue_l365d',
    'bathrooms_text', 'has_availability',
]

# Plot colors
C_DET = '#FF6B6B'   # deterministic model
C_BAY = '#6200EA'   # Bayesian model
C_AUX = '#4ECDC4'   # auxiliary / uncertainty
