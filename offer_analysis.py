import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Clean portfolio dataset
def cln_portfolio(portfolio):
    ''' Clean portfolio dataframe
    INPUT:
    portfolio - portfolio dataframe
    RETURNS:
    portfolio_cln - clean dataframe
    '''
    # Copy portfolio df
    portfolio_cln = portfolio.copy()
    # Rename 'id' column to 'offer_id'
    portfolio_cln.rename(columns={'id':'offer_id'}, inplace=True)
    # Create dummy variables for 'channels' column
    dummy_channel = pd.get_dummies(portfolio_cln.channels.apply(pd.Series).stack(),
                                  prefix='channel').sum(level=0)
    portfolio_cln = pd.concat([portfolio_cln, dummy_channel], axis=1, sort=False)
    portfolio_cln.drop(columns='channels', inplace=True)

    return portfolio_cln


# Clean profile dataframe
def cln_profile(profile):
    ''' Clean profile df
    INPUT:
    profile - profile dataset
    RETURNS:
    profile_cln - clean dataset
    '''
    # Copy profile df
    profile_cln = profile.copy()
    # Drop rows with missing age, gender, and income
    profile_cln = profile.drop(profile[profile['gender'].isnull()].index)
    # Replace 'age' value of 118 to NaN
    profile_cln.age.replace(118, np.nan, inplace=True)
    # Transform date from int to datetime
    date = lambda x: pd.to_datetime(str(x), format='%Y%m%d')
    profile_cln.became_member_on = profile_cln.became_member_on.apply(date)
    # Rename 'id' column to 'customer_id'
    profile_cln.rename(columns={'id':'customer_id'}, inplace=True)
    # Create dummy columns for the gender column
    dummy_gender = pd.get_dummies(profile_cln.gender, prefix='gender')
    profile_cln = pd.concat([profile_cln, dummy_gender], axis=1, sort=False)

    return profile_cln


# Clean transcript dataset
def cln_transcript(transcript):
    ''' Clean transcript df
    INPUT:
    transcript - profile dataframe
    RETURNS:
    transcript_cln - clean dataframe
    '''
    # Copy Transcript df
    transcript_cln = transcript.copy()
    # Split 'event' into separate columns using dummy variables
    transcript_cln.event = transcript_cln.event.str.replace(' ', '_')
    dummy_event = pd.get_dummies(transcript_cln.event, prefix='event')
    transcript_cln = pd.concat([transcript_cln, dummy_event], axis=1, sort=False)
    transcript_cln.drop(columns='event', inplace=True)
    # Extract offer data from 'value' column
    transcript_cln['offer_id'] = [[*v.values()][0]
                                   if [*v.keys()][0] in ['offer id', 'offer_id'] else None
                                   for v in transcript_cln.value]
    # Extract transaction amounts from 'value' column
    transcript_cln['amount'] = [np.round([*v.values()][0], decimals=2)
                                 if [*v.keys()][0] == 'amount' else None
                                 for v in transcript_cln.value]
    transcript_cln.drop(columns='value', inplace=True)
    # Rename 'person' to 'customer_id'
    transcript_cln.rename(columns={'person':'customer_id'}, inplace=True)
    # Drop customer_id values that are absent in the profile dataset
    transcript_cln = transcript_cln[transcript_cln.customer_id.isin(profile_cln.customer_id)]
    # Drop duplicates and reset index
    transcript_cln.drop_duplicates(inplace=True)
    transcript_cln.reset_index(drop=True, inplace=True)

    return transcript_cln


# Merge dataframes for exploratory data analyis
def df_cleaned(portfolio_cln, profile_cln, transcript_cln):
    ''' Merge cleaned dataframes
    INPUT:
    portfolio_cln - cleaned portfolio dataset
    profile_cln - cleaned profile dataset
    transcript_cln - cleaned transcript dataset
    RETURNS:
    cleaned_df - merged dataset
    '''
    # Merge df
    trans_prof = pd.merge(transcript_cln, profile_cln, on='customer_id', how='left')
    df = pd.merge(trans_prof, portfolio_cln, on='offer_id', how='left')
    # Label offer ids by offer type
    offer_id = {'ae264e3637204a6fb9bb56bc8210ddfd':'bogo_1',
                '4d5c57ea9a6940dd891ad53e9dbe8da0':'bogo_2',
                '9b98b8c7a33c4b65b9aebfe6a799e6d9':'bogo_3',
                'f19421c1d4aa40978ebb69ca19b0e20d':'bogo_4',
                '0b1e1539f2cc45b7b9fa7c272da2e1d7':'discount_1',
                '2298d6c36e964ae4a3e7e9706d1fb8c2':'discount_2',
                'fafdcd668e3743c1bb461111dcafc2a4':'discount_3',
                '2906b810c7d4411798c6938adc9daaa5':'discount_4',
                '3f207df678b143eea3cee63160fa8bed':'info_1',
                '5a8bc65990b245e5a138643cd4eb9837':'info_2'}
    df.offer_id = df.offer_id.apply(lambda x: offer_id[x] if x else None)

    return df


# Get offer data (received, viewed and completed) per customer by offer type
def cust_offer_type(cleaned_df, offer_type=None):
    ''' Get offer data (received, viewed and completed) per customer by offer type
    INPUT:
    cleaned_df - merged transactions, portfolio, and profile datasets
    offer_type - bogo, discount, informational
    RETURNS:
    offer_type_cust - offer type data (received, viewed and completed) per customer
    '''
    # Define dict
    data = dict()
    for e in ['received', 'viewed', 'completed']:
        # Get 'completed' data for informational offers
        if offer_type == 'informational' and e == 'completed':
            continue
        flag = (cleaned_df['event_offer_{}'.format(e)] == 1)
        key = e
        if offer_type:
            flag = flag & (cleaned_df.offer_type == offer_type)
            key = '{}_'.format(offer_type) + key
        data[key] = cleaned_df[flag].groupby('customer_id').offer_id.count()
        # Get 'reward' data for informational offers
        flag = (cleaned_df.event_offer_completed == 1)
        if offer_type != 'informational':
            key = 'reward'
            if offer_type:
                flag = flag & (cleaned_df.offer_type == offer_type)
                key = '{}_'.format(offer_type) + key
            data[key] = cleaned_df[flag].groupby('customer_id').reward.sum()

    return data


# Get offer data (received, viewed and completed) per customer by offer id
def cust_offer_id(cleaned_df, offer_id):
    ''' Get offer data (received, viewed and completed) per customer by offer id
    INPUT:
    cleaned_df - merged transactions, portfolio, and profile datasets
    offer_id - 'bogo_1','bogo_2','bogo_3','bogo_4','discount_1','discount_2','discount_3','discount_4','info_1','info_2'
    RETURNS:
    cust_offer_id - offer id data per customer
    '''
    data = dict()
    for e in ['received', 'viewed', 'completed']:
        # Get 'completed' data for informational offers
        if offer_id in ['info_1', 'info_2'] and e == 'completed':
            continue
        event = 'event_offer_{}'.format(e)
        flag = (cleaned_df[event] == 1) & (cleaned_df.offer_id == offer_id)
        key = '{}_{}'.format(offer_id, e)
        data[key] = cleaned_df[flag].groupby('customer_id').offer_id.count()
        # Get 'reward' data for informational offers
        flag = (cleaned_df.event_offer_completed == 1) & (cleaned_df.offer_id == offer_id)
        if offer_id not in ['info_1', 'info_2']:
            key = '{}_reward'.format(offer_id)
            data[key] = cleaned_df[flag].groupby('customer_id').reward.sum()

    return data



# Group income by rounding to the lower 10000th
def group_income(x):
    ''' Round income to the lower 10000th
    INPUT:
    x - income
    RETURNS:
    rounded_income - returns 0 if the income is less than 30,000 or greater than 120,000
    '''
    for y in range(30, 130, 10):
        if x >= y*1000 and x < (y+10)*1000:
            return y*1000
    return 0
# Group age by rounding to the 5th of each 10th
def group_age(x):
    ''' Round age to the 5th of each 10th (15, 25,..., 105)
    INPUT:
    x - age

    RETURNS:
    rounded_age - returns 0 if the value is less than 15 or greater than 105
    '''
    for y in range(15, 106, 10):
        if x >= y and x < y+10:
            return y
    return 0


# Build customer dataframe with aggregated purchase data, offer data, and demographic data
def merged_cust(cleaned_df, profile_cln):
    ''' Build a dataframe with aggregated purchase and offer data and demographics
    INPUT:
    cleaned_df - merged transactions, portfolio, and profile datasets
    RETURNS:
    merged_cust - df with aggregated customer data
    '''
    cust_dict = dict()
    # Get total transaction data
    transactions = cleaned_df[cleaned_df.event_transaction == 1].groupby('customer_id')
    cust_dict['total_expense'] = transactions.amount.sum()
    cust_dict['total_transactions'] = transactions.amount.count()
    # Get aggregate offer data
    cust_dict.update(cust_offer_type(cleaned_df))
    # Get offer type data
    for ot in ['bogo', 'discount', 'informational']:
        cust_dict.update(cust_offer_type(cleaned_df, ot))
    # Get offer id data
    for oi in ['bogo_1','bogo_2','bogo_3','bogo_4',
               'discount_1','discount_2','discount_3','discount_4',
               'info_1','info_2']:
        cust_dict.update(cust_offer_id(cleaned_df, oi))
    # Build df, aggregate dict values and demographic data
    merged_cust = pd.concat(cust_dict.values(), axis=1, sort=False);
    merged_cust.columns = cust_dict.keys()
    merged_cust.fillna(0, inplace=True)
    merged_cust = pd.merge(merged_cust, profile_cln.set_index('customer_id'),
                         left_index=True, right_index=True)
    # Add columns for net expense, age, and income groups
    merged_cust['age_group'] = merged_cust.age.apply(group_age)
    merged_cust['income_group'] = merged_cust.income.apply(group_income)
    merged_cust['net_expense'] = merged_cust['total_expense'] - merged_cust['reward']

    return merged_cust


# Get any column for customers that received but not viewed an offer, viewed but not completed the offer, and those that viewed and completed the offer, grouped by a column
def offer_status(merged_cust, stat, offer, by_col, aggr='sum'):
    ''' Get any column for customers that received but not viewed an offer, viewed but not completed the offer, and those that viewed and completed the offer, grouped by a column
    INPUT:
    merged_cust - aggregated offer and demographic df
    stat - column of interest
    offer - offer of interest
    by_col - column used to group the data
    aggr - aggregation method sum or mean
    RETURNS:
    (received_agg, viewed_agg, completed) - tuple with sum aggregation
    '''
    # Define dict
    received_col = '{}_received'.format(offer)
    viewed_col = '{}_viewed'.format(offer)
    received = (merged_cust[received_col] > 0) & (merged_cust[viewed_col] == 0)
    completed = None
    # Aggregate customer behavior data for schema
    if offer not in ['informational', 'info_1', 'info_2']:
        completed_col = '{}_completed'.format(offer)
        viewed = (merged_cust[viewed_col] > 0) & (merged_cust[completed_col] == 0)
        completed_off = (merged_cust[completed_col] > 0)
        if aggr == 'sum':
            completed = merged_cust[completed_off].groupby(by_col)[stat].sum()
        elif aggr == 'mean':
            completed = merged_cust[completed_off].groupby(by_col)[stat].mean()
    else:
        viewed = (merged_cust[viewed_col] > 0)
    if aggr == 'sum':
        received_agg = merged_cust[received].groupby(by_col)[stat].sum()
        viewed_agg = merged_cust[viewed].groupby(by_col)[stat].sum()
    elif aggr == 'mean':
        received_agg = merged_cust[received].groupby(by_col)[stat].mean()
        viewed_agg = merged_cust[viewed].groupby(by_col)[stat].mean()

    return received_agg, viewed_agg, completed


# Get the average expense for customers that received but not viewed an offer, viewed but not completed the offer, and those that viewed and completed the offer, group by a column
def avg_expense(merged_cust, offer, by_col):
    ''' Get the average expense for customers that received but not viewed an offer, viewed but not completed the offer, and those that viewed and completed the offer, group by a column
    INPUT:
    merged_cust - aggregated offer and demographic df
    offer - offer of interest
    by_col - column used to group the data
    RETURNS:
    (received, viewed, completed) - tuple with the average expense
    '''
    # Get totals
    received_total, viewed_total, completed_total = offer_status(merged_cust,
                                                        'total_expense',
                                                        offer, by_col)
    received_trans, viewed_trans, completed_trans = offer_status(merged_cust,
                                                        'total_transactions',
                                                        offer, by_col)
    # Calculate averages for received and viewed offers
    received_avg = received_total / received_trans
    received_avg.fillna(0, inplace=True)
    viewed_avg = viewed_total / viewed_trans
    viewed_avg.fillna(0, inplace=True)
    completed_avg = None
    if offer not in ['informational', 'info_1', 'info_2']:
        completed_avg = completed_total / completed_trans

    return received_avg, viewed_avg, completed_avg


# Plot the total expense and the average expense per transaction incurred by customers that have received, viewed and completed an offer.
def offer_earnings_plot(merged_cust, offer):
    ''' Plot the average expense per transaction for customersthat have received, viewed and completed an offer.
    INPUT:
    merged_cust - customer df
    offer - offer type
    RETURNS:
    (age, incomme, and gender) - plots
    '''
    # Set palette
    plt.rcParams["image.cmap"] = "Set1"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    # Define variables
    received_by = dict()
    viewed_by = dict()
    completed_by = dict()
    received_avg_by = dict()
    viewed_avg_by = dict()
    completed_avg_by = dict()
    # Aggregate data by age, income, and gender
    for key in ['age_group', 'income_group', 'gender']:
        received_by[key], viewed_by[key], completed_by[key] = offer_status(merged_cust,
                                                                  'net_expense',
                                                                  offer, key,
                                                                  aggr='mean')
        by = avg_expense(merged_cust, offer, key)
        received_avg_by[key], viewed_avg_by[key], completed_avg_by[key] = by
    # Plot layout
    plt.figure(figsize=(16, 10))
    # Plot offer expense by 'bogo'
    plt.subplot(231)
    plt.plot(received_avg_by['age_group'], label='{}-received'.format(offer))
    plt.plot(viewed_avg_by['age_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'info_1', 'info_2']:
        plt.plot(completed_avg_by['age_group'], label='{}-completed'.format(offer))
    plt.legend(loc='upper left')
    plt.title('Average Transaction Value by Age')
    plt.xlabel('Age')
    plt.ylabel('USD');
    # Plot offer expense by 'discount'
    plt.subplot(232)
    plt.plot(received_avg_by['income_group'], label='{}-received'.format(offer))
    plt.plot(viewed_avg_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'info_1', 'info_2']:
        plt.plot(completed_avg_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='upper left')
    plt.title('Average Transaction Value by Income')
    plt.xlabel('Income')
    plt.ylabel('USD');
    # Plot offer expense by 'informational'
    plt.subplot(233)
    index = np.array([0, 1, 2])
    bar_width = 0.3
    plt.bar(index, received_avg_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label='{}-received'.format(offer))
    plt.bar(index + bar_width, viewed_avg_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'info_1', 'info_2']:
        plt.bar(index+2*bar_width, completed_avg_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.legend(loc='upper left')
    plt.title('Average Transaction Value by Gender')
    plt.xticks(index + bar_width, ('Male', 'Female', 'Other'))
    plt.xlabel('Gender')
    plt.ylabel('USD');


# Get the net earnings for customers that viewed and completed offers
def net_earnings(merged_cust, offer, q=0.5):
    '''Get the net_earnings for customers that viewed and completed offers
    INPUT:
    offer - offer of interest
    q - quantile to be used
    RETURNS:
    net_earnings - median of total transaction value
    '''
    # Flag customers that viewed offers
    flag = (merged_cust['{}_viewed'.format(offer)] > 0)
    # Sort through positive net earnings
    flag = flag & (merged_cust.net_expense > 0)
    # Aggregate those with at least 5 transactions
    flag = flag & (merged_cust.total_transactions >= 5)
    # Aggregate viewed and completed offers
    if offer not in ['info_1', 'info_2']:
        flag = flag & (merged_cust['{}_completed'.format(offer)] > 0)

    return merged_cust[flag].net_expense.quantile(q)


# Define loop that sorts by highest earnings
def greatest_earnings(merged_cust, n_top=2, q=0.5, offers=None):
    '''Sort offers based on the ones that result in the highest net_expense
    INPUT:
    customers - dataframe with aggregated data of the offers
    n_top - number of offers to be returned (default: 2)
    q - quantile used for sorting
    offers - list of offers to be sorted
    RETURNS:
    sorted list of offers, in descending order according to the median net_expense
    '''
    # Sort for offers earnings in second quantile
    if not offers:
        offers = ['bogo_1','bogo_2','bogo_3','bogo_4',
               'discount_1','discount_2','discount_3','discount_4',
               'info_1','info_2']
    offers.sort(key=lambda x: net_earnings(merged_cust, x, q), reverse=True)
    offers_dict = {o: net_earnings(merged_cust, o, q) for o in offers}

    return offers[:n_top], offers_dict

# Print 10 offers by most to least popular, highest to least highest earnings
offers = greatest_earnings(merged_cust, n_top=10)
print(offers[0])
print(offers[1])



# Calculate percent success by offer
def calculate_percentage_success():
    '''Calculate percent success by offer
    INPUT:
    cleaned_df - dataframe with merged transaction, offer, and demographic data
    RETURNS:
    percent success by offer
    '''
    # Define variables, calculate percent success
    successful_count = cleaned_df[['offer_id', 'event_offer_completed']].groupby(
        'offer_id').sum().reset_index()
    offer_count = cleaned_df['offer_id'].value_counts()
    offer_count = pd.DataFrame(list(zip(offer_count.index.values,
                                        offer_count.values)),
                               columns=['offer_id', 'count'])
    successful_count = successful_count.sort_values('offer_id')
    offer_count = offer_count.sort_values('offer_id')
    percent_success = pd.merge(offer_count, successful_count, on="offer_id")
    percent_success['percent_success'] = (100 * percent_success['event_offer_completed'] / percent_success['count'])
    percent_success = percent_success.drop(columns=['event_offer_completed'])
    percent_success = percent_success.sort_values('percent_success', ascending=False)

    return percent_success.reset_index(drop=True)



# Plot offer id count and percent success
fig, ax = plt.subplots(figsize=(28, 8), nrows=1, ncols=2)
# Plot offer type by count
ax[0].bar(percent_success.index + 1, percent_success['count'], color='teal')
ax[0].set_xticks(np.arange(0,8) + 1)
ax[0].set_xlabel('Offer Type',size = 15)
ax[0].set_ylabel('Count',size = 15)
ax[0].set_title('Offer Type by Count', size = 20)
# Renaming x tick marks
group_labels = ['bogo_3', 'discount_4', 'discount_3',
                'discount_2', 'discount_1', 'bogo_4',
                'bogo_1', 'bogo_2']
ax[0].set_xticklabels(group_labels, rotation=-45)
# Plot percent success by offer type
ax[1].plot(percent_success.index + 1, percent_success['percent_success'], linewidth=4.0, color='black')
ax[1].set_xticks(np.arange(0,8) + 1)
ax[1].set_xlabel('Offer Type',size = 15)
ax[1].set_ylabel('Percent Success',size = 15)
ax[1].set_title('Percent Success by Offer Type', size = 20)
# Renaming x tick marks
group_labels = ['bogo_3', 'discount_4', 'discount_3',
                'discount_2', 'discount_1', 'bogo_4',
                'bogo_1', 'bogo_2']
ax[1].set_xticklabels(group_labels, rotation=-45)


# Clean Portfolio dataset by offer type
def modeling_portfolio(portfolio):
    ''' Preprocess portfolio df
    INPUT:
    portfolio - portfolio dataset
    RETURNS:
    portfolio_clean - clean dataset
    '''
    # Copy portfolio df
    portfolio_modeling = portfolio.copy()
    # portfolio: one-hot encode channels
    for index, row in portfolio_modeling.iterrows():
        for channel in ['web', 'email', 'social', 'mobile']:
            if channel in portfolio_modeling.loc[index, 'channels']:
                portfolio_modeling.loc[index, channel] = 1
            else:
                portfolio_modeling.loc[index, channel] = 0
    portfolio_modeling.drop(columns='channels', inplace=True)
    # One-hot encode offer_type column
    for index, row in portfolio_modeling.iterrows():
        for offertype in ['bogo', 'informational', 'discount']:
            if offertype in portfolio_modeling.loc[index, 'offer_type']:
                portfolio_modeling.loc[index, offertype] = 1
            else:
                portfolio_modeling.loc[index, offertype] = 0
    portfolio_modeling.drop(columns='offer_type', inplace=True)
    # Rename 'id' column to 'offer_id'
    portfolio_modeling.rename(columns={'id':'offer_id'}, inplace=True)

    return portfolio_modeling


# Separate offer and transaction data
def modeling_transcript(transcript):
    ''' Preprocess transcript df
    INPUT:
    transcript - profile dataset
    RETURNS:
    transcript_clean - clean dataset
    '''
    # Copy Transcript df
    transcript_modeling = transcript.copy()
    # Rename 'person' to 'customer_id'
    transcript_modeling.rename(columns={'person':'customer_id'}, inplace=True)
    # Extract offer data from 'value' column
    transcript_modeling['offer_id'] = [[*v.values()][0]
                                   if [*v.keys()][0] in ['offer id', 'offer_id'] else None
                                   for v in transcript_modeling.value]
    # Extract transaction amounts from 'value' column
    transcript_modeling['amount'] = [np.round([*v.values()][0], decimals=2)
                                 if [*v.keys()][0] == 'amount' else None
                                 for v in transcript_modeling.value]
    transcript_modeling.drop(columns='value', inplace=True)
    # Drop customer_id values that are absent in the profile dataset
    transcript_modeling = transcript_modeling[transcript_modeling.customer_id.isin(profile_clean.customer_id)]
    # Convert time in hours to time in days
    transcript_modeling['time'] /= 24.0
    # Drop duplicates and reset index
    transcript_modeling.drop_duplicates(inplace=True)
    transcript_modeling.reset_index(drop=True, inplace=True)

    return transcript_modeling


# Use progressbar to combine modeling dataframes
def combine_modeling_data(profile_modeling, portfolio_modeling, transaction_modeling, offers_modeling):
    ''' Combine modeling dataframes
    INPUT:
    profile_modeling - profile dataset
    portfolio_modeling - portfolio dataset
    transaction_modeling - transaction dataset
    offers_modeling - offers dataset
    RETURNS:
    data - pd df of merged datasets
    '''
    data = []
    customer_ids = offers_modeling['customer_id'].unique()
    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    # Loop through all customer ids in offers_df
    for ind in progressbar.progressbar(range(len(customer_ids)), widgets=widgets):
        # Get customer id from the list
        cust_id = customer_ids[ind]
        # Extract customer profile from profile data
        customer = profile_modeling[profile_modeling['customer_id']==cust_id]
        # Extract offers associated with the customer from offers_df
        cust_offer_data = offers_modeling[offers_modeling['customer_id']==cust_id]
        # Extract transactions associated with the customer from transactions_df
        cust_transaction_data = transaction_modeling[transaction_modeling['customer_id']==cust_id]
        # Extract received, completed, viewed offer data from customer offers
        offer_received_data = cust_offer_data[cust_offer_data['received'] == 1]
        offer_completed_data = cust_offer_data[cust_offer_data['completed'] == 1]
        offer_viewed_data = cust_offer_data[cust_offer_data['viewed'] == 1]
        rows = []

        # Loop through each received offer
        for i in range(offer_received_data.shape[0]):
            # Fetch an offer id
            offer_id = offer_received_data.iloc[i]['offer_id']
            # Extract offer row from portfolio
            offer_row = portfolio_modeling.loc[portfolio_modeling['offer_id'] == offer_id]
            # Extract duration days of an offer from offer row
            duration_days = offer_row['duration'].values[0]
            # Initialize start and end time of an offer
            start_time = offer_received_data.iloc[i]['time']
            end_time = start_time + duration_days
            # Get completed offers by end date
            off_completed_withintime = np.logical_and(
                offer_completed_data['time'] >= start_time, offer_completed_data['time'] <= end_time)
            # Get offers viewed by end date
            off_viewed_withintime = np.logical_and(
                offer_viewed_data['time'] >= start_time, offer_viewed_data['time'] <=end_time)
            # Flag offer_successful to 1 if an offer is viewed and completed within end time else to 0
            offer_successful = off_completed_withintime.sum() > 0 and off_viewed_withintime.sum() > 0
            # Get transactions occured within time
            transaction_withintime = np.logical_and(
                cust_transaction_data['time'] >= start_time, cust_transaction_data['time'] <= end_time)
            transaction_data = cust_transaction_data[transaction_withintime]
            # Get total amount spent by a customer from given offers
            transaction_total_amount = transaction_data['amount'].sum()
            row = {
                'offer_id': offer_id,
                'customer_id': cust_id,
                'time': start_time,
                'total_amount': transaction_total_amount,
                'offer_successful': int(offer_successful),
            }
            row.update(offer_row.iloc[0,0:].to_dict())
            row.update(customer.iloc[0,:].to_dict())
            rows.append(row)
        data.extend(rows)
    data = pd.DataFrame(data)
    return data

# Prediction modeling: create train and test datasets
# Factors/features that influence the label variable
features = modeling_df.drop(columns=['offer_successful'])
feature_names = features.columns[2:]
# Label variable to predict
label = modeling_df.filter(['offer_successful'])
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features.values, label.values, test_size=0.3, random_state=42)
# Train-data: extract offer_id, total_amount and convert other features to float
offer_id_train = X_train[:, 0]
total_amount_train = X_train[:, 1]
X_train = X_train[:, 2:].astype('float64')
# Test-data: extract offer_id, total_amount and convert other features to float
offer_id_test = X_test[:, 0]
total_amount_test = X_test[:, 1]
X_test = X_test[:, 2:].astype('float64')
# Convert train and test labels to array
y_train = y_train.ravel()
y_test = y_test.ravel()


# Evaluate naive predictor performance
naive_predictor_accuracy = accuracy_score(y_train, np.ones(len(y_train)))
naive_predictor_f1score = f1_score(y_train, np.ones(len(y_train)))
print("Naive predictor accuracy: %.3f" % (naive_predictor_accuracy))
print("Naive predictor f1-score: %.3f" % (naive_predictor_f1score))


# Construct logistic regression model
scorer = make_scorer(fbeta_score, beta=0.5)
# Instantiate a logistic regression classifer object
lr_clf = LogisticRegression(random_state=42, solver='liblinear')
# Construct a params dict to tune the model
grid_params = {
    'penalty': ['l1', 'l2'],
    'C': [1.0, 0.1, 0.01]}
lr_random = RandomizedSearchCV(
    estimator = lr_clf, param_distributions = grid_params,
    scoring=scorer, n_iter = 6, cv = 3, verbose=2,
    random_state=42, n_jobs = 3)
# Fit train data to the model
lr_random.fit(X_train, y_train)


# Define model performance evaluation function
def evaluate_model_performance(clf, X_train, y_train):
    '''Prints a model's accuracy and F1-score
    INPUT:
    clf - Model object
    X_train - Training data matrix
    y_train - Expected model output vector
    OUTPUT:
    clf_accuracy: Model accuracy
    clf_f1_score: Model F1-score
    '''
    class_name = re.sub("[<>']", '', str(clf.__class__))
    class_name = class_name.split(' ')[1]
    class_name = class_name.split('.')[-1]
    y_pred_rf = clf.predict(X_train)
    clf_accuracy = accuracy_score(y_train, y_pred_rf)
    clf_f1_score = f1_score(y_train, y_pred_rf)
    print("%s model accuracy: %.3f" % (class_name, clf_accuracy))
    print("%s model f1-score: %.3f" % (class_name, clf_f1_score))

    return clf_accuracy, clf_f1_score


# Evaluate logistic regression model performance
evaluate_model_performance(lr_random.best_estimator_, X_train, y_train)


# Instantiate a random forest classifier obj
rf_clf = RandomForestClassifier(random_state=42)
# Number of trees in random forest
n_estimators = [10, 50, 100, 150, 200, 250, 300]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.arange(3, 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Create the random grid
grid_params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
# Tune the classifer
rf_random = RandomizedSearchCV(estimator = rf_clf,
                               param_distributions = grid_params,
                               scoring=scorer,
                               n_iter = 100,
                               cv = 3,
                               verbose=2,
                               random_state=42,
                               n_jobs = 3)
# Fit train data to the classifier
rf_random.fit(X_train, y_train)


# Evaluate random forest classifier model's performance
evaluate_model_performance(rf_random.best_estimator_, X_train, y_train)


# Define variables
relative_importance = rf_random.best_estimator_.feature_importances_
relative_importance = relative_importance / np.sum(relative_importance)
feature_importance =\
    pd.DataFrame(list(zip(feature_names,
                          relative_importance)),
                 columns=['feature', 'relativeimportance'])
feature_importance = feature_importance.sort_values('relativeimportance',
                                                    ascending=False)
feature_importance = feature_importance.reset_index(drop=True)
palette = sns.color_palette("Blues_r", feature_importance.shape[0])
# Plot Estimated Feature Importance
plt.figure(figsize=(8, 8))
sns.barplot(x='relativeimportance',
            y='feature',
            data=feature_importance,
            palette=palette)
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.title('Estimated Feature Importance Based on Random Forest')


# Create the random grid
gb_clf = GradientBoostingClassifier(random_state=42)
random_grid = {'loss': ['deviance', 'exponential'],
               'learning_rate': [0.1, 0.01, 0.001],
               'n_estimators': [10, 30, 50, 100, 150, 200, 250, 300],
               'min_samples_leaf': min_samples_leaf,
               'min_samples_split': min_samples_split}
gb_random = RandomizedSearchCV(estimator = gb_clf,
                               param_distributions = random_grid,
                               scoring=scorer,
                               n_iter = 100,
                               cv = 3,
                               verbose=2,
                               random_state=42,
                               n_jobs = 3)
gb_random.fit(X_train, y_train)


# Evaluate gradient boosting model performance
evaluate_model_performance(gb_random.best_estimator_, X_train, y_train)


# Build df of ranked model performance
model_performance = []
classifier_type = ['naivepredictor','logisticregression','randomforest','gradientboosting']
model_performance.append((naive_predictor_accuracy, naive_predictor_f1score))
model_performance.append(evaluate_model_performance(lr_random.best_estimator_, X_train, y_train))
model_performance.append(evaluate_model_performance(rf_random.best_estimator_, X_train, y_train))
model_performance.append(evaluate_model_performance(gb_random.best_estimator_, X_train, y_train))
model_performance = pd.DataFrame(model_performance, columns=['accuracy', 'f1score'])
classifier_type = pd.DataFrame(classifier_type, columns=['classifiertype'])
model_performance = pd.concat([classifier_type, model_performance], axis=1)
model_performance = model_performance.sort_values('f1score', ascending=False)
model_performance = model_performance.reset_index(drop=True)
model_performance


# Print the best model's hyperparameters
print(rf_random.best_estimator_)


# Refine model hyperparameter space
parameters = {'n_estimators': [300, 350, 400, 450, 500],
              'max_depth': [10, 11, 12, 13, 14, 15],
              'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split,
              'random_state': [42]}
grid_obj = GridSearchCV(rf_clf,
                        parameters,
                        scoring=scorer,
                        cv=5,
                        n_jobs=3,
                        verbose=2)
grid_fit = grid_obj.fit(X_train, y_train)
# Get the estimator
best_clf = grid_fit.best_estimator_
# Evaluate model performance
evaluate_model_performance(best_clf, X_train, y_train)


# Print the refined random forest model's hyperparameters
best_clf


# Evaluate model performance
evaluate_model_performance(best_clf, X_test, y_test)
