'''
Name:   Susan Dawson
CS602:  Section: SN1
Data: Boston /Crime:

This program reads in crime data, removes unnecessary columns and cleans the data.
'''

import pandas as pd
import datetime


def filter_df_by_end_date(df, year, month, day):
    through_date = datetime.date(year, month, day)
    # Filter for date range
    df = df.loc[(df["date"] <= through_date)]

    return df


def process_file_data(filename, year):

    crime_df = pd.read_csv(filename, header=0, dtype={'INCIDENT_NUMBER': 'string'})

    crime_df.set_index('INCIDENT_NUMBER')
    crime_df.pop('OFFENSE_CODE')
    crime_df.pop('OFFENSE_CODE_GROUP')
    crime_df.pop('UCR_PART')
    crime_df.pop('REPORTING_AREA')
    crime_df.pop('Location')
    crime_df.pop('YEAR')
    crime_df.pop('MONTH')
    crime_df['OCCURRED_ON_DATE'] = crime_df['OCCURRED_ON_DATE'].apply(lambda x: pd.to_datetime(str(x)))
    crime_df['date'] = crime_df['OCCURRED_ON_DATE'].dt.date
    crime_df.pop('OCCURRED_ON_DATE')
    crime_df.columns = ['incident_num', 'offense', 'district', 'shooting', 'weekday', 'hour', 'street', 'lat',
                        'long', 'date']
    # Replace missing locations
    crime_df['lat'] = crime_df['lat'].replace(0, 42.35866)
    crime_df['long'] = crime_df['long'].replace(0, -71.05674)
    # crime_df['location'] = crime_df['location'].replace('[0, 0]', '[42.35866, -71.05674]')
    crime_df = crime_df.dropna()

    # Sort based on date column**
    crime_df = crime_df[['incident_num', 'offense', 'date', 'street', 'lat', 'long', 'district',
                                         'shooting', 'weekday', 'hour']]
    crime_df.sort_values(by='date')
    crime_df.set_index('incident_num', inplace=True)

    return crime_df


filenames = ['BostonCrime2021_sample_orig.csv', 'Crime2020.csv', 'Crime2019.csv']

# read in 2021 data frame
df_2021 = process_file_data(filenames[0], 2021)
pd.DataFrame.to_csv(df_2021, "BostonCrime2021_sample.csv")

# read in 2020 data and create a sample file
df_2020 = process_file_data(filenames[1], 2020)
df_2020 = filter_df_by_end_date(df_2020, 2020, 6, 30)
sample_df = df_2020.sample(n=7000, random_state=4)
sample_df = sample_df.loc[:, ~sample_df.columns.str.contains('^Unnamed')]
pd.DataFrame.to_csv(sample_df, "BostonCrime2020_sample.csv")

# read in 2019 data and create a sample file
df_2019 = process_file_data(filenames[2], 2019)
sample_df2 = df_2019.sample(n=7000, random_state=4)
sample_df2 = sample_df2.loc[:, ~sample_df.columns.str.contains('^Unnamed')]
pd.DataFrame.to_csv(sample_df2, "BostonCrime2019_sample.csv")

