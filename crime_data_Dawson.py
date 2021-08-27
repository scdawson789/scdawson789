'''
Name:   Susan Dawson
CS602:  Section: SN1
Data: Boston /Crime:

This program allows users to view crime data for Boston with a primary focus on 2021 data,
but also to compare 2019 and 2020 to 2021.
The user can filter the data based on date, crime categories and districts.
Data is displayed in data frame and can have the filter changed.
'''
import pandas as pd
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_folium import folium_static
import folium
import math
import seaborn as sns

st.set_page_config(layout="wide")

# Categorize data crimes for filtering
data_categories = ['threats', 'drugs', 'fraud', 'theft', 'physical_confrontation', 'murder', 'suicide',
                   'missing', 'fire', 'vandalism']
data_categories = sorted(data_categories)
data_type = {}
data_type['threats'] = ['threats', 'firearm', 'weapon', 'disorderly conduct', 'trespassing', 'verbal dispute',
                        'harassment', 'bomb', 'burglarious']
data_type['drugs'] = ['drug', 'drugs', 'alcohol', 'under the influence']
data_type['fraud'] = ['fraud', 'forgery']
data_type['theft'] = ['robbery', 'burglary', 'breaking and entering', 'larceny', 'theft', 'extortion']
data_type['physical_confrontation'] = ['assault', 'affray', 'animal abuse']
data_type['murder'] = ['murder']
data_type['suicide'] = ['suicide']
data_type['missing'] = ['missing']
data_type['fire'] = ['arson', 'fire']
data_type['vandalism'] = ['vandalism', 'graffiti']


def process_file_data(filename, district_df, filter_out_data, crime_keywords):
    '''
    Function: Process_file_data
    Inputs: Filename of crime data to read in, the district name data file to be merged
    crime keywords (based on the categorizations above)
    Outputs: Returns merged and cleaned data frame
    '''
    df = pd.read_csv(filename, header=0)
    df['date'] = pd.to_datetime(df['date']).dt.date

    if filter_out_data:
        df = exec_crimes_filter(df, crime_keywords)
    df = merge_dataframes(district_df, df)
    return df


def process_districts():
    '''
    Function: process_districts
    Inputs: None
    Output: Dataframe from the BostonPoliceDistricts.csv file
    '''
    district_df = pd.read_csv('BostonPoliceDistricts.csv', header=0)
    return district_df


def merge_dataframes(district_df, crime_df):
    '''
    Function: merge_dataframes
    Inputs: 2 data frames: the districts data frame and the crime data frame
    Output: merged data frame
    Also some of the crime data had the district code as External, but I am using District Name
    in drop lists.  The district data file whichc brought in the district Name did not include
    external.  So the merged data frame is updated to copy external to the District Name column
    when the district column shows 'External'.
    '''
    complete_crime = district_df.merge(crime_df, how="outer", left_on='District', right_on='district')
    complete_crime.pop('District')
    complete_crime.loc[(complete_crime.district == "External"), 'District Name'] = "External"
    return complete_crime


def merge_df_totals(left_df, right_df, merge_col):
    '''
    Function: merge_df_totals
    Input: 2 data frames for merging and the merge column
    Output: Merged data frame
    '''
    merged_df = left_df.merge(right_df, how="outer", left_on=merge_col, right_on=merge_col)
    return merged_df


def filter_df_from_through(df, from_date, through_date):
    '''
    Function: filter db based on from and through date
    Input: data frame for filtering, from date and through date to filter on
    Output: filtered data frame (based on dates entered)
    If invalid date range entered, the date range is ignored.
    '''
    datatypes = df.dtypes
    if from_date <= through_date:
        df = df.loc[(df['date'] >= from_date) & (df['date'] <= through_date)]

    return df


def filter_df_by_district(df, district_name):
    '''
    Function: filter_df_by_district
    Input: data frame to filter and district name to filter on
    Output: filtered data frame
    '''
    df = df.loc[(df['District Name'] == district_name)]
    return df


def filter_df_by_crime(df, crime_list):
    '''
    Function: filter_df_by_crime
    Input: data frame for filtering and crimelist keywords to filter on
    Output: filtered df
    '''
    df['offense'] = df['offense'].str.lower()
    df = df.loc[df['offense'].str.contains('|'.join(crime_list))]
    df.sort_values(by='offense')
    return df


def get_frequency_totals_one_column(df, year):
    '''
    Function: get_frequency_totals_one_column
    Input: data frame and year for the total count column
    Output: frequency table
    Also fills in Nan fields created by merge (outer join) of frequency tables
    '''
    new_df = pd.value_counts(df.offense).reset_index()
    new_df.columns = ['offense', 'count_' + str(year)]
    new_df['count_' + str(year)] = new_df['count_' + str(year)].fillna(0)
    return new_df


def get_freq_totals_orig(df, column):
    '''
    Function: get_freq_totals_orig
    Input data frame and column
    Returns frequency counts for the column
    '''
    value_counts = df[column].value_counts()
    return value_counts


def get_offense_descriptions(data_frame):
    '''
    Function:get_offense_descriptions
    Inputs: crime data frame
    Outputs:list of offense descriptions (for filtering)
    '''
    offense_desc = data_frame['offense'].unique()
    return offense_desc


def get_crime_keywords():
    '''
    Function:get_crime_keywords
    Output: flat list of crime keywords that are presented to user for confirmation
    these values will be passed to the crimes_not_found_by_keywords
    '''
    flat_list = []
    for crime in data_type.values():
        flat_list += crime
    return flat_list


def crimes_not_found_by_keywords(crimes, keywords):
    '''
    Function: crimes_not_found_by_keywords
    Input: list of crimes and keywords
    Output: Which offense are found by keywords (retained in the data)
    which offenses are not found by keywords (filtered out of the data)
    '''
    found_items = []
    not_found = []
    found = False
    for crime in crimes:
        for key in keywords:
            found = crime.lower().count(key.lower())
            if found > 0:
                found = True
                break
        if found:
            found_items.append(crime)
        else:
            not_found.append(crime)
    return not_found, found_items


def get_unique_column_values(data_frame, column):
    '''
    Function:get_unique_column_values
    Input: data frame and column
    Output: unique values list for that column
    '''
    column_values = data_frame[column].unique()
    return column_values


def filter_data_question(df):
    '''
    Function:filter_data_question
    Input data frame for filtering
    output: filtered data frame based on side bar selections
    This is a filter of the offenses based on keyword.  User is allowed to override
    the keyword filter, by adding offenses slated for filtering into the keywords list
    '''
    # get keywords for filtering
    crime_keywords = get_crime_keywords()
    crime_keywords.sort()
    full_list_crimes = get_offense_descriptions(df)
    full_list_crimes.sort()
    notfound, found = crimes_not_found_by_keywords(full_list_crimes, crime_keywords)
    st.subheader("Would you like the 'crimes' below filtered out of data")
    filter_out_data = st.selectbox("Would you like the 'crimes' below filtered out of data", ["Yes", "No"])
    st.write(notfound)
    st_ms = st.multiselect("offenses you would like to keep in the dataset", notfound)
    for item in st_ms:
        crime_keywords.append(item.lower())
    if filter_out_data == "Yes":
        df = exec_crimes_filter(df, crime_keywords)
    return df, filter_out_data, crime_keywords


def exec_crimes_filter(df, crime_keywords):
    '''
    Function:exec_crimes_filter
    Input: data frame and crime keywords list
    Output: returns filtered data frame
    '''
    df = filter_df_by_crime(df, crime_keywords)
    return df


def filter_bar(df):
    '''
    Function:filter_bar
    Input: data frame for filtering
    Output: filtered data frame
    User is allowed to filter on from and through date, show raw data and sort it by
    either offense or date
    '''
    from_date = st.sidebar.date_input('From Date')
    through_date = st.sidebar.date_input('Through Date')
    if from_date != through_date:
        df = filter_df_from_through(df, from_date, through_date)
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader('Raw Data')
        chosen = st.radio('Sorting Options', ('offense', 'date'))
        if chosen == 'offense':
            df = df.sort_values(by='offense')
        elif chosen == 'date':
            df = df.sort_values(by='date')
        st.dataframe(df)
    return df


def multi_year_filter(final_2021, df_2020, df_2019, neighborhoods):
    '''
    Function: multi_year_filter
    Input: data frames for 2019, 2020 and 2021, as well as a list of districts/ neighborhoods
    Output: 3 filtered data frames:
    Filtering options: - districts and offense
    User can choose not to filer on offenses main categories (all option)
    User can choose to show raw data and sorting options are provided.
    Since there are 3 data frames (3 years), if raw data displayed, beta_columns are used
    '''
    if st.sidebar.checkbox("Filter Data districts and crime types: "):
        # filter for districts
        view_district = st.sidebar.selectbox("What neighborhood are you interested in?", neighborhoods)
        final_2021 = filter_df_by_district(final_2021, view_district)
        df_2020 = filter_df_by_district(df_2020, view_district)
        df_2019 = filter_df_by_district(df_2019, view_district)

        data_categories.append('all')
        crime_family = st.sidebar.selectbox("What category of crimes would you like to look at?", data_categories)
        if crime_family != 'all':
            final_2021 = filter_df_by_crime(final_2021, data_type.get(crime_family))
            df_2020 = filter_df_by_crime(df_2020, data_type.get(crime_family))
            df_2019 = filter_df_by_crime(df_2019, data_type.get(crime_family))

    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader('Raw Data')
        chosen2 = st.sidebar.radio('Sort Options', ('offense', 'date'))
        if chosen2 == 'offense':
            final_2021 = final_2021.sort_values(by='offense')
            df_2020 = df_2020.sort_values(by='offense')
            df_2019 = df_2019.sort_values(by='offense')
        elif chosen2 == 'date':
            final_2021 = final_2021.sort_values(by='date')
            df_2020 = df_2020.sort_values(by='date')
            df_2019 = df_2019.sort_values(by='date')
        yr1_name, yr2_name, yr3_name = st.beta_columns(3)
        yr1_name.write("2021")
        yr2_name.write("2020")
        yr3_name.write("2019")
        yr2021, yr2020, yr2019 = st.beta_columns(3)
        yr2021.write(final_2021)
        yr2020.write(df_2020)
        yr2019.write(df_2019)

    return final_2021, df_2020, df_2019


def group_by_bar_chart(final_2021, df_2020, df_2019):
    '''
    Function:group_by_bar_chart
    Input: filtered data frames for 2019, 2020 and 2021
    Output: Group by chart showing crime frequencies across the years
    '''
    # build chart
    font = {'family': 'monospace',
            'color': 'black',
            'weight': 'normal',
            'size': 12}
    #
    group_by_2021 = get_frequency_totals_one_column(final_2021, 2021)
    group_by_2020 = get_frequency_totals_one_column(df_2020, 2020)
    group_by_2019 = get_frequency_totals_one_column(df_2019, 2019)
    merged_df = merge_df_totals(group_by_2021, group_by_2020, 'offense')
    merged_df = merge_df_totals(merged_df, group_by_2019, 'offense')
    merged_df['count_2019'].fillna(0, inplace=True)
    merged_df['count_2020'].fillna(0, inplace=True)
    merged_df = merged_df.nlargest(10, 'count_2021')
    st.dataframe(merged_df)

    plt.rcParams["figure.figsize"] = (10, 2)
    fig, ax = plt.subplots()

    width = .3

    x_labels = merged_df['offense'].tolist()
    trunc_labels = [x[:25] for x in x_labels]

    bar1 = np.arange(len(x_labels))
    bar2 = [i + width for i in bar1]
    bar3 = [i + width for i in bar2]
    frequencies_2021 = merged_df['count_2021'].tolist()
    frequencies_2020 = merged_df['count_2020'].tolist()
    frequencies_2019 = merged_df['count_2019'].tolist()

    rects1 = ax.bar(bar1, frequencies_2021, width, label="2021")
    rects2 = ax.bar(bar2, frequencies_2020, width, label="2020")
    rects3 = ax.bar(bar3, frequencies_2019, width, label="2019")

    ax.bar_label(rects1, padding=3, rotation=90)
    ax.bar_label(rects2, padding=3, rotation=90)
    ax.bar_label(rects3, padding=3, rotation=90)

    # determine y axis height
    merged_df['max_value'] = merged_df[['count_2019', 'count_2020', 'count_2021']].max(axis=1)
    high = max(merged_df['max_value'])
    step = math.ceil(high/8)
    y_ticks = np.arange(0, math.ceil(high * 1.3), step)

    # set x and y ticks
    ax.set_xticks(bar2)
    ax.set_xticklabels(trunc_labels, rotation=90, fontsize=10)
    ax.set_yticks(y_ticks)

    plt.legend(['2021', '2020', '2019'], loc='upper center', bbox_to_anchor=(.5, 1.5))

    st.pyplot(fig)


def seaborn_bar_chart(df, crime):
    '''
    Function: seaborn_bar_chart
    Input: data frame, crime list
    Output: bar plot showing the frequency for an individual crime across neighborhoods
    '''
    font = {'family': 'monospace',
            'color': 'black',
            'weight': 'normal',
            'size': 12}
    chart_data = df[df['offense'] == crime]
    chart_data.insert(5, 'occurrences', chart_data['District Name'].count())
    chart_data['occurrences'] = chart_data.groupby(by='District Name')['District Name'].transform('count')
    st.dataframe(chart_data)

    # determine y axis height
    high = max(chart_data['occurrences'])
    step = math.ceil(high/8)
    y_ticks = np.arange(0, math.ceil(high * 1.1), step)

    # create figure and set the size
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 1.5)

    ax.set_ylabel('Frequency', fontdict=font)
    ax.set_xlabel('Districts of Boston', fontdict=font)

    g = sns.catplot(data=chart_data, kind='count', x='District Name', hue='offense')
    plt.yticks(y_ticks)
    g.fig.set_figwidth(5)
    g.fig.set_figheight(1)
    g._legend.remove()
    plt.xticks(rotation=90)
    plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.4))
    st.pyplot(g)


def main():

    fileNames = ['BostonCrime2021_sample.csv', 'BostonCrime2020_sample.csv', 'BostonCrime2019_sample.csv']
    # Data Loading Section (Initial)
    # Get 2021 Sample Data and accept filtering
    crime_df_2021 = pd.read_csv(fileNames[0], header=0)
    crime_df_2021['date'] = pd.to_datetime(crime_df_2021['date']).dt.date

    st.title("Boston Crime Data")
    crime_df_2021, filter_out_data, crime_keywords = filter_data_question(crime_df_2021)

    # Get District Data and merge / Data Gathering
    district_df = process_districts()
    complete_2021 = merge_dataframes(district_df, crime_df_2021)
    neighborhoods = get_unique_column_values(district_df, 'District Name')
    neighborhoods = sorted(neighborhoods)

    st.subheader("Questions to Ask of the Data")
    st.write("What would you like to know about crime?")
    questionOptions = {"View all crimes in a particular neighborhood": 1, "Compare crimes across neigborhoods": 2,
                       "Compare Crime Frequency between 2021 and either 2020 or 2019": 3,
                       "Location of Crimes": 4}
    questionSelection = st.selectbox('Select a question to answer.', list(questionOptions.keys()))

    if questionSelection == "View all crimes in a particular neighborhood":
        view_district = st.sidebar.selectbox("What neighborhood are you interested in?", neighborhoods)
        filtered_df = filter_df_by_district(complete_2021, view_district)
        filtered_df = filter_bar(filtered_df)
        count_df = get_frequency_totals_one_column(filtered_df, 2021)
        count_df.columns = ['offense', 'frequency']

        top_n = st.text_input("How many of the top locations would you like to see?", 10)
        top_n = int(top_n)
        count_df = count_df.head(top_n)
        total_count = count_df['frequency'].sum()
        count_df['total'] = total_count
        count_df['percent'] = (count_df['frequency'] / count_df['total'] * 100)

        # input_col, pie_col = st.beta_columns(2)
        fig = px.pie(count_df, values='frequency', names='offense', title="Crime in " + view_district)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.dataframe(count_df)
        st.write(fig)

    elif questionSelection == "Compare crimes across neigborhoods":

        data_categories.insert(0, 'all')
        crime_family = st.sidebar.selectbox("What category of crimes would you like to look at?", data_categories)
        if crime_family != 'all':
            filtered_df = filter_df_by_crime(complete_2021, data_type.get(crime_family))

        else:
            filtered_df = complete_2021

        filtered_df = filter_bar(filtered_df)

        crimes = get_unique_column_values(filtered_df, 'offense')
        crimes = sorted(crimes)
        st.subheader("Select Crime to Compare across Districts")
        crime = st.selectbox("What crime would you like to see data for?", crimes)
        seaborn_bar_chart(filtered_df, crime)

    elif questionSelection == "Compare Crime Frequency between 2021 and either 2020 or 2019":
        df_2020 = process_file_data(fileNames[1], district_df, filter_out_data, crime_keywords)
        df_2019 = process_file_data(fileNames[2], district_df, filter_out_data, crime_keywords)

        final_2021, df_2020, df_2019 = multi_year_filter(complete_2021, df_2020, df_2019, neighborhoods)
        st.subheader("Comparing Crime changes between 2019 and 2021")
        st.info("Note that the date range for 2019 is the 2nd half of the year.  2021 "
                "only includes the first half of the year so I filtered the 2020 data set "
                "to include only data from the 1st half of the year.  Also, a sample data set "
                "of 7000 records was used for 2021.  So I created a sample data file for 2020"
                "and 2019 with the same number of records")
        group_by_bar_chart(final_2021, df_2020, df_2019)

    elif questionSelection == "Location of Crimes":
        crime_family = st.sidebar.selectbox("What category of crimes would you like to look at?", data_categories)
        filtered_df = filter_df_by_crime(complete_2021, data_type.get(crime_family))
        filtered_df = filter_bar(filtered_df)
        crime_map = folium.Map(location=[42.35866, -71.05674], zoom_start=13, width=700, height=500,
                               control_scale=True)
        folium.TileLayer('stamentoner').add_to(crime_map)
        for (index, row) in filtered_df.iterrows():
            folium.Marker(location=[row.loc['lat'], row.loc['long']],
                          popup=row.loc['offense'] + ' ' + row.loc['District Name'] + ' ' + row.loc['street'],
                          tooltip='click for info').add_to(crime_map)
        folium_static(crime_map)


main()



