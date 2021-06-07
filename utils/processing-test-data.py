import os
import re
import pandas as pd

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]


if __name__ == "__main__":
    test_folder = os.path.join(
        os.getcwd(), "COVID-19", "csse_covid_19_data", "csse_covid_19_daily_reports_us"
    )
    all_files = os.listdir(test_folder)

    # ? Extract file_name from 04/01/2021 to 04/30/2021
    pattern = "04-\w+-2021.csv"
    test_date = []
    for file_name in sorted(all_files):
        index = re.search(pattern=pattern, string=file_name)
        if index:
            test_date.append(file_name)

    # ? Merge data
    test_dataframe = None
    for file_name in test_date:
        print(file_name)
        dataset = pd.read_csv(os.path.join(test_folder, file_name))

        # ? The dataset has more than 50 states => need to filter
        dataset = dataset[dataset["Province_State"].isin(states)]
        dataset["Date"] = file_name.replace(".csv", "")

        if test_dataframe is None:
            test_dataframe = dataset
        else:
            test_dataframe = test_dataframe.append(dataset, ignore_index=True)

    test_dataframe = test_dataframe.drop(
        columns=[
            "Country_Region",
            "Last_Update",
            "Lat",
            "Long_",
            "FIPS",
            "UID",
            "ISO3",
            "Hospitalization_Rate",
            "People_Hospitalized",
        ]
    )
    test_dataframe.to_csv("test_data.csv")
