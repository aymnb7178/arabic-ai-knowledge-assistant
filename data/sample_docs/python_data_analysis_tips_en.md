# Python Data Analysis Tips

Practical habits for analysing tabular data with Python and pandas. Original notes written for this demo.

## Look at the data first

Start every analysis by inspecting the table. Use `df.head()`, `df.info()` and `df.describe()` to see column types, missing values and ranges. Most analysis bugs come from data that was not what you assumed, such as numbers stored as text or duplicated rows.

## Handle missing values deliberately

Count missing values with `df.isna().sum()` before deciding what to do. You can drop rows with `dropna()`, fill them with `fillna()` or keep them as a separate category. Write down the choice, because filling with the mean changes the distribution of the column.

## Fix types and dates

Convert date strings with `pd.to_datetime()` and set categorical columns to the `category` dtype. Correct types make grouping faster, use less memory and allow date functions such as resampling by month.

## Group and aggregate

`groupby()` followed by `agg()` is the core pandas pattern for summaries, for example total sales per region. Use named aggregation to give the output columns clear names, and call `reset_index()` when you need a plain table.

## Merge tables carefully

Use `merge()` with an explicit `how` argument and the `validate` option, for example `validate="many_to_one"`, so pandas raises an error if the keys are not unique as expected. After a merge, compare the row count with the original, because an unexpected increase usually means duplicate keys.

## Avoid loops over rows

Looping over rows with `iterrows()` is slow. Prefer vectorised column operations, `np.where()` for conditions and `map()` for lookups. Vectorised code is also shorter and easier to read.

## Visualise before modelling

A histogram, a box plot or a scatter plot often shows outliers and skew that summary numbers hide. Label axes and units, and keep one message per chart. With matplotlib, call `plt.tight_layout()` before saving a figure to avoid cut-off labels.

## Make results reproducible

Set random seeds for sampling and model training, keep the code in a script or notebook that runs from top to bottom, and save the cleaned data separately from the raw data. Never edit raw files by hand.

## Validate before sharing

Check totals against a known number, such as the sum of sales in the source system. Test a few rows manually and add simple assertions in the code, for example that an identifier column has no duplicates.
