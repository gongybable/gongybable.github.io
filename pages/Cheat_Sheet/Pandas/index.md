# Pandas

## Series
<details>

```python
'''
A Series is a one-dimensional labelled array-like object. It contains two arrays:
 - Index, or labels
 - The actual data
'''
S = pd.Series([11, 28, 72, 3, 5, 8])
S.index         # with default index 0, 1, 2, ... 
S.values        # the values

S = pd.Series(val_arr, index=index_arr) # user defined indices, can be a list of strings

S1 + S2         # add values of two series with the same indices;
                # value set to NaN for indice only appear in one series

S[indice1, indice2]     # access values in series by indices
np.sin((S + 3) * 4)     # can apply operations on series as well
S[S>20]                 # only return the series (values) meet the condition

cities = { "London": 8615246, "Berlin": 3562166, "Milan": 1350680 }
city_series = pd.Series(cities) # creating a series
"London" in city_series         # True
my_cities = ["London", "Paris", "Berlin"]
my_city_series = pd.Series(cities, index=my_cities)
# London     8615246
# Paris      NaN
# Berlin     3562166
# dtype: float64
my_city_series.isnull()     # or we can use my_city_series.notnull()
# London       False
# Paris        True
# Berlin       False
# dtype: bool
my_city_series.dropna()     # to drop NaN rows;
my_city_series.fillna(0).astype(int)    # fill NaN, and now we can convert back to int
```
</details>

## DataFrame
<details>

```python
'''
A DataFrame can be seen as a concatenation of Series, each Series having the same index.
'''
df = pd.concat([s1, s2, s3], axis=1)
df.columns = col_arr        # getting a df with column names

cities = {
    "name": ["London", "Berlin", "Madrid"],
    "population": [8615246, 3562166, 3165235],
    "country": ["England", "Germany", "Spain"]
}
city_frame = pd.DataFrame(cities)   # getting df from dict
city_frame.columns.values           # getting cols: array(['country', 'name', 'population'], dtype=object)
pd.DataFrame(cities,
             columns=["name",
                      "country",
                      "population"])                # reorder cols
city_frame = pd.DataFrame(cities, index=my_index)   # custom index
city_frame.reindex(index=my_index, 
                   columns=['country', 'name', 'population'])   #reindex
city_frame = pd.DataFrame(cities,
                          columns=["name", "population"],
                          index=cities["country"])  #use the col as index, or use city_frame.set_index("country")
city_frame.loc[["Germany", "France"]]               # return all the rows with index == "Germany" or "France"
city_frame["population"].sum()      # getting the sum
city_frame.insert(loc=idx, column='area', value=area)   #insert a new col

df.T        # transpose data
df.replace(val1, val2)  # replace val1 in the df with val2
```
</details>