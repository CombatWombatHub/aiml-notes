# Using DataCamp

- have access to DataCamp through work
- you can't download the workspace from the site
- if I want to use it as notes, I'll need to get the stuff for it myself

## Extracting DataFrames from DataCamp

1. Run `pd.to_csv` on the DataFrame in the interactive window
2. Use `Shift` + `Down` to select the text
3. Copy and Paste into a `.csv` file
4. Find and Replace `\n` with `/`
5. Activate Regex mode (`.*`)
6. Find and Replace `/` with `\n`
7. remove the extra apostrophes (`'`) from the start and end of the text