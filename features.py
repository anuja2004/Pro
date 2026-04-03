# # After preprocessing, check the processed data files
# python3 -c "
# import pandas as pd
# baf = pd.read_csv('data/processed/baf_processed.csv')
# ieee = pd.read_csv('data/processed/ieee_processed.csv')
# print('BAF features:', list(baf.columns))
# print('IEEE features:', list(ieee.columns))
# print('Common features:', set(baf.columns) & set(ieee.columns))
# "