import pandas as pd
import logging

def read_local_file(file):
  logging.info("read local file")
  dataframe = pd.read_csv(file)
  return dataframe

def read_remote_file(url):
  logging.info("read remote file")

