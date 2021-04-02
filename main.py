import symbol_retriever as sr
import pandas as pd
import logging
import prediction

def main():
  logging.basicConfig(level=logging.INFO)

  # read workflow

  logging.info("read csv file with all symbols")
  stock_data = pd.read_csv("data/nasdaq_screener_1617395525969.csv")
  logging.info(stock_data.columns.tolist())
  logging.info(stock_data.shape)
  
  # apply symbol filter
  # sample data (market cap > 10 billion, volume > 100000)
  filtered_stock_data = stock_data[(stock_data["Market Cap"] > 50000000) & (stock_data["Country"] == "United States")]
  logging.info(filtered_stock_data.shape)

  # retrieve historical data
  
  # do prediction
  prediction.predict("PFE")

if __name__ == "__main__":

  main()
  
