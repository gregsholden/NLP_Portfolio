#  Can NLP Techniques Be Used To Enhance Investment Portfolios?

## Introduction

This project analyzes whether NLP analysis can be used to enhance the returns of a investment portfolio which otherwise relies solely on traditional technical indicators.  Our NLP analyses are conducted on text data from the Federal Open Market Committiee's meeting minutes ("FOMC Minutes").  Our results show that NLP can contribute to higher risk-adjusted investment returns.

## Background

Much of the existing analysis combining NLP and investing uses sentiment parsed from social media to make predictions about future prices of a SINGLE STOCK (Tesla, Apple, etc.).

But this approach has its limitations:

First, investors rarely deploy all of their investable cash into a single stock. 

Second, outside of a few high-profile technology and consumer goods companies, very few single stocks generate enough social media dialogue to be useful for NLP modeling.

So how did we address these issues:

First, we focused on a PORTFOLIO of investments rather than a single stock.  We started with a base case investment portfolio consisting of 60% equity (in the form of "SPY" - an Exchange-Traded Fund ("ETF") mirroring the S&P 500 Index) and 40% debt investments (in the form of "TLT", an ETF mirroring long-dated U.S Treasury bonds).  

This "60/40" portfolio is a commonly-used investing benchmark that balances the upside potential of stocks with the stability of bonds.

Second, we use FOMC Minutes as the basis for our NLP modeling efforts.  These documents – released 8 times each year – provide detailed macroeconomic commentary on the state of the U.S. economy.  Releases are highly-anticipated and carefully scrutinized by investors, who search for miniscule clues about potential changes in economic (and therefore investing) conditions.

## Investment Strategy

Our approach compares the investment return of TWO PORTFOLIOS – one that relies solely on traditional stock market technical indicators to make investment decisions, and a second that uses these same technical indicators BUT ALSO INCLUDES NLP SIGNALS generated from different models.

## Target Audience

1.  Investors who are interested in harnessing the power of NLP to potentially increase portfolio returns while being mindful of risk.

2.  Machine Learning enthusiasts interested in comparing the relative performance of a series of NLP (and ML) models on the same dataset.

## Description of Dataset

Our dataset consists of FOMC Minutes from 2008 through June of 2023.   Upon the completion of each FOMC meeting a short summary is produced.   Approximately 3 weeks later a more extensive summary of the meeting (the FOMC Minutes) is released.  FOMC Minutes documents average ~10,000 words each.  Given our evaluation horizon of 2008 -  June 2023, this results in 123 documents.

## Data Access

FOMC Minutes documents are freely available to download from the FedTools Python packag.  Historical stock price and technical indicator information is also freely avaialble to download from yFinance and Ta-lib.

## List of Files

This repository contains the following files:

### NLP Analysis

mins_dataset_2008_2023.csv:  Contains 123 FOMC Minutes transcripts downloaded from the FedTools python package.  These are the inputs for the NLP models.

NLP_Analyis_Final.ipynb:  A Jupyter Notebook which performs multiple NLP analyses on the FOMC Minutes, then translates the outputs into numerical vectors that can be used as feature inputs in the Portfolio Optimization models.

NLP_output_NEW_TEST_DATE.csv:  The output from the Jupyter Notebook of all NLP analyses except for Universal Sentence Encoder and Named Entity Recognition.  Consists of numerical vector representations for each of the FOMC Minutes documents.  Used in the Portfolio Optimization models.  

use_scores_NEW_TEST_DATE.csv:  Numerical similarity scores between subsequent FOMC Minutes documents using the Universal Sentence Encoder model.  Used in the Portfolio Optimization model.

NLP_FOMC_people_NEW_TEST_DATE.csv:  Counts of how many times each FOMC Board Member is mentioned in each of the 123 FOMC Minutes documents. Used in the Portfolio Optimization models.

### Transformers & NLP Data Aggregation:

Transformer_FED.ipynb: Due to the computational demands of transformer LLM and a reliance on CUDA, the transformer sentiment analysis is contained in its own sub folder with a batch pre populated vector array stored as an output. Transformer_FED.ipynb extracts FOMC meeting transcripts and batch scores them for sentiment. A pre populated output of FOMC sentiment is located in a sub folder titled sentiment_minutes. 

FED_NLP_Combine.ipynb: Aggregates all NLP outputs into a singular dataframe. This notebooks inputs are sentiment_minutes.. (multiple batch files), use_scores_07_20,NLP_output_07_19,FOMC_people_07_19. This noteboot outputs the an aggregated csv file titled Full_Fed_Minutes.csv. All outputs and inputs are located in a sub folder titled Batch_Transformer_files.

### Portfolio Analysis:

ETF_data_fetch: Extracts ETF ticker data along with technical indicators for later use in downstream classification models. This file also combines dividend distributions with ETF prices to calculate total returns. It also exports a .csv file for use in the downstream Trading_Window.ipynb "titled test_60_40_advanced.csv". Users have the ability to pick their desired ETFs within this notebook.

Trading_Window.ipynb: This file combines both the ETF ticker and technical data with the vectors created from upstream NLP models. The files required to run this script are test_60_40_advanced.csv (stock ticker trading test data), Full_Fed_minutes_test.csv which is comprised of combined NLP metric vectors and NLP_FOMC_People.csv which provides named entity recognition NLP data. Trading_window.ipynb combines the financial data and NLP data into a singular dataframe. Once combined, this script calculates the 5 day average return values from FOMC meetings to provide point estimates to the downstream classification model with respect to bull and bear markets. This script also plots historical subsector bull and bear returns with respect to the S&P 500 and exports a dataset called Fed_Window_df.ipynb which extracts all of the 5 day averages around FOMC releases dating back to 2008. 

Classifiers.ipynb: This script runs the final 3 classifiers used in the NLP portfolios rebalancing strategy. This file performs a random grid search through a Monte Carlo analysis for a Logistic Regression, Random Forest, and XGBoost classifier. It takes in the previously aggregated Fed_Window_df.ipynb for use as training and testing data and exports the best classifiers predictions for a desired trade window. This file is currently set to train on data from 2008 to 2019 and tests on data from 2020 to 2023. The training data was utilized for A/B testing to determine the best model and NLP combinations. A static .csv file which was generated from this notebook is provided in this files sub folder called nlp_mc_final.csv which plots the batch A/B testing findings. Classifiers_Final also produces a final prediction dataframe which was generated from the best NLP and Classifier model combination (XGBoost) called Predictions (we placed a static Predictions_best_tune.csv file in the sub folder to prevent overwriting of our best findings). This file is used downstream in the Performance notebook to generate trade signals. 

Performance.ipynb: This notebook takes in the trade signals generated by the upstream Classifiers_Final notebook and uses them to rebalance an ETF portfolio around each FOMC release for a desired trading range. This file takes in the Predictions.csv previously discussed in the Classifiers.ipynb and also takes in test_60_40_advanced.csv for historical ETF prices for plotting purposes only. This file produces a list of the top ETF equity and debt combinations with respect to total return through a backtesting function. It also plots 3 portfolio scenarios, a passive portfolio with no adjustments throughout the backtest, a portfolio with a combination of NLP and financial metrics, and a portfolio with only financial metrics. The latter two portfolios mentioned above are provided the ability to rebalance after each FOMC release. Performance.ipynb also measures each portfolios risk adjusted performance throughout a desired trading range by calculating Sharpe ratios for each. 



## Getting Started

### Part 1 of 2:  Running the NLP Models

Inputs:  FOMC Minutes transcripts are the primary inputs to the NLP models.  This info can be accessed in two ways:

1.	Using the FedTools python package to set a custom date range, and then downloading new data:

<img width="359" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/7daf2bf7-26af-42e8-a147-676ec6be5a2d">

2.	 Alternatively, a pre-existing dataset containing all FOMC Minutes from 2008-23 YTD can be retrieved from a .csv file within the github repository:

   <img width="464" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/a4bf4577-f78c-4767-8b62-42fa9c1c053d">

In either case, this process will return a pandas df that looks like this:

 <img width="340" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/c26a16db-9f1e-43ae-ba23-88e5d9ecea69">

Outputs:   The NLP model notebook creates 3 separate .csv files which can be used in the Portfolio Optimization models: (a) one containing vectors from the Universal Sentence Encoder (2) a second file containing results of the Named Entity Recognition modeling, and (3) a third file containing the results of all other NLP analyses.

## Part 2 of 2:  Running the Portfolio Optimization Models

## Requirements: 
The portfolio analysis section is reliant on the combined aggregate .csv file generated from the Transformer NLP folder titled Full_Fed_Minutes_test.csv. 
The Yahoo Finance (yfinance) library is required to extract ticker data. 

1. Run ETF_Data_Fetch to retrieve desired ticker data and generate the test_60_40_advanced.csv file. Ticker inputs can be modified in the Equity_Tickers & Debt_Tickers inputs at the top of the file. 


<img width="700" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/1784be78-0e4b-420f-84c7-7efed73e6c69">


2. Run the Trading_window_Final.ipynb with the upstream generated test_60_40_advanced.csv file along with the Full_Fed_Minutes_test.csv file produced from the previous NLP vector aggregation the NLP_FOMC_People.csv named entity recognition .csv file ( Full samples of both NLP .csv files is provided in the sub folder to allow this section to run as a standalone from the upstream NLP data generation portion. Verify the Ticker_list, bond_list, and equities_list match the ETF tickers generated in the upstream ETF_Data_Fetch notebook. This notebook produces an output called Fed_Window_df.csv used in the Classifiers downstream notebook.

   
<img width="700" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/9080eb21-e60f-4e49-90e3-f99730d38968">

<img width="750" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/74e7e371-910d-4667-a23b-e9a4560ecfa7)">


3. Run Classifiers.ipynb with the Fed_Window_df.csv as input ( A pre populated .csv is provided in a sub folder). Specify which NLP metrics to include to be evaluated in model performance. Run the notebook to conduct a Monte Carlo analysis for all 3 classifiers. After A/B testing, XGBoost is considered to be the superior classifier for this task across all NLP metrics, and is the default model for bull/bear prediction. Additionally, this notebook uses previous Monte Carlo data from nlp_mc_Final which was batch processed. These findings are plotted within this notebook. This notebook outputs the predictions of the provided classifier and chosen NLP metrics as Predictions_.csv ( A final version is provided as Predictions_best_tune.csv).

<img width="750" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/0be0a659-3abb-492f-9f18-43fbcddfbe59">



4. Run the Performance_Final.ipynb with the test_60_40_advanced stock ticker .csv file generated in Step 1 along with the Predictions_.csv generated in step 3. Verify the Ticker_data, equity_tickers and bond_tickers inputs match from Step 1 and choose preferred equity upweighting (bond weighting is calculated as the inverse) This notebook generates optimum pairings of ETF tickers in a tops variable. It also plots aggregate return and calculates Sharpe Ratio. It is currently set to plot the S&P 500 – TLT baseline combination for the NLP, non NLP and passive portfolios as an override. 


Top-Combinations

<img width="200" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/e6391f01-1c7a-4cc1-91fd-f4cdc6e0b991">

Aggregate Return Results


<img width="900" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/af11ae81-cdf9-4f25-9465-c198d9d7dd9c">



Sharpe 

<img width="600" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/103608779/20c6d401-b9fe-4174-be4c-4eca9084165d">



### PIPREQS FILE
PipReqs files for each section of this Project are located in their respective sub folders. 



