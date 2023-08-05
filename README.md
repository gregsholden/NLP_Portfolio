#  Can NLP Techniques Be Used To Enhance Investment Portfolios?

## Introduction

This project analyzes whether NLP analysis can be used to enhance the returns of a investment portfolio which otherwise relies solely on traditional technical indicators.  Our NLP analyses are conducted on text data from the Federal Open Market Committiee's meeting minutes ("FOMC Minutes").  Our results show that NLP can contribute to higher risk-adjusted investment returns.

## Background

Much of the existing analysis combining NLP and investing uses sentiment parsed from social media to make predictions about future prices of a SINGLE STOCK (Tesla, Apple, etc.).

But this approach has its limitations:

First, investors rarely deploy all their investable cash into a single stock. 

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

## List of Files

This repository contains the following files:

### NLP Analysis

mins_dataset_2008_2023.csv:  Contains 123 FOMC Minutes transcripts downloaded from the FedTools python package.  These are the inputs for the NLP models.

NLP_Final.ipynb:  A Jupyter Notebook which performs multiple NLP analyses on the FOMC Minutes, then translates the outputs into numerical vectors that can be used as feature inputs in the Portfolio model.

NLP_output_NEW_TEST_DATE.csv:  The output from the Jupyter Notebook of all NLP analyses except for Universal Sentence Encoder and Named Entity Recognition.  Consists of numerical vector representations for each of the FOMC Minutes documents.  Used in the Portfolio model.  

use_scores_NEW_TEST_DATE.csv:  Numerical similarity scores between subsequent FOMC Minutes documents using the Universal Sentence Encoder model.  Used in the Portfolio model.

NLP_FOMC_people_newtext.csv:  Counts of how many times each FOMC Board Member is mentioned in each of the 123 FOMC Minutes documents. Used in the Portfolio model.

## Portfolio Analysis:

Portfolio_model.ipynb:


## Getting Started

### Part 1 of 2:  Running the NLP Models

Inputs:  FOMC Minutes transcripts are the primary inputs to the NLP models.  This info can be accessed in two ways:

1.	Using the FedTools python package to set a custom date range, and then downloading new data:

<img width="359" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/7daf2bf7-26af-42e8-a147-676ec6be5a2d">

2.	 Alternatively, a pre-existing dataset containing all FOMC Minutes from 2008-23 YTD can be retrieved from a .csv file within the github repository:

   <img width="464" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/a4bf4577-f78c-4767-8b62-42fa9c1c053d">

In either case, this process will return a pandas df that looks like this:

 <img width="340" alt="image" src="https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/c26a16db-9f1e-43ae-ba23-88e5d9ecea69">

Outputs:   The NLP model notebook creates 3 separate .csv files which can be used in the Portfolio Optimization model: (a) one containing vectors from the Universal Sentence Encoder (2) a second file containing results of the Named Entity Recognition modeling, and (3) a third file containing the results of all other NLP analyses.

## Part 2 of 2:  Running the Portfolio Optimization Models


## Requirements

### PIPREQS FILE


## Project Structure

DIAGRAM SHOWING HOW IT WORKS?  MAY NOT NEED THIS

## Sample Outputs - Portfolio Model

Code samples
Output charts – stock price returns

