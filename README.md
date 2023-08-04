# NLP_Portfolio
This project attempts to fuse NLP data from FOMC minutes to produce signals used for portfolio allocation

Introduction and Background

This library is intended to show how NLP analysis can be used to enhance and optimize an investment portfolio.

Much of the existing analysis combining NLP and investing uses sentiment parsed from social media to make predictions about future prices of a SINGLE STOCK (Tesla, Apple, etc.).

But this approach has its limitations:

1..   Investors rarely deploy all their investable cash into a single stock. 
2…  Outside of a few high-profile technology and consumer goods companies, very few single stocks generate enough social media dialogue to be useful in NLP modeling.

So how did we address these issues:

First, we focused on a PORTFOLIO of investments rather than a single stock.  Our base case portfolio assumes a split of 60% equity and 40% debt investments, which is a common benchmark.

Second, we use documents produced at the Federal Open Market Committee’s (‘FOMC”) periodic meetings as the basis for our NLP modeling efforts.  These documents – released 8 times each year – are hotly-anticipated and carefully scrutinized by investors, who search for clues about potential changes in macroeconomic (and therefore investing) conditions.

Our approach compares the investment return of TWO PORTFOLIOS – one that relies solely on traditional stock market technical indicators to make investment decisions, and a second that uses these same technical indicators BUT ALSO INCLUDES NLP SIGNALS generated from different models.

Target Audience

1.. Investors who are interested in harnessing the power of NLP to potentially increase portfolio returns while being mindful of risk.

2.  Machine Learning enthusiasts interested in comparing the relative performance of a series of NLP (and ML) models on the same dataset.

Description of Dataset

Our primary dataset consists of the minutes taken from meetings of the Federal Open Market Committee (“FOMC Minutes”).  The FOMC meets 8 times each year.  Upon completion of the meeting a short summary is produced.   Approximately 3 weeks later the FOMC Minutes are released.  These documents average ~10,000 words each.  Given our evaluation horizon of 2008 -  June 2023, this results in 123 documents.

INSERT PIC OF DOC SAMPLE??

List of Files

This repository contains the following files:

NLP Analysis

mins_dataset_2008_2023.csv:  Contains FOMC Minutes transcripts downloaded from the FedTools python package.  These are the inputs for the NLP models.

NLP_Final.ipynb:  A Jupyter Notebook which performs multiple NLP analyses on the FOMC Minutes, then translates the outputs into numerical vectors that can be used as features in the Portfolio model.

NLP_output_final.csv:  The output of the NLP_Final Jupyter Notebook.  Used in the Portolfio model.

Portfolio Analysis:

Portfolio_model.ipynb:


Getting Started

Part 1 of 2:  Running the NLP Models

Inputs:  FOMC Minutes transcripts are the primary inputs to the NLP models.  This info can be accessed in two ways:

1.	Using the FedTools python package to set a custom date range, and then downloading new data:

 

2.	 Alternatively, a pre-existing dataset containing all FOMC Minutes from 2008-23 YTD can be retrieved from a .csv file within the github repository:

 

In either case, this process will return a pandas df that looks like this:

 

Outputs:   The NLP model notebook creates 3 separate .csv files which can be used in the Portfolio Optimization model: (a) one containing vectors from the Universal Sentence Encoder modeling (2) a second file containing results of the Named Entity Recognition modeling, and (3) a third file containing the results of all other NLP analyses.


![image](https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/641e3ee7-622d-4f5a-818b-d57be9d69fe9)



Requirements


Project Structure

DIAGRAM SHOWING HOW IT WORKS?  MAY NOT NEED THIS

Sample Outputs

Code samples
Output charts – stock price returns

![image](https://github.com/JPeloquin13/NLP_Portfolio/assets/127001653/8cd3f108-9c39-4dd2-bd95-5c1f54e966f3)

