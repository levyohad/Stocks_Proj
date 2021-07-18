import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools as it
import math
import yfinance as yf
from typing import List
import matplotlib.pyplot as plt
import time


class PortfolioBuilder:

    def __init__(self):
        self.tickers_list = []
        self.amount_of_stocks = 0
        self.my_df = []
        self.my_portfolio = []
        self.x_t = []
        self.total_x_t = []
        self.legend = []
        self.start_date = ''
        self.end_date = ''
        self.dates = []

    def calculate_wealth(self, last_wealth):
        return last_wealth*(np.dot(self.my_portfolio, self.x_t))

    def calculate_wealth_given_port(self, portfolio):
        return self.total_x_t * np.array(portfolio)

    def calculate_x_t_given_port(self,stockrates_beginning, stockrates_end, portfolio):
        return np.dot((stockrates_beginning / stockrates_end), portfolio)

    def sanity_positivity_check(self, portfolio):
        if self.sanity_check(portfolio):
            for i in range(len(portfolio)):
                if round(portfolio[i], 7) < 0:
                    break
            else:
                return True
        else:
            return False

    def sanity_check(self, portfolio):
        the_sum = 0
        for fractionvalue in portfolio:
            the_sum += fractionvalue
        if 1.01 > the_sum > 0.99:
            return True
        else:
            return False

    def normalize(self, portfolio):
        the_sum = 0
        for fractionvalue in portfolio:
            the_sum += fractionvalue
        j = 0
        for fractionvalue in portfolio:
            portfolio[j] = fractionvalue / the_sum
            j += 1

    def calculate_x_t(self, stockrates_today, stockrates_yest):
        """ calcultes the x_t of today. meaning what was today's stock change"""
        self.x_t = stockrates_today / stockrates_yest
        return self.x_t

    def get_daily_data(self, tickers_list: List[str],
                       start_date: date,
                       end_date: date = date.today()
                       ) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """
        self.tickers_list = tickers_list
        self.legend += tickers_list
        self.amount_of_stocks = len(tickers_list)
        self.start_date = str(start_date)
        self.end_date = str(end_date)

        try:
            self.my_df = web.DataReader(self.tickers_list, start = start_date, end= end_date, data_source="yahoo")['Adj Close']
        except:
            raise ValueError
        for stock in self.my_df:
            if not (self.my_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all()):
                raise ValueError
        return self.my_df

    def find_legal_permutations(self, port_quant):
        all_valid_fractions = [fraction/port_quant for fraction in range(port_quant+1)]
        legal_permutations = [list(combi) for combi in it.product(all_valid_fractions, repeat=self.amount_of_stocks) if self.sanity_positivity_check(combi)]
        return legal_permutations

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        self.my_portfolio = [1 / self.amount_of_stocks for stock in range(self.amount_of_stocks)]
        list_of_wealth_attained = [1.0]
        my_array = np.array(self.my_df)  # creating a copy of the DataFrame as a numpy array
        legal_permutations = self.find_legal_permutations(portfolio_quantization)  # Creates all legal permutation.


        for day in range(1, len(my_array)):  # For every day of trading
            temp_portfolio = [0*stock for stock in range(self.amount_of_stocks)]  # initiates and empty list.
            self.calculate_x_t(my_array[day], my_array[day-1])
            list_of_wealth_attained.append(float(self.calculate_wealth(list_of_wealth_attained[-1])))

            for permut in legal_permutations: # Here we calculate b_t+1
                self.total_x_t = self.calculate_x_t_given_port(my_array[day], my_array[0], permut)
                temp_portfolio += self.calculate_wealth_given_port(permut)
            self.normalize(temp_portfolio)
            self.my_portfolio = temp_portfolio.copy()
        return list_of_wealth_attained

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.2) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """
        #self.initiate_default_porfolio(self.amount_of_stocks)
        self.my_portfolio = [1/self.amount_of_stocks for stock in range(self.amount_of_stocks)]
        list_of_wealth_attained = [1]
        my_array = np.array(self.my_df)  # creating a copy of the DataFrame as a numpy array
        temp_portfolio = [1 for x in range(self.amount_of_stocks)]
        for day in range(len(my_array)-1):  # For every day of trading
            self.calculate_x_t(my_array[day+1], my_array[day])
            for stock in range(self.amount_of_stocks):
                temp_portfolio[stock] = self.my_portfolio[stock] * math.exp((learn_rate*self.x_t[stock])/(np.dot(self.my_portfolio, self.x_t)))
            if not self.sanity_check(temp_portfolio):
                self.normalize(temp_portfolio)
            list_of_wealth_attained.append(float(self.calculate_wealth(list_of_wealth_attained[-1])))
            self.my_portfolio = temp_portfolio.copy()
        return list_of_wealth_attained

    def neutral(self):
        """
        calculates the profit earned in a "no-buy,no-sell" strategy. Used on order to compare other strategies.
        :return: vector of wealth for each day of trading.
        """
        pb.my_portfolio = [1/self.amount_of_stocks for stock in range(self.amount_of_stocks)] # Dividing protfolio to shares
        list_of_wealth_attained = [1]
        my_array = np.array(self.my_df)  # creating a copy of the DataFrame as a numpy array
        temp_portfolio = [1 for x in range(self.amount_of_stocks)]
        for day in range(len(my_array) - 1):  # For every day of trading
            self.calculate_x_t(my_array[day + 1], my_array[day])
            list_of_wealth_attained.append(float(self.calculate_wealth(list_of_wealth_attained[-1])))
        return list_of_wealth_attained

    def gradient(self, tickers, start_date=date(2020, 1, 1), end_date=date.today()):
        """a quick function to see the profit could've been gained in a specific period of time.
        the function must be used after using get_daily_data function.
        unlike other functions, this one plots automatically with legend and ticks."""

        for i in range(len(tickers)):
            df = self.get_daily_data([tickers[i]], start_date, end_date)
            grad = self.find_exponential_gradient_portfolio()
            plt.plot(grad)
        plt.xlabel("Trading Days Since " + str(start_date))
        plt.ylabel("Growth")
        plt.xticks()
        plt.yticks()
        title = "Growth of 1$ Investment Since {}".format(start_date)
        plt.title(title)
        plt.grid(color='black', linestyle='--', linewidth= 0.4)
        plt.legend(self.legend)
        plt.show()


if __name__ == '__main__':
    pb = PortfolioBuilder()
    pb.get_daily_data(["CCL", "AAL", "TSLA","BA"], date(2020, 1, 1), date(2020, 12, 15))

    start_time = time.time()
    a = pb.find_exponential_gradient_portfolio(0.6)
    print("expo algo takes %s seconds " % (time.time() - start_time))

    start_time = time.time()
    b = pb.find_universal_portfolio(30)
    print("uni algo takes %s seconds " % (time.time() - start_time))

    start_time = time.time()
    c = pb.neutral()
    print("neutral calc takes %s seconds" % (time.time() - start_time))

    plt.plot(a, label="expo", color="blue")
    plt.plot(b, label="universal", color="orange")
    plt.plot(c, label="neutral", color="gray")
    plt.ylabel("Growth")
    plt.xlabel("Trading Days Since " + str(pb.start_date), size=14)
    plt.yticks()
    title = "Growth of 1$ Investment of {}".format(pb.tickers_list)
    plt.title(title)
    plt.grid(color='black', linestyle='--', linewidth=0.4)
    plt.legend()
    plt.show()




