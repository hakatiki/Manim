from manim import *
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class EfficientFrontier(Scene):
    def construct(self):
        # Step 1: Download data
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        start_date = '2023-01-01'
        end_date = '2024-01-01'
        data = self.load_yfinance_data(tickers, start_date, end_date)
        returns = self.calculate_returns(data)
        
        # Step 2: Simulate portfolios
        results = self.efficient_frontier(returns, num_portfolios=20)  # Increased number of portfolios for better visualization
        
        # Step 3: Calculate exact efficient frontier
        desired_returns, exact_risks = self.exact_efficient_frontier(returns)
        
        # Step 4: Plot results using Manim
        self.plot_efficient_frontier(results, desired_returns, exact_risks)

    def load_yfinance_data(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data

    def calculate_returns(self, data):
        return data.pct_change().dropna()

    def portfolio_statistics(self, weights, returns, covariance):
        portfolio_return = np.dot(weights, returns.mean()) * 252  # Annualized return
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)  # Annualized volatility
        return portfolio_return, portfolio_volatility

    def efficient_frontier(self, returns, num_portfolios=10000, risk_free_rate=0.0):
        cov_matrix = returns.cov() * 252  # Annualize the covariance matrix
        expected_returns = returns.mean() * 252  # Annualize the returns
        num_assets = len(expected_returns)
        results = np.zeros((4, num_portfolios))

        for i in range(num_portfolios):
            weights = np.random.randn(num_assets)
            weights /= np.sum(weights)        
            portfolio_return, portfolio_volatility = self.portfolio_statistics(weights, returns, cov_matrix)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio

        return results

    def exact_efficient_frontier(self, returns):
        mu = returns.mean() * 252  # Annualize the daily returns
        Sigma = returns.cov() * 252  # Annualize the daily covariances
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(len(mu))
        U = np.vstack([mu, ones]).T
        M = U.T @ (Sigma_inv @ U)
        M_inv = np.linalg.inv(M)
        desired_returns = np.linspace(0.35,0.66, 100)
        exact_risks = []

        for desired_return in desired_returns:
            u = np.array([desired_return, 1])
            w_star = Sigma_inv @ U @ M_inv @ u
            portfolio_risk = np.sqrt(np.dot(w_star.T, np.dot(Sigma, w_star))) * np.sqrt(252)
            exact_risks.append(portfolio_risk)

        return desired_returns, exact_risks

    def plot_efficient_frontier(self, results, exact_returns, exact_risks):
        min_x, max_x = 2.6, 6.5
        min_y, max_y =0.35,0.65
        steps_x = 0.2
        steps_y = 0.04
        lables_x =  np.arange(min_x, max_x,steps_x) 
        lables_y =  np.arange(min_y, max_y,steps_y) 
         
        axes = Axes(
            x_range=[min_x, max_x, steps_x],
            y_range=[min_y, max_y, steps_y],
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": lables_x},
            y_axis_config={"numbers_to_include": lables_y},
            tips=False,

        )        
        axes.scale(0.9)  # Adjust the scale factor as needed

        axes.shift(UP*0.5 + RIGHT*0.5)
        x_label = Tex("Volatility").next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Tex("Expected Return").next_to(axes.y_axis, LEFT, buff=-1).rotate(90 * DEGREES)

        self.add(axes, x_label, y_label)

        # counter = Integer(0).to_corner(UP)#.shift(RIGHT*8)
        # counter_label = Tex("Portfolios Added: ").next_to(counter, LEFT)
        # self.add(counter_label, counter)

        # Plot simulated portfolios
        sharpe_ratios = results[2]
        min_sharpe = np.min(sharpe_ratios)
        max_sharpe = np.max(sharpe_ratios)
        normalized_sharpes = (sharpe_ratios - min_sharpe) / (max_sharpe - min_sharpe)
        
        num_dots_middle = 15
        num_dots = len(results[0])

        scatter_slow = VGroup()
        for i in range(10):
            if results[1][i] > 0.65 or results[1][i] < 0.36 or  results[0][i] > 6.4:
                continue
            dot = Dot(point=axes.c2p(results[0][i], results[1][i]), color=interpolate_color(RED, GREEN, normalized_sharpes[i]), radius=0.08)
            scatter_slow.add(dot)
            
            # scatter_slow.add( counter)
            # self.play(Create(dot), counter.animate.set_value(counter.get_value()), run_time=0.5)
        self.add(axes, scatter_slow)
        self.play(Create(scatter_slow),  run_time=5)
        
        scatter_middle = VGroup()
        for i in range(10, num_dots_middle):
            if results[1][i] > 0.65 or results[1][i] < 0.36 or  results[0][i] > 6.4  :
                continue
            dot = Dot(point=axes.c2p(results[0][i], results[1][i]), color=interpolate_color(RED, GREEN, normalized_sharpes[i]), radius=0.08)
            scatter_middle.add(dot)
        #     counter.increment_value()
        #     self.play(Create(dot), counter.animate.set_value(counter.get_value()), run_time=0.05)
        self.add(axes, scatter_middle)
        self.play(Create(scatter_middle),  run_time=5)

        scatter_fast = VGroup()
        for i in range(num_dots_middle, num_dots):
            if results[1][i] > 0.65 or results[1][i] < 0.36 or  results[0][i] > 6.4  :
                continue
            dot = Dot(point=axes.c2p(results[0][i], results[1][i]), color=interpolate_color(RED, GREEN, normalized_sharpes[i]), radius=0.08)
            scatter_fast.add(dot)
            # counter.increment_value()
            # self.play(Create(dot), counter.animate.set_value(counter.get_value()), run_time=0.01)
        self.add(axes, scatter_fast)
        self.play(Create(scatter_fast),  run_time=10)

        scatter = scatter_fast + scatter_slow + scatter_middle

        # Plot exact efficient frontier
        exact_points = VGroup()
        for x, y in zip(exact_risks, exact_returns):
            point = Dot(point=axes.c2p(x, y), color=RED, radius=0.05)
            exact_points.add(point)
        self.play(Create(exact_points), run_time=0.5)
        
        exact_frontier = VMobject()
        exact_frontier.set_points_as_corners([axes.c2p(x, y) for x, y in zip(exact_risks, exact_returns)])
        exact_frontier.set_color(RED)

        self.add(exact_points, exact_frontier)
        self.play(Create(exact_frontier), run_time=2)
        self.play(FadeOut(scatter))#,íFadeOut(counter_label) ,FadeOut(counter))
        # self.play(FadeOut(counter_label), )
        
        self.wait(2)

