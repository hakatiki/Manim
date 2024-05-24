from manim import *

class PortfolioWeights(Scene):
    def construct(self):
        title = Text("Understanding Portfolio Weights").scale(1.2)
        self.play(Write(title))
        self.wait(2)
        self.play(title.animate.to_edge(UP))

        # Define table data
        stocks = ["AAPL", "TSLA", "NVDA", "META", "MSFT"]
        stock_prices = [150, 700, 500, 250, 300]
        quantities_owned = [4, 2, 6, 3, 5]
        stock_values = [p * q for p, q in zip(stock_prices, quantities_owned)]

        # Create table rows
        table_data = [
            ["Stocks"] + stocks,
            ["Price"] + [f"${price}" for price in stock_prices],
            ["Quantities"] + [str(quantity) for quantity in quantities_owned],
            ["Value"] + [""] * len(stocks),
            ["Weights"] + [""] * len(stocks)
        ]

        # Create and display the table
        table = Table(
            table_data,
            include_outer_lines=True
        ).scale(0.6)

        self.play(table.create())
        self.wait(2)

        # Animate Value row calculation
        for i in range(len(stocks)):
            price_cell = table.get_cell((2, i + 2))
            quantity_cell = table.get_cell((3, i + 2))
            value = stock_values[i]
            value_text = Text(f" ${value}").scale(0.5)
            value_text.move_to(table.get_cell((4, i + 2)).get_center())

            self.play(Indicate(price_cell), Indicate(quantity_cell))
            self.play(TransformFromCopy(quantity_cell, value_text))
            self.wait(1)
            table.add(value_text)

        self.wait(2)
        
        value_cells = table.get_cell((4, 1))
        # for i in range(0,len(stocks)):
        #     # value_cells += table.get_cell((4, i + 2))
        #     self.play(Indicate(table.get_cell((4, i + 2)), color=RED))
        # table.get_columns()
        # Calculate total portfolio value
        total_value = sum(stock_values)
        total_value_text = Text(f"Total Portfolio Value: ${total_value}").scale(0.7).to_edge(DOWN)
        # text_write = Write(total_value_text)
        self.play(TransformFromCopy(table.get_rows()[3], total_value_text))
        self.wait(2)

        # Calculate and display portfolio weights
        for i in range(len(stocks)):
            weight = (stock_values[i] / total_value)
            weight_text = Text(f"{weight:.2%}").scale(0.5)
            weight_text.move_to(table.get_cell((5, i + 2)).get_center())
            self.play(TransformFromCopy(table.get_cell((4, i + 2)), weight_text))
            table.add(weight_text)

        self.wait(3)

# To run this script, save it as a .py file and execute it with Manim using:
# manim -pql your_script.py PortfolioWeights
